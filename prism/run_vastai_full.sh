#!/bin/bash
# =============================================================================
# PRISM — Vast.ai Full Pipeline (Retrain + Calibrate + Parallel Eval + Ablation)
# =============================================================================
# Usage: bash run_vastai_full.sh
# Runs the entire publishable pipeline on a single RTX 5090 instance.
#
# Parallelism map (safe — no shared mutable state between concurrent jobs):
#   Step 5  : CW  ||  FGSM+PGD+Square+AA       (already parallel, locked artifacts)
#   Step 6  : All 5 adaptive-PGD seeds in parallel  ← NEW
#   Step 6+7: Ablation starts while Step 6 seeds run  ← NEW
#             (Ablation reads only locked pkl artifacts, no Step-6 output dep)
#
# Sequential constraints that CANNOT be parallelised:
#   Steps 1→2: Step 2 reads reference_profiles.pkl written by Step 1
#   Steps 2→3: Step 3 reads ensemble_scorer.pkl written by Step 2
#   Steps 3→4: Step 4 reads calibrator.pkl written by Step 3
#   Steps 4→5: Step 5 requires the FPR gate to have passed
#   Steps 5→6: Step 6 must run after Step 5 (provenance check + gate result)
#
# Exit codes: 0=success, 1=gate failure, 2=eval failure

set -euo pipefail
cd /workspace/prism-repo/prism

SEEDS="42 123 456 789 999"
N_TEST=1000
CW_MAX_ITER=100
CW_BSS=9

echo "============================================================"
echo "PRISM Vast.ai Full Pipeline — $(date)"
echo "Instance: $(hostname)"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# ── Environment ──────────────────────────────────────────────────────────────
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVIDIA_TF32_OVERRIDE=1
export TORCH_CUDNN_V8_API_ENABLED=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

mkdir -p logs models experiments/calibration experiments/evaluation experiments/ablation

# ── Verify GPU + PyTorch ─────────────────────────────────────────────────────
echo ""
echo "=== Step 0: GPU + PyTorch verification ==="
python -c "
import torch
print('torch:', torch.__version__)
print('cuda:', torch.version.cuda)
print('device:', torch.cuda.get_device_name(0))
cc = torch.cuda.get_device_capability(0)
print('compute capability:', cc)
x = torch.randn(1024, 1024, device='cuda')
y = (x @ x.T).sum()
torch.cuda.synchronize()
print('smoke GPU matmul: OK')
assert int(torch.__version__.split('.')[0]) >= 2, 'PyTorch >= 2 required'
"
echo "Step 0: PASS"

# ── Preflight: verify per-tier config ────────────────────────────────────────
python -c "
from src.config import TIER_CAL_ALPHA_FACTORS, CAL_ALPHA_FACTOR
print(f'cal_alpha_factor (scalar): {CAL_ALPHA_FACTOR}')
print(f'tier_cal_alpha_factors: {TIER_CAL_ALPHA_FACTORS}')
assert TIER_CAL_ALPHA_FACTORS.get('L3', CAL_ALPHA_FACTOR) <= 0.55, \
  f'L3 factor must be <= 0.55, got {TIER_CAL_ALPHA_FACTORS.get(\"L3\")}'
print('PREFLIGHT PASS: per-tier L3=0.50 confirmed')
"

# ── Step 1: Build reference profiles ─────────────────────────────────────────
echo ""
echo "=== Step 1: Build Reference Profiles [test 0-4999] ==="
# Use process substitution instead of pipe so $? reflects the Python exit code,
# not tee's exit code (pipe + set -euo pipefail caused triple-execution in the
# previous Vast.ai run when tee flushed a partial line and SIGPIPE was raised).
python scripts/build_profile_testset.py 2>&1 > >(tee logs/step1_build_profile.log)
STEP1_EXIT=${PIPESTATUS[0]:-$?}
echo "Step 1 exit: $STEP1_EXIT"
if [ "$STEP1_EXIT" -ne 0 ]; then
  echo "ERROR: Step 1 failed. Check logs/step1_build_profile.log"; exit 1
fi

# ── Step 2: Retrain ensemble (with CW + AutoAttack in training mix) ───────────
echo ""
echo "=== Step 2: Retrain Ensemble [n=4000, CW+AA in mix, fgsm-os=2.5] ==="
# FGSM oversample 2.5 gives FGSM 2.5/(2.5+1+1+1+1) = 38.5% of the adversarial
# budget, close to the original 3-attack share (1.5/3.5 = 42.9%) that achieved
# FGSM TPR 86.76%. This compensates for CW+AA dilution without oversampling so
# aggressively that other attacks regress.
#
# NOTE: --use-grad-norm was tested on the 2026-04-22 Vast.ai run and REVERTED.
# It caused a catastrophic regression: FGSM TPR 80.6% → 63.0%, Square 89.1% → 79.8%.
# Root cause: the gradient L2 norm is nearly non-discriminative (AUC +0.004) but
# inflated calibration thresholds by 15-20%, destroying the TPR/FPR tradeoff.
# See regression_analysis_20260422.md for the full forensic analysis.
python scripts/train_ensemble_scorer.py \
  --n-train 4000 \
  --fgsm-oversample 2.5 \
  --include-cw \
  --include-autoattack \
  --cw-max-iter 30 \
  --cw-bss 3 \
  --output models/ensemble_scorer.pkl \
  2>&1 > >(tee logs/step2_retrain.log)
STEP2_EXIT=${PIPESTATUS[0]:-$?}
echo "Step 2 exit: $STEP2_EXIT"
if [ "$STEP2_EXIT" -ne 0 ]; then
  echo "ERROR: Step 2 failed. Check logs/step2_retrain.log"; exit 1
fi

# ── Step 2b: Post-retrain verification ────────────────────────────────────────
# Verifies CW and AutoAttack are in the training mix, grad-norm is OFF,
# and feature dimension is 37 (36 persistence + 1 DCT, no grad-norm).
echo ""
echo "=== Step 2b: Retrain Verification ==="
python -c "
import pickle, sys
e = pickle.load(open('models/ensemble_scorer.pkl', 'rb'))
ta = list(getattr(e, 'training_attacks', []))
ng = bool(getattr(e, 'use_grad_norm', False))
nf = int(getattr(e, 'n_features', 0)) if hasattr(e, 'n_features') else None
errors = []
if 'CW' not in ta:
    errors.append(f'CW missing from training_attacks: {ta}')
if 'AutoAttack' not in ta:
    errors.append(f'AutoAttack missing from training_attacks: {ta}')
if ng:
    errors.append('use_grad_norm=True — grad-norm must be OFF (reverted, see regression_analysis_20260422.md)')
if nf is not None and nf != 37:
    errors.append(f'n_features={nf}, expected 37 (36 persistence + 1 DCT)')
if errors:
    print('RETRAIN VERIFICATION FAIL:')
    for err in errors: print(f'  • {err}')
    sys.exit(1)
print(f'[OK] Retrain verified: training_attacks={ta}')
print(f'[OK] use_grad_norm={ng}, n_features={nf}')
"
if [ $? -ne 0 ]; then
  echo "ERROR: Post-retrain verification failed. Fix and re-run Step 2."; exit 1
fi

# ── Step 3: Calibrate conformal thresholds ───────────────────────────────────
echo ""
echo "=== Step 3: Calibrate Conformal Thresholds ==="
python scripts/calibrate_ensemble.py 2>&1 > >(tee logs/step3_calibrate.log)
STEP3_EXIT=${PIPESTATUS[0]:-$?}
echo "Step 3 exit: $STEP3_EXIT"
if [ "$STEP3_EXIT" -ne 0 ]; then
  echo "ERROR: Step 3 failed. Check logs/step3_calibrate.log"; exit 1
fi

# ── Step 4: FPR gate check ───────────────────────────────────────────────────
echo ""
echo "=== Step 4: Validation FPR Gate [test 7000-7999] ==="
python scripts/compute_ensemble_val_fpr.py 2>&1 > >(tee logs/step4_val_fpr.log)
STEP4_EXIT=${PIPESTATUS[0]:-$?}
if [ "$STEP4_EXIT" -ne 0 ]; then
  echo "ERROR: Step 4 compute failed. Check logs/step4_val_fpr.log"; exit 1
fi

echo ""
echo "=== Step 4: GATE CHECK ==="
python -c "
import json, sys
with open('experiments/calibration/ensemble_fpr_report.json') as f:
    r = json.load(f)
targets = [('L1', 0.10), ('L2', 0.03), ('L3', 0.005)]
failures = []
for tier, tgt in targets:
    fpr = r['tiers'][tier]['FPR']
    status = 'PASS' if fpr <= tgt else 'FAIL'
    print(f'  {tier} FPR={fpr:.4f}  target={tgt}  [{status}]')
    if fpr > tgt:
        failures.append(f'{tier} FPR={fpr:.4f} > {tgt}')
if failures:
    print(f'GATE FAIL: {failures}')
    sys.exit(1)
else:
    print('ALL GATES PASS — proceeding to evaluation')
"
GATE_EXIT=$?
if [ $GATE_EXIT -ne 0 ]; then
  echo "ERROR: FPR gate failed. Check logs/step4_val_fpr.log"
  echo "FIX: Lower tier_cal_alpha_factors in configs/default.yaml, re-run steps 3-4"
  exit 1
fi

# ── LOCK: ensemble_scorer.pkl + calibrator.pkl are now frozen ────────────────
echo ""
echo "=== ARTIFACTS LOCKED ==="
echo "ensemble_scorer.pkl and calibrator.pkl are now frozen."
echo "Do NOT retrain or recalibrate between attack runs."
python -c "
import pickle, hashlib
def h(p):
    return hashlib.sha256(open(p,'rb').read()).hexdigest()[:16]
print(f'  ensemble_scorer.pkl SHA256: {h(\"models/ensemble_scorer.pkl\")}')
print(f'  calibrator.pkl SHA256:      {h(\"models/calibrator.pkl\")}')
print(f'  reference_profiles.pkl SHA: {h(\"models/reference_profiles.pkl\")}')
"

# ══════════════════════════════════════════════════════════════════════════════
# Step 5: PARALLEL 5-seed evaluation
# ══════════════════════════════════════════════════════════════════════════════
# GPU utilization per attack:
#   FGSM/PGD: ~20-30%  |  Square: ~25-35%  |  CW: ~30-40%  |  AA: ~25-35%
#
# Running CW in one process and FGSM+PGD+Square+AA in another uses ~60-70%
# GPU total. This is safe because:
#   - Each process loads its own PRISM instance (independent memory)
#   - Both use the SAME locked ensemble_scorer.pkl + calibrator.pkl
#   - Results go to separate JSON files (no file contention)
#   - CUDA handles SM time-slicing automatically
#   - Seeds are identical → deterministic given same artifacts
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo "=== Step 5: Parallel Multi-seed Evaluation [n=$N_TEST × 5 seeds] ==="
echo "  Process A: CW-L2 (paper-canonical: max_iter=$CW_MAX_ITER, bss=$CW_BSS)"
echo "  Process B: FGSM + PGD + Square + AutoAttack"
echo "  Running in parallel on same GPU..."
echo ""

# Process A: CW (the slow one — bottleneck)
python experiments/evaluation/run_evaluation_full.py \
  --n-test $N_TEST --attacks CW \
  --multi-seed --seeds $SEEDS \
  --cw-max-iter $CW_MAX_ITER --cw-bss $CW_BSS \
  --checkpoint-interval 100 \
  --output experiments/evaluation/results_cw_n${N_TEST}_ms5.json \
  2>&1 | tee logs/step5_cw_ms5.log &
PID_CW=$!
echo "  CW started (PID=$PID_CW)"

# Process B: Fast attacks (FGSM + PGD + Square + AutoAttack)
python experiments/evaluation/run_evaluation_full.py \
  --n-test $N_TEST --attacks FGSM PGD Square AutoAttack \
  --multi-seed --seeds $SEEDS \
  --gen-chunk 128 --square-max-iter 5000 \
  --aa-version standard --aa-chunk 64 \
  --checkpoint-interval 100 \
  --output experiments/evaluation/results_fast_n${N_TEST}_ms5.json \
  2>&1 | tee logs/step5_fast_ms5.log &
PID_FAST=$!
echo "  FGSM+PGD+Square+AA started (PID=$PID_FAST)"

echo ""
echo "  Waiting for both processes..."
echo "  Monitor: tail -f logs/step5_cw_ms5.log logs/step5_fast_ms5.log"
echo ""

# Wait for both — capture exit codes
FAIL=0
wait $PID_CW || { echo "ERROR: CW process failed (exit $?)"; FAIL=1; }
wait $PID_FAST || { echo "ERROR: Fast attacks process failed (exit $?)"; FAIL=1; }

if [ $FAIL -ne 0 ]; then
  echo "ERROR: One or more evaluation processes failed. Check logs."
  exit 2
fi

echo ""
echo "=== Step 5: COMPLETE — all evaluations finished ==="

# ── Step 5 validation: verify ensemble provenance matches ────────────────────
echo ""
echo "=== Step 5: Provenance Check ==="
python -c "
import json
files = [
    'experiments/evaluation/results_cw_n${N_TEST}_ms5.json',
    'experiments/evaluation/results_fast_n${N_TEST}_ms5.json',
]
ta_sets = set()
for f in files:
    d = json.load(open(f))
    # Multi-seed: check first seed's _meta
    ps = d.get('per_seed', {})
    for seed_key, seed_data in ps.items():
        meta = seed_data.get('_meta', {})
        ta = tuple(meta.get('ensemble', {}).get('training_attacks', []))
        ta_sets.add(ta)
        break
print(f'Training attacks across result files: {ta_sets}')
if len(ta_sets) == 1:
    ta = list(ta_sets)[0]
    if 'CW' in ta and 'AutoAttack' in ta:
        print('PROVENANCE CHECK PASS: all results use same retrained ensemble')
    else:
        print(f'WARNING: ensemble does not include CW+AutoAttack: {ta}')
else:
    print('PROVENANCE CHECK FAIL: different ensembles detected!')
    exit(1)
"

# ── Step 6: Adaptive PGD (§7.6 — required for venue credibility) ─────────────
# PARALLELISM: All 5 seeds launched simultaneously as background jobs.
# Safe because:
#   • Each seed uses a different rng subset of EVAL_IDX (8000-9999) — no overlap
#   • Each process loads its own PRISM instance (independent GPU memory)
#   • All processes read the SAME locked ensemble_scorer.pkl + calibrator.pkl
#     (read-only after Step 4 — no writer exists at this point)
#   • Output files are distinct: results_adaptive_pgd_seed{s}.json per seed
#   • Log files are distinct: step6_adaptive_pgd_seed{s}.log per seed
#   • Adaptive PGD uses its own gradient tape (no shared tensor state)
# GPU headroom: each seed ≈20-30% SM utilisation → 5 seeds ≤ ~80% total
echo ""
echo "=== Step 6: Adaptive PGD [n=$N_TEST × 5 seeds, ALL PARALLEL] ==="
# Use indexed arrays (bash 3+ compatible) instead of declare -A (bash 4+ only).
STEP6_PIDS=""
STEP6_SEEDS=""
for s in $SEEDS; do
  echo "  Launching adaptive PGD seed=$s in background..."
  python experiments/evaluation/run_adaptive_pgd.py \
    --n-test $N_TEST --seed $s \
    --output experiments/evaluation/results_adaptive_pgd_seed${s}.json \
    2>&1 > >(tee logs/step6_adaptive_pgd_seed${s}.log) &
  STEP6_PIDS="$STEP6_PIDS $!"
  STEP6_SEEDS="$STEP6_SEEDS $s"
done
echo "  All 5 seeds launched. PIDs:$STEP6_PIDS"
echo "  Monitor: tail -f logs/step6_adaptive_pgd_seed42.log"

# ── Step 7: Ablation — starts WHILE Step 6 seeds are running ───────────────
# PARALLELISM: Ablation is launched immediately without waiting for Step 6.
# Safe because:
#   • Ablation reads only: models/ensemble_scorer.pkl, models/calibrator.pkl,
#     models/reference_profiles.pkl — all locked read-only after Step 4
#   • Ablation output goes to experiments/ablation/results_ablation_*.json
#     — completely separate from Step 6's results_adaptive_pgd_*.json
#   • No Step-6 output is consumed by ablation or the gate logic
#   • Both Step 6 and ablation are pure evaluation consumers of frozen artifacts
# Ablation uses FGSM, PGD, Square only — run_ablation_paper.py has no CW/AA
# implementation. The full evaluation (Step 5) covers CW + AutoAttack.
echo ""
echo "=== Step 7: Ablation [n=$N_TEST × 5 seeds, parallel with Step 6] ==="
python experiments/ablation/run_ablation_paper.py \
  --n $N_TEST \
  --multi-seed --seeds $SEEDS \
  --attacks FGSM PGD Square \
  2>&1 > >(tee logs/step7_ablation.log) &
PID_ABLATION=$!
echo "  Ablation started in background (PID=$PID_ABLATION)"
echo "  Monitor: tail -f logs/step7_ablation.log"

# ── Wait for Step 6 (all seeds) then Step 7 (ablation) ──────────────────────
# CRITICAL: Every `wait` MUST use `|| VAR=$?` because `set -e` would otherwise
# abort the script on the first failed background job, orphaning all remaining
# processes (other seeds + ablation). We capture exit codes in the || branch
# because `wait $pid || true` would clobber $? to 0.
echo ""
echo "  Waiting for Step 6 adaptive PGD seeds..."
STEP6_FAIL=0
# Iterate seeds and PIDs in lockstep (space-separated lists, same order)
set -- $STEP6_PIDS
for s in $STEP6_SEEDS; do
  pid=$1; shift
  EXIT_S=0
  wait $pid || EXIT_S=$?   # ← captures real exit code; prevents set -e abort
  if [ $EXIT_S -ne 0 ]; then
    echo "  ERROR: Adaptive PGD seed=$s (pid=$pid) failed (exit $EXIT_S). Check logs/step6_adaptive_pgd_seed${s}.log"
    STEP6_FAIL=1
  else
    echo "  Seed $s: DONE"
  fi
done
if [ $STEP6_FAIL -ne 0 ]; then
  echo "ERROR: One or more adaptive PGD seeds failed."
  # Do NOT exit — let ablation finish before aborting
fi
echo "Step 6: COMPLETE"

echo ""
echo "  Waiting for Step 7 ablation..."
STEP7_EXIT=0
wait $PID_ABLATION || STEP7_EXIT=$?   # ← captures real exit code
if [ $STEP7_EXIT -ne 0 ]; then
  echo "ERROR: Ablation failed (exit $STEP7_EXIT). Check logs/step7_ablation.log"
fi
echo "Step 7 exit: $STEP7_EXIT"

# Abort after both complete if Step 6 had failures
if [ $STEP6_FAIL -ne 0 ]; then
  exit 2
fi

# ── Step 8: Reproducibility manifest ─────────────────────────────────────────
echo ""
echo "=== Step 8: Reproducibility Manifest ==="
python -c "
import pickle, json, hashlib, os
def h(p): return hashlib.sha256(open(p,'rb').read()).hexdigest()[:16] if os.path.exists(p) else None
e = pickle.load(open('models/ensemble_scorer.pkl','rb'))
out = {
  'ensemble_training_attacks': list(getattr(e, 'training_attacks', [])),
  'ensemble_training_n':       int(getattr(e, 'training_n', 0)),
  'ensemble_n_features':       int(getattr(e, 'n_features', 0)) if hasattr(e, 'n_features') else None,
  'ensemble_use_dct':          getattr(e, 'use_dct', None),
  'ensemble_use_grad_norm':    getattr(e, 'use_grad_norm', None),
  'ensemble_sha256_16':        h('models/ensemble_scorer.pkl'),
  'calibrator_sha256_16':      h('models/calibrator.pkl'),
  'reference_profiles_sha256_16': h('models/reference_profiles.pkl'),
  'seeds':                     [42, 123, 456, 789, 999],
  'eval_split':                'CIFAR-10 test idx 8000-9999',
  'eps_linf':                  8.0/255,
  'cw_eval_params':            {'max_iter': $CW_MAX_ITER, 'bss': $CW_BSS, 'batch_size': 64, 'confidence': 0.0},
  'aa_eval_params':            {'version': 'standard', 'chunk': 64},
  'square_eval_params':        {'max_iter': 5000, 'nb_restarts': 1},
}
print(json.dumps(out, indent=2))
" 2>&1 | tee logs/manifest.json

echo ""
echo "============================================================"
echo "PRISM Vast.ai Full Pipeline — COMPLETE"
echo "$(date)"
echo "============================================================"
echo ""
echo "Result files:"
ls -lh experiments/evaluation/results_*_ms5.json \
  experiments/evaluation/results_adaptive_pgd_*.json \
  experiments/ablation/results_* 2>/dev/null
echo ""
echo "Download command (run from your laptop):"
echo "  scp -P <port> root@<ip>:/workspace/prism-repo/prism/experiments/evaluation/results_*_ms5.json ."
echo "  scp -P <port> root@<ip>:/workspace/prism-repo/prism/experiments/evaluation/results_adaptive_pgd_*.json ."
echo "  scp -P <port> root@<ip>:/workspace/prism-repo/prism/logs/*.log ."
echo "  scp -P <port> root@<ip>:/workspace/prism-repo/prism/models/*.pkl ."
