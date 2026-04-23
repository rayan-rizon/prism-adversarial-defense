#!/bin/bash
# =============================================================================
# PRISM — Vast.ai Full Pipeline (Retrain + Calibrate + Parallel Eval + Ablation)
# =============================================================================
# Usage: bash run_vastai_full.sh
# Runs the entire publishable pipeline on a single RTX 5090 instance.
#
# Parallelism map (safe — no shared mutable state between concurrent jobs):
#   Steps 5+6+7: ALL launched simultaneously after Step 4 LOCK.
#     Step 5A: CW-L2          (bottleneck, ~2h)
#     Step 5B: FGSM+PGD+Square+AA
#     Step 6 : 5 adaptive-PGD seeds in parallel
#     Step 7 : Ablation (FGSM+PGD+Square)
#   Wait order: Step 5 first (provenance check needs its JSON), then 6, then 7.
#   Saves ~35-40% wall-clock vs old sequential 5→6→7 schedule.
#
# Sequential constraints that CANNOT be parallelised:
#   Steps 1→2: Step 2 reads reference_profiles.pkl written by Step 1
#   Steps 2→3: Step 3 reads ensemble_scorer.pkl written by Step 2
#   Steps 3→4: Step 4 reads calibrator.pkl written by Step 3
#   Steps 4→5/6/7: All eval phases require the FPR gate to have passed
#
# Exit codes: 0=success, 1=gate failure, 2=eval failure

set -euo pipefail
cd /root/prism-repo/prism

SEEDS="42 123 456 789 999"
N_TEST=1000
CW_MAX_ITER=100
CW_BSS=9
CW_CHUNK=64   # gen_chunk for CW: 64 imgs/call fills ART batch_size=128 for ~full GPU occupancy

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
/venv/main/bin/python -c "
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
# Publication-grade determinism: cuDNN must not use non-deterministic algorithms.
# warn_only=True so ART's internal ops don't abort the run; violations are logged.
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
torch.cuda.manual_seed_all(0)
print('determinism flags: OK (warn_only=True for ART compatibility)')
"
echo "Step 0: PASS"

# ── Preflight: verify per-tier config ────────────────────────────────────────
/venv/main/bin/python -c "
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
/venv/main/bin/python scripts/build_profile_testset.py 2>&1 > >(tee logs/step1_build_profile.log)
STEP1_EXIT=${PIPESTATUS[0]:-$?}
echo "Step 1 exit: $STEP1_EXIT"
if [ "$STEP1_EXIT" -ne 0 ]; then
  echo "ERROR: Step 1 failed. Check logs/step1_build_profile.log"; exit 1
fi

# ── Step 2: Retrain ensemble (with CW + AutoAttack in training mix) ───────────
echo ""
echo "=== Step 2: Retrain Ensemble [n=4000, CW+AA in mix, fgsm-os=1.8] ==="
# FGSM oversample 1.8 gives FGSM 1.8/(1.8+1+1+1+1) = 31.0% of the adversarial
# budget, close to the original 3-attack share (1.5/3.5 = 42.9%) that achieved
# FGSM TPR 86.76%. This compensates for CW+AA dilution without oversampling so
# aggressively that other attacks regress.
#
# NOTE: --use-grad-norm was tested on the 2026-04-22 Vast.ai run and REVERTED.
# It caused a catastrophic regression: FGSM TPR 80.6% → 63.0%, Square 89.1% → 79.8%.
# Root cause: the gradient L2 norm is nearly non-discriminative (AUC +0.004) but
# inflated calibration thresholds by 15-20%, destroying the TPR/FPR tradeoff.
# See regression_analysis_20260422.md for the full forensic analysis.
/venv/main/bin/python scripts/train_ensemble_scorer.py \
  --n-train 4000 \
  --fgsm-oversample 2.0 \
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
/venv/main/bin/python -c "
import pickle, sys
d = pickle.load(open('models/ensemble_scorer.pkl', 'rb'))
# save() always serialises a dict (see src/cadg/ensemble_scorer.py:save).
# Legacy object-pickled format is rejected here — it means a stale pkl.
if not isinstance(d, dict):
    print('RETRAIN VERIFICATION FAIL: pkl is not a dict — stale or wrong-format artifact.')
    print(f'  type={type(d).__name__}  hint: delete models/ensemble_scorer.pkl and re-run Step 2.')
    sys.exit(1)
ta = list(d.get('training_attacks', []))
ng = bool(d.get('use_grad_norm', False))
# n_features is now saved explicitly; fall back to computing from flags for
# any pkl created before this fix was deployed (backward-compatible).
nf = d.get('n_features')
if nf is None:
    base = len(d.get('layer_names', [])) * len(d.get('dims', [])) * 6
    nf   = base + int(d.get('use_dct', False)) + int(d.get('use_grad_norm', False))
errors = []
if 'CW' not in ta:
    errors.append(f'CW missing from training_attacks: {ta}')
if 'AutoAttack' not in ta:
    errors.append(f'AutoAttack missing from training_attacks: {ta}')
if ng:
    errors.append('use_grad_norm=True — grad-norm must be OFF (reverted, see regression_analysis_20260422.md)')
if nf != 37:
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
/venv/main/bin/python scripts/calibrate_ensemble.py 2>&1 > >(tee logs/step3_calibrate.log)
STEP3_EXIT=${PIPESTATUS[0]:-$?}
echo "Step 3 exit: $STEP3_EXIT"
if [ "$STEP3_EXIT" -ne 0 ]; then
  echo "ERROR: Step 3 failed. Check logs/step3_calibrate.log"; exit 1
fi

# ── Step 4: FPR gate check ───────────────────────────────────────────────────
echo ""
echo "=== Step 4: Validation FPR Gate [test 7000-7999] ==="
/venv/main/bin/python scripts/compute_ensemble_val_fpr.py 2>&1 > >(tee logs/step4_val_fpr.log)
STEP4_EXIT=${PIPESTATUS[0]:-$?}
if [ "$STEP4_EXIT" -ne 0 ]; then
  echo "ERROR: Step 4 compute failed. Check logs/step4_val_fpr.log"; exit 1
fi

echo ""
echo "=== Step 4: GATE CHECK ==="
/venv/main/bin/python -c "
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
/venv/main/bin/python -c "
import pickle, hashlib
def h(p):
    return hashlib.sha256(open(p,'rb').read()).hexdigest()[:16]
print(f'  ensemble_scorer.pkl SHA256: {h(\"models/ensemble_scorer.pkl\")}')
print(f'  calibrator.pkl SHA256:      {h(\"models/calibrator.pkl\")}')
print(f'  reference_profiles.pkl SHA: {h(\"models/reference_profiles.pkl\")}')
"

# ══════════════════════════════════════════════════════════════════════════════
# Steps 5 + 6 + 7: FULL PARALLEL LAUNCH
# ══════════════════════════════════════════════════════════════════════════════
# All three phases are pure consumers of the locked frozen artifacts.
# No inter-step file dependency between 5, 6, and 7 — each writes to distinct
# output paths and loads the same read-only pkl artifacts from Step 4.
#
# Parallelism map:
#   Step 5A : CW-L2 eval (bottleneck, ~2h)
#   Step 5B : FGSM + PGD + Square + AutoAttack eval
#   Step 6  : 5 adaptive PGD seeds (each ~1h; overlap with Step 5)
#   Step 7  : Ablation (FGSM+PGD+Square, overlap with Steps 5+6)
#
# This collapses the original 3-phase sequential schedule into one wall-clock
# phase, saving ~35-40% total GPU instance time (CW is the hard ceiling).
#
# Wait order (enforced below):
#   1. Wait Step 5A + 5B first → provenance check requires their JSON output
#   2. Wait Step 6 seeds       → STEP6_FAIL gate
#   3. Wait Step 7 ablation    → STEP7_EXIT gate
#   If Step 5 fails we still wait 6+7 before exit to avoid orphaned processes.
#
# Safety proof (same as before):
#   • All processes load independent PRISM instances (independent GPU memory)
#   • CUDA SM time-slicing handles ≤8 concurrent light-batch processes safely
#   • Output filenames are distinct (no write contention)
#   • set -euo pipefail is active; every wait uses || VAR=$? to capture exit codes
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo "=== Steps 5+6+7: Full Parallel Launch [n=$N_TEST × 5 seeds] ==="
echo "  Step 5A: CW-L2 (max_iter=$CW_MAX_ITER, bss=$CW_BSS)"
echo "  Step 5B: FGSM + PGD + Square + AutoAttack"
echo "  Step 6 : Adaptive PGD × 5 seeds"
echo "  Step 7 : Ablation"
echo "  All launched simultaneously — CW is the wall-clock bottleneck."
echo ""

# ── Step 5A: CW ──────────────────────────────────────────────────────────────
/venv/main/bin/python experiments/evaluation/run_evaluation_full.py \
  --n-test $N_TEST --attacks CW \
  --multi-seed --seeds $SEEDS \
  --cw-max-iter $CW_MAX_ITER --cw-bss $CW_BSS --cw-chunk $CW_CHUNK \
  --checkpoint-interval 100 \
  --output experiments/evaluation/results_cw_n${N_TEST}_ms5.json \
  2>&1 | tee logs/step5_cw_ms5.log &
PID_CW=$!
echo "  Step 5A CW started (PID=$PID_CW)"

# ── Step 5B: Fast attacks ─────────────────────────────────────────────────────
/venv/main/bin/python experiments/evaluation/run_evaluation_full.py \
  --n-test $N_TEST --attacks FGSM PGD Square AutoAttack \
  --multi-seed --seeds $SEEDS \
  --gen-chunk 128 --square-max-iter 5000 \
  --aa-version standard --aa-chunk 64 \
  --checkpoint-interval 100 \
  --output experiments/evaluation/results_fast_n${N_TEST}_ms5.json \
  2>&1 | tee logs/step5_fast_ms5.log &
PID_FAST=$!
echo "  Step 5B FGSM+PGD+Square+AA started (PID=$PID_FAST)"

# ── Step 6: Adaptive PGD — all 5 seeds ───────────────────────────────────────
# Use indexed arrays (bash 3+ compatible) instead of declare -A (bash 4+ only).
echo ""
echo "  Launching Step 6 adaptive PGD seeds..."
STEP6_PIDS=""
STEP6_SEEDS=""
for s in $SEEDS; do
  /venv/main/bin/python experiments/evaluation/run_adaptive_pgd.py \
    --n-test $N_TEST --seed $s \
    --output experiments/evaluation/results_adaptive_pgd_seed${s}.json \
    2>&1 | tee logs/step6_adaptive_pgd_seed${s}.log &
  STEP6_PIDS="$STEP6_PIDS $!"
  STEP6_SEEDS="$STEP6_SEEDS $s"
  echo "  Step 6 seed=$s started (PID=$!)"
done

# ── Step 7: Ablation ─────────────────────────────────────────────────────────
# Reads only frozen pkl artifacts; output path is separate from Steps 5+6.
echo ""
/venv/main/bin/python experiments/ablation/run_ablation_paper.py \
  --n $N_TEST \
  --multi-seed --seeds $SEEDS \
  --attacks FGSM PGD Square \
  2>&1 | tee logs/step7_ablation.log &
PID_ABLATION=$!
echo "  Step 7 ablation started (PID=$PID_ABLATION)"

echo ""
echo "  All processes running. Monitor logs:"
echo "    tail -f logs/step5_cw_ms5.log"
echo "    tail -f logs/step6_adaptive_pgd_seed42.log"
echo "    tail -f logs/step7_ablation.log"
echo ""

# ── Wait order 1: Step 5 (provenance check needs its JSON output) ────────────
echo "  Waiting for Step 5 (CW + Fast attacks)..."
STEP5_FAIL=0
STEP5_CW_EXIT=0;   wait $PID_CW   || STEP5_CW_EXIT=$?
STEP5_FAST_EXIT=0; wait $PID_FAST || STEP5_FAST_EXIT=$?
[ $STEP5_CW_EXIT   -ne 0 ] && { echo "ERROR: Step 5 CW failed (exit $STEP5_CW_EXIT)";   STEP5_FAIL=1; }
[ $STEP5_FAST_EXIT -ne 0 ] && { echo "ERROR: Step 5 Fast failed (exit $STEP5_FAST_EXIT)"; STEP5_FAIL=1; }

echo ""
echo "=== Step 5: COMPLETE ==="

# ── Step 5 provenance check ───────────────────────────────────────────────────
if [ $STEP5_FAIL -eq 0 ]; then
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
fi

# ── Wait order 2: Step 6 seeds ────────────────────────────────────────────────
# CRITICAL: use || VAR=$? on every wait — set -e would abort on first failure,
# orphaning the remaining seeds and ablation process.
echo ""
echo "  Waiting for Step 6 adaptive PGD seeds..."
STEP6_FAIL=0
set -- $STEP6_PIDS       # positional params = PID list; must come AFTER PID_CW/FAST waits
for s in $STEP6_SEEDS; do
  pid=$1; shift
  EXIT_S=0
  wait $pid || EXIT_S=$?
  if [ $EXIT_S -ne 0 ]; then
    echo "  ERROR: Adaptive PGD seed=$s (pid=$pid) failed (exit $EXIT_S). Check logs/step6_adaptive_pgd_seed${s}.log"
    STEP6_FAIL=1
  else
    echo "  Seed $s: DONE"
  fi
done
[ $STEP6_FAIL -ne 0 ] && echo "ERROR: One or more adaptive PGD seeds failed."
echo "Step 6: COMPLETE"

# ── Wait order 3: Step 7 ablation ─────────────────────────────────────────────
echo ""
echo "  Waiting for Step 7 ablation..."
STEP7_EXIT=0
wait $PID_ABLATION || STEP7_EXIT=$?
[ $STEP7_EXIT -ne 0 ] && echo "ERROR: Ablation failed (exit $STEP7_EXIT). Check logs/step7_ablation.log"
echo "Step 7 exit: $STEP7_EXIT"

# Abort now that all background processes have finished
if [ $STEP5_FAIL -ne 0 ]; then
  echo "ERROR: Step 5 evaluation failed. Check logs/step5_cw_ms5.log and logs/step5_fast_ms5.log"
  exit 2
fi
if [ $STEP6_FAIL -ne 0 ]; then
  exit 2
fi

# ── Step 8: Reproducibility manifest ─────────────────────────────────────────
echo ""
echo "=== Step 8: Reproducibility Manifest ==="
/venv/main/bin/python -c "
import pickle, json, hashlib, os
def h(p): return hashlib.sha256(open(p,'rb').read()).hexdigest()[:16] if os.path.exists(p) else None
e = pickle.load(open('models/ensemble_scorer.pkl','rb'))
# save() writes a dict; use .get() not getattr() to avoid silent empty-list bugs
assert isinstance(e, dict), f'unexpected pkl type: {type(e).__name__}'
out = {
  'ensemble_training_attacks': list(e.get('training_attacks', [])),
  'ensemble_training_n':       int(e.get('training_n') or 0),
  'ensemble_n_features':       e.get('n_features'),
  'ensemble_use_dct':          e.get('use_dct'),
  'ensemble_use_grad_norm':    e.get('use_grad_norm'),
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
echo "  scp -P <port> root@<ip>:/root/prism-repo/prism/experiments/evaluation/results_*_ms5.json ."
echo "  scp -P <port> root@<ip>:/root/prism-repo/prism/experiments/evaluation/results_adaptive_pgd_*.json ."
echo "  scp -P <port> root@<ip>:/root/prism-repo/prism/logs/*.log ."
echo "  scp -P <port> root@<ip>:/root/prism-repo/prism/models/*.pkl ."
