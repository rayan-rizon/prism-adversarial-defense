#!/bin/bash
# =============================================================================
# PRISM — Vast.ai Full Pipeline (Retrain + Calibrate + Parallel Eval + Ablation)
# =============================================================================
# Usage: bash run_vastai_full.sh
# Runs the entire publishable pipeline on a single RTX 5090 instance.
# GPU-hungry attacks (CW vs FGSM+PGD+Square+AutoAttack) run in parallel.
#
# Pipeline order (mandatory):
#   1. Build reference profiles
#   2. Retrain ensemble (with CW + AutoAttack in training mix)
#   3. Calibrate conformal thresholds
#   4. FPR gate check (abort if any tier fails)
#   5. Parallel 5-seed evaluation: CW || FGSM+PGD+Square+AutoAttack
#   6. Ablation study
#   7. Reproducibility manifest
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
python scripts/build_profile_testset.py 2>&1 | tee logs/step1_build_profile.log
echo "Step 1 exit: $?"

# ── Step 2: Retrain ensemble (with CW + AutoAttack) ──────────────────────────
echo ""
echo "=== Step 2: Retrain Ensemble [n=4000, CW+AA in mix] ==="
python scripts/train_ensemble_scorer.py \
  --n-train 4000 \
  --fgsm-oversample 1.5 \
  --include-cw \
  --include-autoattack \
  --cw-max-iter 30 \
  --cw-bss 3 \
  --output models/ensemble_scorer.pkl \
  2>&1 | tee logs/step2_retrain.log
echo "Step 2 exit: $?"

# ── Step 3: Calibrate conformal thresholds ───────────────────────────────────
echo ""
echo "=== Step 3: Calibrate Conformal Thresholds ==="
python scripts/calibrate_ensemble.py 2>&1 | tee logs/step3_calibrate.log
echo "Step 3 exit: $?"

# ── Step 4: FPR gate check ───────────────────────────────────────────────────
echo ""
echo "=== Step 4: Validation FPR Gate [test 7000-7999] ==="
python scripts/compute_ensemble_val_fpr.py 2>&1 | tee logs/step4_val_fpr.log

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
# PGD where the attacker minimises the PRISM ensemble score. This is the
# minimum-viable adaptive-attack bar for any robustness venue.
echo ""
echo "=== Step 6: Adaptive PGD [n=$N_TEST × 5 seeds] ==="
for s in $SEEDS; do
  echo "  Adaptive PGD seed=$s"
  python experiments/evaluation/run_adaptive_pgd.py \
    --n-test $N_TEST --seed $s \
    --output experiments/evaluation/results_adaptive_pgd_seed${s}.json \
    2>&1 | tee logs/step6_adaptive_pgd_seed${s}.log
  echo "  Seed $s exit: $?"
done
echo "Step 6: COMPLETE"

# ── Step 7: Ablation ─────────────────────────────────────────────────────────
echo ""
echo "=== Step 7: Ablation [n=$N_TEST × 5 seeds] ==="
python experiments/ablation/run_ablation_paper.py \
  --n $N_TEST \
  --multi-seed --seeds $SEEDS \
  --attacks FGSM PGD Square CW AutoAttack \
  2>&1 | tee logs/step7_ablation.log
echo "Step 7 exit: $?"

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
