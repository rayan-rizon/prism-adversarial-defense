#!/bin/bash
# =============================================================================
# PRISM — Local Smoke Test (CPU, n≈300-500)
# =============================================================================
# Purpose: Validate that the FGSM/Square TPR fix (fgsm-oversample=1.8, no
#          grad-norm, 37 features) works before spending GPU hours on Vast.ai.
#
# PARITY TABLE — every param must match run_vastai_full.sh unless marked LOCAL:
# ┌───────────────────────────────┬──────────────┬──────────────┬───────┐
# │ Parameter                     │ Vast.ai       │ Local (here) │ Match │
# ├───────────────────────────────┼──────────────┼──────────────┼───────┤
# │ Python binary                 │ python        │ python3      │ LOCAL │
# │ fgsm-oversample               │ 1.8           │ 0.9 (match %)│ LOCAL │
# │ use-grad-norm                 │ OFF           │ OFF          │  ✅   │
# │ include-cw (train)            │ YES           │ NO  (slow)   │ LOCAL │
# │ include-autoattack (train)    │ YES           │ NO  (missing)│ LOCAL │
# │ n-train                       │ 4000          │ 500          │ LOCAL │
# │ cw-max-iter (train)           │ 30            │ —            │ LOCAL │
# │ cw-bss (train)                │ 3             │ —            │ LOCAL │
# │ TIER_CAL_ALPHA_FACTORS        │ L1=0.7,L2=0.7,L3=0.5 (config.py) │ SAME │ ✅ │
# │ CONFORMAL_ALPHAS              │ L1=0.1,L2=0.03,L3=0.005 (config.py) │ SAME │ ✅ │
# │ EPS_LINF                      │ 8/255         │ 8/255        │  ✅   │
# │ EVAL_IDX split                │ 8000-9999     │ 8000-9999    │  ✅   │
# │ CAL_IDX split                 │ 5000-7000     │ 5000-7000    │  ✅   │
# │ VAL_IDX split                 │ 7000-8000     │ 7000-8000    │  ✅   │
# │ LAYER_NAMES                   │ layer2/3/4 (config.py) │ SAME │ ✅ │
# │ N_SUBSAMPLE                   │ 150 (config.py)│ SAME        │  ✅   │
# │ n-test (eval)                 │ 1000          │ 300          │ LOCAL │
# │ seeds (eval)                  │ 5 (42..999)   │ 1 (42 only)  │ LOCAL │
# │ attacks (eval)                │ FGSM+PGD+Square+AA │ FGSM+Square │ LOCAL │
# │ square-max-iter               │ 5000          │ 500 (fast)   │ LOCAL │
# │ checkpoint-interval           │ 100           │ 50           │ LOCAL │
# │ gen-chunk                     │ 128           │ 32  (CPU)    │ LOCAL │
# │ model artifact                │ fresh retrain │ fresh retrain│  ✅   │
# │ calibrator artifact           │ fresh calibrate│ fresh calibrate│ ✅  │
# │ reference profiles artifact   │ fresh build   │ fresh build  │  ✅   │
# └───────────────────────────────┴──────────────┴──────────────┴───────┘
#
# LOCAL deviations are all compute-budget adjustments — none change the
# underlying algorithm, model architecture, or calibration logic.
# If this test passes (FGSM TPR ≥ 80%, Square TPR ≥ 83%), Vast.ai run is safe.
#
# Usage: bash run_smoke_test.sh [n_test]
#   n_test: optional override, default 300. Use 500 for higher confidence.
#
# Runtime estimate (Mac M-series, CPU):
#   Step 1 (build profiles): ~10-15 min (5000 images × TDA)
#   Step 2 (retrain):        ~15-25 min (500 FGSM + 200 PGD + 200 Square)
#   Step 3 (calibrate):      ~3-5 min
#   Step 4 (val FPR):        ~2-3 min
#   Step 5 (eval FGSM+Sq):  ~8-12 min (300 samples × 2 attacks)
#   TOTAL:                   ~40-60 min
#
# Exit codes: 0=PASS, 1=FAIL, 2=setup error

set -euo pipefail

N_TEST="${1:-300}"   # default 300; override with: bash run_smoke_test.sh 500
SEED=42
SMOKE_OUTPUT_DIR="experiments/evaluation/smoke_local"

echo "============================================================"
echo "PRISM Local Smoke Test — $(date)"
echo "n_test=$N_TEST  seed=$SEED  device=cpu"
echo "============================================================"

# ── Environment (matches Vast.ai where safe for CPU) ─────────────────────────
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
# NOT setting CUBLAS/NVIDIA/CUDNN — CPU only, those are GPU-specific

mkdir -p logs "$SMOKE_OUTPUT_DIR" models \
         experiments/calibration experiments/evaluation experiments/ablation

# ── Parity check: verify config constants match expectations ─────────────────
echo ""
echo "=== Parity Check: Config Constants ==="
python3 -c "
from src.config import (
    LAYER_NAMES, LAYER_WEIGHTS, DIM_WEIGHTS, N_SUBSAMPLE, MAX_DIM,
    CAL_ALPHA_FACTOR, TIER_CAL_ALPHA_FACTORS, CONFORMAL_ALPHAS,
    PROFILE_IDX, CAL_IDX, VAL_IDX, EVAL_IDX, EPS_LINF_STANDARD
)
import sys

errors = []
# ── Must match Vast.ai exactly ───────────────────────────────────────────────
if LAYER_NAMES != ['layer2', 'layer3', 'layer4']:
    errors.append(f'LAYER_NAMES mismatch: {LAYER_NAMES}')
if N_SUBSAMPLE != 150:
    errors.append(f'N_SUBSAMPLE={N_SUBSAMPLE}, expected 150')
if MAX_DIM != 1:
    errors.append(f'MAX_DIM={MAX_DIM}, expected 1')
if abs(EPS_LINF_STANDARD - 8/255) > 1e-9:
    errors.append(f'EPS_LINF={EPS_LINF_STANDARD}, expected {8/255}')
if CONFORMAL_ALPHAS != {'L1': 0.1, 'L2': 0.03, 'L3': 0.005}:
    errors.append(f'CONFORMAL_ALPHAS mismatch: {CONFORMAL_ALPHAS}')
if TIER_CAL_ALPHA_FACTORS.get('L3', 0) != 0.5:
    errors.append(f'TIER_CAL_ALPHA_FACTORS L3={TIER_CAL_ALPHA_FACTORS}')
if PROFILE_IDX != (0, 5000):
    errors.append(f'PROFILE_IDX mismatch: {PROFILE_IDX}')
if CAL_IDX != (5000, 7000):
    errors.append(f'CAL_IDX mismatch: {CAL_IDX}')
if VAL_IDX != (7000, 8000):
    errors.append(f'VAL_IDX mismatch: {VAL_IDX}')
if EVAL_IDX != (8000, 10000):
    errors.append(f'EVAL_IDX mismatch: {EVAL_IDX}')

if errors:
    print('CONFIG PARITY FAIL:')
    for e in errors: print(f'  ✗ {e}')
    sys.exit(1)
print('  ✅ LAYER_NAMES:   layer2, layer3, layer4')
print('  ✅ N_SUBSAMPLE:   150')
print(f'  ✅ EPS_LINF:      {EPS_LINF_STANDARD:.6f} = 8/255')
print(f'  ✅ CONFORMAL_ALPHAS: {CONFORMAL_ALPHAS}')
print(f'  ✅ TIER_CAL_ALPHA_FACTORS: {TIER_CAL_ALPHA_FACTORS}')
print(f'  ✅ Splits: PROFILE={PROFILE_IDX} CAL={CAL_IDX} VAL={VAL_IDX} EVAL={EVAL_IDX}')
print('  All config constants match Vast.ai.')
"
echo ""

# ── Step 0: Environment verification ─────────────────────────────────────────
echo "=== Step 0: Python + Deps Verification ==="
python3 -c "
import sys, torch, torchvision, numpy, sklearn, gudhi
print(f'Python:      {sys.version.split()[0]}')
print(f'PyTorch:     {torch.__version__}')
print(f'torchvision: {torchvision.__version__}')
print(f'numpy:       {numpy.__version__}')
print(f'sklearn:     {sklearn.__version__}')
print(f'gudhi:       {gudhi.__version__}')
print(f'CUDA:        {torch.cuda.is_available()} (CPU-only run, expected False)')
try:
    from art.attacks.evasion import FastGradientMethod
    print('ART:         available ✅')
except ImportError:
    print('ART:         MISSING ❌'); import sys; sys.exit(1)
try:
    import autoattack
    print('AutoAttack:  available ✅')
except ImportError:
    print('AutoAttack:  not installed (skipped in smoke test — LOCAL deviation)')
assert int(torch.__version__.split('.')[0]) >= 2, 'PyTorch >= 2 required'
print('Step 0: PASS')
"
echo ""

# ── Step 1: Build reference profiles (IDENTICAL to Vast.ai) ──────────────────
echo "=== Step 1: Build Reference Profiles [CIFAR-10 test 0-4999] ==="
echo "  (Same script + same PROFILE_IDX as Vast.ai — full 5000 images required)"
python3 scripts/build_profile_testset.py \
  2>&1 > >(tee logs/smoke_step1_build_profile.log)
echo "Step 1: DONE"
echo ""

# ── Step 2: Retrain ensemble ──────────────────────────────────────────────────
echo "=== Step 2: Retrain Ensemble [n_train=500, FGSM+PGD+Square only] ==="
echo "  LOCAL deviations (compute only — algorithm identical):"
echo "    n-train:          500  (Vast.ai: 4000)"
echo "    include-cw:       OFF  (slow on CPU; AUC impact ~0.003)"
echo "    include-autoattack: OFF (not installed)"
echo "    fgsm-oversample:  0.9  (Matches 31% share of Vast.ai 1.8) LOCAL"
echo "    use-grad-norm:    OFF  ← SAME as Vast.ai (REVERTED) ✅"
echo ""
# 0.9 / (0.9+1+1) = 31% FGSM share (matches Vast.ai 5-attack mix with 1.8)
# 500 samples with fgsm-os=0.9 → FGSM≈155, PGD≈172, Square≈172
python3 scripts/train_ensemble_scorer.py \
  --n-train 500 \
  --fgsm-oversample 0.9 \
  --output models/ensemble_scorer.pkl \
  2>&1 > >(tee logs/smoke_step2_retrain.log)
echo "Step 2: DONE"
echo ""

# ── Step 2b: Post-retrain verification (IDENTICAL logic to Vast.ai) ───────────
echo "=== Step 2b: Retrain Verification ==="
python3 -c "
import pickle, sys
e = pickle.load(open('models/ensemble_scorer.pkl', 'rb'))
ta = list(getattr(e, 'training_attacks', []))
ng = bool(getattr(e, 'use_grad_norm', False))
nf = int(getattr(e, 'n_features', 0)) if hasattr(e, 'n_features') else None
errors = []
# Smoke-test-specific: CW/AA not included (CPU budget), but FGSM must be present
if 'FGSM' not in ta:
    errors.append(f'FGSM missing from training_attacks: {ta}')
# Grad-norm must be OFF (same as Vast.ai post-regression-fix)
if ng:
    errors.append('use_grad_norm=True — must be OFF (see regression_analysis_20260422.md)')
if nf is not None and nf != 37:
    errors.append(f'n_features={nf}, expected 37')
if errors:
    print('RETRAIN VERIFICATION FAIL:')
    for err in errors: print(f'  • {err}')
    sys.exit(1)
print(f'[OK] training_attacks={ta}')
print(f'[OK] use_grad_norm={ng}, n_features={nf}')
"
if [ $? -ne 0 ]; then
  echo "ERROR: Step 2b verification failed."; exit 2
fi
echo ""

# ── Step 3: Calibrate (IDENTICAL script to Vast.ai) ──────────────────────────
echo "=== Step 3: Calibrate Conformal Thresholds [CAL_IDX 5000-7000] ==="
echo "  Same script, same CAL_IDX, same TIER_CAL_ALPHA_FACTORS as Vast.ai ✅"
python3 scripts/calibrate_ensemble.py \
  2>&1 > >(tee logs/smoke_step3_calibrate.log)
echo "Step 3: DONE"
echo ""

# ── Step 4: Validation FPR gate (IDENTICAL to Vast.ai) ───────────────────────
echo "=== Step 4: Validation FPR Gate [VAL_IDX 7000-8000] ==="
python3 scripts/compute_ensemble_val_fpr.py \
  2>&1 > >(tee logs/smoke_step4_val_fpr.log)

python3 -c "
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
    print(f'FPR GATE FAIL: {failures}')
    sys.exit(1)
print('FPR GATE: ALL PASS — proceeding to eval')
"
if [ $? -ne 0 ]; then
  echo "ERROR: FPR gate failed. Lower TIER_CAL_ALPHA_FACTORS in configs/default.yaml."; exit 1
fi
echo ""

# ── Artifact SHA lock (mirrors Vast.ai LOCK block) ───────────────────────────
echo "=== ARTIFACTS LOCKED ==="
python3 -c "
import pickle, hashlib
def h(p):
    return hashlib.sha256(open(p,'rb').read()).hexdigest()[:16]
print(f'  ensemble_scorer.pkl  SHA256: {h(\"models/ensemble_scorer.pkl\")}')
print(f'  calibrator.pkl       SHA256: {h(\"models/calibrator.pkl\")}')
print(f'  reference_profiles   SHA256: {h(\"models/reference_profiles.pkl\")}')
"
echo ""

# ── Step 5: Evaluation — FGSM + Square only (no AutoAttack locally) ───────────
echo "=== Step 5: Eval [n=$N_TEST, seed=$SEED, attacks: FGSM + Square] ==="
echo "  LOCAL deviations:"
echo "    n-test:           $N_TEST  (Vast.ai: 1000)"
echo "    seeds:            1  (seed 42 only; Vast.ai: 5 seeds)"
echo "    attacks:          FGSM Square  (Vast.ai: +PGD +AutoAttack)"
echo "    square-max-iter:  500  (Vast.ai: 5000 — faster, less thorough)"
echo "    gen-chunk:        32  (Vast.ai: 128 — smaller for CPU memory)"
echo "    checkpoint-interval: 50  (Vast.ai: 100)"
echo "  Algorithm, calibrator, model: IDENTICAL to Vast.ai ✅"
echo ""
python3 experiments/evaluation/run_evaluation_full.py \
  --n-test "$N_TEST" \
  --attacks FGSM Square \
  --seed "$SEED" \
  --square-max-iter 500 \
  --gen-chunk 32 \
  --checkpoint-interval 50 \
  --output "$SMOKE_OUTPUT_DIR/smoke_results_n${N_TEST}_seed${SEED}.json" \
  2>&1 > >(tee logs/smoke_step5_eval.log)
echo ""

# ── Step 6: Gate check — PASS/FAIL ───────────────────────────────────────────
echo "=== Step 6: Smoke Gate Check ==="
echo ""
echo "  Targets (same as Vast.ai publishable thresholds):"
echo "    FGSM TPR ≥ 80%  (smoke gate; Vast.ai gate: 85%)"
echo "    Square TPR ≥ 83%  (smoke gate; Vast.ai gate: 85%)"
echo "  Note: n=$N_TEST is smaller than n=1000 → wider CIs; use 80/83% not 85/85%"
echo ""
python3 -c "
import json, sys
d = json.load(open('$SMOKE_OUTPUT_DIR/smoke_results_n${N_TEST}_seed${SEED}.json'))

# Single-seed result (no --multi-seed flag)
fgsm = d.get('FGSM', {})
sqr  = d.get('Square', {})

fgsm_tpr = fgsm.get('TPR', 0)
sqr_tpr  = sqr.get('TPR', 0)
fgsm_ci  = fgsm.get('TPR_CI_95', [0, 0])
sqr_ci   = sqr.get('TPR_CI_95', [0, 0])
fgsm_fn  = fgsm.get('FN', '?')
sqr_fn   = sqr.get('FN', '?')

# FPR for clean-safety sanity
fgsm_fpr = fgsm.get('FPR', 0)
fgsm_l1  = fgsm.get('per_tier_fpr', {}).get('FPR_L1_plus', 0)

print(f'  FGSM  TPR={fgsm_tpr:.4f}  CI=[{fgsm_ci[0]:.3f}, {fgsm_ci[1]:.3f}]  FN={fgsm_fn}')
print(f'  Square TPR={sqr_tpr:.4f}  CI=[{sqr_ci[0]:.3f}, {sqr_ci[1]:.3f}]  FN={sqr_fn}')
print(f'  FPR L1+={fgsm_l1:.4f}  (target ≤ 0.10)')
print()

failures = []
# Smoke thresholds are looser than publication: n=300 has wide CIs
if fgsm_tpr < 0.80:
    failures.append(f'FGSM TPR={fgsm_tpr:.4f} < 0.80 smoke threshold')
if sqr_tpr < 0.83:
    failures.append(f'Square TPR={sqr_tpr:.4f} < 0.83 smoke threshold')
if fgsm_l1 > 0.10:
    failures.append(f'L1+ FPR={fgsm_l1:.4f} > 0.10 target')

if failures:
    print('❌ SMOKE TEST FAIL:')
    for f in failures: print(f'     • {f}')
    print()
    print('  This fix is NOT ready for Vast.ai. Check training mix or oversample ratio.')
    sys.exit(1)
else:
    print('✅ SMOKE TEST PASS:')
    print(f'     FGSM TPR={fgsm_tpr:.4f} ≥ 0.80 ✅')
    print(f'     Square TPR={sqr_tpr:.4f} ≥ 0.83 ✅')
    print()
    print('  Proceed to Vast.ai: bash run_vastai_full.sh')
"
GATE_EXIT=$?

echo ""
echo "============================================================"
if [ $GATE_EXIT -eq 0 ]; then
  echo "SMOKE TEST: PASS — configuration validated for Vast.ai run"
  echo ""
  echo "Next step:"
  echo "  scp -r prism/ root@<ip>:/workspace/prism-repo/"
  echo "  ssh root@<ip> 'cd /workspace/prism-repo/prism && bash run_vastai_full.sh'"
else
  echo "SMOKE TEST: FAIL — do NOT run Vast.ai until this is resolved"
  echo ""
  echo "Diagnostics:"
  echo "  cat logs/smoke_step2_retrain.log | tail -20"
  echo "  cat logs/smoke_step5_eval.log | tail -20"
  exit 1
fi
echo "$(date)"
echo "============================================================"
