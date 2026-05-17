#!/bin/bash
# =============================================================================
# PRISM — Local Smoke Test (CPU, n≈300-500)
# =============================================================================
# Purpose: Validate that the local fast-attack detector contract (balanced
#          FGSM/PGD/Square, no grad-norm, 54 raw features with entropy,
#          logit-profile, stability-v2, plus side-quadratic scorer expansion)
#          works before spending
#          GPU hours on Vast.ai.
#
# PARITY TABLE — every param must match run_vastai_full.sh unless marked LOCAL:
# ┌───────────────────────────────┬──────────────┬──────────────┬───────┐
# │ Parameter                     │ Vast.ai       │ Local (here) │ Match │
# ├───────────────────────────────┼──────────────┼──────────────┼───────┤
# │ Python binary                 │ python        │ python3      │ LOCAL │
# │ attack mix                    │ balanced      │ balanced     │  ✅   │
# │ use-grad-norm                 │ OFF           │ OFF          │  ✅   │
# │ include-cw (train)            │ NO            │ NO           │  ✅   │
# │ include-autoattack (train)    │ NO            │ NO           │  ✅   │
# │ n-train                       │ 2400          │ 500          │ LOCAL │
# │ cw-max-iter (train)           │ 40            │ —            │ LOCAL │
# │ cw-bss (train)                │ 5             │ —            │ LOCAL │
# │ TIER_CAL_ALPHA_FACTORS        │ L1=0.75,L2=0.7,L3=0.5 (config.py) │ SAME │ ✅ │
# │ CONFORMAL_ALPHAS              │ L1=0.1,L2=0.03,L3=0.005 (config.py) │ SAME │ ✅ │
# │ EPS_LINF                      │ 8/255         │ 8/255        │  ✅   │
# │ EVAL_IDX split                │ 8000-9999     │ 8000-9999    │  ✅   │
# │ CAL_IDX split                 │ 5000-7000     │ 5000-7000    │  ✅   │
# │ VAL_IDX split                 │ 7000-8000     │ 7000-8000    │  ✅   │
# │ LAYER_NAMES                   │ layer2/3/4 (config.py) │ SAME │ ✅ │
# │ N_SUBSAMPLE                   │ 150 (config.py)│ SAME        │  ✅   │
# │ n-test (eval)                 │ 1000          │ 300          │ LOCAL │
# │ seeds (eval)                  │ 5 (42..999)   │ 1 (42 only)  │ LOCAL │
# │ attacks (eval)                │ FGSM+PGD+Square+AA │ FGSM+PGD+Square │ LOCAL │
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
# If this test passes at the publishable FGSM/PGD/Square gates, Vast.ai CW-L2
# and AutoAttack runs are worth launching.
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

# ── Step 0b: Backbone accuracy gate ─────────────────────────────────────────
# Hard precondition for the entire detector pipeline. An undertrained
# backbone produces noisy decision boundaries; attacks computed against it
# do not yield meaningfully adversarial activations, so the TDA + entropy
# features can't separate clean from adv and the detector collapses to
# TPR ≈ FPR ≈ random. This gate makes that failure mode unreachable.
#
# The 0.93 floor matches the publishable Madry-recipe target (94-95% acc);
# below it the FGSM/Square TPR gate downstream is statistically meaningless.
echo "=== Step 0b: Backbone Accuracy Gate ==="
if [ ! -f models/cifar_resnet18.pt ] || [ ! -f models/cifar_resnet18.acc.json ]; then
  echo "❌ Backbone checkpoint missing:"
  [ -f models/cifar_resnet18.pt ]         || echo "     - models/cifar_resnet18.pt not found"
  [ -f models/cifar_resnet18.acc.json ]   || echo "     - models/cifar_resnet18.acc.json not found"
  echo ""
  echo "  This smoke pipeline cannot validate detection quality without a"
  echo "  properly-trained CIFAR-10 backbone (≥ 93% test acc, ~200 epochs)."
  echo ""
  echo "  Fix on a GPU box (~50-70 min on RTX 5090):"
  echo "    python3 scripts/pretrain_cifar_backbone.py"
  echo ""
  echo "  Then scp models/cifar_resnet18.pt and models/cifar_resnet18.acc.json"
  echo "  back to this machine and re-run the smoke pipeline."
  exit 2
fi
python3 scripts/verify_backbone_acc.py \
  --checkpoint models/cifar_resnet18.pt \
  --sidecar    models/cifar_resnet18.acc.json \
  --min-acc 0.93 --n 1000 \
  2>&1 | tee logs/smoke_step0b_backbone_gate.log
if [ "${PIPESTATUS[0]:-1}" -ne 0 ]; then
  echo "❌ Backbone failed accuracy gate. Refusing to run downstream stages."
  echo "  (A poisoned pipeline produces silent TPR collapse — see plan §root-cause.)"
  exit 2
fi
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
echo "    n-train:          500  (Vast.ai: 2400)"
echo "    include-cw:       OFF  (slow on CPU; AUC impact ~0.003)"
echo "    include-autoattack: OFF (not installed)"
echo "    attack mix:       balanced FGSM/PGD/Square  ← SAME local fast-attack contract"
echo "    use-grad-norm:    OFF  ← SAME as Vast.ai (REVERTED) ✅"
echo "    logit-profile:    ON   ← current local FGSM/Square lower-tail fix"
echo "    side-quadratic:   ON   ← current local lower-tail candidate"
echo ""
# 500 samples with --balanced-attacks -> approximately equal FGSM/PGD/Square shares.
python3 scripts/train_ensemble_scorer.py \
  --n-train 500 \
  --source-split profile \
  --balanced-attacks \
  --pgd-train-steps 40 \
  --square-train-max-iter 500 \
  --selection-objective worst_case_tpr \
  --use-stability-features \
  --use-logit-profile-features \
  --use-side-quadratic-features \
  --output models/ensemble_scorer.pkl \
  2>&1 > >(tee logs/smoke_step2_retrain.log)
echo "Step 2: DONE"
echo ""

# ── Step 2b: Post-retrain verification (IDENTICAL logic to Vast.ai) ───────────
echo "=== Step 2b: Retrain Verification ==="
python3 -c "
import pickle, sys
# ensemble_scorer.save() pickles a dict (see ensemble_scorer.py:417). Use
# dict.get(...) — getattr() on a dict silently returns the default and would
# false-fail the FGSM-presence check below.
e = pickle.load(open('models/ensemble_scorer.pkl', 'rb'))
if not isinstance(e, dict):
    print(f'RETRAIN VERIFICATION FAIL: ensemble_scorer.pkl is {type(e).__name__}, expected dict')
    sys.exit(1)
ta = list(e.get('training_attacks', []))
ng = bool(e.get('use_grad_norm', False))
se = bool(e.get('use_softmax_entropy', False))
sf = bool(e.get('use_stability_features', False))
lp = bool(e.get('use_logit_profile_features', False))
sq = bool(e.get('use_side_quadratic_features', False))
nf = int(e.get('n_features', 0)) if 'n_features' in e else None
model_dim = int(e.get('logistic_input_dim') or 0)
errors = []
# Smoke-test-specific: CW/AA not included (CPU budget), but FGSM must be present
if 'FGSM' not in ta:
    errors.append(f'FGSM missing from training_attacks: {ta}')
# Grad-norm must be OFF (same as Vast.ai post-regression-fix)
if ng:
    errors.append('use_grad_norm=True — must be OFF (see regression_analysis_20260422.md)')
if not se:
    errors.append('use_softmax_entropy=False — must be ON')
if not sf:
    errors.append('use_stability_features=False - must be ON for canonical local detector')
if not lp:
    errors.append('use_logit_profile_features=False - current local winner requires logit-profile features')
if nf is not None and nf != 54:
    errors.append(f'n_features={nf}, expected 54')
if int(e.get('stability_feature_count', 0)) != 8:
    errors.append(f'stability_feature_count={e.get(\"stability_feature_count\")}, expected 8')
if int(e.get('logit_profile_feature_count', 0)) != 8:
    errors.append(f'logit_profile_feature_count={e.get(\"logit_profile_feature_count\")}, expected 8')
if e.get('feature_space_version') != 'pixel-stability-v2+logitprofile+sidequad':
    errors.append(f'feature_space_version={e.get(\"feature_space_version\")}, expected pixel-stability-v2+logitprofile+sidequad')
if not sq:
    errors.append('use_side_quadratic_features=False - current local winner requires side-quadratic expansion')
if model_dim <= (nf or 0):
    errors.append(f'logistic_input_dim={model_dim}, expected expanded input > n_features={nf}')
if e.get('attack_head_mode', 'off') not in ('off', None):
    errors.append(f'attack_head_mode={e.get(\"attack_head_mode\")}, expected off for current local winner')
if int(e.get('pgd_train_steps', -1)) != 40:
    errors.append(f'pgd_train_steps={e.get(\"pgd_train_steps\")}, expected 40')
if int(e.get('square_train_max_iter', -1)) != 500:
    errors.append(f'square_train_max_iter={e.get(\"square_train_max_iter\")}, expected 500')
if errors:
    print('RETRAIN VERIFICATION FAIL:')
    for err in errors: print(f'  • {err}')
    sys.exit(1)
print(f'[OK] training_attacks={ta}')
print(f'[OK] use_grad_norm={ng}, entropy={se}, stability={sf}, logit_profile={lp}, sidequad={sq}, n_features={nf}, model_dim={model_dim}')
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
echo "=== Step 5: Eval [n=$N_TEST, seed=$SEED, attacks: FGSM + PGD + Square] ==="
echo "  LOCAL deviations:"
echo "    n-test:           $N_TEST  (Vast.ai: 1000)"
echo "    seeds:            1  (seed 42 only; Vast.ai: 5 seeds)"
echo "    attacks:          FGSM PGD Square  (Vast.ai: +CW +AutoAttack)"
echo "    square-max-iter:  500  (Vast.ai: 5000 — faster, less thorough)"
echo "    gen-chunk:        32  (Vast.ai: 128 — smaller for CPU memory)"
echo "    checkpoint-interval: 50  (Vast.ai: 100)"
echo "  Algorithm, calibrator, model: IDENTICAL to Vast.ai ✅"
echo ""
python3 experiments/evaluation/run_evaluation_full.py \
  --n-test "$N_TEST" \
  --attacks FGSM PGD Square \
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
echo "    FGSM   TPR >= 85%"
echo "    PGD    TPR >= 90%"
echo "    Square TPR >= 85%"
echo "  Note: n=$N_TEST is smaller than n=1000, so this gate is diagnostic; promotion still requires the multi-seed gate."
echo ""
python3 -c "
import json, sys
d = json.load(open('$SMOKE_OUTPUT_DIR/smoke_results_n${N_TEST}_seed${SEED}.json'))

# Single-seed result (no --multi-seed flag)
fgsm = d.get('FGSM', {})
pgd  = d.get('PGD', {})
sqr  = d.get('Square', {})

fgsm_tpr = fgsm.get('TPR', 0)
pgd_tpr  = pgd.get('TPR', 0)
sqr_tpr  = sqr.get('TPR', 0)
fgsm_ci  = fgsm.get('TPR_CI_95', [0, 0])
pgd_ci   = pgd.get('TPR_CI_95', [0, 0])
sqr_ci   = sqr.get('TPR_CI_95', [0, 0])
fgsm_fn  = fgsm.get('FN', '?')
pgd_fn   = pgd.get('FN', '?')
sqr_fn   = sqr.get('FN', '?')

# FPR for clean-safety sanity
fgsm_fpr = fgsm.get('FPR', 0)
fgsm_l1  = fgsm.get('per_tier_fpr', {}).get('FPR_L1_plus', 0)

print(f'  FGSM  TPR={fgsm_tpr:.4f}  CI=[{fgsm_ci[0]:.3f}, {fgsm_ci[1]:.3f}]  FN={fgsm_fn}')
print(f'  PGD   TPR={pgd_tpr:.4f}  CI=[{pgd_ci[0]:.3f}, {pgd_ci[1]:.3f}]  FN={pgd_fn}')
print(f'  Square TPR={sqr_tpr:.4f}  CI=[{sqr_ci[0]:.3f}, {sqr_ci[1]:.3f}]  FN={sqr_fn}')
print(f'  FPR L1+={fgsm_l1:.4f}  (target ≤ 0.10)')
print()

failures = []
if fgsm_tpr < 0.85:
    failures.append(f'FGSM TPR={fgsm_tpr:.4f} < 0.85 target')
if pgd_tpr < 0.90:
    failures.append(f'PGD TPR={pgd_tpr:.4f} < 0.90 target')
if sqr_tpr < 0.85:
    failures.append(f'Square TPR={sqr_tpr:.4f} < 0.85 target')
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
    print(f'     FGSM TPR={fgsm_tpr:.4f} ≥ 0.85 ✅')
    print(f'     PGD TPR={pgd_tpr:.4f} ≥ 0.90 ✅')
    print(f'     Square TPR={sqr_tpr:.4f} ≥ 0.85 ✅')
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
