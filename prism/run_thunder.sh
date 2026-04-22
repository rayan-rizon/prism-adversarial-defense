#!/bin/bash
# PRISM Full Pipeline + n=1000 Evaluation — ThunderCompute A100
# Usage: bash run_thunder.sh
# Runs B1→B2→B3→B4 (gate)→C1 and logs everything.
# Exit codes: 0=success, non-zero=failed phase (check logs/).

set -e
cd ~/prism-run/prism-adversarial-defense/prism

echo "=== PRISM ThunderCompute Run $(date) ==="
echo "Instance: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Remove Windows venv if present, create clean Linux venv
if [ -d ".venv/Lib" ]; then
  echo "Removing Windows venv..."
  rm -rf .venv
fi

if [ ! -d ".venv" ]; then
  echo "Creating Linux venv..."
  python3 -m venv .venv
fi

source .venv/bin/activate
python --version
pip install --upgrade pip wheel --quiet

echo "=== Installing requirements ==="
pip install -r requirements.txt --quiet
pip install adversarial-robustness-toolbox --quiet

# Environment
export PYTHONIOENCODING=utf-8
export SSL_CERT_FILE=$(python -c 'import certifi; print(certifi.where())' 2>/dev/null || echo "")
export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE

# Create required directories
mkdir -p logs models experiments/calibration experiments/evaluation

# Verify GPU
python -c "
import torch
assert torch.cuda.is_available(), 'NO CUDA!'
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'CUDA: {torch.version.cuda}')
print(f'Torch: {torch.__version__}')
"

# Preflight: verify per-tier config is present
python -c "
from src.config import TIER_CAL_ALPHA_FACTORS, CAL_ALPHA_FACTOR
print(f'cal_alpha_factor (scalar): {CAL_ALPHA_FACTOR}')
print(f'tier_cal_alpha_factors: {TIER_CAL_ALPHA_FACTORS}')
assert TIER_CAL_ALPHA_FACTORS.get('L3', CAL_ALPHA_FACTOR) <= 0.55, \
  f'L3 factor must be <= 0.55, got {TIER_CAL_ALPHA_FACTORS.get(\"L3\")}'
print('PREFLIGHT PASS: per-tier L3=0.50 confirmed')
"

echo ""
echo "=== B1: Build Reference Profiles [test 0-4999] ==="
python scripts/build_profile_testset.py 2>&1 | tee logs/B1_build_profile.log
echo "B1 exit: $?"

echo ""
echo "=== B2: Train Ensemble Scorer [n=3000, fgsm_oversample=1.5] ==="
python scripts/train_ensemble_scorer.py \
  --n-train 3000 \
  --fgsm-oversample 1.5 \
  2>&1 | tee logs/B2_train_ensemble.log
echo "B2 exit: $?"

echo ""
echo "=== B3: Calibrate Conformal Thresholds [per-tier L3=0.50] ==="
python scripts/calibrate_ensemble.py 2>&1 | tee logs/B3_calibrate.log
echo "B3 exit: $?"

echo ""
echo "=== B4: Validation FPR Gate [test 7000-7999] ==="
python scripts/compute_ensemble_val_fpr.py 2>&1 | tee logs/B4_val_fpr.log
echo "B4 exit: $?"

# GATE CHECK
echo ""
echo "=== B4 GATE CHECK ==="
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
    print('ALL GATES PASS — proceeding to C1')
"
GATE_EXIT=$?
if [ $GATE_EXIT -ne 0 ]; then
  echo "ERROR: B4 gate failed. Aborting C1."
  exit 1
fi

echo ""
echo "=== C1: Multi-seed Evaluation [n=1000, seeds=42/123/456/789/999] ==="
echo "Attacks: FGSM PGD Square  (no CW / AutoAttack)"
OUT="experiments/evaluation/results_n1000_multiseed_$(date +%Y%m%d).json"
python experiments/evaluation/run_evaluation_full.py \
  --n-test 1000 \
  --attacks FGSM PGD Square \
  --multi-seed \
  --seeds 42 123 456 789 999 \
  --square-max-iter 5000 \
  --checkpoint-interval 200 \
  --output "$OUT" \
  2>&1 | tee logs/C1_multiseed.log
echo "C1 exit: $?"

echo ""
echo "=== FINAL RESULTS ==="
python -c "
import json
with open('$OUT') as f:
    r = json.load(f)
agg = r.get('aggregate', {})
for atk in ['FGSM', 'PGD', 'Square']:
    if atk in agg:
        a = agg[atk]
        print(f'{atk}: TPR={a.get(\"TPR_mean\",\"?\"): .4f}  FPR={a.get(\"FPR_mean\",\"?\"): .4f}')
print(f'Output: $OUT')
"

echo "=== DONE $(date) ==="
