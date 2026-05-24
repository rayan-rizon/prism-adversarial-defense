#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/prism"
export PRISM_CONFIG=configs/wrn_cifar10.yaml
mkdir -p logs experiments/wrn/evaluation experiments/wrn/calibration

echo "[1/3] Recalibrate WRN conformal thresholds with patched L3 factor"
python scripts/calibrate_ensemble.py 2>&1 | tee logs/recalibrate_wrn_l3_new_gpu.log

echo "[2/3] Recompute validation FPR gate"
python scripts/compute_ensemble_val_fpr.py 2>&1 | tee logs/recheck_wrn_l3_new_gpu.log

echo "[gate] require L1<=0.10, L2<=0.03, L3<=0.005"
python - <<'PY'
import json, sys
with open('experiments/wrn/calibration/ensemble_fpr_report.json') as f:
    r = json.load(f)
targets = {'L1': 0.10, 'L2': 0.03, 'L3': 0.005}
failed = []
for tier, target in targets.items():
    fpr = float(r['tiers'][tier]['FPR'])
    ok = fpr <= target
    print(f'{tier}: FPR={fpr:.4f}, target<={target:.4f}, status={"PASS" if ok else "FAIL"}')
    if not ok:
        failed.append(tier)
if failed:
    raise SystemExit(f'FPR gate failed: {failed}')
PY

echo "[3/3] Rerun FGSM + PGD only across 5 seeds"
python experiments/evaluation/run_evaluation_full.py \
  --n-test 1000 \
  --attacks FGSM PGD \
  --multi-seed \
  --seeds 42 123 456 789 999 \
  --gen-chunk 128 \
  --pgd-max-iter 50 \
  --pgd-restarts 10 \
  --skip-latency \
  --checkpoint-interval 100 \
  --output experiments/wrn/evaluation/results_fgsm_pgd_l3fix_wrn.json \
  2>&1 | tee logs/eval_fgsm_pgd_l3fix_new_gpu.log
