set -euo pipefail
cd /workspace/prism-repo/prism
export PRISM_CONFIG=configs/wrn_cifar10.yaml

mkdir -p logs backups

echo "[queue] waiting for active evaluation jobs to finish..."
while pgrep -f "python experiments/evaluation/run_evaluation_full.py" >/dev/null; do
  date -u
  pgrep -af "run_evaluation_full.py" || true
  sleep 300
done

ts="$(date -u +%Y%m%d_%H%M%S)"
backup_dir="backups/recalibration_l3_${ts}"
mkdir -p "$backup_dir"

cp -a models/wrn/calibrator*.pkl "$backup_dir"/ 2>/dev/null || true
cp -a experiments/wrn/calibration/ensemble_fpr_report.json "$backup_dir"/ 2>/dev/null || true

echo "[recalibrate] using WRN L3 factor:"
grep -n -E "tier_cal_alpha_factors|L1:|L2:|L3:" configs/wrn_cifar10.yaml | head -20

python scripts/calibrate_ensemble.py 2>&1 | tee "logs/recalibrate_wrn_l3_${ts}.log"
python scripts/compute_ensemble_val_fpr.py 2>&1 | tee "logs/recheck_wrn_l3_fpr_${ts}.log"

python - <<'PY'
import json, sys
with open("experiments/wrn/calibration/ensemble_fpr_report.json") as f:
    r = json.load(f)

targets = {"L1": 0.10, "L2": 0.03, "L3": 0.005}
failed = []
for tier, target in targets.items():
    fpr = float(r["tiers"][tier]["FPR"])
    ok = fpr <= target
    print(f"{tier}: FPR={fpr:.4f}, target<={target:.4f}, status={PASS if ok else FAIL}")
    if not ok:
        failed.append(tier)

if failed:
    print("FPR gate failed:", failed)
    sys.exit(1)
print("FPR gate passed.")
PY

echo "[targeted rerun] FGSM+PGD only, 5 seeds, validates L3 fix without rerunning slow Square/AutoAttack"
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
  2>&1 | tee "logs/eval_fgsm_pgd_l3fix_${ts}.log"

echo "[done] recalibration + targeted FGSM/PGD validation complete"
