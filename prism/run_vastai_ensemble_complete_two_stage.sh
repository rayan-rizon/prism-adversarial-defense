#!/bin/bash
set -euo pipefail

cd /workspace/prism-repo/prism/prism
. /workspace/prism-venv/bin/activate

SEEDS="${SEEDS:-42 123 456 789 999}"
SCAN_TAG="${SCAN_TAG:-ensemble_complete_lambda_scan_n50}"
CONFIRM_TAG="${CONFIRM_TAG:-ensemble_complete_worst_lambda_n200}"
SCAN_LAMBDAS="${SCAN_LAMBDAS:-0.0 0.5 1.0 2.0 5.0 10.0}"
SCAN_N="${SCAN_N:-50}"
CONFIRM_N="${CONFIRM_N:-200}"
STEPS="${STEPS:-100}"
RESTARTS="${RESTARTS:-10}"

run_phase() {
  local tag="$1"
  local n_test="$2"
  local lambdas="$3"
  mkdir -p "logs/${tag}" "experiments/evaluation/${tag}"
  echo "PHASE_START tag=${tag} n=${n_test} lambdas=${lambdas} $(date -Is)" | tee -a "logs/${tag}/phase.log"

  local pids=""
  for seed in $SEEDS; do
    (
      echo "SEED_RUN_START seed=${seed} tag=${tag} $(date -Is)"
      echo "git=$(git -C /workspace/prism-repo/prism rev-parse --short HEAD)"
      echo "gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
      echo "n_test=${n_test} steps=${STEPS} restarts=${RESTARTS} lambdas=${lambdas}"
      TAG="$tag" OUTDIR="experiments/evaluation/${tag}" LOGDIR="logs/${tag}" \
      SEEDS="$seed" N_TEST="$n_test" STEPS="$STEPS" RESTARTS="$RESTARTS" LAMBDAS="$lambdas" \
      bash run_vastai_ensemble_complete_adaptive.sh
    ) > "logs/${tag}/seed${seed}.outer.log" 2>&1 &
    pids="$pids $!"
    echo "  launched seed=${seed} pid=$!" | tee -a "logs/${tag}/phase.log"
    sleep 2
  done

  local fail=0
  for pid in $pids; do
    wait "$pid" || fail=1
  done
  if [ "$fail" -ne 0 ]; then
    echo "PHASE_FAILED tag=${tag} $(date -Is)" | tee -a "logs/${tag}/phase.log"
    exit 1
  fi
  echo "PHASE_DONE tag=${tag} $(date -Is)" | tee -a "logs/${tag}/phase.log"
}

choose_worst_lambda() {
  python - <<'PY'
import json
from pathlib import Path

root = Path("experiments/evaluation/ensemble_complete_lambda_scan_n50")
rows = {}
for path in sorted(root.glob("results_ensemble_complete_adaptive_pgd_seed*.json")):
    data = json.loads(path.read_text())
    for key, value in data.items():
        if key.startswith("AdaptivePGD_lambda_"):
            rows.setdefault(str(value["lambda"]), []).append(value)

if not rows:
    raise SystemExit("No scan result JSON files found")

summary = []
for lam, vals in sorted(rows.items(), key=lambda kv: float(kv[0])):
    tp = sum(int(v.get("TP", 0)) for v in vals)
    fp = sum(int(v.get("FP", 0)) for v in vals)
    fn = sum(int(v.get("FN", 0)) for v in vals)
    tn = sum(int(v.get("TN", 0)) for v in vals)
    n_adv = max(tp + fn, 1)
    n_clean = max(fp + tn, 1)
    n_success = sum(int(v.get("n_successful_adv", 0)) for v in vals)
    detected_success = sum(int(v.get("detected_successful_adv", 0)) for v in vals)
    evaded_success = sum(int(v.get("evaded_successful_adv", 0)) for v in vals)
    item = {
        "lambda": lam,
        "n_seeds": len(vals),
        "n_adv": n_adv,
        "n_clean": n_clean,
        "TPR": tp / n_adv,
        "FPR": fp / n_clean,
        "model_ASR": n_success / n_adv,
        "TPR_on_successful_attacks": detected_success / max(n_success, 1),
        "evasion_rate_on_successful_attacks": evaded_success / max(n_success, 1),
        "undetected_success_rate": evaded_success / n_adv,
        "detected_successful_adv": detected_success,
        "evaded_successful_adv": evaded_success,
        "n_successful_adv": n_success,
    }
    summary.append(item)

# Worst deployment risk: maximize successful classifier fooling that also
# evades PRISM. This avoids selecting a low-ASR lambda just because its few
# successful adversarials evade more often.
worst = max(summary, key=lambda x: (
    x["undetected_success_rate"],
    x["model_ASR"],
    x["evasion_rate_on_successful_attacks"],
    -x["TPR_on_successful_attacks"],
    -x["TPR"],
))

out = Path("/workspace/prism_ec_worst_lambda.json")
out.write_text(json.dumps({"summary": summary, "worst": worst}, indent=2))
Path("/workspace/prism_ec_worst_lambda.txt").write_text(worst["lambda"] + "\n")
print(json.dumps({"worst": worst, "summary": summary}, indent=2))
PY
}

run_phase "$SCAN_TAG" "$SCAN_N" "$SCAN_LAMBDAS"
choose_worst_lambda | tee -a "logs/${SCAN_TAG}/worst_lambda.log"
WORST_LAMBDA="$(cat /workspace/prism_ec_worst_lambda.txt)"
echo "CONFIRM_WORST_LAMBDA ${WORST_LAMBDA}" | tee -a "logs/${CONFIRM_TAG}.launch.log"
run_phase "$CONFIRM_TAG" "$CONFIRM_N" "$WORST_LAMBDA"
echo "TWO_STAGE_DONE $(date -Is) worst_lambda=${WORST_LAMBDA}" | tee -a "logs/${CONFIRM_TAG}.launch.log"
