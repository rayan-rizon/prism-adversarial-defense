#!/bin/bash
# =============================================================================
# PRISM - Vast.ai ensemble-complete adaptive PGD
# =============================================================================
# Runs the adaptive attack against all deployed differentiable PRISM side
# channels: activation matching for TDA, DCT, entropy, logit profile,
# stability-v2, grad-norm, and the fitted side-quadratic logistic surrogate.
#
# Usage:
#   bash run_vastai_ensemble_complete_adaptive.sh
#
# Quick smoke:
#   SMOKE=1 bash run_vastai_ensemble_complete_adaptive.sh
#
# Useful overrides:
#   CONFIG=configs/cifar100.yaml TAG=cifar100 bash run_vastai_ensemble_complete_adaptive.sh
#   N_TEST=1000 STEPS=100 RESTARTS=10 SEEDS="42 123 456 789 999" bash run_vastai_ensemble_complete_adaptive.sh
#   SKIP_GRADNORM=1 bash run_vastai_ensemble_complete_adaptive.sh

set -euo pipefail

if [ -d /workspace/prism-repo/prism/prism/src ] && [ -f /workspace/prism-repo/prism/prism/requirements.txt ]; then
  PRISM_ROOT=/workspace/prism-repo/prism/prism
elif [ -d /workspace/prism-repo/prism/src ] && [ -f /workspace/prism-repo/prism/requirements.txt ]; then
  PRISM_ROOT=/workspace/prism-repo/prism
elif [ -d "$(pwd)/src" ] && [ -f "$(pwd)/requirements.txt" ]; then
  PRISM_ROOT="$(pwd)"
elif [ -d "$(dirname "$0")/src" ] && [ -f "$(dirname "$0")/requirements.txt" ]; then
  PRISM_ROOT="$(cd "$(dirname "$0")" && pwd)"
else
  echo "ERROR: Could not locate PRISM root (expected src/ and requirements.txt)."
  exit 1
fi
cd "$PRISM_ROOT"

unset PYTHONSAFEPATH || true
export PYTHONPATH="$PRISM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}"

CONFIG="${CONFIG:-${PRISM_CONFIG:-configs/vastai_cw_full.yaml}}"
TAG="${TAG:-ensemble_complete}"
export PRISM_CONFIG="$CONFIG"

SEEDS="${SEEDS:-42 123 456 789 999}"
N_TEST="${N_TEST:-1000}"
STEPS="${STEPS:-100}"
RESTARTS="${RESTARTS:-10}"
LAMBDAS="${LAMBDAS:-0.0 0.5 1.0 2.0 5.0 10.0}"
EOT_SAMPLES="${EOT_SAMPLES:-1}"
EOT_VERIFY_SAMPLES="${EOT_VERIFY_SAMPLES:-20}"
EC_MATCH_COEFF="${EC_MATCH_COEFF:-0.5}"
EC_SCORE_COEFF="${EC_SCORE_COEFF:-0.25}"
SKIP_GRADNORM="${SKIP_GRADNORM:-0}"
OUTDIR="${OUTDIR:-experiments/evaluation/${TAG}}"
LOGDIR="${LOGDIR:-logs/${TAG}}"

if [ "${SMOKE:-0}" = "1" ]; then
  first_seed=""
  for s in $SEEDS; do first_seed="$s"; break; done
  SEEDS="$first_seed"
  N_TEST="${SMOKE_N_TEST:-4}"
  STEPS="${SMOKE_STEPS:-2}"
  RESTARTS="${SMOKE_RESTARTS:-1}"
  LAMBDAS="${SMOKE_LAMBDAS:-0.0 10.0}"
fi

mkdir -p "$OUTDIR" "$LOGDIR"

echo "============================================================"
echo "PRISM Vast.ai ensemble-complete adaptive PGD"
echo "Repo root: $PRISM_ROOT"
echo "Config: $CONFIG"
echo "Seeds: $SEEDS"
echo "n_test=$N_TEST steps=$STEPS restarts=$RESTARTS lambdas=[$LAMBDAS]"
echo "ec_match_coeff=$EC_MATCH_COEFF ec_score_coeff=$EC_SCORE_COEFF skip_gradnorm=$SKIP_GRADNORM"
echo "Output: $OUTDIR"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

if ! python -c "import torch" 2>/dev/null; then
  echo "PyTorch not found; installing requirements.txt ..."
  pip install --no-cache-dir --upgrade pip setuptools wheel
  pip install --no-cache-dir -r requirements.txt
fi

python - <<'PY'
import importlib
import os
import sys

required = ["torch", "torchvision", "numpy", "scipy", "sklearn", "yaml", "tqdm", "ripser", "gudhi"]
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append((name, str(exc).splitlines()[0]))
if missing:
    print("Missing modules:")
    for name, err in missing:
        print(f"  - {name}: {err}")
    sys.exit(1)

from src.config import PATHS
needed = ["reference_profiles", "ensemble_scorer", "calibrator"]
missing_files = [PATHS[k] for k in needed if not os.path.exists(PATHS[k])]
if missing_files:
    print("Missing PRISM artifacts:")
    for path in missing_files:
        print(f"  - {path}")
    print("Run the training/calibration pipeline first, then rerun this attack.")
    sys.exit(2)

import pickle
ensemble_path = PATHS["ensemble_scorer"]
with open(ensemble_path, "rb") as f:
    ensemble = pickle.load(f)
if not isinstance(ensemble, dict):
    print(f"Ensemble artifact is not metadata dict: {ensemble_path}")
    sys.exit(3)

errors = []
if ensemble.get("n_features") != 55:
    errors.append(f"n_features={ensemble.get('n_features')}, expected 55")
if ensemble.get("feature_space_version") != "pixel-stability-v2+logitprofile+sidequad+gradnorm":
    errors.append(
        "feature_space_version="
        f"{ensemble.get('feature_space_version')}, expected "
        "pixel-stability-v2+logitprofile+sidequad+gradnorm"
    )
for key in [
    "use_dct",
    "use_softmax_entropy",
    "use_logit_profile_features",
    "use_stability_features",
    "use_grad_norm",
    "use_side_quadratic_features",
]:
    if not bool(ensemble.get(key, False)):
        errors.append(f"{key}=False")
if int(ensemble.get("logit_profile_feature_count", 0)) != 8:
    errors.append(
        f"logit_profile_feature_count={ensemble.get('logit_profile_feature_count')}, expected 8"
    )
if int(ensemble.get("stability_feature_count", 0)) != 8:
    errors.append(
        f"stability_feature_count={ensemble.get('stability_feature_count')}, expected 8"
    )
if int(ensemble.get("logistic_input_dim", 0)) <= int(ensemble.get("n_features", 0)):
    errors.append(
        f"logistic_input_dim={ensemble.get('logistic_input_dim')}, expected side-quadratic expanded dimension"
    )
if errors:
    print("Ensemble artifact is not the promoted 55-feature contract:")
    for err in errors:
        print(f"  - {err}")
    print(f"Artifact: {ensemble_path}")
    sys.exit(4)

print("Preflight: dependencies, PRISM artifacts, and 55-feature ensemble contract found.")
PY

python experiments/evaluation/run_adaptive_pgd.py --help | grep -q -- '--ensemble-complete' || {
  echo "ERROR: run_adaptive_pgd.py does not expose --ensemble-complete."
  exit 5
}

CONFIG_ARGS=(--config "$CONFIG")
EXTRA_ARGS=(
  --ensemble-complete
  --through-scorer
  --ec-match-coeff "$EC_MATCH_COEFF"
  --ec-score-coeff "$EC_SCORE_COEFF"
)
if [ "$SKIP_GRADNORM" = "1" ]; then
  EXTRA_ARGS+=(--skip-gradnorm-surrogate)
fi

for s in $SEEDS; do
  out_json="$OUTDIR/results_ensemble_complete_adaptive_pgd_seed${s}.json"
  out_jsonl="$OUTDIR/results_ensemble_complete_adaptive_pgd_seed${s}.jsonl"
  log_file="$LOGDIR/ensemble_complete_adaptive_seed${s}.log"
  echo ""
  echo "=== Seed $s -> $out_json ==="
  python experiments/evaluation/run_adaptive_pgd.py \
    "${CONFIG_ARGS[@]}" \
    --n-test "$N_TEST" \
    --seed "$s" \
    --lambdas $LAMBDAS \
    --pgd-steps "$STEPS" \
    --pgd-restarts "$RESTARTS" \
    --eot-samples "$EOT_SAMPLES" \
    --eot-verify-samples "$EOT_VERIFY_SAMPLES" \
    "${EXTRA_ARGS[@]}" \
    --checkpoint-jsonl "$out_jsonl" \
    --resume \
    --output "$out_json" \
    2>&1 | tee "$log_file"
done

echo ""
echo "DONE: ensemble-complete adaptive PGD outputs in $OUTDIR"
