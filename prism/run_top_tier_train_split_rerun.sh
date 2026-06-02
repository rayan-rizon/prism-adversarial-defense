#!/bin/bash
# Strict CIFAR-10 top-tier rerun.
#
# Purpose:
#   Rebuild profile, scorer, and conformal calibration from CIFAR-10 train data,
#   then run the official CIFAR-10 test evaluation.
#
# This does not overwrite submitted artifacts; full outputs live under
# models/top_tier_train_split and experiments/top_tier_train_split. Smoke
# outputs live under *_smoke paths.

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

SEEDS="${SEEDS:-42 123 456 789 999}"
N_TEST="${N_TEST:-1000}"
TRAIN_SOURCE_START="${TRAIN_SOURCE_START:-8000}"
TRAIN_SOURCE_END="${TRAIN_SOURCE_END:-50000}"
ENSEMBLE_N_TRAIN="${ENSEMBLE_N_TRAIN:-1500}"
GEN_CHUNK="${GEN_CHUNK:-512}"
AA_CHUNK="${AA_CHUNK:-64}"
CW_CHUNK="${CW_CHUNK:-128}"
ATTACKS="${ATTACKS:-FGSM PGD Square CW AutoAttack}"
TRAIN_PGD_STEPS="${TRAIN_PGD_STEPS:-40}"
TRAIN_SQUARE_MAX_ITER="${TRAIN_SQUARE_MAX_ITER:-500}"
PGD_MAX_ITER="${PGD_MAX_ITER:-50}"
PGD_RESTARTS="${PGD_RESTARTS:-10}"
SQUARE_MAX_ITER="${SQUARE_MAX_ITER:-5000}"
CW_MAX_ITER="${CW_MAX_ITER:-100}"
CW_BSS="${CW_BSS:-9}"
CW_CONFIDENCE="${CW_CONFIDENCE:-1.0}"

CONFIG_WAS_SET="${CONFIG+x}"
RUN_TAG_WAS_SET="${RUN_TAG+x}"
SMOKE_ARGS=()
EVAL_EXTRA_ARGS=()
if [ "${SMOKE:-0}" = "1" ]; then
  if [ -z "$CONFIG_WAS_SET" ]; then
    CONFIG="configs/top_tier_train_split_cifar10_smoke.yaml"
  fi
  if [ -z "$RUN_TAG_WAS_SET" ]; then
    RUN_TAG="top_tier_train_split_smoke"
  fi
  SMOKE_ARGS=(--allow-undertrained-smoke)
  EVAL_EXTRA_ARGS=(--skip-latency)
  first_seed=""
  for s in $SEEDS; do first_seed="$s"; break; done
  SEEDS="${SMOKE_SEEDS:-$first_seed}"
  N_TEST="${SMOKE_N_TEST:-4}"
  TRAIN_SOURCE_START="${SMOKE_TRAIN_SOURCE_START:-132}"
  TRAIN_SOURCE_END="${SMOKE_TRAIN_SOURCE_END:-500}"
  ENSEMBLE_N_TRAIN="${SMOKE_ENSEMBLE_N_TRAIN:-8}"
  GEN_CHUNK="${SMOKE_GEN_CHUNK:-8}"
  AA_CHUNK="${SMOKE_AA_CHUNK:-4}"
  CW_CHUNK="${SMOKE_CW_CHUNK:-4}"
  ATTACKS="${SMOKE_ATTACKS:-FGSM PGD Square}"
  TRAIN_PGD_STEPS="${SMOKE_TRAIN_PGD_STEPS:-1}"
  TRAIN_SQUARE_MAX_ITER="${SMOKE_TRAIN_SQUARE_MAX_ITER:-2}"
  PGD_MAX_ITER="${SMOKE_PGD_MAX_ITER:-1}"
  PGD_RESTARTS="${SMOKE_PGD_RESTARTS:-1}"
  SQUARE_MAX_ITER="${SMOKE_SQUARE_MAX_ITER:-2}"
fi
CONFIG="${CONFIG:-configs/top_tier_train_split_cifar10.yaml}"
RUN_TAG="${RUN_TAG:-top_tier_train_split}"
LOG_DIR="${LOG_DIR:-logs/$RUN_TAG}"
EVAL_OUTPUT="${EVAL_OUTPUT:-experiments/$RUN_TAG/evaluation/results_main_multiseed.json}"
export PRISM_CONFIG="$CONFIG"
read -r -a ATTACK_LIST <<< "$ATTACKS"

TRAIN_ATTACK_ARGS=()
for attack in "${ATTACK_LIST[@]}"; do
  case "$attack" in
    CW) TRAIN_ATTACK_ARGS+=(--include-cw --cw-max-iter "$CW_MAX_ITER" --cw-bss "$CW_BSS") ;;
    AutoAttack) TRAIN_ATTACK_ARGS+=(--include-autoattack) ;;
  esac
done

mkdir -p "$LOG_DIR" \
         "experiments/$RUN_TAG/calibration" \
         "experiments/$RUN_TAG/evaluation"

echo "============================================================"
echo "PRISM strict train-split rerun"
echo "Root: $PRISM_ROOT"
echo "Config: $CONFIG"
echo "Run tag: $RUN_TAG"
echo "Mode: ${SMOKE:-0} (SMOKE=1 means code-path verification only)"
echo "Train profile/cal/val: CIFAR train split ranges from config"
echo "Scorer fitting source: train idx [$TRAIN_SOURCE_START,$TRAIN_SOURCE_END)"
echo "Final eval: official CIFAR test eval_idx from config"
echo "Seeds: $SEEDS"
echo "Attacks: $ATTACKS"
echo "Output: $EVAL_OUTPUT"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

python - <<'PY'
import importlib, sys
required = ["torch", "torchvision", "numpy", "scipy", "sklearn", "yaml", "tqdm", "ripser", "gudhi", "art", "autoattack"]
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
print("Dependency preflight: PASS")
PY

echo ""
echo "=== Step 1: train-split profiles and clean calibration scores ==="
python scripts/build_profile_testset.py \
  --config "$CONFIG" \
  --source-split train \
  "${SMOKE_ARGS[@]}" \
  2>&1 | tee "$LOG_DIR/step1_build_profile_train.log"

echo ""
echo "=== Step 2: train-split detector head, disjoint from profile/cal/val ==="
python scripts/train_ensemble_scorer.py \
  --config "$CONFIG" \
  --n-train "$ENSEMBLE_N_TRAIN" \
  --source-split train \
  --source-start-index "$TRAIN_SOURCE_START" \
  --source-end-index "$TRAIN_SOURCE_END" \
  --balanced-attacks \
  --attack-heads \
  --use-grad-norm \
  --use-stability-features \
  --use-logit-profile-features \
  --use-side-quadratic-features \
  --selection-objective worst_case_tpr \
  --pgd-train-steps "$TRAIN_PGD_STEPS" \
  --square-train-max-iter "$TRAIN_SQUARE_MAX_ITER" \
  --gen-chunk "$GEN_CHUNK" \
  "${TRAIN_ATTACK_ARGS[@]}" \
  "${SMOKE_ARGS[@]}" \
  2>&1 | tee "$LOG_DIR/step2_train_scorer_train_window.log"

echo ""
echo "=== Step 3: train-split conformal calibration and validation ==="
python scripts/calibrate_ensemble.py \
  --config "$CONFIG" \
  --source-split train \
  2>&1 | tee "$LOG_DIR/step3_calibrate_train.log"

echo ""
echo "=== Step 4: official-test multi-seed main attack evaluation ==="
python experiments/evaluation/run_evaluation_full.py \
  --config "$CONFIG" \
  --multi-seed \
  --seeds $SEEDS \
  --n-test "$N_TEST" \
  --attacks "${ATTACK_LIST[@]}" \
  --cw-engine torch \
  --cw-max-iter "$CW_MAX_ITER" \
  --cw-bss "$CW_BSS" \
  --cw-confidence "$CW_CONFIDENCE" \
  --cw-chunk "$CW_CHUNK" \
  --pgd-max-iter "$PGD_MAX_ITER" \
  --pgd-restarts "$PGD_RESTARTS" \
  --square-max-iter "$SQUARE_MAX_ITER" \
  --aa-version standard \
  --aa-chunk "$AA_CHUNK" \
  "${EVAL_EXTRA_ARGS[@]}" \
  --output "$EVAL_OUTPUT" \
  2>&1 | tee "$LOG_DIR/step4_eval_official_test.log"

echo ""
echo "DONE. If Step 4 passes, rerun tables/figures from experiments/$RUN_TAG before updating any paper claim."
