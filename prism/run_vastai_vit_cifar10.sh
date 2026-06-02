#!/bin/bash
# =============================================================================
# PRISM - Vast.ai ViT-B/16 CIFAR-10 Architecture-Agnostic Pipeline
# =============================================================================
# Adds the non-CNN backbone test requested for the "architecture-agnostic"
# claim. Outputs are isolated under models/vit_cifar10/ and
# experiments/vit_cifar10/.
#
# Research-standard default:
#   SEEDS="42 123 456 789 999" N_TEST=1000 bash run_vastai_vit_cifar10.sh
#
# Fast integration smoke:
#   SMOKE_ONLY=1 bash run_vastai_vit_cifar10.sh

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
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVIDIA_TF32_OVERRIDE=1
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDA_MODULE_LOADING=LAZY
export CUDA_DEVICE_MAX_CONNECTIONS=32
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP_BIN="${PIP_BIN:-$PYTHON_BIN -m pip}"
WAIT_FOR_PIDS="${WAIT_FOR_PIDS:-}"

wait_for_pids() {
  if [ -z "$WAIT_FOR_PIDS" ]; then
    return 0
  fi
  echo ""
  echo "=== GPU guard: waiting before GPU-heavy ViT stages ==="
  echo "Waiting for PIDs: $WAIT_FOR_PIDS"
  while true; do
    alive=0
    for p in $WAIT_FOR_PIDS; do
      if kill -0 "$p" 2>/dev/null; then
        alive=1
      fi
    done
    if [ "$alive" -eq 0 ]; then
      echo "GPU guard: all watched PIDs finished; continuing."
      return 0
    fi
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader || true
    sleep "${WAIT_POLL_SECONDS:-300}"
  done
}

TAG=vit_cifar10
CONFIG=configs/vit_cifar10.yaml
SEEDS="${SEEDS:-42 123 456 789 999}"
N_TEST="${N_TEST:-1000}"
ATTACKS="${ATTACKS:-FGSM PGD Square}"
VIT_EPOCHS="${VIT_EPOCHS:-20}"
VIT_BATCH="${VIT_BATCH:-96}"
VIT_LR="${VIT_LR:-3e-5}"
ENSEMBLE_N_TRAIN="${ENSEMBLE_N_TRAIN:-1500}"
GEN_CHUNK="${GEN_CHUNK:-128}"
AA_CHUNK="${AA_CHUNK:-16}"
CW_CHUNK="${CW_CHUNK:-64}"
PGD_TRAIN_STEPS="${PGD_TRAIN_STEPS:-40}"
SQUARE_TRAIN_MAX_ITER="${SQUARE_TRAIN_MAX_ITER:-500}"
PGD_EVAL_MAX_ITER="${PGD_EVAL_MAX_ITER:-50}"
PGD_EVAL_RESTARTS="${PGD_EVAL_RESTARTS:-10}"
CW_EVAL_MAX_ITER="${CW_EVAL_MAX_ITER:-100}"
CW_EVAL_BSS="${CW_EVAL_BSS:-9}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"
TRAIN_EXTRA_ARGS=()
PROFILE_EXTRA_ARGS=()
VERIFY_MIN_ACC=0.90
VERIFY_N=1000

if [ "$SMOKE_ONLY" = "1" ]; then
  TAG=vit_cifar10_smoke
  CONFIG=configs/vit_cifar10_smoke.yaml
  SEEDS="${SEEDS_SMOKE:-42}"
  N_TEST="${N_TEST_SMOKE:-8}"
  ATTACKS="${ATTACKS_SMOKE:-FGSM}"
  VIT_EPOCHS="${VIT_EPOCHS_SMOKE:-1}"
  VIT_BATCH="${VIT_BATCH_SMOKE:-16}"
  ENSEMBLE_N_TRAIN="${ENSEMBLE_N_TRAIN_SMOKE:-6}"
  GEN_CHUNK="${GEN_CHUNK_SMOKE:-8}"
  PGD_TRAIN_STEPS="${PGD_TRAIN_STEPS_SMOKE:-2}"
  SQUARE_TRAIN_MAX_ITER="${SQUARE_TRAIN_MAX_ITER_SMOKE:-10}"
  PGD_EVAL_MAX_ITER="${PGD_EVAL_MAX_ITER_SMOKE:-2}"
  PGD_EVAL_RESTARTS="${PGD_EVAL_RESTARTS_SMOKE:-1}"
  CW_EVAL_MAX_ITER="${CW_EVAL_MAX_ITER_SMOKE:-5}"
  CW_EVAL_BSS="${CW_EVAL_BSS_SMOKE:-1}"
  TRAIN_EXTRA_ARGS=(--allow-undertrained-smoke --min-test-acc 0.0 --train-subset 64 --test-subset 32)
  PROFILE_EXTRA_ARGS=(--allow-undertrained-smoke)
  VERIFY_MIN_ACC=0.0
  VERIFY_N=50
fi
export PRISM_CONFIG="$CONFIG"
export PRISM_VAST_TAG="$TAG"
export PRISM_SMOKE_ONLY="$SMOKE_ONLY"
CKPT=models/${TAG}/vit_b16_cifar10.pt
SIDE=models/${TAG}/vit_b16_cifar10.acc.json

mkdir -p logs/${TAG} models/${TAG} experiments/${TAG}/calibration experiments/${TAG}/evaluation

echo "============================================================"
echo "PRISM Vast.ai ViT-B/16 CIFAR-10 Pipeline - $(date)"
echo "Repo root: $PRISM_ROOT"
echo "Config: $PRISM_CONFIG"
echo "Seeds: $SEEDS"
echo "N_TEST: $N_TEST"
echo "Attacks: $ATTACKS"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo ""
echo "=== Pre-flight: dependencies ==="
if [ "$SMOKE_ONLY" = "1" ]; then
  if ! $PYTHON_BIN -c "import torch, torchvision, yaml, certifi, ripser, gudhi, art" 2>/dev/null; then
    $PIP_BIN install --no-cache-dir adversarial-robustness-toolbox certifi pyyaml
  fi
elif ! $PYTHON_BIN -c "import torch, torchvision, yaml, certifi, ripser, gudhi, art, autoattack" 2>/dev/null; then
  $PIP_BIN install --no-cache-dir -r requirements.txt
fi
$PYTHON_BIN - <<'PY'
import torch, torchvision
import yaml, certifi
assert torch.cuda.is_available(), 'CUDA not available'
print('torch:', torch.__version__)
print('torchvision:', torchvision.__version__)
print('cuda:', torch.version.cuda)
print('gpu:', torch.cuda.get_device_name(0))
PY

echo ""
echo "=== Step 0: Train or verify ViT-B/16 backbone ==="
wait_for_pids
if [ ! -f "$CKPT" ] || [ ! -f "$SIDE" ]; then
  $PYTHON_BIN scripts/pretrain_vit_backbone.py \
    --dataset cifar10 \
    --epochs "$VIT_EPOCHS" \
    --batch-size "$VIT_BATCH" \
    --lr "$VIT_LR" \
    --output "$CKPT" \
    "${TRAIN_EXTRA_ARGS[@]}" \
    2>&1 | tee logs/${TAG}/step0_pretrain_vit.log
else
  echo "Found existing $CKPT and $SIDE; verifying instead of retraining."
fi

$PYTHON_BIN scripts/verify_backbone_acc.py \
  --checkpoint "$CKPT" \
  --sidecar "$SIDE" \
  --min-acc "$VERIFY_MIN_ACC" \
  --n "$VERIFY_N" \
  2>&1 | tee logs/${TAG}/step0_verify_vit.log

echo ""
echo "=== Step 0b: ViT hook/TDA compatibility smoke ==="
$PYTHON_BIN - <<'PY'
import os, torch
from src.config import LAYER_NAMES, BACKBONE_INPUT_SIZE
from src.models import load_backbone
from src.tamm.extractor import ActivationExtractor
from src.tamm.tda import TopologicalProfiler
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_backbone(dev)
extractor = ActivationExtractor(model, LAYER_NAMES)
x = torch.rand(1, 3, BACKBONE_INPUT_SIZE, BACKBONE_INPUT_SIZE, device=dev)
acts = extractor.extract(x)
profiler = TopologicalProfiler(n_subsample=32, max_dim=1)
for name in LAYER_NAMES:
    arr = acts[name].squeeze(0).detach().cpu().numpy()
    dgms = profiler.compute_diagram(arr)
    print(name, arr.shape, [len(d) for d in dgms])
extractor.cleanup()
PY

echo ""
echo "=== Step 1: Build ViT reference profiles ==="
$PYTHON_BIN scripts/build_profile_testset.py --config "$PRISM_CONFIG" \
  "${PROFILE_EXTRA_ARGS[@]}" \
  2>&1 | tee logs/${TAG}/step1_build_profile.log

echo ""
echo "=== Step 2: Train ViT ensemble detector ==="
$PYTHON_BIN scripts/train_ensemble_scorer.py \
  --config "$PRISM_CONFIG" \
  --n-train "$ENSEMBLE_N_TRAIN" \
  --source-split profile \
  --balanced-attacks \
  --pgd-train-steps "$PGD_TRAIN_STEPS" \
  --square-train-max-iter "$SQUARE_TRAIN_MAX_ITER" \
  --gen-chunk "$GEN_CHUNK" \
  --selection-objective worst_case_tpr \
  --use-stability-features \
  --use-logit-profile-features \
  --use-side-quadratic-features \
  --use-grad-norm \
  --output models/${TAG}/ensemble_scorer.pkl \
  2>&1 | tee logs/${TAG}/step2_train_ensemble.log

echo ""
echo "=== Step 3: Calibrate conformal thresholds ==="
$PYTHON_BIN scripts/calibrate_ensemble.py --config "$PRISM_CONFIG" \
  2>&1 | tee logs/${TAG}/step3_calibrate.log

echo ""
echo "=== Step 4: Validation FPR gate ==="
$PYTHON_BIN scripts/compute_ensemble_val_fpr.py --config "$PRISM_CONFIG" \
  2>&1 | tee logs/${TAG}/step4_val_fpr.log

$PYTHON_BIN - <<'PY'
import json, sys
import os
tag = os.environ.get('PRISM_VAST_TAG', 'vit_cifar10')
path = f'experiments/{tag}/calibration/{tag}_ensemble_fpr_report.json'
with open(path) as f:
    r = json.load(f)
targets = {'L1': 0.10, 'L2': 0.03, 'L3': 0.005}
bad = []
for tier, target in targets.items():
    fpr = float(r['tiers'][tier]['FPR'])
    print(f'{tier}: FPR={fpr:.4f}, target={target:.4f}')
    if fpr > target:
        bad.append((tier, fpr, target))
if bad:
    print('FPR gate failed. Tighten configs/vit_cifar10.yaml tier_cal_alpha_factors and rerun steps 3-4.')
    if os.environ.get('PRISM_SMOKE_ONLY') == '1':
        print('Smoke mode: continuing after reporting FPR miss; tiny smoke splits are not paper-valid.')
    else:
        sys.exit(1)
else:
    print('FPR gate PASS')
PY

echo ""
echo "=== Step 5: Train optional TAMSH experts ==="
if [ "$SMOKE_ONLY" = "1" ]; then
  echo "Smoke mode: skipping optional TAMSH expert training."
else
  $PYTHON_BIN scripts/train_experts.py --config "$PRISM_CONFIG" --output models/${TAG}/experts.pkl \
    2>&1 | tee logs/${TAG}/step5_train_experts.log
fi

echo ""
echo "=== Step 6: Multi-seed ViT detection evaluation ==="
$PYTHON_BIN experiments/evaluation/run_evaluation_full.py \
  --config "$PRISM_CONFIG" \
  --multi-seed \
  --seeds $SEEDS \
  --n-test "$N_TEST" \
  --attacks $ATTACKS \
  --output experiments/${TAG}/evaluation/results_${TAG}_multiseed.json \
  --gen-chunk "$GEN_CHUNK" \
  --aa-chunk "$AA_CHUNK" \
  --cw-chunk "$CW_CHUNK" \
  --pgd-max-iter "$PGD_EVAL_MAX_ITER" \
  --pgd-restarts "$PGD_EVAL_RESTARTS" \
  --cw-max-iter "$CW_EVAL_MAX_ITER" \
  --cw-bss "$CW_EVAL_BSS" \
  --cw-confidence 1.0 \
  --skip-latency \
  2>&1 | tee logs/${TAG}/step6_eval_multiseed.log

echo ""
echo "[OK] ViT architecture-agnostic pipeline complete."
echo "Results: experiments/${TAG}/evaluation/results_${TAG}_multiseed.json"
echo "Logs:    logs/${TAG}/"
