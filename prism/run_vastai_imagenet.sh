#!/bin/bash
# =============================================================================
# PRISM - Vast.ai ImageNet-100 / ResNet-50 @ 224x224 Pipeline (Exp 2)
# =============================================================================
# Deflects the "CIFAR-only / 32x32 toy benchmark" reviewer objection with ONE
# standard-attack run at ImageNet grade. Standard attacks only (FGSM/PGD/Square
# at eps=8/255); NO adaptive, NO CW, NO AutoAttack, NO latency claim -- scoped
# exactly like the ViT-B/16 transfer row. Outputs isolated under
# models/imagenet/ and experiments/imagenet/.
#
# PREREQUISITE (one-time, on the box): stage an ImageNet-100 ImageFolder at
#   data/imagenet100/  (one subdir per class, >=10k images total).
# Set IMAGENET_DIR below or via env if you stage it elsewhere.
#
# Research-standard default:
#   SEEDS="42 123 456" N_TEST=1000 bash run_vastai_imagenet.sh
#
# Fast integration smoke (no paper validity; just checks the wiring + 224x224
# ResNet-50 TDA extraction):
#   SMOKE_ONLY=1 bash run_vastai_imagenet.sh

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
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP_BIN="${PIP_BIN:-$PYTHON_BIN -m pip}"

TAG=imagenet
CONFIG=configs/imagenet.yaml
SEEDS="${SEEDS:-42 123 456}"
N_TEST="${N_TEST:-1000}"
ATTACKS="${ATTACKS:-FGSM PGD Square}"
IMAGENET_DIR="${IMAGENET_DIR:-data/imagenet100}"
RN50_EPOCHS="${RN50_EPOCHS:-15}"
RN50_BATCH="${RN50_BATCH:-256}"          # RTX 5090 32GB headroom; saturates GPU
RN50_LR="${RN50_LR:-1e-3}"
RN50_WORKERS="${RN50_WORKERS:-32}"       # dataloader workers (box has many cores)
ENSEMBLE_N_TRAIN="${ENSEMBLE_N_TRAIN:-1500}"
GEN_CHUNK="${GEN_CHUNK:-256}"            # big GPU attack-gen batch (5090 32GB)
# Parallel persistent-homology: ripser is CPU-bound and the real bottleneck.
# Fan it across cores. Default = min(64, nproc/2) so we leave cores for the
# GPU dataloader + attack generation. Override with WORKERS=N.
_NPROC="$(nproc 2>/dev/null || echo 8)"
WORKERS="${WORKERS:-$(( _NPROC/2 < 64 ? _NPROC/2 : 64 ))}"
BUILD_GPU_BATCH="${BUILD_GPU_BATCH:-256}"
BUILD_LOADER_WORKERS="${BUILD_LOADER_WORKERS:-16}"
PGD_TRAIN_STEPS="${PGD_TRAIN_STEPS:-40}"
SQUARE_TRAIN_MAX_ITER="${SQUARE_TRAIN_MAX_ITER:-500}"
PGD_EVAL_MAX_ITER="${PGD_EVAL_MAX_ITER:-50}"
PGD_EVAL_RESTARTS="${PGD_EVAL_RESTARTS:-10}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"
TRAIN_EXTRA_ARGS=()
PROFILE_EXTRA_ARGS=()

if [ "$SMOKE_ONLY" = "1" ]; then
  TAG=imagenet_smoke
  CONFIG=configs/imagenet_smoke.yaml
  PROFILE_EXTRA_ARGS=(--allow-undertrained-smoke)  # 1-epoch backbone won't clear the 0.90 profile gate
  SEEDS="${SEEDS_SMOKE:-42}"
  N_TEST="${N_TEST_SMOKE:-8}"
  ATTACKS="${ATTACKS_SMOKE:-FGSM}"
  RN50_EPOCHS="${RN50_EPOCHS_SMOKE:-1}"
  RN50_BATCH="${RN50_BATCH_SMOKE:-32}"
  RN50_WORKERS="${RN50_WORKERS_SMOKE:-4}"
  ENSEMBLE_N_TRAIN="${ENSEMBLE_N_TRAIN_SMOKE:-6}"
  GEN_CHUNK="${GEN_CHUNK_SMOKE:-8}"
  WORKERS="${WORKERS_SMOKE:-4}"
  BUILD_GPU_BATCH="${BUILD_GPU_BATCH_SMOKE:-16}"
  BUILD_LOADER_WORKERS="${BUILD_LOADER_WORKERS_SMOKE:-2}"
  PGD_TRAIN_STEPS="${PGD_TRAIN_STEPS_SMOKE:-2}"
  SQUARE_TRAIN_MAX_ITER="${SQUARE_TRAIN_MAX_ITER_SMOKE:-10}"
  PGD_EVAL_MAX_ITER="${PGD_EVAL_MAX_ITER_SMOKE:-2}"
  PGD_EVAL_RESTARTS="${PGD_EVAL_RESTARTS_SMOKE:-1}"
fi
export PRISM_CONFIG="$CONFIG"
export PRISM_VAST_TAG="$TAG"
export PRISM_SMOKE_ONLY="$SMOKE_ONLY"
CKPT=models/${TAG}/resnet50_imagenet100.pt

mkdir -p logs/${TAG} models/${TAG} experiments/${TAG}/calibration experiments/${TAG}/evaluation

echo "============================================================"
echo "PRISM Vast.ai ImageNet-100 / ResNet-50 Pipeline - $(date)"
echo "Repo root: $PRISM_ROOT"
echo "Config: $PRISM_CONFIG"
echo "ImageNet dir: $IMAGENET_DIR"
echo "Seeds: $SEEDS   N_TEST: $N_TEST   Attacks: $ATTACKS (eps=8/255)"
echo "GPU: fine-tune batch=$RN50_BATCH, gen_chunk=$GEN_CHUNK | PH workers=$WORKERS (nproc=$_NPROC)"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if [ ! -d "$IMAGENET_DIR" ]; then
  echo "ERROR: ImageNet-100 dir '$IMAGENET_DIR' not found."
  echo "Stage an ImageFolder (one subdir per class, >=10k images) there first."
  exit 1
fi

echo ""
echo "=== Pre-flight: dependencies ==="
if ! $PYTHON_BIN -c "import torch, torchvision, yaml, certifi, ripser, gudhi, art" 2>/dev/null; then
  $PIP_BIN install --no-cache-dir -r requirements.txt
fi
$PYTHON_BIN - <<'PY'
import torch, torchvision
assert torch.cuda.is_available(), 'CUDA not available'
print('torch:', torch.__version__, '| torchvision:', torchvision.__version__,
      '| cuda:', torch.version.cuda, '| gpu:', torch.cuda.get_device_name(0))
PY

echo ""
echo "=== Step 0: Fine-tune or reuse ResNet-50 (100-way head) ==="
if [ ! -f "$CKPT" ]; then
  $PYTHON_BIN scripts/pretrain_imagenet100_backbone.py \
    --data-dir "$IMAGENET_DIR" \
    --epochs "$RN50_EPOCHS" \
    --batch-size "$RN50_BATCH" \
    --lr "$RN50_LR" \
    --num-workers "$RN50_WORKERS" \
    --num-classes 100 \
    --output "$CKPT" \
    "${TRAIN_EXTRA_ARGS[@]}" \
    2>&1 | tee logs/${TAG}/step0_pretrain_resnet50.log
else
  echo "Found existing $CKPT; skipping fine-tune."
fi

echo ""
echo "=== Step 0b: ResNet-50 @ 224x224 hook/TDA compatibility smoke ==="
echo "    (the main scientific risk -- confirms persistence separation survives"
echo "     150-point subsampling over large ResNet-50 activation maps)"
$PYTHON_BIN - <<'PY'
import torch
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
    print(name, 'act', arr.shape, 'H0/H1 pts', [len(d) for d in dgms])
extractor.cleanup()
print('OK: ResNet-50 224x224 activation extraction + TDA path works.')
PY

echo ""
echo "=== Step 1: Build ResNet-50 reference profiles (parallel PH: $WORKERS workers) ==="
$PYTHON_BIN scripts/build_profile_testset.py --config "$PRISM_CONFIG" \
  --workers "$WORKERS" \
  --gpu-batch "$BUILD_GPU_BATCH" \
  --loader-workers "$BUILD_LOADER_WORKERS" \
  "${PROFILE_EXTRA_ARGS[@]}" \
  2>&1 | tee logs/${TAG}/step1_build_profile.log

echo ""
echo "=== Step 2: Train ImageNet ensemble detector ==="
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
import json, sys, os
tag = os.environ.get('PRISM_VAST_TAG', 'imagenet')
path = f'experiments/{tag}/calibration/{tag}_ensemble_fpr_report.json'
with open(path) as f:
    r = json.load(f)
targets = {'L1': 0.10, 'L2': 0.03, 'L3': 0.005}
bad = []
for tier, target in targets.items():
    fpr = float(r['tiers'][tier]['FPR'])
    print(f'{tier}: FPR={fpr:.4f}, target={target:.4f}')
    if fpr > target:
        bad.append(tier)
if bad:
    print('FPR gate failed. Tighten configs/imagenet.yaml tier_cal_alpha_factors and rerun steps 3-4.')
    if os.environ.get('PRISM_SMOKE_ONLY') == '1':
        print('Smoke mode: continuing after FPR miss; tiny smoke splits are not paper-valid.')
    else:
        sys.exit(1)
else:
    print('FPR gate PASS')
PY

echo ""
echo "=== Step 5: Multi-seed ImageNet detection evaluation (standard attacks) ==="
$PYTHON_BIN experiments/evaluation/run_evaluation_full.py \
  --config "$PRISM_CONFIG" \
  --multi-seed \
  --seeds $SEEDS \
  --n-test "$N_TEST" \
  --attacks $ATTACKS \
  --output experiments/${TAG}/evaluation/results_${TAG}_multiseed.json \
  --gen-chunk "$GEN_CHUNK" \
  --pgd-max-iter "$PGD_EVAL_MAX_ITER" \
  --pgd-restarts "$PGD_EVAL_RESTARTS" \
  --skip-latency \
  2>&1 | tee logs/${TAG}/step5_eval_multiseed.log

echo ""
echo "[OK] ImageNet-100 standard-attack pipeline complete."
echo "Results: experiments/${TAG}/evaluation/results_${TAG}_multiseed.json"
echo "Logs:    logs/${TAG}/"
