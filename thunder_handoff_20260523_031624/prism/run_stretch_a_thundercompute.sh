#!/bin/bash
# =============================================================================
# PRISM — Stretch A: WRN-28-10 Architecture Validation (ThunderCompute)
# =============================================================================
# Purpose:
#   Run the full PRISM detection pipeline on WRN-28-10 to produce a second
#   architecture data point for the paper's "architecture-agnostic" claim.
#   This script is intentionally simpler than run_vastai_full.sh — it skips
#   campaign detection and L3 recovery (those are ResNet-18 contributions)
#   and focuses on the detection table: TPR/FPR across FGSM/PGD/Square/CW/AA
#   + adaptive PGD sweep.
#
# Output artifacts:
#   experiments/wrn/evaluation/results_detection_wrn.json   (5 seeds, n=1000)
#   experiments/wrn/evaluation/results_adaptive_pgd_wrn.json
#   experiments/wrn/ablation/results_ablation_wrn.json
#   paper/tables/  (auto-updated by build_paper_tables.py)
#
# Hardware: 1× RTX A6000 (48 GB) + 6 vCPU / 48 GB — Prototyping tier.
#   Wall-clock estimate: ~50 min backbone + ~90 min eval = ~2.5 h total (~$1.20)
#
# Usage:
#   bash run_stretch_a_thundercompute.sh
#
# Environment variable overrides (optional):
#   PRISM_CONFIG    override config (default: configs/wrn_cifar10.yaml)
#   SEEDS           space-separated seed list (default: 42 123 456 789 999)
#   N_TEST          samples per seed per attack (default: 1000)

set -euo pipefail

# ── Locate PRISM root (same logic as run_vastai_full.sh) ─────────────────────
if [ -d /workspace/prism-repo/prism/prism/src ] && [ -f /workspace/prism-repo/prism/prism/requirements.txt ]; then
  PRISM_ROOT=/workspace/prism-repo/prism/prism
elif [ -d /workspace/prism-repo/prism/src ] && [ -f /workspace/prism-repo/prism/requirements.txt ]; then
  PRISM_ROOT=/workspace/prism-repo/prism
elif [ -d "$(pwd)/src" ] && [ -f "$(pwd)/requirements.txt" ]; then
  PRISM_ROOT="$(pwd)"
elif [ -d "$(dirname "$0")/src" ] && [ -f "$(dirname "$0")/requirements.txt" ]; then
  PRISM_ROOT="$(cd "$(dirname "$0")" && pwd)"
else
  echo "ERROR: Could not locate PRISM root. Checked: /workspace/prism-repo/prism, cwd, script dir."
  exit 1
fi
cd "$PRISM_ROOT"

unset PYTHONSAFEPATH || true
export PYTHONPATH="$PRISM_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# ── Config + parameters ──────────────────────────────────────────────────────
PRISM_CONFIG="${PRISM_CONFIG:-configs/wrn_cifar10.yaml}"
export PRISM_CONFIG

SEEDS="${SEEDS:-42 123 456 789 999}"
N_TEST="${N_TEST:-1000}"

# CW: same research-standard params as run_vastai_full.sh
CW_MAX_ITER=100
CW_BSS=9
CW_CONFIDENCE=1.0
CW_CHUNK=128
CW_ENGINE=torch

PGD_MAX_ITER=50
PGD_RESTARTS=10

ADAPTIVE_LAMBDAS="0.0 0.5 1.0 2.0 5.0 10.0"
ADAPTIVE_STEPS=100
ADAPTIVE_RESTARTS=10

ENSEMBLE_N_TRAIN=1500
ENSEMBLE_SOURCE_SPLIT=profile
ENSEMBLE_GEN_CHUNK=512

echo "============================================================"
echo "PRISM Stretch A — WRN-28-10 Architecture Validation"
echo "$(date)  |  host: $(hostname)"
echo "Config:  $PRISM_CONFIG"
echo "Seeds:   $SEEDS   N_TEST: $N_TEST"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Pre-flight: dependencies ──────────────────────────────────────────────────
echo ""
echo "=== Pre-flight: dependencies ==="
if ! python -c "import torch" 2>/dev/null; then
  echo "  PyTorch not found — installing requirements.txt ..."
  pip install --no-cache-dir --upgrade pip setuptools wheel
  pip install --no-cache-dir -r requirements.txt
fi
python -c "
import importlib, sys
required = ['torch','torchvision','numpy','scipy','sklearn','yaml','tqdm',
            'ripser','gudhi','art','autoattack']
missing = [(m, str(e).splitlines()[0])
           for m in required
           for e in [None]
           if (lambda mm: (importlib.import_module(mm), None)[-1]
               if (lambda: importlib.import_module(mm)) and True
               else None)(m) is None
           and not (lambda mm: __import__(mm) or True)(m)]
" 2>/dev/null || python -c "
import importlib, sys
bad = []
for m in ['torch','torchvision','numpy','scipy','sklearn','yaml','tqdm',
          'ripser','gudhi','art','autoattack']:
    try: importlib.import_module(m)
    except Exception as e: bad.append(f'{m}: {e}')
if bad:
    for b in bad: print(' MISSING:', b)
    sys.exit(1)
print('  All required modules OK')
" || { pip install --no-cache-dir -r requirements.txt; }
echo "Pre-flight: PASS"

# ── Env flags (same as run_vastai_full.sh) ───────────────────────────────────
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVIDIA_TF32_OVERRIDE=1
export TORCH_CUDNN_V8_API_ENABLED=1
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
export OMP_NUM_THREADS=4

# ── Create all output directories upfront ────────────────────────────────────
mkdir -p logs \
         models/wrn \
         experiments/wrn/calibration \
         experiments/wrn/evaluation \
         experiments/wrn/ablation

# ── Step 0: GPU + PyTorch check ───────────────────────────────────────────────
echo ""
echo "=== Step 0: GPU + PyTorch ==="
python -c "
import torch
print('torch:', torch.__version__, '| cuda:', torch.version.cuda)
print('device:', torch.cuda.get_device_name(0))
x = torch.randn(512, 512, device='cuda')
torch.cuda.synchronize()
print('smoke GPU matmul: OK')
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
print('determinism flags: OK')
"
echo "Step 0: PASS"

# ── Step 0a: Preflight config check ──────────────────────────────────────────
echo ""
echo "=== Step 0a: WRN config preflight ==="
python -c "
import os, sys
os.environ['PRISM_CONFIG'] = 'configs/wrn_cifar10.yaml'
from src.config import BACKBONE_ARCH, LAYER_NAMES, BACKBONE_CHECKPOINT_PATH
assert BACKBONE_ARCH == 'wrn28_10', f'Expected wrn28_10, got {BACKBONE_ARCH}'
assert 'layer1' in LAYER_NAMES, f'Expected layer1 in LAYER_NAMES, got {LAYER_NAMES}'
assert 'layer4' not in LAYER_NAMES, f'layer4 should not be in WRN config'
assert 'wrn' in BACKBONE_CHECKPOINT_PATH, \
    f'checkpoint path should contain wrn, got {BACKBONE_CHECKPOINT_PATH}'
print(f'[OK] arch={BACKBONE_ARCH}  layers={LAYER_NAMES}  ckpt={BACKBONE_CHECKPOINT_PATH}')
"
echo "Step 0a: PASS"

# ── Step 1: Pretrain WRN-28-10 backbone ──────────────────────────────────────
echo ""
echo "=== Step 1: Pretrain WRN-28-10 backbone (~50 min on A6000) ==="
mkdir -p models/wrn
REUSE_BACKBONE=0
if [ -f models/cifar_wrn28_10.pt ] && [ -f models/cifar_wrn28_10.acc.json ]; then
  echo "  Found existing checkpoint — verifying accuracy..."
  if PRISM_CONFIG="$PRISM_CONFIG" python scripts/verify_backbone_acc.py \
       --checkpoint models/cifar_wrn28_10.pt \
       --sidecar    models/cifar_wrn28_10.acc.json \
       --min-acc 0.93 --n 1000 \
       2>&1 | tee logs/step1_verify_wrn.log; then
    REUSE_BACKBONE=1
    echo "  Existing WRN-28-10 checkpoint PASSES accuracy gate — reusing."
  else
    echo "  Checkpoint failed gate — retraining."
    rm -f models/cifar_wrn28_10.pt models/cifar_wrn28_10.acc.json
  fi
fi
if [ "$REUSE_BACKBONE" -eq 0 ]; then
  PRISM_CONFIG="$PRISM_CONFIG" \
  python scripts/pretrain_wrn_backbone.py \
    2>&1 | tee logs/step1_pretrain_wrn.log
  STEP1_EXIT=${PIPESTATUS[0]:-$?}
  if [ "$STEP1_EXIT" -ne 0 ]; then
    echo "ERROR: WRN backbone training failed. Check logs/step1_pretrain_wrn.log"
    exit 1
  fi
  # Post-train accuracy gate.
  PRISM_CONFIG="$PRISM_CONFIG" python scripts/verify_backbone_acc.py \
    --checkpoint models/cifar_wrn28_10.pt \
    --sidecar    models/cifar_wrn28_10.acc.json \
    --min-acc 0.93 --n 1000 \
    || { echo "ERROR: post-train WRN backbone gate failed"; exit 1; }
fi
echo "Step 1: PASS — WRN-28-10 backbone ready"

# ── Step 2: Build reference profiles ─────────────────────────────────────────
echo ""
echo "=== Step 2: Build WRN reference profiles [test 0-4999] ==="
PRISM_CONFIG="$PRISM_CONFIG" \
python scripts/build_profile_testset.py \
  2>&1 | tee logs/step2_build_profile_wrn.log
if [ ${PIPESTATUS[0]:-$?} -ne 0 ]; then
  echo "ERROR: build_profile_testset failed. Check logs/step2_build_profile_wrn.log"
  exit 1
fi
echo "Step 2: DONE"

# ── Steps 3 + 3b: Train ensemble scorer + no-TDA variant (parallel) ──────────
echo ""
echo "=== Step 3: Train WRN ensemble scorer ==="

# 3b: ensemble-no-TDA in background (overlaps with 3)
PID_3B=""
if python scripts/train_ensemble_scorer.py --help 2>&1 | grep -q -- '--no-tda-features'; then
  PRISM_CONFIG="$PRISM_CONFIG" \
  python scripts/train_ensemble_scorer.py \
    --config "$PRISM_CONFIG" \
    --n-train $ENSEMBLE_N_TRAIN \
    --source-split $ENSEMBLE_SOURCE_SPLIT \
    --balanced-attacks \
    --pgd-train-steps 40 \
    --square-train-max-iter 500 \
    --gen-chunk $ENSEMBLE_GEN_CHUNK \
    --selection-objective worst_case_tpr \
    --use-stability-features \
    --use-logit-profile-features \
    --use-side-quadratic-features \
    --no-tda-features \
    --output models/wrn/ensemble_no_tda.pkl \
    > logs/step3b_retrain_no_tda_wrn.log 2>&1 &
  PID_3B=$!
  echo "  Step 3b (no-TDA) launched in background (PID=$PID_3B)"
fi

# 3: main scorer — foreground
PRISM_CONFIG="$PRISM_CONFIG" \
python scripts/train_ensemble_scorer.py \
  --config "$PRISM_CONFIG" \
  --n-train $ENSEMBLE_N_TRAIN \
  --source-split $ENSEMBLE_SOURCE_SPLIT \
  --balanced-attacks \
  --pgd-train-steps 40 \
  --square-train-max-iter 500 \
  --gen-chunk $ENSEMBLE_GEN_CHUNK \
  --selection-objective worst_case_tpr \
  --use-stability-features \
  --use-logit-profile-features \
  --use-side-quadratic-features \
  --use-grad-norm \
  --output models/wrn/ensemble_scorer.pkl \
  2>&1 | tee logs/step3_retrain_wrn.log
if [ ${PIPESTATUS[0]:-$?} -ne 0 ]; then
  echo "ERROR: ensemble scorer training failed"; exit 1
fi
echo "Step 3: DONE — models/wrn/ensemble_scorer.pkl"

# ── Step 4: Calibrate conformal thresholds ───────────────────────────────────
echo ""
echo "=== Step 4: Calibrate conformal thresholds (WRN) ==="
PRISM_CONFIG="$PRISM_CONFIG" \
python scripts/calibrate_ensemble.py \
  2>&1 | tee logs/step4_calibrate_wrn.log
if [ ${PIPESTATUS[0]:-$?} -ne 0 ]; then
  echo "ERROR: calibration failed"; exit 1
fi
echo "Step 4: DONE"

# ── Step 5: Validation FPR gate ───────────────────────────────────────────────
echo ""
echo "=== Step 5: Validation FPR gate [test 7000-7999] ==="
PRISM_CONFIG="$PRISM_CONFIG" \
python scripts/compute_ensemble_val_fpr.py \
  2>&1 | tee logs/step5_val_fpr_wrn.log
if [ ${PIPESTATUS[0]:-$?} -ne 0 ]; then
  echo "ERROR: val FPR computation failed"; exit 1
fi

# FPR gate check — same targets as ResNet-18 (L1≤0.10, L2≤0.03, L3≤0.005)
python -c "
import json, sys
with open('experiments/wrn/calibration/ensemble_fpr_report.json') as f:
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
print('FPR gate: ALL PASS — proceeding to evaluation')
" || {
  echo "FIX: Lower tier_cal_alpha_factors in configs/wrn_cifar10.yaml, re-run steps 4-5"
  exit 1
}

# Wait for no-TDA scorer if it's still running
if [ -n "$PID_3B" ]; then
  wait $PID_3B && echo "  Step 3b (no-TDA): DONE" || \
    echo "  Step 3b (no-TDA): failed — C1/no-TDA ablation arm unavailable (expected)."
fi

# ── LOCK ─────────────────────────────────────────────────────────────────────
echo ""
echo "=== ARTIFACTS LOCKED ==="
python -c "
import pickle, hashlib
def h(p):
    return hashlib.sha256(open(p,'rb').read()).hexdigest()[:16]
print(f'  ensemble_scorer.pkl : {h(\"models/wrn/ensemble_scorer.pkl\")}')
print(f'  calibrator.pkl      : {h(\"models/wrn/calibrator.pkl\")}')
print(f'  reference_profiles  : {h(\"models/wrn/reference_profiles.pkl\")}')
"

# ── Step 6: Parallel evaluation — CW + fast attacks + adaptive PGD ───────────
echo ""
echo "=== Step 6: Parallel evaluation (CW + FGSM/PGD/Square/AA + adaptive PGD) ==="

# 6A: CW-L2
PRISM_CONFIG="$PRISM_CONFIG" \
python experiments/evaluation/run_evaluation_full.py \
  --n-test $N_TEST --attacks CW \
  --multi-seed --seeds $SEEDS \
  --cw-max-iter $CW_MAX_ITER --cw-bss $CW_BSS --cw-chunk $CW_CHUNK \
  --cw-confidence $CW_CONFIDENCE \
  --cw-engine $CW_ENGINE \
  --skip-latency \
  --checkpoint-interval 100 \
  --output experiments/wrn/evaluation/results_cw_wrn.json \
  2>&1 | tee logs/step6a_cw_wrn.log &
PID_CW=$!
echo "  Step 6A CW started (PID=$PID_CW)"

# 6B: Fast attacks
PRISM_CONFIG="$PRISM_CONFIG" \
python experiments/evaluation/run_evaluation_full.py \
  --n-test $N_TEST --attacks FGSM PGD Square AutoAttack \
  --multi-seed --seeds $SEEDS \
  --gen-chunk 128 --square-max-iter 5000 \
  --pgd-max-iter $PGD_MAX_ITER --pgd-restarts $PGD_RESTARTS \
  --aa-version standard --aa-chunk 64 \
  --skip-latency \
  --checkpoint-interval 100 \
  --output experiments/wrn/evaluation/results_fast_wrn.json \
  2>&1 | tee logs/step6b_fast_wrn.log &
PID_FAST=$!
echo "  Step 6B fast attacks started (PID=$PID_FAST)"

# 6C: Adaptive PGD — 5 seeds in parallel
STEP6C_PIDS=""
for s in $SEEDS; do
  PRISM_CONFIG="$PRISM_CONFIG" \
  python experiments/evaluation/run_adaptive_pgd.py \
    --n-test $N_TEST --seed $s \
    --lambdas $ADAPTIVE_LAMBDAS \
    --pgd-steps $ADAPTIVE_STEPS \
    --pgd-restarts $ADAPTIVE_RESTARTS \
    --through-scorer \
    --output experiments/wrn/evaluation/results_adaptive_pgd_wrn_seed${s}.json \
    2>&1 | tee logs/step6c_adaptive_wrn_seed${s}.log &
  STEP6C_PIDS="$STEP6C_PIDS $!"
done
echo "  Step 6C adaptive PGD (5 seeds) started"

# 6D: Ablation
PRISM_CONFIG="$PRISM_CONFIG" \
python experiments/ablation/run_ablation_paper.py \
  --n $N_TEST \
  --multi-seed --seeds $SEEDS \
  --attacks FGSM PGD Square CW \
  --output experiments/wrn/ablation/results_ablation_wrn.json \
  2>&1 | tee logs/step6d_ablation_wrn.log &
PID_ABLATION=$!
echo "  Step 6D ablation started (PID=$PID_ABLATION)"

echo ""
echo "  Monitor logs:"
echo "    tail -f logs/step6a_cw_wrn.log"
echo "    tail -f logs/step6b_fast_wrn.log"
echo "    tail -f logs/step6c_adaptive_wrn_seed42.log"
echo ""

# Wait for all evaluation jobs
EVAL_FAIL=0
wait $PID_CW   || { echo "ERROR: Step 6A CW failed";           EVAL_FAIL=1; }
wait $PID_FAST || { echo "ERROR: Step 6B fast attacks failed";  EVAL_FAIL=1; }
for pid in $STEP6C_PIDS; do
  wait $pid    || { echo "ERROR: Step 6C adaptive PGD seed failed"; EVAL_FAIL=1; }
done
wait $PID_ABLATION || echo "WARN: Step 6D ablation failed (non-fatal)"
echo "Step 6: evaluation complete"

if [ $EVAL_FAIL -ne 0 ]; then
  echo "ERROR: One or more evaluation steps failed. Check logs above."
  exit 2
fi

# ── Step 7: Build paper tables ───────────────────────────────────────────────
echo ""
echo "=== Step 7: Build LaTeX tables ==="
if [ -f scripts/build_paper_tables.py ]; then
  PRISM_CONFIG="$PRISM_CONFIG" \
  python scripts/build_paper_tables.py \
    --results-dir experiments/wrn \
    --out-dir paper/tables \
    --arch-tag wrn \
    2>&1 | tee logs/step7_tables_wrn.log || \
    echo "WARN: build_paper_tables failed — run manually after download."
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "PRISM Stretch A — COMPLETE"
echo "$(date)"
echo "============================================================"
echo ""
echo "Result files:"
ls -lh experiments/wrn/evaluation/*.json \
       experiments/wrn/ablation/*.json 2>/dev/null || true
echo ""
echo "Download command (run from your laptop after the run):"
echo "  scp -r <user>@<host>:/path/to/prism/experiments/wrn ./stretch_a_results/"
echo "  scp -r <user>@<host>:/path/to/prism/logs/step6*wrn*.log ./stretch_a_logs/"
