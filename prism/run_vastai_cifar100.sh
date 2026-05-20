#!/bin/bash
# =============================================================================
# PRISM — Vast.ai CIFAR-100 Full Pipeline (Research-Plan P1.1)
# =============================================================================
# Mirror of run_vastai_full.sh for the CIFAR-100 second-dataset evaluation.
# Uses configs/cifar100.yaml; all artifacts land under models/cifar100/ and
# experiments/*_cifar100/ so the canonical CIFAR-10 run is not clobbered.
#
# Expected wall-clock: ~1 GPU-week on a 5090-class card (full retraining of
# reference profiles + ensemble + calibrator + experts against the CIFAR-100
# clean-score distribution, then all evaluation phases).
#
# If CIFAR-100 cal→val FPR overruns target by >1 pp, tighten
# conformal.tier_cal_alpha_factors.L3 from 0.50 → 0.45 in configs/cifar100.yaml
# and re-run from Step 3.

set -euo pipefail

# Resolve PRISM root robustly for both common Vast.ai layouts:
#   /workspace/prism-repo/prism
#   /workspace/prism-repo/prism/prism
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
  echo "       Checked: /workspace/prism-repo/prism, /workspace/prism-repo/prism/prism, cwd, script dir."
  exit 1
fi
cd "$PRISM_ROOT"

# Ensure local package imports (e.g., from src.models...) always work,
# even when Python runs with safe-path mode enabled by the image.
unset PYTHONSAFEPATH || true
export PYTHONPATH="$PRISM_ROOT${PYTHONPATH:+:$PYTHONPATH}"

CONFIG=configs/cifar100.yaml
TAG=cifar100

SEEDS="42 123 456 789 999"
N_TEST=1000

# Research-standard CW (post-audit 2026-05-18): max_iter=100, bss=9, κ=1.0.
CW_MAX_ITER=100
CW_BSS=9
CW_CONFIDENCE=1.0
CW_CHUNK=128

# Research-standard PGD (RobustBench): 50 iter × 10 random restarts.
PGD_MAX_ITER=50
PGD_RESTARTS=10

ADAPTIVE_LAMBDAS="0.0 0.5 1.0 2.0 5.0 10.0"
ADAPTIVE_STEPS=100
ADAPTIVE_RESTARTS=10
FGSM_OVERSAMPLE=2.5

echo "============================================================"
echo "PRISM Vast.ai CIFAR-100 Pipeline — $(date)"
echo "Config: $CONFIG"
echo "Repo root: $PRISM_ROOT"
echo "Instance: $(hostname)"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# ── Pre-flight: PyTorch + deps installed? ─────────────────────────────────────
echo ""
echo "=== Pre-flight: verify dependencies ==="
if ! python -c "import torch" 2>/dev/null; then
  echo "  PyTorch NOT FOUND — installing from requirements.txt ..."
  pip install --no-cache-dir --upgrade pip setuptools wheel
  pip install --no-cache-dir -r requirements.txt || {
    echo "ERROR: pip install -r requirements.txt failed."
    exit 1
  }
fi
python -c "import torch" 2>/dev/null || {
  echo "ERROR: PyTorch import still fails. Aborting."
  exit 1
}
python -c "
import importlib, sys
required = ['torch','torchvision','numpy','scipy','sklearn','yaml','tqdm','ripser','gudhi','art','autoattack']
missing = []
for m in required:
    try: importlib.import_module(m)
    except Exception as e: missing.append((m, str(e).splitlines()[0]))
if missing:
    print('MISSING modules:')
    for m, e in missing: print(f'  - {m}: {e}')
    sys.exit(1)
print('  All required modules import OK.')
" || {
  echo "  Re-running pip install -r requirements.txt ..."
  pip install --no-cache-dir -r requirements.txt
  python -c "import torch, ripser, art, autoattack" || { echo "ERROR: deps missing"; exit 1; }
}
echo "Pre-flight: PASS"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVIDIA_TF32_OVERRIDE=1
export TORCH_CUDNN_V8_API_ENABLED=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

mkdir -p logs/${TAG} \
         models/${TAG} \
         experiments/calibration \
         experiments/evaluation \
         experiments/ablation \
         experiments/campaign \
         experiments/recovery

# ── Step 0: GPU + PyTorch verification ───────────────────────────────────────
echo ""
echo "=== Step 0: GPU + PyTorch verification ==="
python -c "
import torch
print('torch:', torch.__version__)
print('cuda:', torch.version.cuda)
assert torch.cuda.is_available(), 'CUDA not available'
print('gpu:', torch.cuda.get_device_name(0))
print('vram:', round(torch.cuda.get_device_properties(0).total_mem / 1024**3, 1), 'GB')
assert int(torch.__version__.split('.')[0]) >= 2, f'Need PyTorch >= 2.0, got {torch.__version__}'
print('OK')
"
echo "Step 0: PASS"

# ── Step 0a: Pretrain the CIFAR-100 ResNet-18 backbone ───────────────────────
CKPT=models/${TAG}/cifar_resnet18_c100.pt

echo ""
echo "=== Step 0a: Pretrain CIFAR-100 ResNet-18 backbone ==="
if [ -f "$CKPT" ]; then
  echo "  Checkpoint exists: $CKPT — skipping pretraining."
  echo "  (Delete $CKPT to force retrain.)"
else
  python scripts/pretrain_cifar_backbone.py \
    --dataset cifar100 --num-classes 100 \
    --output "$CKPT" --min-test-acc 0.73 \
    2>&1 > >(tee logs/${TAG}/step0a_pretrain_backbone.log)
  STEP0A_EXIT=${PIPESTATUS[0]:-$?}
  if [ "$STEP0A_EXIT" -ne 0 ]; then
    echo "ERROR: Step 0a failed. Check logs/${TAG}/step0a_pretrain_backbone.log"
    exit 1
  fi
fi

# Post-step verification
python -c "
import torch, sys
sd = torch.load('$CKPT', map_location='cpu', weights_only=True)
from src.models.cifar_resnet import cifar_resnet18
m = cifar_resnet18(num_classes=100)
m.load_state_dict(sd)
out = m(torch.randn(1, 3, 32, 32))
assert out.shape == (1, 100), f'Expected (1,100), got {out.shape}'
print(f'Backbone OK: {sum(p.numel() for p in m.parameters())/1e6:.2f}M params, output {out.shape}')
" || { echo "ERROR: Backbone verification failed."; exit 1; }

# ── Step 1: Build reference profiles (CIFAR-100) ─────────────────────────────
echo ""
echo "=== Step 1: Build Reference Profiles [$TAG] ==="
python scripts/build_profile_testset.py --config $CONFIG \
  2>&1 > >(tee logs/${TAG}/step1_build_profile.log)

# ── Step 2: Train ensemble (CIFAR-100) ───────────────────────────────────────
echo ""
echo "=== Step 2: Train Ensemble [$TAG, fgsm-os=$FGSM_OVERSAMPLE] ==="
python scripts/train_ensemble_scorer.py \
  --config $CONFIG \
  --fgsm-oversample $FGSM_OVERSAMPLE \
  --include-cw --include-autoattack \
  --cw-max-iter 40 --cw-bss 5 \
  --output models/${TAG}/ensemble_scorer.pkl \
  2>&1 > >(tee logs/${TAG}/step2_retrain.log)

# Step 2c: ensemble-no-TDA variant (P0.6) — CIFAR-100
if python scripts/train_ensemble_scorer.py --help 2>&1 | grep -q -- '--no-tda-features'; then
  python scripts/train_ensemble_scorer.py \
    --config $CONFIG \
    --fgsm-oversample $FGSM_OVERSAMPLE \
    --include-cw --include-autoattack \
    --cw-max-iter 40 --cw-bss 5 \
    --no-tda-features \
    --output models/${TAG}/ensemble_no_tda.pkl \
    2>&1 > >(tee logs/${TAG}/step2c_no_tda.log)
fi

# Step 2d: differentiated experts (P0.5) — CIFAR-100
if [ -f scripts/train_experts.py ]; then
  python scripts/train_experts.py \
    --config $CONFIG \
    --output models/${TAG}/experts.pkl \
    2>&1 > >(tee logs/${TAG}/step2d_experts.log)
fi

# ── Step 3: Calibrate (CIFAR-100) ────────────────────────────────────────────
echo ""
echo "=== Step 3: Calibrate Conformal Thresholds [$TAG] ==="
python scripts/calibrate_ensemble.py --config $CONFIG \
  2>&1 > >(tee logs/${TAG}/step3_calibrate.log)

echo ""
echo "=== Step 3b: Calibrate Ensemble-no-TDA Arm [$TAG, C1] ==="
python scripts/calibrate_ensemble.py \
  --config $CONFIG \
  --ensemble-path models/${TAG}/ensemble_no_tda.pkl \
  --output models/${TAG}/calibrator_no_tda.pkl \
  2>&1 > >(tee logs/${TAG}/step3b_calibrate_no_tda.log)

python -c "
import pickle, sys
exp = pickle.load(open('models/${TAG}/experts.pkl', 'rb'))
if not isinstance(exp, dict):
    sys.exit('experts.pkl must be a dict artifact')
if int(exp.get('output_dim', -1)) != 100:
    sys.exit(f'experts output_dim={exp.get(\"output_dim\")}, expected 100 for CIFAR-100')
for p in ['models/${TAG}/calibrator_base.pkl', 'models/${TAG}/calibrator_no_tda.pkl', 'models/${TAG}/ensemble_no_tda.pkl']:
    open(p, 'rb').close()
print('[OK] C1/C4 CIFAR-100 artifacts verified')
"

# ── Step 4: FPR gate (CIFAR-100) ─────────────────────────────────────────────
echo ""
echo "=== Step 4: FPR Gate [$TAG] ==="
python scripts/compute_ensemble_val_fpr.py --config $CONFIG \
  2>&1 > >(tee logs/${TAG}/step4_val_fpr.log)

python -c "
import json, sys
with open('experiments/calibration/${TAG}_ensemble_fpr_report.json') as f:
    r = json.load(f)
targets = [('L1', 0.10), ('L2', 0.03), ('L3', 0.005)]
failures = []
for tier, tgt in targets:
    fpr = r['tiers'][tier]['FPR']
    status = 'PASS' if fpr <= tgt else 'FAIL'
    print(f'  {tier} FPR={fpr:.4f}  target={tgt}  [{status}]')
    if fpr > tgt: failures.append(f'{tier} FPR={fpr:.4f} > {tgt}')
if failures:
    print(f'GATE FAIL: {failures}')
    print('FIX: tighten tier_cal_alpha_factors.L3 0.50 → 0.45 in $CONFIG, re-run steps 3-4')
    sys.exit(1)
print('ALL GATES PASS')
" || exit 1

# ── Steps 5–6–7: parallel eval (CIFAR-100) ───────────────────────────────────
echo ""
echo "=== Steps 5+6+7: Parallel Eval [$TAG, n=$N_TEST × 5 seeds] ==="

python experiments/evaluation/run_evaluation_full.py \
  --config $CONFIG \
  --n-test $N_TEST --attacks CW \
  --multi-seed --seeds $SEEDS \
  --cw-max-iter $CW_MAX_ITER --cw-bss $CW_BSS --cw-chunk $CW_CHUNK \
  --cw-confidence $CW_CONFIDENCE \
  --cw-engine torch \
  --skip-latency \
  --checkpoint-interval 100 \
  --output experiments/evaluation/results_${TAG}_cw_n${N_TEST}_ms5.json \
  2>&1 | tee logs/${TAG}/step5_cw_ms5.log &
PID_CW=$!

python experiments/evaluation/run_evaluation_full.py \
  --config $CONFIG \
  --n-test $N_TEST --attacks FGSM PGD Square AutoAttack \
  --multi-seed --seeds $SEEDS \
  --gen-chunk 128 --square-max-iter 5000 \
  --pgd-max-iter $PGD_MAX_ITER --pgd-restarts $PGD_RESTARTS \
  --aa-version standard --aa-chunk 64 \
  --skip-latency \
  --checkpoint-interval 100 \
  --output experiments/evaluation/results_${TAG}_fast_n${N_TEST}_ms5.json \
  2>&1 | tee logs/${TAG}/step5_fast_ms5.log &
PID_FAST=$!

STEP6_PIDS=""
for s in $SEEDS; do
  python experiments/evaluation/run_adaptive_pgd.py \
    --config $CONFIG \
    --n-test $N_TEST --seed $s \
    --lambdas $ADAPTIVE_LAMBDAS \
    --pgd-steps $ADAPTIVE_STEPS \
    --pgd-restarts $ADAPTIVE_RESTARTS \
    --eot-samples 1 \
    --eot-verify-samples 20 \
    --output experiments/evaluation/results_${TAG}_adaptive_pgd_seed${s}.json \
    2>&1 | tee logs/${TAG}/step6_adaptive_pgd_seed${s}.log &
  STEP6_PIDS="$STEP6_PIDS $!"
done

python experiments/ablation/run_ablation_paper.py \
  --config $CONFIG \
  --n $N_TEST --multi-seed --seeds $SEEDS \
  --attacks FGSM PGD Square \
  --output experiments/ablation/results_${TAG}_ablation_multiseed.json \
  2>&1 | tee logs/${TAG}/step7_ablation.log &
PID_ABLATION=$!

STEP5_FAIL=0
STEP5_CW_EXIT=0;   wait $PID_CW   || STEP5_CW_EXIT=$?
STEP5_FAST_EXIT=0; wait $PID_FAST || STEP5_FAST_EXIT=$?
if [ $STEP5_CW_EXIT -ne 0 ]; then
  echo "ERROR: CIFAR-100 CW eval failed (exit $STEP5_CW_EXIT). Check logs/${TAG}/step5_cw_ms5.log"
  STEP5_FAIL=1
fi
if [ $STEP5_FAST_EXIT -ne 0 ]; then
  echo "ERROR: CIFAR-100 fast eval failed (exit $STEP5_FAST_EXIT). Check logs/${TAG}/step5_fast_ms5.log"
  STEP5_FAIL=1
fi

STEP6_FAIL=0
for pid in $STEP6_PIDS; do
  STEP6_EXIT=0
  wait $pid || STEP6_EXIT=$?
  if [ $STEP6_EXIT -ne 0 ]; then
    echo "ERROR: CIFAR-100 adaptive PGD job pid=$pid failed (exit $STEP6_EXIT)."
    STEP6_FAIL=1
  fi
done

STEP7_EXIT=0
wait $PID_ABLATION || STEP7_EXIT=$?
if [ $STEP7_EXIT -ne 0 ]; then
  echo "ERROR: CIFAR-100 ablation failed (exit $STEP7_EXIT). Check logs/${TAG}/step7_ablation.log"
fi

if [ $STEP5_FAIL -ne 0 ] || [ $STEP6_FAIL -ne 0 ] || [ $STEP7_EXIT -ne 0 ]; then
  exit 2
fi

# ── Step 7a: Campaign-stream eval (CIFAR-100) ────────────────────────────────
echo ""
echo "=== Step 7a: Campaign-stream eval [$TAG] ==="
if [ -f experiments/evaluation/run_campaign_eval.py ]; then
  for s in $SEEDS; do
    python experiments/evaluation/run_campaign_eval.py \
      --config $CONFIG --seed $s \
      --output experiments/campaign/results_${TAG}_campaign_seed${s}.json \
      2>&1 | tee logs/${TAG}/step7a_campaign_seed${s}.log
  done
fi

# ── Step 7b: Recovery eval (CIFAR-100) ───────────────────────────────────────
echo ""
echo "=== Step 7b: Recovery eval [$TAG] ==="
if [ -f experiments/evaluation/run_recovery_eval.py ]; then
  for s in $SEEDS; do
    python experiments/evaluation/run_recovery_eval.py \
      --config $CONFIG --seed $s --n-test $N_TEST \
      --output experiments/recovery/results_${TAG}_recovery_seed${s}.json \
      2>&1 | tee logs/${TAG}/step7b_recovery_seed${s}.log
  done
fi

# ── Step 7c: Baselines (CIFAR-100) ───────────────────────────────────────────
echo ""
echo "=== Step 7c: Baselines [$TAG] ==="
if [ -f experiments/evaluation/run_baselines.py ]; then
  for s in $SEEDS; do
    python experiments/evaluation/run_baselines.py \
      --config $CONFIG --seed $s --n-test $N_TEST \
      --methods lid mahalanobis odin energy \
      --output experiments/evaluation/results_${TAG}_baselines_seed${s}.json \
      2>&1 | tee logs/${TAG}/step7c_baselines_seed${s}.log
  done
fi

# ── Step 7d: Paper tables (combined CIFAR-10 + CIFAR-100) ────────────────────
echo ""
echo "=== Step 7d: Rebuild paper tables (combined datasets) ==="
if [ -f scripts/build_paper_tables.py ]; then
  python scripts/build_paper_tables.py \
    --results-dir experiments \
    --out-dir paper/tables \
    2>&1 | tee logs/${TAG}/step7d_paper_tables.log
fi

# ── Step 8: Manifest ─────────────────────────────────────────────────────────
echo ""
echo "=== Step 8: CIFAR-100 Manifest ==="
python -c "
import hashlib, json, os, glob
def h(p): return hashlib.sha256(open(p,'rb').read()).hexdigest()[:16] if os.path.exists(p) else None
out = {
  'dataset': 'cifar100',
  'config': '$CONFIG',
  'ensemble_sha256_16':    h('models/${TAG}/ensemble_scorer.pkl'),
  'ensemble_no_tda_sha256_16': h('models/${TAG}/ensemble_no_tda.pkl'),
  'calibrator_sha256_16':  h('models/${TAG}/calibrator.pkl'),
  'calibrator_base_sha256_16': h('models/${TAG}/calibrator_base.pkl'),
  'calibrator_no_tda_sha256_16': h('models/${TAG}/calibrator_no_tda.pkl'),
  'reference_sha256_16':   h('models/${TAG}/reference_profiles.pkl'),
  'experts_sha256_16':     h('models/${TAG}/experts.pkl'),
  'seeds':                 [42, 123, 456, 789, 999],
  'cw_eval_params':        {'max_iter': $CW_MAX_ITER, 'bss': $CW_BSS, 'batch_size': 256},
  'result_files':          sorted(glob.glob('experiments/**/results_${TAG}_*.json', recursive=True)),
}
print(json.dumps(out, indent=2))
" 2>&1 | tee logs/${TAG}/manifest.json

echo ""
echo "============================================================"
echo "PRISM CIFAR-100 Pipeline — COMPLETE"
echo "$(date)"
echo "============================================================"
