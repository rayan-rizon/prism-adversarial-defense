#!/bin/bash
# =============================================================================
# PRISM — Local Smoke Test (bounded local integration run)
# =============================================================================
# Runs the full pipeline with minimal subsets to verify:
#   1. Backbone trains and checkpoints correctly
#   2. Profile building works
#   3. Ensemble trains and scores
#   4. Conformal calibration meets FPR targets
#   5. Evaluation reports TPR/FPR metrics
#
# NOT a metric gate. This path intentionally uses a tiny local backbone and
# reduced splits to verify train -> profile -> calibrate wiring without using
# any Vast.ai artifacts.
#
# Usage:
#   cd prism/
#   bash scripts/run_local_smoke_test.sh
#
# Pass --clean to delete smoke artifacts afterwards.
# =============================================================================
set -euo pipefail

SMOKE_DIR="models/smoke_test"
SMOKE_CONFIG="configs/smoke_test.yaml"
LOG_DIR="logs/smoke"
CLEAN=0

for arg in "$@"; do
  [ "$arg" = "--clean" ] && CLEAN=1
done

mkdir -p "$SMOKE_DIR" "$LOG_DIR"

echo "============================================================"
echo "  PRISM Local Smoke Test — $(date)"
echo "============================================================"
python3 -c "
import torch, sys
print(f'  Python:  {sys.version.split()[0]}')
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA:    {\"available\" if torch.cuda.is_available() else \"CPU only\"}')
"

# ── Generate minimal smoke config ────────────────────────────────────────────
python3 - <<'PYEOF'
import yaml, os

cfg = {
    'model': {
        'name': 'cifar_resnet18',
        'backbone_checkpoint': 'models/smoke_test/cifar_resnet18_smoke.pt',
        'num_classes': 10,
        'layer_names': ['layer2', 'layer3', 'layer4'],
        'layer_weights': {'layer2': 0.30, 'layer3': 0.30, 'layer4': 0.40},
    },
    'tda': {
        'n_subsample': 30,
        'max_dim': 1,
        'n_reference': 5,
        'dim_weights': [0.70, 0.30],
    },
    'conformal': {
        'alphas': {'L1': 0.10, 'L2': 0.03, 'L3': 0.005},
        'calibration_size': 100,
        'validation_size': 50,
        'cal_alpha_factor': 0.7,
        'tier_cal_alpha_factors': {'L1': 0.70, 'L2': 0.70, 'L3': 0.50},
    },
    # Use a smaller slice of the test set for speed (same non-overlapping splits)
    'splits': {
        'profile_idx': [0,    100],
        'cal_idx':     [100,  200],
        'val_idx':     [200,  250],
        'eval_idx':    [250,  350],
    },
    'campaign': {
        'window_size': 50, 'cp_threshold': 0.3, 'hazard_rate': 0.033,
        'mu0': 7.0, 'beta0': 15.0, 'alert_run_length': 10,
        'alert_run_prob': 0.60, 'warmup_steps': 10, 'l0_factor': 0.8,
    },
    'moe': {'n_experts': 2, 'hidden_dim': 64},   # smaller MoE
    'memory': {'match_threshold': 0.5},
    'data': {
        'dataset': 'cifar10',
        'image_size': 32,
        'mean': [0.4914, 0.4822, 0.4465],
        'std':  [0.2470, 0.2435, 0.2616],
        'data_root': './data',
    },
    'profiling': {'n_clean_images': 100},
    'evaluation': {'n_test_images': 50, 'attacks': ['FGSM', 'PGD']},
    'paths': {
        'reference_profiles': 'models/smoke_test/reference_profiles.pkl',
        'calibrator':         'models/smoke_test/calibrator.pkl',
        'ensemble_scorer':    'models/smoke_test/ensemble_scorer.pkl',
        'experts':            'models/smoke_test/experts.pkl',
        'clean_scores':       'models/smoke_test/clean_scores.npy',
    },
}

with open('configs/smoke_test.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
print('  Smoke config written: configs/smoke_test.yaml')
PYEOF

echo ""
echo "=== Step 0: Pretrain backbone (bounded local smoke) ==="
CKPT="$SMOKE_DIR/cifar_resnet18_smoke.pt"
if [ -f "$CKPT" ]; then
  echo "  Checkpoint exists — skipping backbone pretrain."
else
  python3 scripts/pretrain_cifar_backbone.py \
    --dataset cifar10 \
    --epochs 1 \
    --batch-size 256 \
    --num-workers 0 \
    --train-subset 2000 \
    --test-subset 500 \
    --output "$CKPT" \
    --min-test-acc 0.0 \
    --allow-undertrained-smoke \
    2>&1 | tee "$LOG_DIR/step0_backbone.log"
  echo "  Backbone saved: $CKPT"
fi

# Quick architecture check
python3 -c "
import torch
from src.models.cifar_resnet import cifar_resnet18
sd = torch.load('$CKPT', map_location='cpu', weights_only=True)
m = cifar_resnet18(num_classes=10)
m.load_state_dict(sd)
out = m(torch.randn(1, 3, 32, 32))
assert out.shape == (1, 10)
print(f'  Backbone: OK  params={sum(p.numel() for p in m.parameters())/1e6:.2f}M  output={out.shape}')
"

echo ""
echo "=== Step 1: Build reference profiles ==="
python3 scripts/build_profile_testset.py \
  --config "$SMOKE_CONFIG" \
  --allow-undertrained-smoke \
  2>&1 | tee "$LOG_DIR/step1_profiles.log"
echo "  Profiles built."

echo ""
echo "=== Step 2: Train ensemble scorer ==="
python3 scripts/train_ensemble_scorer.py \
  --config "$SMOKE_CONFIG" \
  --n-train 30 \
  --pgd-train-steps 5 \
  --fgsm-oversample 1.5 \
  --allow-undertrained-smoke \
  2>&1 | tee "$LOG_DIR/step2_ensemble.log"
echo "  Ensemble trained."

echo ""
echo "=== Step 3: Calibrate conformal thresholds ==="
python3 scripts/calibrate_ensemble.py \
  --config "$SMOKE_CONFIG" \
  2>&1 | tee "$LOG_DIR/step3_calibrate.log"
echo "  Calibrated."

echo ""
echo "=== Step 4: FPR gate check (val split) ==="
python3 scripts/compute_ensemble_val_fpr.py \
  --config "$SMOKE_CONFIG" \
  --output "$SMOKE_DIR/fpr_report.json" \
  2>&1 | tee "$LOG_DIR/step4_val_fpr.log"

echo ""
echo "=== Smoke Test Results ==="
python3 - <<'PYEOF'
import json, os

# FPR report
fpr_path = 'models/smoke_test/fpr_report.json'
if os.path.exists(fpr_path):
    fpr = json.load(open(fpr_path))
    print(f"  Val FPR (n={fpr['n_val']}):")
    for tier, info in fpr['tiers'].items():
        status = 'PASS' if info['passed'] else 'FAIL'
        print(f"    {tier}: FPR={info['FPR']:.4f}  target<={info['target']:.3f}  [{status}]")
else:
    print("  FPR report not found.")

print("\n  NOTE: this run uses a tiny local backbone subset and reduced splits.")
print("  Purpose = pipeline correctness, artifact generation, and calibration wiring.")
print("  It is not a publishable metric run.")
PYEOF

if [ "$CLEAN" -eq 1 ]; then
  echo ""
  echo "=== Cleanup (--clean) ==="
  rm -rf models/smoke_test configs/smoke_test.yaml logs/smoke
  echo "  Smoke artifacts removed."
fi

echo ""
echo "============================================================"
echo "  Smoke test complete — $(date)"
echo "  Logs: logs/smoke/"
echo "  Check above for PASS/FAIL on each step."
echo "============================================================"
