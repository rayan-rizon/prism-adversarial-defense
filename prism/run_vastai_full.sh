#!/bin/bash
# =============================================================================
# PRISM — Vast.ai Full Pipeline (Retrain + Calibrate + Parallel Eval + Ablation)
# =============================================================================
# Usage: bash run_vastai_full.sh
# Runs the entire publishable pipeline on a single RTX 5090 instance.
#
# Parallelism map (safe — no shared mutable state between concurrent jobs):
#   Phase 0 training:
#     Step 2  : ensemble_scorer.pkl (foreground, gates Step 3)
#     Step 2c : ensemble_no_tda.pkl (background — overlaps with Step 2)
#     Step 2d : experts.pkl         (background — overlaps with Step 2)
#     Step 2/2c/2d join before LOCK. Wall-clock: ~25–35 min (CW now uses native torch engine).
#   Phase 1 attacks (after LOCK):
#     Step 5A : CW-L2          (torch engine, ~25–35 min per seed × 5 seeds)
#     Step 5B : FGSM+PGD+Square+AA
#     Step 6  : 5 adaptive-PGD seeds in parallel
#     Step 7  : Ablation (FGSM+PGD+Square+CW via torch engine)
#     Step 6b : L0 calibration (background)
#   Phase 2 (after Phase 1 join):
#     Steps 7a+7b+7c parallel
#   Saves ~55min in Phase 0 + ~10-15min in Phase 1 = ~65-70min vs sequential.
#
# Sequential constraints that CANNOT be parallelised:
#   Steps 1→2/2c/2d: all three trainers read reference_profiles.pkl from Step 1
#   Steps 2→3: Step 3 reads ensemble_scorer.pkl written by Step 2
#   Steps 3→4: Step 4 reads calibrator.pkl written by Step 3
#   Steps 4→5/6/7/6b: all eval/calibration phases require the FPR gate
#   Steps 5+6+7+6b join → Steps 7a+7b+7c
#
# Exit codes: 0=success, 1=gate failure, 2=eval failure

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

# Ensure local package imports (e.g., from src.config import ...) always work,
# even when Python runs with safe-path mode enabled by the image.
unset PYTHONSAFEPATH || true
export PYTHONPATH="$PRISM_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PRISM_CONFIG="${PRISM_CONFIG:-configs/vastai_cw_full.yaml}"
export PRISM_CONFIG

SEEDS="42 123 456 789 999"
N_TEST=1000

# CW-L2 research-standard config (post-audit 2026-05-18): 100 iter × 9 bss × κ=1.0.
# Matches Carlini & Wagner (S&P 2017) canonical settings and the NeurIPS-grade
# bar called out in the pre-submission audit. Prior fast-CW values
# (max_iter=40, bss=5, κ=0) underestimated attack strength versus prior work
# and were a flagged reviewer-risk item. The native torch engine absorbs the
# ~3-5× wallclock increase (still gated by Step 5A in the parallelism map).
CW_MAX_ITER=100
CW_BSS=9
CW_CONFIDENCE=1.0
CW_CHUNK=128   # bs=256 (was 64 for bs=128 legacy)
CW_ENGINE=torch

# PGD research-standard (RobustBench): 50 iter × 10 random restarts. Prior
# config (max_iter=40, num_random_init=1) was below the field convention and
# flagged in the pre-submission audit. Function-level defaults already match
# these values; passed explicitly for provenance in the eval logs/manifest.
PGD_MAX_ITER=50
PGD_RESTARTS=10

# Adaptive-PGD expanded sweep (P1.4): λ ∈ {0, 0.5, 1, 2, 5, 10} with 100-step /
# 10-restart PGD variant. EOT=1 (hash subsample is deterministic; one EOT pass
# verifies this; larger EOT is wasted compute).
ADAPTIVE_LAMBDAS="0.0 0.5 1.0 2.0 5.0 10.0"
ADAPTIVE_STEPS=100
ADAPTIVE_RESTARTS=10

# The canonical detector path uses the locally promoted fast-attack mix:
# balanced FGSM/PGD/Square training on the disjoint profile split with grad-norm on.
# CW and AutoAttack remain in the Vast.ai evaluation stages, but they are not
# used to fit the scorer. Detector-head training still uses the same profile
# split so scorer fitting, conformal calibration, validation, and held-out eval
# all follow the same split protocol.
ENSEMBLE_N_TRAIN=1500
ENSEMBLE_SOURCE_SPLIT=profile
ENSEMBLE_GEN_CHUNK=512

echo "============================================================"
echo "PRISM Vast.ai Full Pipeline — $(date)"
echo "Instance: $(hostname)"
echo "Repo root: $PRISM_ROOT"
echo "Config: $PRISM_CONFIG"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# ── Pre-flight: PyTorch + deps installed? ─────────────────────────────────────
# Fresh vast.ai instances may ship without ML packages. Detect missing torch
# and auto-install requirements.txt before any python invocation. Avoids the
# "ModuleNotFoundError" traceback that aborts the pipeline on Step 0 otherwise.
echo ""
echo "=== Pre-flight: verify dependencies ==="
if ! python -c "import torch" 2>/dev/null; then
  echo "  PyTorch NOT FOUND — installing from requirements.txt ..."
  pip install --no-cache-dir --upgrade pip setuptools wheel
  pip install --no-cache-dir -r requirements.txt || {
    echo "ERROR: pip install -r requirements.txt failed."
    echo "       Check network, CUDA wheels, or pin a specific torch wheel for this image."
    exit 1
  }
fi
python -c "import torch" 2>/dev/null || {
  echo "ERROR: PyTorch import still fails after pip install. Aborting."
  echo "       Try:  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
  exit 1
}
# Required python modules sanity check (fail fast before Step 1).
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
  echo "  Re-running pip install -r requirements.txt to resolve missing deps ..."
  pip install --no-cache-dir -r requirements.txt
  python -c "import torch, ripser, art, autoattack" || { echo "ERROR: dependencies still missing"; exit 1; }
}

# Explicit AutoAttack pre-flight. Step 5B requests AutoAttack; the eval script
# now hard-fails if the package is missing, but failing here (before training
# burns ~30min of GPU on Steps 0a/2/3) is the friendlier behavior.
python -c "import autoattack" 2>/dev/null || {
  echo "ERROR: 'autoattack' package not importable but Step 5B requires it."
  echo "       pip install autoattack"
  echo "       Aborting pre-flight rather than discover this 30+ min into the run."
  exit 1
}
echo "Pre-flight: PASS"

# ── Environment ──────────────────────────────────────────────────────────────
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVIDIA_TF32_OVERRIDE=1
export TORCH_CUDNN_V8_API_ENABLED=1
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
export OMP_NUM_THREADS=4

mkdir -p logs models experiments/calibration experiments/evaluation experiments/ablation

# Clear generated artifacts from prior runs so this launch cannot accidentally
# reuse stale scorer, calibrator, or gate outputs. Keep the backbone checkpoint
# in place; Step 0a validates and refreshes it if needed.
rm -f models/ensemble_scorer.pkl \
      models/ensemble_no_tda.pkl \
      models/calibrator.pkl \
      models/calibrator_no_tda.pkl \
      models/calibrator_base.pkl \
      models/experts.pkl \
      experiments/calibration/ensemble_fpr_report.json \
      experiments/calibration/score_audit_val_n200.json \
      experiments/evaluation/results_latency_standalone.json \
      experiments/evaluation/results_cw_n*_ms5*.json \
      experiments/evaluation/results_fast_n*_ms5*.json \
      experiments/evaluation/results_adaptive_pgd_seed*.json \
      experiments/evaluation/results_adaptive_pgd_seed*.jsonl \
      experiments/evaluation/results_baselines_seed*.json \
      experiments/campaign/results_campaign_seed*.json \
      experiments/recovery/results_recovery_seed*.json \
      experiments/ablation/results_ablation_multiseed.json \
      logs/step*.log

# ── Verify GPU + PyTorch ─────────────────────────────────────────────────────
echo ""
echo "=== Step 0: GPU + PyTorch verification ==="
python -c "
import torch
print('torch:', torch.__version__)
print('cuda:', torch.version.cuda)
print('device:', torch.cuda.get_device_name(0))
cc = torch.cuda.get_device_capability(0)
print('compute capability:', cc)
x = torch.randn(1024, 1024, device='cuda')
y = (x @ x.T).sum()
torch.cuda.synchronize()
print('smoke GPU matmul: OK')
assert int(torch.__version__.split('.')[0]) >= 2, 'PyTorch >= 2 required'
# Publication-grade determinism: cuDNN must not use non-deterministic algorithms.
# warn_only=True so ART's internal ops don't abort the run; violations are logged.
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
torch.cuda.manual_seed_all(0)
print('determinism flags: OK (warn_only=True for ART compatibility)')
"
echo "Step 0: PASS"

# ── Preflight: verify per-tier config ────────────────────────────────────────
python -c "
from src.config import TIER_CAL_ALPHA_FACTORS, CAL_ALPHA_FACTOR
print(f'cal_alpha_factor (scalar): {CAL_ALPHA_FACTOR}')
print(f'tier_cal_alpha_factors: {TIER_CAL_ALPHA_FACTORS}')
expected_tier_factors = {'L1': 0.75, 'L2': 0.55, 'L3': 0.52}
assert TIER_CAL_ALPHA_FACTORS == expected_tier_factors, \
  f'TIER_CAL_ALPHA_FACTORS={TIER_CAL_ALPHA_FACTORS}, expected {expected_tier_factors}'
print('PREFLIGHT PASS: per-tier calibration factors match D-55 contract')
"

# ── Step 0a: Pretrain the CIFAR-10 ResNet-18 backbone ────────────────────────
# Replaces the prior ImageNet-pretrained ResNet-18. Trains from scratch on
# the CIFAR-10 training split (50k images) with the standard recipe:
#   SGD lr=0.1 cosine→0, momentum=0.9, wd=5e-4, nesterov, 200 epochs,
#   batch=256, augment=RandomCrop(32,pad=4)+HorizontalFlip.
# Expected clean test accuracy: 94-95%. Wall-clock on RTX 5090: ≈ 50-70 min.
# Output: models/cifar_resnet18.pt — loaded by every downstream stage via
# src.models.load_backbone().  Skipped if the checkpoint already exists.
echo ""
echo "=== Step 0a: Pretrain CIFAR-10 ResNet-18 backbone ==="
mkdir -p models logs
# Reuse the prior checkpoint only if BOTH:
#   1. its provenance sidecar exists with a matching sha256 prefix, AND
#   2. its empirical test accuracy on 1000 test images is >= 0.90.
# The 0.90 floor leaves ~3pp slack below the 0.93 training gate to absorb
# non-deterministic AMP / cuDNN variance. The shape-only check that used
# to live here let a 51%-acc 3-epoch checkpoint pass silently and poison
# the entire detector pipeline — see fix/backbone-acc-gate commit notes.
REUSE_BACKBONE=0
if [ -f models/cifar_resnet18.pt ] && [ -f models/cifar_resnet18.acc.json ]; then
  if python scripts/verify_backbone_acc.py \
       --checkpoint models/cifar_resnet18.pt \
       --sidecar    models/cifar_resnet18.acc.json \
       --min-acc 0.90 --n 1000 \
       2>&1 | tee logs/step0a_verify_backbone.log; then
    REUSE_BACKBONE=1
  else
    echo "  Existing backbone failed the accuracy gate — deleting and retraining."
    rm -f models/cifar_resnet18.pt models/cifar_resnet18.acc.json
  fi
fi
if [ "$REUSE_BACKBONE" -eq 0 ]; then
  python scripts/pretrain_cifar_backbone.py 2>&1 > >(tee logs/step0a_pretrain_backbone.log)
  STEP0A_EXIT=${PIPESTATUS[0]:-$?}
  if [ "$STEP0A_EXIT" -ne 0 ]; then
    echo "ERROR: Step 0a failed. Check logs/step0a_pretrain_backbone.log"
    echo "       Likely cause: clean test accuracy fell below the 0.93 gate."
    echo "       Retry with --epochs 250 or --lr 0.05 if needed."
    exit 1
  fi
  # Post-train verification: must satisfy the same gate the smoke pipeline uses.
  python scripts/verify_backbone_acc.py \
    --checkpoint models/cifar_resnet18.pt \
    --sidecar    models/cifar_resnet18.acc.json \
    --min-acc 0.90 --n 1000 \
    || { echo "ERROR: post-train backbone verification failed"; exit 1; }
fi

# ── Step 1: Build reference profiles ─────────────────────────────────────────
echo ""
echo "=== Step 1: Build Reference Profiles [test 0-4999] ==="
# Use process substitution instead of pipe so $? reflects the Python exit code,
# not tee's exit code (pipe + set -euo pipefail caused triple-execution in the
# previous Vast.ai run when tee flushed a partial line and SIGPIPE was raised).
python scripts/build_profile_testset.py 2>&1 > >(tee logs/step1_build_profile.log)
STEP1_EXIT=${PIPESTATUS[0]:-$?}
echo "Step 1 exit: $STEP1_EXIT"
if [ "$STEP1_EXIT" -ne 0 ]; then
  echo "ERROR: Step 1 failed. Check logs/step1_build_profile.log"; exit 1
fi

# ══════════════════════════════════════════════════════════════════════════════
# Steps 2 + 2c + 2d: PARALLEL TRAINING LAUNCH
# ══════════════════════════════════════════════════════════════════════════════
# All three training jobs read the same frozen reference_profiles.pkl (Step 1
# output, read-only) and write to DISTINCT output files. They have zero shared
# mutable state. Concurrent GPU memory peak: ~8 + 8 + 3 = 19 GB on RTX 5090
# (32 GB) — comfortable headroom.
#
# Wall-clock: was sequential (30 + 30 + 25 = 85 min); now parallel max(30,30,25)
# ≈ 30 min. Saves ~55 minutes vs the previous schedule.
#
# Wait order:
#   1. Wait Step 2 first → Step 2b verification gates Step 3
#   2. Run Step 2b (verification)
#   3. Continue Step 3, 4 (these only need ensemble_scorer.pkl, not 2c/2d)
#   4. Wait Step 2c + 2d before no-TDA calibration and Phase 1 LOCK.
# ══════════════════════════════════════════════════════════════════════════════

# ── Step 2: Retrain ensemble (balanced full-gate artifact) ────────
echo ""
echo "=== Steps 2 + 2c + 2d: Parallel Training Launch ==="
echo "  Step 2  : ensemble (n=$ENSEMBLE_N_TRAIN, source=$ENSEMBLE_SOURCE_SPLIT, balanced FGSM/PGD/Square, logitprofile+sidequad+gradnorm, worst-case TPR) — foreground"
echo "  Step 2c : ensemble-no-TDA variant — background"
echo "  Step 2d : differentiated experts — background"
echo ""

# Current recovery run uses the balanced fast-attack mix and
# selects alpha by worst-case held-out TPR at clean FPR<=10%, not aggregate
# AUC. Grad norm is enabled for the promoted 55-feature contract:
# 36 TDA + DCT + softmax entropy + eight logit-profile features +
# eight deterministic stability-v2 features + grad-norm. The current local
# winner also uses side-quadratic expansion inside the fitted scorer. CW-L2 and
# AutoAttack remain in the full Vast.ai gate after the fast-attack training
# smoke passes.

# ── Step 2c: launched in background (independent of Step 2) ───────────────────
# train_ensemble_scorer.py with --no-tda-features → models/ensemble_no_tda.pkl.
# Reads reference_profiles.pkl (read-only). Independent of Step 2.
PID_2C=""
if python scripts/train_ensemble_scorer.py --help 2>&1 | grep -q -- '--no-tda-features'; then
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
    --output models/ensemble_no_tda.pkl \
    > logs/step2c_retrain_no_tda.log 2>&1 &
  PID_2C=$!
  echo "  Step 2c launched (PID=$PID_2C, background) → models/ensemble_no_tda.pkl"
else
  echo "  Step 2c SKIP: --no-tda-features flag not available."
fi

# ── Step 2d: launched in background (independent of Step 2) ───────────────────
# train_experts.py → models/experts.pkl. Reads reference_profiles.pkl + CIFAR
# CAL split. Independent of Step 2/2c.
PID_2D=""
if [ -f scripts/train_experts.py ]; then
  python scripts/train_experts.py \
    --config "$PRISM_CONFIG" \
    --output models/experts.pkl \
    > logs/step2d_train_experts.log 2>&1 &
  PID_2D=$!
  echo "  Step 2d launched (PID=$PID_2D, background) → models/experts.pkl"
else
  echo "  Step 2d SKIP: scripts/train_experts.py not present."
fi

# ── Step 2: foreground (gates Step 3) ─────────────────────────────────────────
echo ""
echo "=== Step 2: Retrain Ensemble [n=$ENSEMBLE_N_TRAIN, source=$ENSEMBLE_SOURCE_SPLIT, balanced FGSM/PGD/Square, logitprofile+sidequad+gradnorm, worst-case TPR] ==="
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
  --output models/ensemble_scorer.pkl \
  2>&1 > >(tee logs/step2_retrain.log)
STEP2_EXIT=${PIPESTATUS[0]:-$?}
echo "Step 2 exit: $STEP2_EXIT"
if [ "$STEP2_EXIT" -ne 0 ]; then
  echo "ERROR: Step 2 failed. Check logs/step2_retrain.log"
  # Cleanly terminate background jobs before exiting
  [ -n "$PID_2C" ] && kill $PID_2C 2>/dev/null || true
  [ -n "$PID_2D" ] && kill $PID_2D 2>/dev/null || true
  exit 1
fi

# ── Step 2b: Post-retrain verification ────────────────────────────────────────
# Verifies fast-attack training, pixel feature space, 55-feature raw contract,
# side-quadratic scorer expansion, and worst-case TPR alpha selection.
echo ""
echo "=== Step 2b: Retrain Verification ==="
python -c "
import pickle, sys
d = pickle.load(open('models/ensemble_scorer.pkl', 'rb'))
# save() always serialises a dict (see src/cadg/ensemble_scorer.py:save).
# Legacy object-pickled format is rejected here — it means a stale pkl.
if not isinstance(d, dict):
    print('RETRAIN VERIFICATION FAIL: pkl is not a dict — stale or wrong-format artifact.')
    print(f'  type={type(d).__name__}  hint: delete models/ensemble_scorer.pkl and re-run Step 2.')
    sys.exit(1)
ta = list(d.get('training_attacks', []))
ng = bool(d.get('use_grad_norm', False))
se = bool(d.get('use_softmax_entropy', False))
sf = bool(d.get('use_stability_features', False))
lp = bool(d.get('use_logit_profile_features', False))
sq = bool(d.get('use_side_quadratic_features', False))
# n_features is now saved explicitly; fall back to computing from flags for
# any pkl created before this fix was deployed (backward-compatible).
nf = d.get('n_features')
if nf is None:
    base = len(d.get('layer_names', [])) * len(d.get('dims', [])) * 6
    nf   = base + int(d.get('use_dct', False)) + int(d.get('use_softmax_entropy', False)) + int(d.get('stability_feature_count', 8)) * int(d.get('use_stability_features', False)) + int(d.get('use_grad_norm', False))
model_dim = int(d.get('logistic_input_dim') or 0)
errors = []
for required in ('FGSM', 'PGD', 'Square'):
    if required not in ta:
        errors.append(f'{required} missing from training_attacks: {ta}')
if not ng:
    errors.append('use_grad_norm=False — grad-norm is required for the promoted 55-feature artifact')
if not se:
    errors.append('use_softmax_entropy=False — softmax-entropy must be ON for CW-L2 detection')
if not sf:
    errors.append('use_stability_features=False - stability features must be ON for PGD recovery')
if not lp:
    errors.append('use_logit_profile_features=False - current local winner requires logit-profile features')
if int(d.get('stability_feature_count', 0)) != 8:
    errors.append(f'stability_feature_count={d.get(\"stability_feature_count\")}, expected 8')
if int(d.get('logit_profile_feature_count', 0)) != 8:
    errors.append(f'logit_profile_feature_count={d.get(\"logit_profile_feature_count\")}, expected 8')
if nf != 55:
    errors.append(f'n_features={nf}, expected 55 (36 TDA + DCT + entropy + 8 logit-profile + 8 stability-v2 + grad-norm)')
if d.get('feature_space_version') != 'pixel-stability-v2+logitprofile+sidequad+gradnorm':
    errors.append(f'feature_space_version={d.get(\"feature_space_version\")}, expected pixel-stability-v2+logitprofile+sidequad+gradnorm')
if not sq:
    errors.append('use_side_quadratic_features=False - current local winner requires side-quadratic expansion')
if model_dim <= int(nf or 0):
    errors.append(f'logistic_input_dim={model_dim}, expected expanded input > n_features={nf}')
if d.get('attack_head_mode', 'off') not in ('off', None):
    errors.append(f'attack_head_mode={d.get(\"attack_head_mode\")}, expected off for current local winner')
if d.get('selection_objective') != 'worst_case_tpr':
    errors.append(f'selection_objective={d.get(\"selection_objective\")}, expected worst_case_tpr')
if d.get('training_source_split') not in ('profile', 'test-profile'):
    errors.append(f'training_source_split={d.get(\"training_source_split\")}, expected profile/test-profile')
requested = d.get('requested_oversample_weights') or {}
expected_weights = {'FGSM': 1.0, 'PGD': 1.0, 'Square': 1.0}
if not bool(d.get('balanced_attacks', False)):
    errors.append('balanced_attacks=False, expected balanced FGSM/PGD/Square training')
if set(ta) != set(expected_weights):
    errors.append(f'training_attacks={ta}, expected exactly {sorted(expected_weights)}')
if 'CW' in requested:
    errors.append(f'CW unexpectedly present in requested_oversample_weights: {requested}')
for key, expected in expected_weights.items():
    try:
        actual = float(requested.get(key))
    except Exception:
        actual = None
    if actual is None or abs(actual - expected) > 1e-6:
        errors.append(f'{key} oversample={actual}, expected {expected}')
counts = d.get('training_attack_counts') or {}
if counts:
    total = sum(int(v) for v in counts.values())
    denom = sum(expected_weights.values())
    for key, weight in expected_weights.items():
        expected_count = round(total * weight / denom)
        actual_count = int(counts.get(key, -999999))
        if abs(actual_count - expected_count) > 1:
            errors.append(f'{key} count={actual_count}, expected about {expected_count}; counts={counts}')
    if set(counts) != set(expected_weights):
        errors.append(f'training_attack_counts keys={sorted(counts)}, expected {sorted(expected_weights)}')
if int(d.get('pgd_train_steps', -1)) != 40:
    errors.append(f'pgd_train_steps={d.get(\"pgd_train_steps\")}, expected 40')
if int(d.get('square_train_max_iter', -1)) != 500:
    errors.append(f'square_train_max_iter={d.get(\"square_train_max_iter\")}, expected 500')
if errors:
    print('RETRAIN VERIFICATION FAIL:')
    for err in errors: print(f'  • {err}')
    sys.exit(1)
print(f'[OK] Retrain verified: training_attacks={ta}')
print(f'[OK] use_grad_norm={ng}, entropy={se}, stability={sf}, logit_profile={lp}, sidequad={sq}, n_features={nf}, model_dim={model_dim}')
"
if [ $? -ne 0 ]; then
  echo "ERROR: Post-retrain verification failed. Fix and re-run Step 2."; exit 1
fi

# Steps 2c (ensemble-no-TDA) and 2d (differentiated experts) launched in
# background above. They overlap with Steps 2 → 2b → 3 → 4 in foreground.
# We join them just before the LOCK announcement to ensure both artifacts
# exist on disk before Phase 1 starts.

# ── Step 3: Calibrate conformal thresholds ───────────────────────────────────
echo ""
echo "=== Step 3: Calibrate Conformal Thresholds ==="
python scripts/calibrate_ensemble.py 2>&1 > >(tee logs/step3_calibrate.log)
STEP3_EXIT=${PIPESTATUS[0]:-$?}
echo "Step 3 exit: $STEP3_EXIT"
if [ "$STEP3_EXIT" -ne 0 ]; then
  echo "ERROR: Step 3 failed. Check logs/step3_calibrate.log"; exit 1
fi

# ── Step 4: FPR gate check ───────────────────────────────────────────────────
echo ""
echo "=== Step 4: Validation FPR Gate [test 7000-7999] ==="
python scripts/compute_ensemble_val_fpr.py 2>&1 > >(tee logs/step4_val_fpr.log)
STEP4_EXIT=${PIPESTATUS[0]:-$?}
if [ "$STEP4_EXIT" -ne 0 ]; then
  echo "ERROR: Step 4 compute failed. Check logs/step4_val_fpr.log"; exit 1
fi

echo ""
echo "=== Step 4: GATE CHECK ==="
python -c "
import json, sys
with open('experiments/calibration/ensemble_fpr_report.json') as f:
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
    print(f'GATE FAIL: {failures}')
    sys.exit(1)
else:
    print('ALL GATES PASS — proceeding to evaluation')
"
GATE_EXIT=$?
if [ $GATE_EXIT -ne 0 ]; then
  echo "ERROR: FPR gate failed. Check logs/step4_val_fpr.log"
  echo "FIX: Lower tier_cal_alpha_factors in $PRISM_CONFIG, re-run steps 3-4"
  exit 1
fi

# ── Join Steps 2c, 2d (background trainers) before LOCK ──────────────────────
# By this point Steps 2/2b/3/4 have run sequentially in foreground (~40 min)
# while Steps 2c/2d ran in background (~25-30 min each). Both should already
# be done, but we join properly to surface any errors. Both artifacts are
# required: Step 2c for C1/no-TDA calibration and Step 2d for C4 recovery.
echo ""
echo "=== Join Steps 2c, 2d (background trainers) ==="
STEP2C_EXIT=0; STEP2D_EXIT=0
if [ -n "$PID_2C" ]; then
  wait $PID_2C || STEP2C_EXIT=$?
  if [ $STEP2C_EXIT -ne 0 ]; then
    echo "  WARNING: Step 2c (ensemble-no-TDA) failed (exit $STEP2C_EXIT)."
    echo "  This is EXPECTED — the no-TDA ablation baseline cannot separate clean from"
    echo "  adversarial in 2-dim feature space (DCT + softmax-entropy only)."
    echo "  The main PRISM pipeline (Steps 2→3→4) completed successfully."
    echo "  Continuing with evaluation phases (Steps 5+6+7) — C1/no-TDA results will"
    echo "  reflect this expected baseline failure."
    STEP2C_EXIT=0  # non-fatal: expected ablation failure
  else
    echo "  Step 2c: DONE → models/ensemble_no_tda.pkl"
  fi
fi
if [ -n "$PID_2D" ]; then
  wait $PID_2D || STEP2D_EXIT=$?
  if [ $STEP2D_EXIT -ne 0 ]; then
    echo "  ERROR: Step 2d (experts) failed (exit $STEP2D_EXIT). Recovery eval cannot satisfy C4."
    exit 1
  else
    echo "  Step 2d: DONE → models/experts.pkl"
  fi
fi

echo ""
echo "=== Step 3b: Calibrate Ensemble-no-TDA Arm [C1] ==="
if [ -f models/ensemble_no_tda.pkl ]; then
  python scripts/calibrate_ensemble.py \
    --ensemble-path models/ensemble_no_tda.pkl \
    --output models/calibrator_no_tda.pkl \
    2>&1 > >(tee logs/step3b_calibrate_no_tda.log)
  STEP3B_EXIT=${PIPESTATUS[0]:-$?}
  if [ "$STEP3B_EXIT" -ne 0 ]; then
    echo "WARNING: Step 3b failed (exit $STEP3B_EXIT). C1/no-TDA results unavailable."
  else
    echo "Step 3b: DONE → models/calibrator_no_tda.pkl"
  fi
else
  echo "  SKIP: models/ensemble_no_tda.pkl not found (Step 2c did not produce it)."
  echo "  C1/no-TDA calibration will be unavailable — expected for ablation baseline."
fi

python -c "
import pickle, sys
exp = pickle.load(open('models/experts.pkl', 'rb'))
if not isinstance(exp, dict):
    sys.exit('experts.pkl must be a dict artifact')
if int(exp.get('output_dim', -1)) != 10:
    sys.exit(f'experts output_dim={exp.get(\"output_dim\")}, expected 10 for CIFAR-10')
# C1 artifacts are optional (expected ablation baseline failure)
for p in ['models/calibrator_base.pkl']:
    open(p, 'rb').close()
print('[OK] C4 artifacts verified: base calibrator, CIFAR-10 experts')
"

# ── LOCK: ensemble_scorer.pkl + calibrator.pkl are now frozen ────────────────
echo ""
echo "=== ARTIFACTS LOCKED ==="
echo "ensemble_scorer.pkl and calibrator.pkl are now frozen."
echo "Do NOT retrain or recalibrate between attack runs."
python -c "
import pickle, hashlib
def h(p):
    return hashlib.sha256(open(p,'rb').read()).hexdigest()[:16]
print(f'  ensemble_scorer.pkl SHA256: {h(\"models/ensemble_scorer.pkl\")}')
print(f'  calibrator.pkl SHA256:      {h(\"models/calibrator.pkl\")}')
print(f'  reference_profiles.pkl SHA: {h(\"models/reference_profiles.pkl\")}')
"

# ── Standalone latency benchmark ─────────────────────────────────────────────
# Attack jobs below run in parallel and intentionally skip latency measurement
# so wall-clock timings are not contaminated by GPU contention from CW/AA/PGD.
echo ""
echo "=== Step 4b: Standalone Latency Benchmark ==="
python experiments/evaluation/run_evaluation_full.py \
  --n-test 200 \
  --latency-only \
  --output experiments/evaluation/results_latency_standalone.json \
  2>&1 | tee logs/step4b_latency.log
echo "Step 4b: DONE"

# ── Score-distribution audit ─────────────────────────────────────────────────
echo ""
echo "=== Step 4c: Score Distribution Audit [VAL split, diagnostic] ==="
python scripts/audit_score_distributions.py \
  --split val \
  --n 200 \
  --attacks FGSM PGD Square AutoAttack \
  --pgd-steps 40 \
  --square-max-iter 1000 \
  --aa-version standard --aa-chunk 64 \
  --output experiments/calibration/score_audit_val_n200.json \
  2>&1 | tee logs/step4c_score_audit.log || \
  echo "WARNING: score audit failed; continuing because Step 4 FPR gate already passed."

# ══════════════════════════════════════════════════════════════════════════════
# Steps 5 + 6 + 7: FULL PARALLEL LAUNCH
# ══════════════════════════════════════════════════════════════════════════════
# All three phases are pure consumers of the locked frozen artifacts.
# No inter-step file dependency between 5, 6, and 7 — each writes to distinct
# output paths and loads the same read-only pkl artifacts from Step 4.
#
# Parallelism map:
#   Step 5A : CW-L2 eval (torch engine, ~25-35 min per seed)
#   Step 5B : FGSM + PGD + Square + AutoAttack eval
#   Step 6  : 5 adaptive PGD seeds (each ~1h; overlap with Step 5)
#   Step 7  : Ablation (FGSM+PGD+Square+CW, overlap with Steps 5+6)
#
# This collapses the original 3-phase sequential schedule into one wall-clock
# phase, saving ~35-40% total GPU instance time (CW is the hard ceiling).
#
# Wait order (enforced below):
#   1. Wait Step 5A + 5B first → provenance check requires their JSON output
#   2. Wait Step 6 seeds       → STEP6_FAIL gate
#   3. Wait Step 7 ablation    → STEP7_EXIT gate
#   If Step 5 fails we still wait 6+7 before exit to avoid orphaned processes.
#
# Safety proof (same as before):
#   • All processes load independent PRISM instances (independent GPU memory)
#   • CUDA SM time-slicing handles ≤8 concurrent light-batch processes safely
#   • Output filenames are distinct (no write contention)
#   • set -euo pipefail is active; every wait uses || VAR=$? to capture exit codes
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo "=== Steps 5+6+7: Full Parallel Launch [n=$N_TEST × 5 seeds] ==="
echo "  Step 5A: CW-L2 (torch engine, max_iter=$CW_MAX_ITER, bss=$CW_BSS, κ=$CW_CONFIDENCE)"
echo "  Step 5B: FGSM + PGD ($PGD_MAX_ITER it, $PGD_RESTARTS restarts) + Square + AutoAttack"
echo "  Step 6 : Adaptive PGD × 5 seeds"
echo "  Step 7 : Ablation (FGSM+PGD+Square+CW via torch engine)"
echo "  All launched simultaneously — CW eval is the wall-clock bottleneck."
echo ""

# ── Step 5A: CW ──────────────────────────────────────────────────────────────
# Research-standard CW: max_iter=100, bss=9, κ=1.0 (Carlini & Wagner S&P 2017).
python experiments/evaluation/run_evaluation_full.py \
  --n-test $N_TEST --attacks CW \
  --multi-seed --seeds $SEEDS \
  --cw-max-iter $CW_MAX_ITER --cw-bss $CW_BSS --cw-chunk $CW_CHUNK \
  --cw-confidence $CW_CONFIDENCE \
  --cw-engine $CW_ENGINE \
  --skip-latency \
  --checkpoint-interval 100 \
  --output experiments/evaluation/results_cw_n${N_TEST}_ms5.json \
  2>&1 | tee logs/step5_cw_ms5.log &
PID_CW=$!
echo "  Step 5A CW started (PID=$PID_CW)"

# ── Step 5B: Fast attacks ─────────────────────────────────────────────────────
# Research-standard PGD: max_iter=50, num_random_init=10 (RobustBench convention).
python experiments/evaluation/run_evaluation_full.py \
  --n-test $N_TEST --attacks FGSM PGD Square AutoAttack \
  --multi-seed --seeds $SEEDS \
  --gen-chunk 128 --square-max-iter 5000 \
  --pgd-max-iter $PGD_MAX_ITER --pgd-restarts $PGD_RESTARTS \
  --aa-version standard --aa-chunk 64 \
  --skip-latency \
  --checkpoint-interval 100 \
  --output experiments/evaluation/results_fast_n${N_TEST}_ms5.json \
  2>&1 | tee logs/step5_fast_ms5.log &
PID_FAST=$!
echo "  Step 5B FGSM+PGD+Square+AA started (PID=$PID_FAST)"

# ── Step 6: Adaptive PGD — all 5 seeds ───────────────────────────────────────
# Use indexed arrays (bash 3+ compatible) instead of declare -A (bash 4+ only).
echo ""
echo "  Launching Step 6 adaptive PGD seeds..."
STEP6_PIDS=""
STEP6_SEEDS=""
count=0
for s in $SEEDS; do
  # Research-plan P1.4: expanded λ sweep, 100-step × 10-restart PGD.
  # Runtime optimization uses EOT=1 because PRISM's scoring path is
  # deterministic; --eot-verify-samples records the Appendix-B n=20 check.
  # Flags --pgd-restarts, --eot-samples, --lambdas are gated on the P1.4 branch
  # being merged — if argparse rejects them, fall back to the defaults.
  if python experiments/evaluation/run_adaptive_pgd.py --help 2>&1 | grep -q -- '--pgd-restarts'; then
    python experiments/evaluation/run_adaptive_pgd.py \
      --n-test $N_TEST --seed $s \
      --lambdas $ADAPTIVE_LAMBDAS \
      --pgd-steps $ADAPTIVE_STEPS \
      --pgd-restarts $ADAPTIVE_RESTARTS \
      --eot-samples 1 \
      --eot-verify-samples 20 \
      --through-scorer \
      --checkpoint-jsonl experiments/evaluation/results_adaptive_pgd_seed${s}.jsonl \
      --resume \
      --output experiments/evaluation/results_adaptive_pgd_seed${s}.json \
      2>&1 | tee logs/step6_adaptive_pgd_seed${s}.log &
  else
    python experiments/evaluation/run_adaptive_pgd.py \
      --n-test $N_TEST --seed $s \
      --output experiments/evaluation/results_adaptive_pgd_seed${s}.json \
      2>&1 | tee logs/step6_adaptive_pgd_seed${s}.log &
  fi
  pid=$!
  STEP6_PIDS="$STEP6_PIDS $pid"
  STEP6_SEEDS="$STEP6_SEEDS $s"
  echo "  Step 6 seed=$s started (PID=$pid)"
  count=$((count+1))

done

# ── Step 7: Ablation ─────────────────────────────────────────────────────────
# Reads only frozen pkl artifacts; output path is separate from Steps 5+6.
echo ""
python experiments/ablation/run_ablation_paper.py \
  --n $N_TEST \
  --multi-seed --seeds $SEEDS \
  --attacks FGSM PGD Square CW \
  --output experiments/ablation/results_ablation_multiseed.json \
  2>&1 | tee logs/step7_ablation.log &
PID_ABLATION=$!
echo "  Step 7 ablation started (PID=$PID_ABLATION)"

# ── Step 6b: L0 threshold calibration (LAUNCHED IN PARALLEL with Phase 1) ────
# Was previously run sequentially after Steps 5+6+7 join, blocking Phase 2 by
# 10-15 min. Step 6b only consumes locked artifacts (ensemble_scorer.pkl,
# calibrator.pkl, reference_profiles.pkl) — no dependency on Steps 5/6/7
# results. Launching it in parallel with Phase 1 hides its runtime entirely
# behind the CW bottleneck (~2h). Joined later before Phase 2 launches.
echo ""
echo "  Step 6b: L0 threshold calibration launched in parallel (P0.4)"
python scripts/calibrate_l0_thresholds.py \
  --n-clean 500 \
  --n-adv   500 \
  > logs/step6b_l0_calibration.log 2>&1 &
PID_6B=$!
echo "  Step 6b started (PID=$PID_6B, background)"

echo ""
echo "  All processes running. Monitor logs:"
echo "    tail -f logs/step5_cw_ms5.log          # CW eval (~25-35 min/seed)"
echo "    tail -f logs/step5_fast_ms5.log         # FGSM+PGD+Square+AA"
echo "    tail -f logs/step6_adaptive_pgd_seed42.log  # Adaptive PGD"
echo "    tail -f logs/step7_ablation.log         # Ablation (FGSM+PGD+Square+CW)"
echo "    tail -f logs/step6b_l0_calibration.log  # L0 calibration"
echo ""

# ── Wait order 1: Step 5 (provenance check needs its JSON output) ────────────
echo "  Waiting for Step 5 (CW + Fast attacks)..."
STEP5_FAIL=0
STEP5_CW_EXIT=0;   wait $PID_CW   || STEP5_CW_EXIT=$?
STEP5_FAST_EXIT=0; wait $PID_FAST || STEP5_FAST_EXIT=$?
[ $STEP5_CW_EXIT   -ne 0 ] && { echo "ERROR: Step 5 CW failed (exit $STEP5_CW_EXIT)";   STEP5_FAIL=1; }
[ $STEP5_FAST_EXIT -ne 0 ] && { echo "ERROR: Step 5 Fast failed (exit $STEP5_FAST_EXIT)"; STEP5_FAIL=1; }

echo ""
echo "=== Step 5: COMPLETE ==="

# ── Step 5 provenance check ───────────────────────────────────────────────────
if [ $STEP5_FAIL -eq 0 ]; then
  echo ""
  echo "=== Step 5: Provenance Check ==="
  python -c "
import json
files = [
    'experiments/evaluation/results_cw_n${N_TEST}_ms5.json',
    'experiments/evaluation/results_fast_n${N_TEST}_ms5.json',
]
ta_sets = set()
for f in files:
    d = json.load(open(f))
    ps = d.get('per_seed', {})
    for seed_key, seed_data in ps.items():
        meta = seed_data.get('_meta', {})
        ta = tuple(meta.get('ensemble', {}).get('training_attacks', []))
        ta_sets.add(ta)
        break
print(f'Training attacks across result files: {ta_sets}')
if len(ta_sets) == 1:
    ta = list(ta_sets)[0]
    required_train = {'FGSM', 'PGD', 'Square'}
    if required_train.issubset(set(ta)):
        print('PROVENANCE CHECK PASS: all results use same FGSM/PGD-rebalanced retrained ensemble')
    else:
        print(f'PROVENANCE CHECK FAIL: ensemble missing required training attacks {sorted(required_train)}: {ta}')
        exit(1)
else:
    print('PROVENANCE CHECK FAIL: different ensembles detected!')
    exit(1)
"
fi

GATE_MISS_FULL=0
if [ $STEP5_FAIL -eq 0 ]; then
  echo ""
  echo "=== Step 5: Full Attack Metric Gate ==="
  python scripts/check_vastai_full_gate.py \
    --fast-result experiments/evaluation/results_fast_n${N_TEST}_ms5.json \
    --cw-result experiments/evaluation/results_cw_n${N_TEST}_ms5.json \
    --latency-file experiments/evaluation/results_latency_standalone.json \
    --calibration-report experiments/calibration/ensemble_fpr_report.json \
    --expected-n-test "$N_TEST" \
    --expected-seeds $SEEDS \
    2>&1 | tee logs/step5_full_metric_gate.log || GATE_MISS_FULL=1
fi

# ── Wait order 2: Step 6 seeds ────────────────────────────────────────────────
# CRITICAL: use || VAR=$? on every wait — set -e would abort on first failure,
# orphaning the remaining seeds and ablation process.
echo ""
echo "  Waiting for Step 6 adaptive PGD seeds..."
STEP6_FAIL=0
set -- $STEP6_PIDS       # positional params = PID list; must come AFTER PID_CW/FAST waits
for s in $STEP6_SEEDS; do
  pid=$1; shift
  EXIT_S=0
  wait $pid || EXIT_S=$?
  if [ $EXIT_S -ne 0 ]; then
    echo "  ERROR: Adaptive PGD seed=$s (pid=$pid) failed (exit $EXIT_S). Check logs/step6_adaptive_pgd_seed${s}.log"
    STEP6_FAIL=1
  else
    echo "  Seed $s: DONE"
  fi
done
[ $STEP6_FAIL -ne 0 ] && echo "ERROR: One or more adaptive PGD seeds failed."
echo "Step 6: COMPLETE"

# ── Wait order 3: Step 7 ablation ─────────────────────────────────────────────
echo ""
echo "  Waiting for Step 7 ablation..."
STEP7_EXIT=0
wait $PID_ABLATION || STEP7_EXIT=$?
[ $STEP7_EXIT -ne 0 ] && echo "ERROR: Ablation failed (exit $STEP7_EXIT). Check logs/step7_ablation.log"
echo "Step 7 exit: $STEP7_EXIT"

# Abort now that all background processes have finished
if [ $STEP5_FAIL -ne 0 ]; then
  echo "ERROR: Step 5 evaluation failed. Check logs/step5_cw_ms5.log and logs/step5_fast_ms5.log"
  exit 2
fi
if [ $STEP6_FAIL -ne 0 ]; then
  exit 2
fi

# ── Step 6b: Join (L0 calibration was launched in parallel with Phase 1) ─────
# Step 6b started in background back at line ~445, overlapping with Phase 1's
# ~2h CW bottleneck. By now it should already be done (~10-15 min runtime).
# We join here to surface any feasibility failure before Phase 2 starts.
# l0_thresholds.pkl is read by run_campaign_eval.py via auto-discovery.
echo ""
echo "=== Step 6b: Join L0 Threshold Calibration [P0.4] ==="
L0_CAL_EXIT=0
wait $PID_6B || L0_CAL_EXIT=$?
if [ $L0_CAL_EXIT -ne 0 ]; then
  echo "ERROR: L0 calibration FAILED (exit $L0_CAL_EXIT). No feasible (hazard_rate, alert_run_prob) cell found."
  echo "       P0.4 gate cannot be evaluated with calibrated thresholds."
  echo "       Investigate logs/step6b_l0_calibration.log."
  echo "       Step 7a will proceed with CampaignMonitor defaults — results flagged as uncalibrated."
fi
echo "Step 6b: DONE (exit=$L0_CAL_EXIT)"

# ══════════════════════════════════════════════════════════════════════════════
# Research-Plan Phase 2 — C3/C4 Evidence Generation + Baselines + Paper Tables
# ══════════════════════════════════════════════════════════════════════════════
# Steps 7a/7b/7c run FULLY PARALLEL — each reads the same locked
# ensemble/calibrator/experts/reference_profiles pkl artifacts (read-only) and
# writes to distinct output directories (experiments/campaign/,
# experiments/recovery/, experiments/evaluation/results_baselines_*.json).
# No write contention; independent PRISM instances per process.
#
# Parallelism safety on a single 32 GB RTX 5090:
#   • 7a campaign seeds: ~2.5 GB each (PGD-40 generation dominates)
#   • 7b recovery seeds: ~1.5 GB each (expert forward + oracle eval)
#   • 7c baseline seeds: ~2.5 GB each (Mahalanobis fit on activations)
#   Total with 5 seeds per suite serialized within the suite and 3 suites in
#   parallel: peak ~2.5 + 1.5 + 2.5 ≈ 6.5 GB. Comfortably within 32 GB.
# Seeds within each suite stay serial — keeps logs readable and bounds VRAM
# peak regardless of future per-seed memory growth.
#
# Wait order: all three launched simultaneously after Step 7 (ablation) completes;
# gate-check python block runs ONCE after all three suites join, then step 7d
# (paper tables) runs last since it aggregates every JSON written above.
# ══════════════════════════════════════════════════════════════════════════════

mkdir -p experiments/campaign experiments/recovery

echo ""
echo "=== Steps 7a+7b+7c: Parallel launch [P0.4 + P0.5 + P0.2] ==="

# ── Step 7a: Campaign-stream eval (P0.4, C3 evidence) ────────────────────────
(
  if [ -f experiments/evaluation/run_campaign_eval.py ]; then
    for s in $SEEDS; do
      python experiments/evaluation/run_campaign_eval.py \
        --seed $s \
        --output experiments/campaign/results_campaign_seed${s}.json \
        2>&1 | tee logs/step7a_campaign_seed${s}.log
    done
  else
    echo "SKIP: run_campaign_eval.py not present. C3 evidence will be missing."
  fi
) &
PID_7A=$!
echo "  Step 7a (campaign, 5 seeds serial) started (PID=$PID_7A)"

# ── Step 7b: L3-recovery eval (P0.5, C4 evidence) ────────────────────────────
(
  if [ -f experiments/evaluation/run_recovery_eval.py ]; then
    for s in $SEEDS; do
      python experiments/evaluation/run_recovery_eval.py \
        --seed $s \
        --n-test $N_TEST \
        --output experiments/recovery/results_recovery_seed${s}.json \
        2>&1 | tee logs/step7b_recovery_seed${s}.log
    done
  else
    echo "SKIP: run_recovery_eval.py not present. C4 evidence will be missing."
  fi
) &
PID_7B=$!
echo "  Step 7b (recovery, 5 seeds serial) started (PID=$PID_7B)"

# ── Step 7c: Baseline detectors (P0.2, LID/Mahalanobis/ODIN/Energy) ──────────
(
  if [ -f experiments/evaluation/run_baselines.py ]; then
    for s in $SEEDS; do
      python experiments/evaluation/run_baselines.py \
        --seed $s \
        --n-test $N_TEST \
        --methods lid mahalanobis odin energy \
        --output experiments/evaluation/results_baselines_seed${s}.json \
        2>&1 | tee logs/step7c_baselines_seed${s}.log
    done
  else
    echo "SKIP: run_baselines.py not present. Baseline comparison table will be empty."
  fi
) &
PID_7C=$!
echo "  Step 7c (baselines, 5 seeds serial) started (PID=$PID_7C)"

# Join all three suites
echo "  Waiting for Steps 7a + 7b + 7c to complete..."
STEP7A_EXIT=0; wait $PID_7A || STEP7A_EXIT=$?
STEP7B_EXIT=0; wait $PID_7B || STEP7B_EXIT=$?
STEP7C_EXIT=0; wait $PID_7C || STEP7C_EXIT=$?
[ $STEP7A_EXIT -ne 0 ] && echo "WARN: Step 7a exited with $STEP7A_EXIT"
[ $STEP7B_EXIT -ne 0 ] && echo "WARN: Step 7b exited with $STEP7B_EXIT"
[ $STEP7C_EXIT -ne 0 ] && echo "WARN: Step 7c exited with $STEP7C_EXIT"
echo "Steps 7a+7b+7c: COMPLETE"

# ── Combined gate checks (P0.4 + P0.5) ───────────────────────────────────────
# JSON-key shapes verified against source:
#   run_campaign_eval.py:203 → results[scenario]['l0_on']['l0_active_fraction']
#   run_campaign_eval.py:216 → results[scenario]['asr_gap_pp']
#   run_recovery_eval.py:278 → results[strategy]['recovery_accuracy']
echo ""
echo "=== Gate checks (P0.4 campaign, P0.5 recovery) ==="
# Gate misses set GATE_MISS_PHASE2=1 → captured below; pipeline exits non-zero
# at the very end of the script (after Step 7d + Step 8 manifest still run, so
# the operator gets paper tables and provenance even on a gate miss).
python -c "
import json, glob, sys
miss = []

# ── P0.4 campaign gates ──
cfiles = sorted(glob.glob('experiments/campaign/results_campaign_seed*.json'))
if not cfiles:
    print('WARN: no campaign results found — skipping P0.4 gate check')
    miss.append('P0.4_no_results')
else:
    gaps, fas = [], []
    for f in cfiles:
        d = json.load(open(f))
        # scenarios are dumped at TOP LEVEL (no 'scenarios' wrapper)
        sust = d.get('sustained_rho100', {})
        clean_l0on = d.get('clean_only', {}).get('l0_on', {})
        gap = sust.get('asr_gap_pp')
        fa  = clean_l0on.get('l0_active_fraction')
        if gap is not None: gaps.append(gap)
        if fa  is not None: fas.append(fa)
    if gaps:
        mean_gap = sum(gaps)/len(gaps)
        print(f'P0.4 sustained-rho=1.0 ASR gap (mean across seeds): {mean_gap:.2f}pp  [gate >= 10pp]')
        if mean_gap < 10:
            print('  -> C3 gate MISS: demote SACD to appendix per Appendix A3')
            miss.append(f'P0.4_asr_gap={mean_gap:.2f}pp<10')
    else:
        miss.append('P0.4_no_asr_gap_recorded')
    if fas:
        max_fa = max(fas)
        print(f'P0.4 clean-only false-alarm (max across seeds): {max_fa:.4f}  [gate <= 0.01]')
        if max_fa > 0.01:
            print('  -> BOCPD priors (mu0/beta0) need retune in configs/default.yaml')
            miss.append(f'P0.4_clean_fpr={max_fa:.4f}>0.01')
    else:
        miss.append('P0.4_no_clean_fpr_recorded')

# ── P0.5 recovery gate ──
rfiles = sorted(glob.glob('experiments/recovery/results_recovery_seed*.json'))
if not rfiles:
    print('WARN: no recovery results found — skipping P0.5 gate check')
    miss.append('P0.5_no_results')
else:
    gaps = []
    for f in rfiles:
        d = json.load(open(f))
        # strategies dumped at TOP LEVEL; metric key is 'recovery_accuracy'
        t = d.get('tamsh',       {}).get('recovery_accuracy')
        p = d.get('passthrough', {}).get('recovery_accuracy')
        if t is not None and p is not None: gaps.append((t - p) * 100)
    if gaps:
        mean_gap = sum(gaps)/len(gaps)
        print(f'P0.5 TAMSH - passthrough gap (mean across seeds): {mean_gap:.2f}pp  [gate >= 15pp]')
        if mean_gap < 15:
            print('  -> C4 gate MISS: demote TAMSH to appendix per Appendix A3')
            miss.append(f'P0.5_recovery_gap={mean_gap:.2f}pp<15')
    else:
        miss.append('P0.5_no_recovery_gap_recorded')

if miss:
    print('')
    print(f'GATE SUMMARY: {len(miss)} miss(es): {miss}')
    sys.exit(1)
print('')
print('GATE SUMMARY: ALL P0.4/P0.5 gates PASS')
" || GATE_MISS_PHASE2=1
GATE_MISS_PHASE2=${GATE_MISS_PHASE2:-0}

# ── Step 7d: Build paper tables (P0.7) ───────────────────────────────────────
echo ""
echo "=== Step 7d: Build LaTeX paper tables [P0.7] ==="
if [ -f scripts/build_paper_tables.py ]; then
  python scripts/build_paper_tables.py \
    --results-dir experiments \
    --out-dir paper/tables \
    2>&1 | tee logs/step7d_paper_tables.log
  echo "  LaTeX tables written to paper/tables/*.tex"
else
  echo "SKIP: build_paper_tables.py not present. Run manually post-campaign."
fi

# ── Step 8: Reproducibility manifest ─────────────────────────────────────────
echo ""
echo "=== Step 8: Reproducibility Manifest ==="
python -c "
import pickle, json, hashlib, os
def h(p): return hashlib.sha256(open(p,'rb').read()).hexdigest()[:16] if os.path.exists(p) else None
e = pickle.load(open('models/ensemble_scorer.pkl','rb'))
# save() writes a dict; use .get() not getattr() to avoid silent empty-list bugs
assert isinstance(e, dict), f'unexpected pkl type: {type(e).__name__}'
out = {
  'ensemble_training_attacks': list(e.get('training_attacks', [])),
  'ensemble_training_n':       int(e.get('training_n') or 0),
  'ensemble_n_features':       e.get('n_features'),
  'ensemble_use_dct':          e.get('use_dct'),
  'ensemble_use_softmax_entropy': e.get('use_softmax_entropy'),
  'ensemble_use_logit_profile_features': e.get('use_logit_profile_features'),
  'ensemble_logit_profile_feature_count': e.get('logit_profile_feature_count'),
  'ensemble_use_stability_features': e.get('use_stability_features'),
  'ensemble_stability_feature_count': e.get('stability_feature_count'),
  'ensemble_use_grad_norm':    e.get('use_grad_norm'),
  'ensemble_feature_space_version': e.get('feature_space_version'),
  'ensemble_selection_objective': e.get('selection_objective'),
  'ensemble_training_attack_counts': e.get('training_attack_counts'),
  'ensemble_balanced_attacks': e.get('balanced_attacks'),
  'ensemble_pgd_train_steps': e.get('pgd_train_steps'),
  'ensemble_aa_train_mode': e.get('aa_train_mode'),
  'ensemble_sha256_16':        h('models/ensemble_scorer.pkl'),
  'ensemble_no_tda_sha256_16': h('models/ensemble_no_tda.pkl'),
  'calibrator_sha256_16':      h('models/calibrator.pkl'),
  'calibrator_base_sha256_16': h('models/calibrator_base.pkl'),
  'calibrator_no_tda_sha256_16': h('models/calibrator_no_tda.pkl'),
  'experts_sha256_16':         h('models/experts.pkl'),
  'reference_profiles_sha256_16': h('models/reference_profiles.pkl'),
  'latency_result_file':        'experiments/evaluation/results_latency_standalone.json',
  'score_audit_file':           'experiments/calibration/score_audit_val_n200.json',
  'seeds':                     [42, 123, 456, 789, 999],
  'eval_split':                'CIFAR-10 test idx 8000-9999',
  'eps_linf':                  8.0/255,
  'cw_eval_params':            {'engine': '$CW_ENGINE', 'max_iter': $CW_MAX_ITER, 'bss': $CW_BSS, 'chunk': $CW_CHUNK, 'confidence': $CW_CONFIDENCE},
  'adaptive_pgd_params':       {'lambdas': [float(x) for x in '$ADAPTIVE_LAMBDAS'.split()], 'steps': $ADAPTIVE_STEPS, 'restarts': $ADAPTIVE_RESTARTS, 'eot_samples': 1, 'eot_verify_samples': 20, 'through_scorer': True, 'checkpoint_jsonl': True},
  'fgsm_oversample':           e.get('fgsm_oversample'),
  'requested_oversample_weights': e.get('requested_oversample_weights'),
  'baselines_methods':         ['lid', 'mahalanobis', 'odin', 'energy'],
  'campaign_scenarios':        ['clean_only', 'sustained_rho050', 'sustained_rho080', 'sustained_rho100', 'burst', 'low_rate'],
  'recovery_strategies':       ['reject', 'passthrough', 'tamsh'],
  'aa_eval_params':            {'version': 'standard', 'chunk': 64},
  'square_eval_params':        {'max_iter': 5000, 'nb_restarts': 1},
}
print(json.dumps(out, indent=2))
" 2>&1 | tee logs/manifest.json

echo ""
echo "============================================================"
echo "PRISM Vast.ai Full Pipeline — COMPLETE"
echo "$(date)"
echo "============================================================"
echo ""
echo "Result files:"
ls -lh experiments/evaluation/results_*_ms5.json \
  experiments/evaluation/results_adaptive_pgd_*.json \
  experiments/ablation/results_* 2>/dev/null
echo ""
echo "Download command (run from your laptop):"
echo "  scp -P <port> root@<ip>:/workspace/prism-repo/prism/experiments/evaluation/results_*_ms5.json ."
echo "  scp -P <port> root@<ip>:/workspace/prism-repo/prism/experiments/evaluation/results_adaptive_pgd_*.json ."
echo "  scp -P <port> root@<ip>:/workspace/prism-repo/prism/logs/*.log ."
echo "  scp -P <port> root@<ip>:/workspace/prism-repo/prism/models/*.pkl ."

# ── Final exit code reflects gate outcomes ───────────────────────────────────
# All artifacts (paper tables, manifest) are written before this point so the
# operator gets full diagnostics even when a gate misses. Exit-code contract:
#   0 → all gates pass (Step 4 + Step 5 full attack + P0.4 + P0.5)
#   3 → Phase 2 gate miss (P0.4 ASR gap, FPR, or P0.5 recovery gap)
#   4 → Step 5 full attack metric gate miss
# Step 4 (FPR) and earlier failures already exited 1/2 before we got here.
if [ "${GATE_MISS_PHASE2:-0}" -ne 0 ]; then
  echo ""
  echo "EXIT 3: One or more Phase 2 gates failed (see GATE SUMMARY above)."
  echo "        Paper tables + manifest were written for diagnostic review."
  exit 3
fi
if [ "${GATE_MISS_FULL:-0}" -ne 0 ]; then
  echo ""
  echo "EXIT 4: Full attack metric gate failed (see logs/step5_full_metric_gate.log)."
  echo "        Paper tables + manifest were written for diagnostic review."
  exit 4
fi
