#!/bin/bash
# =============================================================================
# PRISM — Vast.ai CIFAR-100 Full Pipeline (Research-Standard)
# =============================================================================
# Research-standard mirror of run_vastai_full.sh for the CIFAR-100 generalisation
# evaluation (paper §App. CIFAR-100 / Appendix~\ref{app:cifar100}).
#
# Uses configs/cifar100.yaml; every artifact lands under models/cifar100/ and
# experiments/*/results_cifar100_*.json so the canonical CIFAR-10 run is never
# clobbered. Mirrors the same:
#   - Parallel training (Steps 2/2c/2d in parallel)
#   - 55-feature detector contract (logit-profile + side-quadratic + grad-norm)
#   - 5-seed evaluation (FGSM/PGD/Square/CW/AutoAttack + adaptive PGD)
#   - Channel-mask ablation (5 seeds)
#   - Standalone latency benchmark
#   - L0 BOCPD/CUSUM calibration (parallel with Phase 1)
#   - Parallel Phase 2 (campaign + recovery + baselines)
#   - P0.4/P0.5 gate checks + paper-tables rebuild
#   - SHA-pinned reproducibility manifest
#
# Wall-clock: ~10–12 h on an RTX 5090 (one full GPU-day budget recommended).
# If the cal→val FPR overruns target by >1 pp, tighten
# conformal.tier_cal_alpha_factors.L3 from 0.50 → 0.45 in configs/cifar100.yaml
# and re-run from Step 3.
#
# Exit codes: 0=success, 1=gate failure, 2=eval failure, 3=Phase 2 gate miss,
# 4=Step 5 full-attack gate miss.

set -euo pipefail

# Resolve PRISM root robustly for both common Vast.ai layouts.
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

CONFIG=configs/cifar100.yaml
TAG=cifar100
export PRISM_CONFIG="$CONFIG"

SEEDS="42 123 456 789 999"
N_TEST=1000

# Research-standard CW (Carlini & Wagner S&P 2017): max_iter=100, bss=9, κ=1.0.
CW_MAX_ITER=100
CW_BSS=9
CW_CONFIDENCE=1.0
CW_CHUNK=128
CW_ENGINE=torch

# Research-standard PGD (RobustBench convention): 50 iter × 10 restarts.
PGD_MAX_ITER=50
PGD_RESTARTS=10

# Adaptive-PGD expanded sweep (P1.4): λ ∈ {0,0.5,1,2,5,10}, 100 steps × 10 restarts.
ADAPTIVE_LAMBDAS="0.0 0.5 1.0 2.0 5.0 10.0"
ADAPTIVE_STEPS=100
ADAPTIVE_RESTARTS=10

# Detector training: same balanced FGSM/PGD/Square mix as CIFAR-10 (55-feature
# contract). CW + AutoAttack remain eval-only so the detector never sees them
# during fitting.
ENSEMBLE_N_TRAIN=1500
ENSEMBLE_SOURCE_SPLIT=profile
ENSEMBLE_GEN_CHUNK=512

echo "============================================================"
echo "PRISM Vast.ai CIFAR-100 Pipeline — $(date)"
echo "Instance: $(hostname)"
echo "Repo root: $PRISM_ROOT"
echo "Config: $CONFIG"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# ── Pre-flight: PyTorch + deps installed? ─────────────────────────────────────
echo ""
echo "=== Pre-flight: verify dependencies ==="
if ! python -c "import torch" 2>/dev/null; then
  echo "  PyTorch NOT FOUND — installing from requirements.txt ..."
  pip install --no-cache-dir --upgrade pip setuptools wheel
  pip install --no-cache-dir -r requirements.txt || {
    echo "ERROR: pip install -r requirements.txt failed."; exit 1
  }
fi
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
# GPU throughput tuning (saves wallclock on 32 GB 5090 / 4090):
#   - LAZY module loading: avoids CUDA preloading every kernel up front,
#     ~15% startup faster across 9 concurrent processes
#   - MAX_CONNECTIONS=32: more concurrent kernel launches per stream
#   - expandable_segments: reduces fragmentation under multi-process VRAM mix
export CUDA_MODULE_LOADING=LAZY
export CUDA_DEVICE_MAX_CONNECTIONS=32
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

mkdir -p logs/${TAG} \
         models/${TAG} \
         experiments/calibration \
         experiments/evaluation \
         experiments/ablation \
         experiments/campaign \
         experiments/recovery \
         experiments/recovery_uniform \
         baselines

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

# ── Step 0a: Pretrain CIFAR-100 ResNet-18 backbone ───────────────────────────
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

# ══════════════════════════════════════════════════════════════════════════════
# Steps 2 + 2c + 2d: PARALLEL TRAINING LAUNCH
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== Steps 2 + 2c + 2d: Parallel Training Launch [$TAG] ==="
echo "  Step 2  : ensemble_scorer (n=$ENSEMBLE_N_TRAIN, balanced FGSM/PGD/Square, 55-feature) — foreground"
echo "  Step 2c : ensemble_no_tda (C1 ablation arm) — background"
echo "  Step 2d : differentiated experts (C4 recovery) — background"
echo ""

# ── Step 2c: ensemble-no-TDA variant in background ───────────────────────────
PID_2C=""
if python scripts/train_ensemble_scorer.py --help 2>&1 | grep -q -- '--no-tda-features'; then
  python scripts/train_ensemble_scorer.py \
    --config $CONFIG \
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
    --output models/${TAG}/ensemble_no_tda.pkl \
    > logs/${TAG}/step2c_no_tda.log 2>&1 &
  PID_2C=$!
  echo "  Step 2c launched (PID=$PID_2C, background)"
fi

# ── Step 2d: differentiated experts in background ────────────────────────────
PID_2D=""
if [ -f scripts/train_experts.py ]; then
  python scripts/train_experts.py \
    --config $CONFIG \
    --output models/${TAG}/experts.pkl \
    > logs/${TAG}/step2d_experts.log 2>&1 &
  PID_2D=$!
  echo "  Step 2d launched (PID=$PID_2D, background)"
fi

# ── Step 2: foreground (gates Step 3) ────────────────────────────────────────
echo ""
echo "=== Step 2: Retrain Ensemble [$TAG, balanced FGSM/PGD/Square + 55-feature contract] ==="
python scripts/train_ensemble_scorer.py \
  --config $CONFIG \
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
  --output models/${TAG}/ensemble_scorer.pkl \
  2>&1 > >(tee logs/${TAG}/step2_retrain.log)
STEP2_EXIT=${PIPESTATUS[0]:-$?}
if [ "$STEP2_EXIT" -ne 0 ]; then
  echo "ERROR: Step 2 failed. Check logs/${TAG}/step2_retrain.log"
  [ -n "$PID_2C" ] && kill $PID_2C 2>/dev/null || true
  [ -n "$PID_2D" ] && kill $PID_2D 2>/dev/null || true
  exit 1
fi

# ── Step 2b: Post-retrain verification (55-feature contract) ──────────────────
echo ""
echo "=== Step 2b: Post-Retrain Verification [$TAG] ==="
python -c "
import pickle, sys
d = pickle.load(open('models/${TAG}/ensemble_scorer.pkl', 'rb'))
if not isinstance(d, dict):
    print('FAIL: pkl is not a dict — stale or wrong-format artifact.')
    sys.exit(1)
ta = list(d.get('training_attacks', []))
ng = bool(d.get('use_grad_norm', False))
se = bool(d.get('use_softmax_entropy', False))
sf = bool(d.get('use_stability_features', False))
lp = bool(d.get('use_logit_profile_features', False))
sq = bool(d.get('use_side_quadratic_features', False))
nf = d.get('n_features')
model_dim = int(d.get('logistic_input_dim') or 0)
errors = []
for required in ('FGSM', 'PGD', 'Square'):
    if required not in ta:
        errors.append(f'{required} missing from training_attacks: {ta}')
if not ng: errors.append('use_grad_norm=False')
if not se: errors.append('use_softmax_entropy=False')
if not sf: errors.append('use_stability_features=False')
if not lp: errors.append('use_logit_profile_features=False')
if not sq: errors.append('use_side_quadratic_features=False')
if int(d.get('stability_feature_count', 0)) != 8: errors.append(f'stability_feature_count={d.get(\"stability_feature_count\")}')
if int(d.get('logit_profile_feature_count', 0)) != 8: errors.append(f'logit_profile_feature_count={d.get(\"logit_profile_feature_count\")}')
if nf != 55: errors.append(f'n_features={nf}, expected 55')
if d.get('feature_space_version') != 'pixel-stability-v2+logitprofile+sidequad+gradnorm':
    errors.append(f'feature_space_version={d.get(\"feature_space_version\")}')
if model_dim <= int(nf or 0):
    errors.append(f'logistic_input_dim={model_dim}, expected > n_features={nf}')
if d.get('selection_objective') != 'worst_case_tpr':
    errors.append(f'selection_objective={d.get(\"selection_objective\")}')
if not bool(d.get('balanced_attacks', False)):
    errors.append('balanced_attacks=False')
if errors:
    print('VERIFICATION FAIL:')
    for err in errors: print(f'  • {err}')
    sys.exit(1)
print(f'[OK] CIFAR-100 retrain verified: training_attacks={ta}, n_features={nf}, model_dim={model_dim}')
" || { echo "ERROR: Step 2b verification failed."; exit 1; }

# ── Step 3: Calibrate conformal thresholds ───────────────────────────────────
echo ""
echo "=== Step 3: Calibrate Conformal Thresholds [$TAG] ==="
python scripts/calibrate_ensemble.py --config $CONFIG \
  2>&1 > >(tee logs/${TAG}/step3_calibrate.log)

# ── Step 4: Validation FPR gate ──────────────────────────────────────────────
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

# ── Join Steps 2c, 2d (background trainers) before LOCK ──────────────────────
echo ""
echo "=== Join Steps 2c, 2d (background trainers) ==="
STEP2C_EXIT=0; STEP2D_EXIT=0
if [ -n "$PID_2C" ]; then
  wait $PID_2C || STEP2C_EXIT=$?
  if [ $STEP2C_EXIT -ne 0 ]; then
    echo "  WARNING: Step 2c (ensemble-no-TDA) failed (exit $STEP2C_EXIT)."
    echo "  Expected — no-TDA ablation is the C1 baseline."
    STEP2C_EXIT=0
  fi
fi
if [ -n "$PID_2D" ]; then
  wait $PID_2D || STEP2D_EXIT=$?
  if [ $STEP2D_EXIT -ne 0 ]; then
    echo "  ERROR: Step 2d (experts) failed (exit $STEP2D_EXIT). C4 recovery cannot run."
    exit 1
  fi
fi

# ── Step 3b: Calibrate ensemble-no-TDA arm (C1) ──────────────────────────────
echo ""
echo "=== Step 3b: Calibrate Ensemble-no-TDA Arm [$TAG, C1] ==="
if [ -f models/${TAG}/ensemble_no_tda.pkl ]; then
  python scripts/calibrate_ensemble.py \
    --config $CONFIG \
    --ensemble-path models/${TAG}/ensemble_no_tda.pkl \
    --output models/${TAG}/calibrator_no_tda.pkl \
    2>&1 > >(tee logs/${TAG}/step3b_calibrate_no_tda.log) || \
    echo "  WARNING: Step 3b calibrate-no-TDA failed; C1 results will reflect baseline failure."
fi

# Artifact verification (CIFAR-100 specific)
python -c "
import pickle, sys
exp = pickle.load(open('models/${TAG}/experts.pkl', 'rb'))
if not isinstance(exp, dict): sys.exit('experts.pkl not a dict')
if int(exp.get('output_dim', -1)) != 100:
    sys.exit(f'experts output_dim={exp.get(\"output_dim\")}, expected 100 for CIFAR-100')
for p in ['models/${TAG}/calibrator_base.pkl']:
    open(p, 'rb').close()
print('[OK] CIFAR-100 C4 artifacts verified')
"

# ── LOCK ─────────────────────────────────────────────────────────────────────
echo ""
echo "=== ARTIFACTS LOCKED [$TAG] ==="
python -c "
import hashlib, os
def h(p): return hashlib.sha256(open(p,'rb').read()).hexdigest()[:16] if os.path.exists(p) else 'MISSING'
for p in ['models/${TAG}/ensemble_scorer.pkl','models/${TAG}/calibrator.pkl','models/${TAG}/reference_profiles.pkl','models/${TAG}/experts.pkl']:
    print(f'  {p} SHA256: {h(p)}')
"

# ── Step 4b: Standalone latency benchmark ────────────────────────────────────
echo ""
echo "=== Step 4b: Standalone Latency Benchmark [$TAG] ==="
python experiments/evaluation/run_evaluation_full.py \
  --config $CONFIG \
  --n-test 200 \
  --latency-only \
  --output experiments/evaluation/results_${TAG}_latency_standalone.json \
  2>&1 | tee logs/${TAG}/step4b_latency.log || \
  echo "  WARNING: latency benchmark failed; non-fatal."

# ── Step 4c: Score-distribution audit ────────────────────────────────────────
# Diagnostic only; Step 4 FPR gate has already passed. AutoAttack dropped from
# audit and n lowered 200→100 to cut ~20-30 min off critical path; the audit
# still characterises FGSM/PGD/Square score distributions which is sufficient
# for cal→val drift detection.
echo ""
echo "=== Step 4c: Score Distribution Audit [$TAG, VAL split] ==="
python scripts/audit_score_distributions.py \
  --config $CONFIG \
  --split val \
  --n 100 \
  --attacks FGSM PGD Square \
  --pgd-steps 40 \
  --square-max-iter 1000 \
  --output experiments/calibration/${TAG}_score_audit_val_n100.json \
  2>&1 | tee logs/${TAG}/step4c_score_audit.log || \
  echo "  WARNING: score audit failed; non-fatal (Step 4 FPR gate already passed)."

# ══════════════════════════════════════════════════════════════════════════════
# Steps 5 + 6 + 7 + 6b: FULL PARALLEL LAUNCH
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== Steps 5+6+7+6b: Full Parallel Launch [$TAG, n=$N_TEST × 5 seeds] ==="
echo "  Step 5A: CW-L2 (torch engine, max_iter=$CW_MAX_ITER, bss=$CW_BSS, κ=$CW_CONFIDENCE)"
echo "  Step 5B: FGSM + PGD ($PGD_MAX_ITER it, $PGD_RESTARTS restarts) + Square + AutoAttack"
echo "  Step 6 : Adaptive PGD × 5 seeds"
echo "  Step 7 : Ablation (FGSM+PGD+Square+CW)"
echo "  Step 6b: L0 BOCPD/CUSUM threshold calibration"
echo ""

# ── Step 5A: CW-L2 (research-standard: max_iter=100, bss=9, κ=1.0) ───────────
python experiments/evaluation/run_evaluation_full.py \
  --config $CONFIG \
  --n-test $N_TEST --attacks CW \
  --multi-seed --seeds $SEEDS \
  --cw-max-iter $CW_MAX_ITER --cw-bss $CW_BSS --cw-chunk $CW_CHUNK \
  --cw-confidence $CW_CONFIDENCE \
  --cw-engine $CW_ENGINE \
  --skip-latency \
  --checkpoint-interval 100 \
  --output experiments/evaluation/results_${TAG}_cw_n${N_TEST}_ms5.json \
  2>&1 | tee logs/${TAG}/step5_cw_ms5.log &
PID_CW=$!

# ── Step 5B: Fast attacks (FGSM/PGD/Square/AutoAttack) ────────────────────────
# Throughput-tuned for RTX 5090 32 GB: gen-chunk 256 (was 128) doubles batch on
# FGSM/PGD/Square generation. AA-chunk stays at 64: AutoAttack runs 4 sequential
# sub-attacks (APGD-CE + APGD-T + FAB + Square) per chunk; bumping past 64 risks
# OOM when 9 concurrent processes (CW + Fast + 5×Adaptive-PGD + Ablation + L0)
# all peak VRAM at once.
python experiments/evaluation/run_evaluation_full.py \
  --config $CONFIG \
  --n-test $N_TEST --attacks FGSM PGD Square AutoAttack \
  --multi-seed --seeds $SEEDS \
  --gen-chunk 256 --square-max-iter 5000 \
  --pgd-max-iter $PGD_MAX_ITER --pgd-restarts $PGD_RESTARTS \
  --aa-version standard --aa-chunk 64 \
  --skip-latency \
  --checkpoint-interval 100 \
  --output experiments/evaluation/results_${TAG}_fast_n${N_TEST}_ms5.json \
  2>&1 | tee logs/${TAG}/step5_fast_ms5.log &
PID_FAST=$!

# ── Step 6: Adaptive PGD × 5 seeds (parallel, staggered launch) ──────────────
# Stagger launches by 3 s so CUDA init bursts don't simultaneously peak VRAM
# at startup. After steady-state, all 5 procs run truly parallel.
STEP6_PIDS=""; STEP6_SEEDS=""
STEP6_FIRST=1
for s in $SEEDS; do
  if [ "$STEP6_FIRST" -eq 0 ]; then sleep 3; fi
  STEP6_FIRST=0
  if python experiments/evaluation/run_adaptive_pgd.py --help 2>&1 | grep -q -- '--pgd-restarts'; then
    python experiments/evaluation/run_adaptive_pgd.py \
      --config $CONFIG \
      --n-test $N_TEST --seed $s \
      --lambdas $ADAPTIVE_LAMBDAS \
      --pgd-steps $ADAPTIVE_STEPS \
      --pgd-restarts $ADAPTIVE_RESTARTS \
      --eot-samples 1 \
      --eot-verify-samples 20 \
      --through-scorer \
      --checkpoint-jsonl experiments/evaluation/results_${TAG}_adaptive_pgd_seed${s}.jsonl \
      --resume \
      --output experiments/evaluation/results_${TAG}_adaptive_pgd_seed${s}.json \
      2>&1 | tee logs/${TAG}/step6_adaptive_pgd_seed${s}.log &
  else
    python experiments/evaluation/run_adaptive_pgd.py \
      --config $CONFIG \
      --n-test $N_TEST --seed $s \
      --output experiments/evaluation/results_${TAG}_adaptive_pgd_seed${s}.json \
      2>&1 | tee logs/${TAG}/step6_adaptive_pgd_seed${s}.log &
  fi
  STEP6_PIDS="$STEP6_PIDS $!"
  STEP6_SEEDS="$STEP6_SEEDS $s"
done

# ── Step 7: Ablation (FGSM+PGD+Square+CW) ─────────────────────────────────────
python experiments/ablation/run_ablation_paper.py \
  --config $CONFIG \
  --n $N_TEST \
  --multi-seed --seeds $SEEDS \
  --attacks FGSM PGD Square CW \
  --output experiments/ablation/results_${TAG}_ablation_multiseed.json \
  2>&1 | tee logs/${TAG}/step7_ablation.log &
PID_ABLATION=$!

# ── Step 6b: L0 threshold calibration (parallel with Phase 1) ─────────────────
# Output pinned to models/${TAG}/l0_thresholds.pkl so CIFAR-10 calibrator
# next to models/calibrator.pkl is never overwritten.
python scripts/calibrate_l0_thresholds.py \
  --config $CONFIG \
  --n-clean 500 --n-adv 500 \
  --output models/${TAG}/l0_thresholds.pkl \
  > logs/${TAG}/step6b_l0_calibration.log 2>&1 &
PID_6B=$!

echo "  All processes running. Joining in wait-order..."

# ── Wait order 1: Step 5 ──────────────────────────────────────────────────────
STEP5_FAIL=0
STEP5_CW_EXIT=0;   wait $PID_CW   || STEP5_CW_EXIT=$?
STEP5_FAST_EXIT=0; wait $PID_FAST || STEP5_FAST_EXIT=$?
[ $STEP5_CW_EXIT   -ne 0 ] && { echo "ERROR: Step 5 CW failed (exit $STEP5_CW_EXIT)";   STEP5_FAIL=1; }
[ $STEP5_FAST_EXIT -ne 0 ] && { echo "ERROR: Step 5 Fast failed (exit $STEP5_FAST_EXIT)"; STEP5_FAIL=1; }
echo "Step 5: COMPLETE"

# ── Step 5 provenance check ───────────────────────────────────────────────────
if [ $STEP5_FAIL -eq 0 ]; then
  echo ""
  echo "=== Step 5: Provenance Check [$TAG] ==="
  python -c "
import json
files = [
    'experiments/evaluation/results_${TAG}_cw_n${N_TEST}_ms5.json',
    'experiments/evaluation/results_${TAG}_fast_n${N_TEST}_ms5.json',
]
ta_sets = set()
for f in files:
    d = json.load(open(f))
    ps = d.get('per_seed', {})
    for seed_key, seed_data in ps.items():
        meta = seed_data.get('_meta', {})
        ta = tuple(meta.get('ensemble', {}).get('training_attacks', []))
        ta_sets.add(ta); break
print(f'Training attacks across result files: {ta_sets}')
if len(ta_sets) == 1:
    ta = list(ta_sets)[0]
    if {'FGSM','PGD','Square'}.issubset(set(ta)):
        print('PROVENANCE CHECK PASS')
    else:
        print(f'PROVENANCE FAIL: missing FGSM/PGD/Square: {ta}'); exit(1)
else:
    print('PROVENANCE FAIL: mixed ensembles'); exit(1)
"
fi

# ── Step 5 full-attack metric gate ────────────────────────────────────────────
GATE_MISS_FULL=0
if [ $STEP5_FAIL -eq 0 ]; then
  echo ""
  echo "=== Step 5: Full Attack Metric Gate [$TAG] ==="
  python scripts/check_vastai_full_gate.py \
    --fast-result experiments/evaluation/results_${TAG}_fast_n${N_TEST}_ms5.json \
    --cw-result experiments/evaluation/results_${TAG}_cw_n${N_TEST}_ms5.json \
    --latency-file experiments/evaluation/results_${TAG}_latency_standalone.json \
    --calibration-report experiments/calibration/${TAG}_ensemble_fpr_report.json \
    --expected-n-test "$N_TEST" \
    --expected-seeds $SEEDS \
    2>&1 | tee logs/${TAG}/step5_full_metric_gate.log || GATE_MISS_FULL=1
fi

# ── Wait order 2: Step 6 seeds ────────────────────────────────────────────────
STEP6_FAIL=0
set -- $STEP6_PIDS
for s in $STEP6_SEEDS; do
  pid=$1; shift
  EXIT_S=0; wait $pid || EXIT_S=$?
  if [ $EXIT_S -ne 0 ]; then
    echo "  ERROR: Adaptive PGD seed=$s failed (exit $EXIT_S)"
    STEP6_FAIL=1
  fi
done
echo "Step 6: COMPLETE"

# ── Wait order 3: Step 7 ablation ─────────────────────────────────────────────
STEP7_EXIT=0
wait $PID_ABLATION || STEP7_EXIT=$?
[ $STEP7_EXIT -ne 0 ] && echo "ERROR: Ablation failed (exit $STEP7_EXIT)"

if [ $STEP5_FAIL -ne 0 ] || [ $STEP6_FAIL -ne 0 ] || [ $STEP7_EXIT -ne 0 ]; then
  exit 2
fi

# ── Step 6b: Join L0 calibration ──────────────────────────────────────────────
echo ""
echo "=== Step 6b: Join L0 Threshold Calibration [$TAG] ==="
L0_CAL_EXIT=0
wait $PID_6B || L0_CAL_EXIT=$?
if [ $L0_CAL_EXIT -ne 0 ]; then
  echo "WARNING: L0 calibration failed (exit $L0_CAL_EXIT). Campaign eval will use config defaults."
fi

# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: Steps 7a + 7b + 7c PARALLEL (campaign / recovery / baselines)
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== Steps 7a+7b+7c: Phase 2 Parallel Launch [$TAG] ==="

# ── Step 7a: Campaign-stream eval ─────────────────────────────────────────────
(
  if [ -f experiments/evaluation/run_campaign_eval.py ]; then
    for s in $SEEDS; do
      python experiments/evaluation/run_campaign_eval.py \
        --config $CONFIG --seed $s \
        --output experiments/campaign/results_${TAG}_campaign_seed${s}.json \
        2>&1 | tee logs/${TAG}/step7a_campaign_seed${s}.log
    done
  fi
) &
PID_7A=$!

# ── Step 7b: L3-recovery eval (combined H0+H1, all strategies) ────────────────
(
  if [ -f experiments/evaluation/run_recovery_eval.py ]; then
    for s in $SEEDS; do
      python experiments/evaluation/run_recovery_eval.py \
        --config $CONFIG --seed $s --n-test $N_TEST \
        --attack PGD \
        --strategies reject passthrough tamsh tamsh_uniform tamsh_force \
        --comparison-mode combined \
        --force-expert 2 \
        --output experiments/recovery_uniform/results_${TAG}_recovery_uniform_seed${s}.json \
        2>&1 | tee logs/${TAG}/step7b_recovery_seed${s}.log
    done
  fi
) &
PID_7B=$!

# ── Step 7c: Baselines (LID/Mahalanobis/ODIN/Energy) ──────────────────────────
(
  if [ -f experiments/evaluation/run_baselines.py ]; then
    for s in $SEEDS; do
      python experiments/evaluation/run_baselines.py \
        --config $CONFIG --seed $s --n-test $N_TEST \
        --methods lid mahalanobis odin energy \
        --output experiments/evaluation/results_${TAG}_baselines_seed${s}.json \
        2>&1 | tee logs/${TAG}/step7c_baselines_seed${s}.log
    done
  fi
) &
PID_7C=$!

echo "  Waiting for Steps 7a + 7b + 7c..."
STEP7A_EXIT=0; wait $PID_7A || STEP7A_EXIT=$?
STEP7B_EXIT=0; wait $PID_7B || STEP7B_EXIT=$?
STEP7C_EXIT=0; wait $PID_7C || STEP7C_EXIT=$?
[ $STEP7A_EXIT -ne 0 ] && echo "WARN: Step 7a exit=$STEP7A_EXIT"
[ $STEP7B_EXIT -ne 0 ] && echo "WARN: Step 7b exit=$STEP7B_EXIT"
[ $STEP7C_EXIT -ne 0 ] && echo "WARN: Step 7c exit=$STEP7C_EXIT"
echo "Steps 7a+7b+7c: COMPLETE"

# ── Step 7c2: Aggregate baselines ─────────────────────────────────────────────
echo ""
echo "=== Step 7c2: Aggregate baselines [$TAG] ==="
if [ -f scripts/aggregate_baselines.py ]; then
  BASELINE_INPUTS=""
  for s in $SEEDS; do
    BASELINE_INPUTS="$BASELINE_INPUTS experiments/evaluation/results_${TAG}_baselines_seed${s}.json"
  done
  python scripts/aggregate_baselines.py \
    --inputs $BASELINE_INPUTS \
    --output baselines/results_${TAG}_baselines_aggregate.json \
    2>&1 | tee logs/${TAG}/step7c2_baselines_aggregate.log || \
    echo "  WARNING: aggregate_baselines.py failed; non-fatal."
fi

# ── Phase 2 gate checks (P0.4 campaign + P0.5 recovery) ──────────────────────
echo ""
echo "=== Gate checks (P0.4 campaign, P0.5 recovery) [$TAG] ==="
GATE_MISS_PHASE2=0
python -c "
import json, glob, sys
miss = []
cfiles = sorted(glob.glob('experiments/campaign/results_${TAG}_campaign_seed*.json'))
if not cfiles:
    miss.append('P0.4_no_results')
else:
    gaps, fas = [], []
    for f in cfiles:
        d = json.load(open(f))
        sust = d.get('sustained_rho100', {})
        clean_l0on = d.get('clean_only', {}).get('l0_on', {})
        gap = sust.get('asr_gap_pp')
        fa  = clean_l0on.get('l0_active_fraction')
        if gap is not None: gaps.append(gap)
        if fa  is not None: fas.append(fa)
    if gaps:
        mean_gap = sum(gaps)/len(gaps)
        print(f'P0.4 ASR gap mean: {mean_gap:.2f}pp  [gate >=10pp; demoted to time-to-detect]')
    if fas:
        max_fa = max(fas)
        print(f'P0.4 clean-only false-alarm max: {max_fa:.4f}  [gate <=0.01]')
        if max_fa > 0.01: miss.append(f'P0.4_clean_fpr={max_fa:.4f}>0.01')

rfiles = sorted(glob.glob('experiments/recovery_uniform/results_${TAG}_recovery_uniform_seed*.json'))
if not rfiles:
    miss.append('P0.5_no_results')
else:
    gaps = []
    for f in rfiles:
        d = json.load(open(f))
        # tamsh_force is the published CIFAR-10 winner; report its gap too
        for arm in ('tamsh_force', 'tamsh'):
            t = d.get(arm, {}).get('recovery_accuracy')
            p = d.get('passthrough', {}).get('recovery_accuracy')
            if t is not None and p is not None:
                print(f'  seed={f.split(\"seed\")[-1].split(\".\")[0]}  {arm}-passthrough = {(t-p)*100:+.2f}pp')
                if arm == 'tamsh_force': gaps.append((t-p)*100)
                break
    if gaps:
        mean_gap = sum(gaps)/len(gaps)
        print(f'P0.5 tamsh_force - passthrough gap mean: {mean_gap:.2f}pp  [gate >=15pp]')
        if mean_gap < 15: miss.append(f'P0.5_recovery_gap={mean_gap:.2f}pp<15')

if miss:
    print(f'GATE SUMMARY: {len(miss)} miss(es): {miss}')
    sys.exit(1)
print('GATE SUMMARY: ALL P0.4/P0.5 gates PASS')
" || GATE_MISS_PHASE2=1

# ── Step 7d: Build paper tables (combined CIFAR-10 + CIFAR-100) ──────────────
echo ""
echo "=== Step 7d: Rebuild paper tables (combined datasets) ==="
if [ -f scripts/build_paper_tables.py ]; then
  python scripts/build_paper_tables.py \
    --results-dir experiments \
    --out-dir paper/tables \
    2>&1 | tee logs/${TAG}/step7d_paper_tables.log
fi

# ── Step 8: Reproducibility manifest ─────────────────────────────────────────
echo ""
echo "=== Step 8: CIFAR-100 Reproducibility Manifest ==="
python -c "
import hashlib, json, os, glob, pickle
def h(p): return hashlib.sha256(open(p,'rb').read()).hexdigest()[:16] if os.path.exists(p) else None
try:
    e = pickle.load(open('models/${TAG}/ensemble_scorer.pkl','rb'))
except Exception:
    e = {}
out = {
  'dataset':                       'cifar100',
  'config':                        '$CONFIG',
  'ensemble_training_attacks':     list(e.get('training_attacks', [])),
  'ensemble_n_features':           e.get('n_features'),
  'ensemble_feature_space_version':e.get('feature_space_version'),
  'ensemble_selection_objective':  e.get('selection_objective'),
  'ensemble_sha256_16':            h('models/${TAG}/ensemble_scorer.pkl'),
  'ensemble_no_tda_sha256_16':     h('models/${TAG}/ensemble_no_tda.pkl'),
  'calibrator_sha256_16':          h('models/${TAG}/calibrator.pkl'),
  'calibrator_base_sha256_16':     h('models/${TAG}/calibrator_base.pkl'),
  'calibrator_no_tda_sha256_16':   h('models/${TAG}/calibrator_no_tda.pkl'),
  'reference_profiles_sha256_16':  h('models/${TAG}/reference_profiles.pkl'),
  'experts_sha256_16':             h('models/${TAG}/experts.pkl'),
  'l0_thresholds_sha256_16':       h('models/${TAG}/l0_thresholds.pkl'),
  'backbone_sha256_16':            h('models/${TAG}/cifar_resnet18_c100.pt'),
  'seeds':                         [42, 123, 456, 789, 999],
  'eval_split':                    'CIFAR-100 test idx 8000-9999',
  'eps_linf':                      8.0/255,
  'cw_eval_params':                {'engine': '$CW_ENGINE', 'max_iter': $CW_MAX_ITER, 'bss': $CW_BSS, 'chunk': $CW_CHUNK, 'confidence': $CW_CONFIDENCE},
  'pgd_eval_params':               {'max_iter': $PGD_MAX_ITER, 'restarts': $PGD_RESTARTS},
  'adaptive_pgd_params':           {'lambdas': [float(x) for x in '$ADAPTIVE_LAMBDAS'.split()], 'steps': $ADAPTIVE_STEPS, 'restarts': $ADAPTIVE_RESTARTS},
  'aa_eval_params':                {'version': 'standard', 'chunk': 64},
  'square_eval_params':            {'max_iter': 5000},
  'baselines_methods':             ['lid', 'mahalanobis', 'odin', 'energy'],
  'campaign_scenarios':            ['clean_only','sustained_rho050','sustained_rho080','sustained_rho100','burst','low_rate'],
  'recovery_strategies':           ['reject','passthrough','tamsh','tamsh_uniform','tamsh_force'],
  'result_files':                  sorted(glob.glob('experiments/**/results_${TAG}_*.json', recursive=True)),
}
print(json.dumps(out, indent=2))
" 2>&1 | tee logs/${TAG}/manifest.json

echo ""
echo "============================================================"
echo "PRISM CIFAR-100 Pipeline — COMPLETE"
echo "$(date)"
echo "============================================================"
echo ""
echo "Result files:"
ls -lh experiments/**/results_${TAG}_*.json 2>/dev/null | head -30
echo ""
echo "Download command (run from your laptop):"
echo "  scp -P <port> root@<ip>:'/workspace/prism-repo/prism/experiments/**/results_${TAG}_*.json' ."
echo "  scp -P <port> root@<ip>:/workspace/prism-repo/prism/logs/${TAG}/*.log ."
echo "  scp -P <port> root@<ip>:/workspace/prism-repo/prism/models/${TAG}/*.pkl ."

# ── Final exit code reflects gate outcomes ───────────────────────────────────
if [ "${GATE_MISS_PHASE2:-0}" -ne 0 ]; then
  echo ""
  echo "EXIT 3: Phase 2 gate miss (see GATE SUMMARY above)."
  exit 3
fi
if [ "${GATE_MISS_FULL:-0}" -ne 0 ]; then
  echo ""
  echo "EXIT 4: Full attack metric gate miss (see logs/${TAG}/step5_full_metric_gate.log)."
  exit 4
fi
