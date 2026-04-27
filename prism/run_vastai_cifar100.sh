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
cd /workspace/prism-repo/prism

CONFIG=configs/cifar100.yaml
TAG=cifar100

SEEDS="42 123 456 789 999"
N_TEST=1000

# CW-L2 research-plan P0.1 config (same as CIFAR-10 run)
CW_MAX_ITER=40
CW_BSS=5
CW_CHUNK=128

ADAPTIVE_LAMBDAS="0.0 0.5 1.0 2.0 5.0 10.0"
ADAPTIVE_STEPS=100
ADAPTIVE_RESTARTS=10
FGSM_OVERSAMPLE=2.5

echo "============================================================"
echo "PRISM Vast.ai CIFAR-100 Pipeline — $(date)"
echo "Config: $CONFIG"
echo "Instance: $(hostname)"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

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
    2>&1 > >(tee logs/${TAG}/step2c_no_tda.log) || echo "WARN: no-TDA variant failed"
fi

# Step 2d: differentiated experts (P0.5) — CIFAR-100
if [ -f scripts/train_experts.py ]; then
  python scripts/train_experts.py \
    --config $CONFIG \
    --output models/${TAG}/experts.pkl \
    2>&1 > >(tee logs/${TAG}/step2d_experts.log) || echo "WARN: experts failed"
fi

# ── Step 3: Calibrate (CIFAR-100) ────────────────────────────────────────────
echo ""
echo "=== Step 3: Calibrate Conformal Thresholds [$TAG] ==="
python scripts/calibrate_ensemble.py --config $CONFIG \
  2>&1 > >(tee logs/${TAG}/step3_calibrate.log)

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
  --output experiments/evaluation/results_${TAG}_cw_n${N_TEST}_ms5.json \
  2>&1 | tee logs/${TAG}/step5_cw_ms5.log &
PID_CW=$!

python experiments/evaluation/run_evaluation_full.py \
  --config $CONFIG \
  --n-test $N_TEST --attacks FGSM PGD Square AutoAttack \
  --multi-seed --seeds $SEEDS \
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

wait $PID_CW $PID_FAST
for pid in $STEP6_PIDS; do wait $pid || true; done
wait $PID_ABLATION

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
    --out paper/tables \
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
  'calibrator_sha256_16':  h('models/${TAG}/calibrator.pkl'),
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
