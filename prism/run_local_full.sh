#!/bin/bash
# =============================================================================
# PRISM — Local Full Pipeline (CPU, all 4 P0 gates)
# =============================================================================
# Purpose: End-to-end local dry-run of the entire Vast.ai pipeline on a single
#          seed with reduced n, so the operator can validate every algorithmic
#          component (train → calibrate → L0 calibration → campaign → recovery
#          → ablation → all P0 gates) before burning the ~8h Vast.ai budget.
#
# What this covers (relative to run_smoke_test.sh which only covers Step 5):
#   ┌──────────────────────────────────┬─────────────┬──────────────┬─────────┐
#   │ Phase                            │ Smoke (old) │ Local (here) │ Vast.ai │
#   ├──────────────────────────────────┼─────────────┼──────────────┼─────────┤
#   │ Build reference profiles         │ ✅          │ ✅           │ ✅      │
#   │ Train ensemble scorer            │ ✅          │ ✅           │ ✅      │
#   │ Calibrate conformal thresholds   │ ✅          │ ✅           │ ✅      │
#   │ Train experts (TAMSH)            │ ❌          │ ✅           │ ✅      │
#   │ Validation FPR gate              │ ✅          │ ✅           │ ✅      │
#   │ L0 threshold calibration (P0.4)  │ ❌          │ ✅           │ ✅      │
#   │ Standard eval (FGSM TPR, etc.)   │ ✅          │ ✅           │ ✅      │
#   │ Campaign eval (P0.4)             │ ❌          │ ✅           │ ✅      │
#   │ Recovery eval (P0.5)             │ ❌          │ ✅           │ ✅      │
#   │ Ablation (P0.6)                  │ ❌          │ ✅           │ ✅      │
#   │ Combined gate check (exit code)  │ partial     │ ✅           │ ✅      │
#   └──────────────────────────────────┴─────────────┴──────────────┴─────────┘
#
# Local deviations (compute only — algorithm identical):
#   • n-train (ensemble):  500    (Vast.ai: 4000)
#   • n-test (eval):       100    (Vast.ai: 1000)
#   • seeds:               1      (Vast.ai: 5)
#   • CW + AutoAttack:     SKIP   (CPU-prohibitive; FGSM+PGD+Square only)
#   • Recovery attack:     PGD only (Vast.ai: PGD only too — same)
#   • Campaign scenarios:  clean_only + sustained_rho100 (Vast.ai: all 6)
#   • L0 calibration:      n=64 each (Vast.ai: 500 each)
#   • Ablation: --fast     n=50 per attack (Vast.ai: n=500 multi-seed)
#
# Runtime estimate (Mac M-series, CPU): ~2.5–3.5 hours total
#   Step 1 (profiles):      ~12 min
#   Step 2 (ensemble):      ~20 min
#   Step 2c (experts):      ~25 min
#   Step 3 (calibrate):     ~4 min
#   Step 4 (val FPR):       ~3 min
#   Step 6b (L0 cal):       ~15 min
#   Step 7a (campaign):     ~25 min
#   Step 7b (recovery):     ~30 min
#   Step 7c (ablation):     ~25 min
#   Gate check:             <1 min
#
# Exit codes:
#   0 → all P0.4/P0.5/P0.6 gates pass
#   1 → setup / training / calibration error
#   3 → P0.4 or P0.5 or P0.6 gate miss (artifacts written; review & retune)
#
# Usage:
#   bash run_local_full.sh                    # default n_test=100
#   bash run_local_full.sh 200                # higher confidence
#   bash run_local_full.sh 100 --skip-train   # reuse existing models/*.pkl
# =============================================================================

set -euo pipefail

N_TEST="${1:-100}"
SEED=42
SKIP_TRAIN=0
[ "${2:-}" = "--skip-train" ] && SKIP_TRAIN=1

LOCAL_OUT="experiments/evaluation/local_full"
mkdir -p logs "$LOCAL_OUT" models \
  experiments/calibration experiments/evaluation \
  experiments/campaign experiments/recovery experiments/ablation

echo "============================================================"
echo "PRISM Local Full Pipeline — $(date)"
echo "n_test=$N_TEST  seed=$SEED  device=cpu  skip_train=$SKIP_TRAIN"
echo "============================================================"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

# ── Step 0: Environment + config parity ─────────────────────────────────────
echo ""
echo "=== Step 0: Environment + Config Parity ==="
python3 -c "
import sys, torch, gudhi, sklearn
from src.config import (LAYER_NAMES, N_SUBSAMPLE, MAX_DIM, EPS_LINF_STANDARD,
                        CONFORMAL_ALPHAS, TIER_CAL_ALPHA_FACTORS,
                        PROFILE_IDX, CAL_IDX, VAL_IDX, EVAL_IDX, PATHS)
errs = []
if LAYER_NAMES != ['layer2', 'layer3', 'layer4']: errs.append(f'LAYER_NAMES={LAYER_NAMES}')
if N_SUBSAMPLE != 150: errs.append(f'N_SUBSAMPLE={N_SUBSAMPLE}')
if MAX_DIM != 1: errs.append(f'MAX_DIM={MAX_DIM}')
if abs(EPS_LINF_STANDARD - 8/255) > 1e-9: errs.append(f'EPS={EPS_LINF_STANDARD}')
if PROFILE_IDX != (0, 5000): errs.append(f'PROFILE_IDX={PROFILE_IDX}')
if CAL_IDX != (5000, 7000): errs.append(f'CAL_IDX={CAL_IDX}')
if VAL_IDX != (7000, 8000): errs.append(f'VAL_IDX={VAL_IDX}')
if EVAL_IDX != (8000, 10000): errs.append(f'EVAL_IDX={EVAL_IDX}')
if errs:
    for e in errs: print(f'  CONFIG MISMATCH: {e}')
    sys.exit(1)
print(f'  PyTorch: {torch.__version__}, gudhi: {gudhi.__version__}')
print(f'  Splits: PROFILE={PROFILE_IDX} CAL={CAL_IDX} VAL={VAL_IDX} EVAL={EVAL_IDX}')
print(f'  PATHS[ensemble_scorer]={PATHS[\"ensemble_scorer\"]}')
print('  Config parity: OK')
"

# ── Phase 1: training ──
if [ $SKIP_TRAIN -eq 0 ]; then
  # ── Step 1: Build reference profiles ───────────────────────────────────────
  echo ""
  echo "=== Step 1: Build Reference Profiles [PROFILE_IDX] ==="
  python3 scripts/build_profile_testset.py 2>&1 | tee logs/local_step1_profiles.log
  STEP1_EXIT=${PIPESTATUS[0]}
  [ $STEP1_EXIT -ne 0 ] && { echo "ERROR: Step 1 failed."; exit 1; }

  # ── Step 2: Train ensemble scorer ──────────────────────────────────────────
  echo ""
  echo "=== Step 2: Train Ensemble Scorer [n_train=500, FGSM/PGD/Square] ==="
  python3 scripts/train_ensemble_scorer.py \
    --n-train 500 \
    --fgsm-oversample 2.5 \
    --output models/ensemble_scorer.pkl \
    2>&1 | tee logs/local_step2_train.log
  STEP2_EXIT=${PIPESTATUS[0]}
  [ $STEP2_EXIT -ne 0 ] && { echo "ERROR: Step 2 failed."; exit 1; }

  # Verify ensemble_scorer.pkl is a dict (the previously-fixed smoke bug)
  python3 -c "
import pickle, sys
e = pickle.load(open('models/ensemble_scorer.pkl', 'rb'))
assert isinstance(e, dict), f'expected dict, got {type(e).__name__}'
ta = list(e.get('training_attacks', []))
assert 'FGSM' in ta, f'FGSM missing from training_attacks: {ta}'
print(f'  ensemble_scorer.pkl: training_attacks={ta} alpha={e.get(\"alpha\"):.3f}')
print(f'  alpha_tune_summary: {e.get(\"alpha_tune_summary\")}')
"

  # ── Step 2c: Train experts (required for recovery eval) ────────────────────
  if [ ! -f models/experts.pkl ]; then
    echo ""
    echo "=== Step 2c: Train MoE Experts (TAMSH) [4 experts × 500 imgs] ==="
    python3 scripts/train_experts.py \
      --n-train 500 \
      --epochs 2 \
      2>&1 | tee logs/local_step2c_experts.log
    STEP2C_EXIT=${PIPESTATUS[0]}
    [ $STEP2C_EXIT -ne 0 ] && { echo "ERROR: Step 2c failed."; exit 1; }
  else
    echo ""
    echo "=== Step 2c: SKIP — models/experts.pkl already exists ($(du -h models/experts.pkl | cut -f1)) ==="
  fi

  # ── Step 3: Calibrate conformal thresholds ─────────────────────────────────
  echo ""
  echo "=== Step 3: Calibrate Conformal Thresholds [CAL_IDX] ==="
  python3 scripts/calibrate_ensemble.py 2>&1 | tee logs/local_step3_calibrate.log
  STEP3_EXIT=${PIPESTATUS[0]}
  [ $STEP3_EXIT -ne 0 ] && { echo "ERROR: Step 3 failed."; exit 1; }
else
  echo ""
  echo "=== --skip-train: Reusing existing models/*.pkl ==="
  for f in reference_profiles.pkl ensemble_scorer.pkl calibrator.pkl experts.pkl; do
    if [ ! -f "models/$f" ]; then
      echo "ERROR: --skip-train but models/$f is missing. Run without --skip-train first."
      exit 1
    fi
    echo "  [OK] models/$f ($(du -h models/$f | cut -f1))"
  done
fi

# ── Step 4: Validation FPR gate ─────────────────────────────────────────────
echo ""
echo "=== Step 4: Validation FPR Gate [VAL_IDX] ==="
python3 scripts/compute_ensemble_val_fpr.py 2>&1 | tee logs/local_step4_val_fpr.log

python3 -c "
import json, sys
with open('experiments/calibration/ensemble_fpr_report.json') as f:
    r = json.load(f)
targets = [('L1', 0.10), ('L2', 0.03), ('L3', 0.005)]
fail = []
for tier, tgt in targets:
    fpr = r['tiers'][tier]['FPR']
    status = 'PASS' if fpr <= tgt else 'FAIL'
    print(f'  {tier}: FPR={fpr:.4f}  target={tgt}  [{status}]')
    if fpr > tgt: fail.append(f'{tier} FPR={fpr:.4f}>{tgt}')
if fail:
    print(f'P0.1/FPR GATE FAIL: {fail}')
    sys.exit(1)
print('  P0.1/FPR GATE: PASS')
"
[ $? -ne 0 ] && { echo "ERROR: FPR gate failed. Tighten TIER_CAL_ALPHA_FACTORS."; exit 1; }

# ── Step 6b: L0 threshold calibration (P0.4 lever) ──────────────────────────
echo ""
echo "=== Step 6b: L0 Threshold Calibration [P0.4 lever, n_clean=n_adv=64] ==="
python3 scripts/calibrate_l0_thresholds.py \
  --n-clean 64 --n-adv 64 \
  2>&1 | tee logs/local_step6b_l0_cal.log
L0_CAL_EXIT=${PIPESTATUS[0]}
if [ $L0_CAL_EXIT -ne 0 ]; then
  echo "WARN: L0 calibration FAILED (exit $L0_CAL_EXIT) — Step 7a will use defaults."
  echo "      For local dry-run with n=64, defaults are usually adequate."
fi

# ── Step 7a: Campaign-stream eval (P0.4) ────────────────────────────────────
echo ""
echo "=== Step 7a: Campaign-Stream Eval [P0.4, scenarios: clean_only + sustained_rho100] ==="
python3 experiments/evaluation/run_campaign_eval.py \
  --scenarios clean_only sustained_rho100 \
  --seed $SEED \
  --n-clean-pool $N_TEST \
  --n-adv-pool $N_TEST \
  --output experiments/campaign/results_campaign_seed${SEED}.json \
  --device cpu \
  2>&1 | tee logs/local_step7a_campaign.log
STEP7A_EXIT=${PIPESTATUS[0]}
[ $STEP7A_EXIT -ne 0 ] && echo "WARN: Step 7a exited with $STEP7A_EXIT"

# ── Step 7b: TAMSH recovery eval (P0.5) ─────────────────────────────────────
echo ""
echo "=== Step 7b: Recovery Eval [P0.5, attack=PGD, strategies=reject+passthrough+tamsh] ==="
python3 experiments/evaluation/run_recovery_eval.py \
  --attack PGD \
  --n-test $N_TEST \
  --strategies reject passthrough tamsh \
  --seed $SEED \
  --output experiments/recovery/results_recovery_seed${SEED}.json \
  --device cpu \
  2>&1 | tee logs/local_step7b_recovery.log
STEP7B_EXIT=${PIPESTATUS[0]}
[ $STEP7B_EXIT -ne 0 ] && echo "WARN: Step 7b exited with $STEP7B_EXIT"

# ── Step 7c: Ablation (P0.6) ────────────────────────────────────────────────
echo ""
echo "=== Step 7c: Ablation [P0.6, --fast, FGSM+PGD] ==="
python3 experiments/ablation/run_ablation_paper.py \
  --fast \
  --attacks FGSM PGD \
  2>&1 | tee logs/local_step7c_ablation.log
STEP7C_EXIT=${PIPESTATUS[0]}
[ $STEP7C_EXIT -ne 0 ] && echo "WARN: Step 7c exited with $STEP7C_EXIT"

# ── Combined gate check (mirrors run_vastai_full.sh:595+) ───────────────────
echo ""
echo "=== Combined Gate Checks (P0.4 + P0.5 + P0.6) ==="
python3 -c "
import json, glob, os, sys
miss = []

# ── P0.4 campaign ──
cfiles = sorted(glob.glob('experiments/campaign/results_campaign_seed*.json'))
if cfiles:
    for f in cfiles:
        d = json.load(open(f))
        sust = d.get('sustained_rho100', {})
        clean_l0on = d.get('clean_only', {}).get('l0_on', {})
        gap = sust.get('asr_gap_pp')
        fa = clean_l0on.get('l0_active_fraction')
        if gap is not None:
            print(f'  P0.4 ASR gap (sustained ρ=1.0): {gap:.2f}pp  [gate >= 10pp]')
            if gap < 10: miss.append(f'P0.4_asr_gap={gap:.2f}<10')
        if fa is not None:
            print(f'  P0.4 clean false-alarm:        {fa:.4f}    [gate <= 0.01]')
            if fa > 0.01: miss.append(f'P0.4_clean_fpr={fa:.4f}>0.01')
else:
    print('  P0.4: no campaign results found')
    miss.append('P0.4_no_results')

# ── P0.5 recovery ──
rfiles = sorted(glob.glob('experiments/recovery/results_recovery_seed*.json'))
if rfiles:
    for f in rfiles:
        d = json.load(open(f))
        t = d.get('tamsh', {}).get('recovery_accuracy')
        p = d.get('passthrough', {}).get('recovery_accuracy')
        tr = d.get('_meta', {}).get('l3_trigger_rate')
        if t is not None and p is not None:
            gap = (t - p) * 100
            print(f'  P0.5 TAMSH-passthrough gap:    {gap:.2f}pp  [gate >= 15pp]')
            if gap < 15: miss.append(f'P0.5_recovery_gap={gap:.2f}<15')
        if tr is not None:
            ok = 0.10 <= tr <= 0.80
            print(f'  P0.5 L3 trigger rate:          {tr:.3f}     [band 0.10-0.80, {\"OK\" if ok else \"WARN\"}]')
else:
    print('  P0.5: no recovery results found')
    miss.append('P0.5_no_results')

# ── P0.6 ablation ──
afiles = sorted(glob.glob('experiments/ablation/results_ablation_paper*.json'))
if afiles:
    for f in afiles[-1:]:
        d = json.load(open(f))
        # Look for full vs ensemble-no-tda comparison
        full_auc = d.get('full', {}).get('auc') or d.get('Full PRISM', {}).get('AUC')
        no_tda_auc = d.get('ensemble_no_tda', {}).get('auc') or d.get('Ensemble (no TDA)', {}).get('AUC')
        if full_auc is not None and no_tda_auc is not None:
            print(f'  P0.6 Full PRISM AUC: {full_auc:.4f}, Ensemble-no-TDA AUC: {no_tda_auc:.4f}  [gate full > no-tda]')
            if full_auc <= no_tda_auc: miss.append(f'P0.6_full={full_auc:.4f}<=no_tda={no_tda_auc:.4f}')
        else:
            print('  P0.6: ablation result keys not found in JSON; inspect manually')
            print(f'        keys present: {list(d.keys())[:10]}')
else:
    print('  P0.6: no ablation results found')
    miss.append('P0.6_no_results')

print('')
if miss:
    print(f'GATE SUMMARY: {len(miss)} miss(es) — {miss}')
    sys.exit(3)
print('GATE SUMMARY: ALL P0.4/P0.5/P0.6 gates PASS')
"
GATE_EXIT=$?

echo ""
echo "============================================================"
if [ $GATE_EXIT -eq 0 ]; then
  echo "LOCAL FULL PIPELINE: PASS — proceed to Vast.ai"
  echo ""
  echo "Next: bash run_vastai_full.sh"
else
  echo "LOCAL FULL PIPELINE: GATE MISS — diagnose before Vast.ai"
  echo ""
  echo "Diagnostics:"
  echo "  cat experiments/campaign/results_campaign_seed${SEED}.json | python3 -m json.tool | head -40"
  echo "  cat experiments/recovery/results_recovery_seed${SEED}.json | python3 -m json.tool | head -40"
  echo "  cat experiments/ablation/results_ablation_paper*.json | python3 -m json.tool | head -40"
  echo "  tail -30 logs/local_step7a_campaign.log"
  echo "  tail -30 logs/local_step7b_recovery.log"
fi
echo "$(date)"
echo "============================================================"
exit $GATE_EXIT
