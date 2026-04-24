#!/bin/bash
# =============================================================================
# PRISM — Vast.ai Full Pipeline (Retrain + Calibrate + Parallel Eval + Ablation)
# =============================================================================
# Usage: bash run_vastai_full.sh
# Runs the entire publishable pipeline on a single RTX 5090 instance.
#
# Parallelism map (safe — no shared mutable state between concurrent jobs):
#   Steps 5+6+7: ALL launched simultaneously after Step 4 LOCK.
#     Step 5A: CW-L2          (bottleneck, ~2h)
#     Step 5B: FGSM+PGD+Square+AA
#     Step 6 : 5 adaptive-PGD seeds in parallel
#     Step 7 : Ablation (FGSM+PGD+Square)
#   Wait order: Step 5 first (provenance check needs its JSON), then 6, then 7.
#   Saves ~35-40% wall-clock vs old sequential 5→6→7 schedule.
#
# Sequential constraints that CANNOT be parallelised:
#   Steps 1→2: Step 2 reads reference_profiles.pkl written by Step 1
#   Steps 2→3: Step 3 reads ensemble_scorer.pkl written by Step 2
#   Steps 3→4: Step 4 reads calibrator.pkl written by Step 3
#   Steps 4→5/6/7: All eval phases require the FPR gate to have passed
#   Steps 5+6+7 join → Step 6b (L0 calibration) → Steps 7a+7b+7c
#
# Exit codes: 0=success, 1=gate failure, 2=eval failure

set -euo pipefail
cd /workspace/prism-repo/prism

SEEDS="42 123 456 789 999"
N_TEST=1000

# CW-L2 research-plan config (P0.1): 40 iter × 5 binary-search steps × batch 256.
# This is the RobustBench detector-evaluation standard; balances ℓ₂ attack
# strength against GPU-hours. Pre-research-plan values (max_iter=100, bss=9)
# are preserved in the legacy block below for reference.
CW_MAX_ITER=40
CW_BSS=5
CW_CHUNK=128   # bs=256 for research-plan P0.1 (was 64 for bs=128 legacy)

# Adaptive-PGD expanded sweep (P1.4): λ ∈ {0, 0.5, 1, 2, 5, 10} with 100-step /
# 10-restart PGD variant. EOT=1 (hash subsample is deterministic; one EOT pass
# verifies this; larger EOT is wasted compute).
ADAPTIVE_LAMBDAS="0.0 0.5 1.0 2.0 5.0 10.0"
ADAPTIVE_STEPS=100
ADAPTIVE_RESTARTS=10

# FGSM oversample (P0.3): restore to 2.5 — the regression to 1.8/2.0 in commits
# dadf2cf/cf854f0 dropped pooled FGSM TPR from 0.87 → 0.806.
FGSM_OVERSAMPLE=2.5

echo "============================================================"
echo "PRISM Vast.ai Full Pipeline — $(date)"
echo "Instance: $(hostname)"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# ── Environment ──────────────────────────────────────────────────────────────
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVIDIA_TF32_OVERRIDE=1
export TORCH_CUDNN_V8_API_ENABLED=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

mkdir -p logs models experiments/calibration experiments/evaluation experiments/ablation

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
assert TIER_CAL_ALPHA_FACTORS.get('L3', CAL_ALPHA_FACTOR) <= 0.55, \
  f'L3 factor must be <= 0.55, got {TIER_CAL_ALPHA_FACTORS.get(\"L3\")}'
print('PREFLIGHT PASS: per-tier L3=0.50 confirmed')
"

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

# ── Step 2: Retrain ensemble (with CW + AutoAttack in training mix) ───────────
echo ""
echo "=== Step 2: Retrain Ensemble [n=4000, CW+AA in mix, fgsm-os=$FGSM_OVERSAMPLE] ==="
# Research-plan P0.3: FGSM oversample 2.5 restores the pre-regression training
# mix share that achieved pooled FGSM TPR 0.87. Commits dadf2cf/cf854f0
# temporarily lowered this to 1.8/2.0 and dropped pooled FGSM TPR to 0.806 —
# below the 0.85 gate. This value is locked by Appendix §A2 of VASTAI_RUN_GUIDE.md.
#
# NOTE: --use-grad-norm was tested on the 2026-04-22 Vast.ai run and REVERTED.
# It caused a catastrophic regression: FGSM TPR 80.6% → 63.0%, Square 89.1% → 79.8%.
# See regression_analysis_20260422.md for the full forensic analysis.
python scripts/train_ensemble_scorer.py \
  --n-train 4000 \
  --fgsm-oversample $FGSM_OVERSAMPLE \
  --include-cw \
  --include-autoattack \
  --cw-max-iter 30 \
  --cw-bss 3 \
  --output models/ensemble_scorer.pkl \
  2>&1 > >(tee logs/step2_retrain.log)
STEP2_EXIT=${PIPESTATUS[0]:-$?}
echo "Step 2 exit: $STEP2_EXIT"
if [ "$STEP2_EXIT" -ne 0 ]; then
  echo "ERROR: Step 2 failed. Check logs/step2_retrain.log"; exit 1
fi

# ── Step 2b: Post-retrain verification ────────────────────────────────────────
# Verifies CW and AutoAttack are in the training mix, grad-norm is OFF,
# and feature dimension is 37 (36 persistence + 1 DCT, no grad-norm).
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
# n_features is now saved explicitly; fall back to computing from flags for
# any pkl created before this fix was deployed (backward-compatible).
nf = d.get('n_features')
if nf is None:
    base = len(d.get('layer_names', [])) * len(d.get('dims', [])) * 6
    nf   = base + int(d.get('use_dct', False)) + int(d.get('use_grad_norm', False))
errors = []
if 'CW' not in ta:
    errors.append(f'CW missing from training_attacks: {ta}')
if 'AutoAttack' not in ta:
    errors.append(f'AutoAttack missing from training_attacks: {ta}')
if ng:
    errors.append('use_grad_norm=True — grad-norm must be OFF (reverted, see regression_analysis_20260422.md)')
if nf != 37:
    errors.append(f'n_features={nf}, expected 37 (36 persistence + 1 DCT)')
if errors:
    print('RETRAIN VERIFICATION FAIL:')
    for err in errors: print(f'  • {err}')
    sys.exit(1)
print(f'[OK] Retrain verified: training_attacks={ta}')
print(f'[OK] use_grad_norm={ng}, n_features={nf}')
"
if [ $? -ne 0 ]; then
  echo "ERROR: Post-retrain verification failed. Fix and re-run Step 2."; exit 1
fi

# ── Step 2c: Ensemble-no-TDA variant (P0.6) ──────────────────────────────────
# Research-plan P0.6: retrain the ensemble scorer with TDA features disabled so
# we can quantify TAMM's (C1) marginal contribution in the ablation table.
# Artifact lands at models/ensemble_no_tda.pkl so the canonical pipeline
# (calibrator, FPR gate, eval) continues to use models/ensemble_scorer.pkl.
# If `--no-tda-features` is not implemented in train_ensemble_scorer.py on this
# branch, the step skips with a warning rather than aborting — the main
# pipeline does not depend on it.
echo ""
echo "=== Step 2c: Ensemble-no-TDA variant [P0.6] ==="
if python scripts/train_ensemble_scorer.py --help 2>&1 | grep -q -- '--no-tda-features'; then
  python scripts/train_ensemble_scorer.py \
    --n-train 4000 \
    --fgsm-oversample $FGSM_OVERSAMPLE \
    --include-cw \
    --include-autoattack \
    --cw-max-iter 30 \
    --cw-bss 3 \
    --no-tda-features \
    --output models/ensemble_no_tda.pkl \
    2>&1 > >(tee logs/step2c_retrain_no_tda.log) || {
      echo "WARN: Step 2c (ensemble-no-TDA) failed. Ablation arm will be missing."
    }
else
  echo "SKIP: --no-tda-features flag not available on this branch. Re-run after merging P0.6 support."
fi

# ── Step 2d: Differentiated experts (P0.5) ───────────────────────────────────
# Research-plan P0.5: the existing MoE experts may be homogeneous (all trained
# on the same distillation target). Re-train four experts with distinct attack
# mixes (clean / fgsm / pgd / mixed) so TAMSH routing has something to choose
# between. Skipped if scripts/train_experts.py is absent.
echo ""
echo "=== Step 2d: Differentiated experts [P0.5] ==="
if [ -f scripts/train_experts.py ]; then
  python scripts/train_experts.py \
    --config configs/default.yaml \
    --output models/experts.pkl \
    2>&1 > >(tee logs/step2d_train_experts.log) || {
      echo "WARN: Step 2d (experts) failed. Recovery eval may produce trivial results."
    }
else
  echo "SKIP: scripts/train_experts.py not present. Recovery eval will use existing experts."
fi

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
  echo "FIX: Lower tier_cal_alpha_factors in configs/default.yaml, re-run steps 3-4"
  exit 1
fi

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

# ══════════════════════════════════════════════════════════════════════════════
# Steps 5 + 6 + 7: FULL PARALLEL LAUNCH
# ══════════════════════════════════════════════════════════════════════════════
# All three phases are pure consumers of the locked frozen artifacts.
# No inter-step file dependency between 5, 6, and 7 — each writes to distinct
# output paths and loads the same read-only pkl artifacts from Step 4.
#
# Parallelism map:
#   Step 5A : CW-L2 eval (bottleneck, ~2h)
#   Step 5B : FGSM + PGD + Square + AutoAttack eval
#   Step 6  : 5 adaptive PGD seeds (each ~1h; overlap with Step 5)
#   Step 7  : Ablation (FGSM+PGD+Square, overlap with Steps 5+6)
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
echo "  Step 5A: CW-L2 (max_iter=$CW_MAX_ITER, bss=$CW_BSS)"
echo "  Step 5B: FGSM + PGD + Square + AutoAttack"
echo "  Step 6 : Adaptive PGD × 5 seeds"
echo "  Step 7 : Ablation"
echo "  All launched simultaneously — CW is the wall-clock bottleneck."
echo ""

# ── Step 5A: CW ──────────────────────────────────────────────────────────────
python experiments/evaluation/run_evaluation_full.py \
  --n-test $N_TEST --attacks CW \
  --multi-seed --seeds $SEEDS \
  --cw-max-iter $CW_MAX_ITER --cw-bss $CW_BSS --cw-chunk $CW_CHUNK \
  --checkpoint-interval 100 \
  --output experiments/evaluation/results_cw_n${N_TEST}_ms5.json \
  2>&1 | tee logs/step5_cw_ms5.log &
PID_CW=$!
echo "  Step 5A CW started (PID=$PID_CW)"

# ── Step 5B: Fast attacks ─────────────────────────────────────────────────────
python experiments/evaluation/run_evaluation_full.py \
  --n-test $N_TEST --attacks FGSM PGD Square AutoAttack \
  --multi-seed --seeds $SEEDS \
  --gen-chunk 128 --square-max-iter 5000 \
  --aa-version standard --aa-chunk 64 \
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
for s in $SEEDS; do
  # Research-plan P1.4: expanded λ sweep, 100-step × 10-restart PGD, EOT=1 (hash
  # subsample is deterministic; one EOT pass is sufficient to verify).
  # Flags --pgd-restarts, --eot-samples, --lambdas are gated on the P1.4 branch
  # being merged — if argparse rejects them, fall back to the defaults.
  if python experiments/evaluation/run_adaptive_pgd.py --help 2>&1 | grep -q -- '--pgd-restarts'; then
    python experiments/evaluation/run_adaptive_pgd.py \
      --n-test $N_TEST --seed $s \
      --lambdas $ADAPTIVE_LAMBDAS \
      --pgd-steps $ADAPTIVE_STEPS \
      --pgd-restarts $ADAPTIVE_RESTARTS \
      --eot-samples 1 \
      --output experiments/evaluation/results_adaptive_pgd_seed${s}.json \
      2>&1 | tee logs/step6_adaptive_pgd_seed${s}.log &
  else
    python experiments/evaluation/run_adaptive_pgd.py \
      --n-test $N_TEST --seed $s \
      --output experiments/evaluation/results_adaptive_pgd_seed${s}.json \
      2>&1 | tee logs/step6_adaptive_pgd_seed${s}.log &
  fi
  STEP6_PIDS="$STEP6_PIDS $!"
  STEP6_SEEDS="$STEP6_SEEDS $s"
  echo "  Step 6 seed=$s started (PID=$!)"
done

# ── Step 7: Ablation ─────────────────────────────────────────────────────────
# Reads only frozen pkl artifacts; output path is separate from Steps 5+6.
echo ""
python experiments/ablation/run_ablation_paper.py \
  --n $N_TEST \
  --multi-seed --seeds $SEEDS \
  --attacks FGSM PGD Square \
  2>&1 | tee logs/step7_ablation.log &
PID_ABLATION=$!
echo "  Step 7 ablation started (PID=$PID_ABLATION)"

echo ""
echo "  All processes running. Monitor logs:"
echo "    tail -f logs/step5_cw_ms5.log"
echo "    tail -f logs/step6_adaptive_pgd_seed42.log"
echo "    tail -f logs/step7_ablation.log"
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
    if 'CW' in ta and 'AutoAttack' in ta:
        print('PROVENANCE CHECK PASS: all results use same retrained ensemble')
    else:
        print(f'WARNING: ensemble does not include CW+AutoAttack: {ta}')
else:
    print('PROVENANCE CHECK FAIL: different ensembles detected!')
    exit(1)
"
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

# ── Step 6b: Calibrate L0 thresholds on real score streams (P0.4 lever) ──────
# Must run after the ensemble scorer + calibrator (Steps 3+4) are written and
# before Step 7a (campaign eval). Reads models/ensemble_scorer.pkl,
# models/calibrator.pkl, and models/reference_profiles.pkl (all read-only) and
# writes models/l0_thresholds.pkl. CampaignMonitor (Step 7a) auto-discovers this
# pkl and overlays its hazard_rate/alert_run_prob/warmup_steps, replacing the
# synthetic-tuned defaults. On any feasibility failure the script exits non-zero.
echo ""
echo "=== Step 6b: L0 Threshold Calibration [P0.4] ==="
python scripts/calibrate_l0_thresholds.py \
  --n-clean 500 \
  --n-adv   500 \
  2>&1 | tee logs/step6b_l0_calibration.log
L0_CAL_EXIT=${PIPESTATUS[0]}
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
python -c "
import json, glob
# ── P0.4 campaign gates ──
cfiles = sorted(glob.glob('experiments/campaign/results_campaign_seed*.json'))
if not cfiles:
    print('WARN: no campaign results found — skipping P0.4 gate check')
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
        if mean_gap < 10: print('  -> C3 gate MISS: demote SACD to appendix per Appendix A3')
    if fas:
        max_fa = max(fas)
        print(f'P0.4 clean-only false-alarm (max across seeds): {max_fa:.4f}  [gate <= 0.01]')
        if max_fa > 0.01: print('  -> BOCPD priors (mu0/beta0) need retune in configs/default.yaml')

# ── P0.5 recovery gate ──
rfiles = sorted(glob.glob('experiments/recovery/results_recovery_seed*.json'))
if not rfiles:
    print('WARN: no recovery results found — skipping P0.5 gate check')
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
        if mean_gap < 15: print('  -> C4 gate MISS: demote TAMSH to appendix per Appendix A3')
"

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
  'ensemble_use_grad_norm':    e.get('use_grad_norm'),
  'ensemble_sha256_16':        h('models/ensemble_scorer.pkl'),
  'calibrator_sha256_16':      h('models/calibrator.pkl'),
  'reference_profiles_sha256_16': h('models/reference_profiles.pkl'),
  'seeds':                     [42, 123, 456, 789, 999],
  'eval_split':                'CIFAR-10 test idx 8000-9999',
  'eps_linf':                  8.0/255,
  'cw_eval_params':            {'max_iter': $CW_MAX_ITER, 'bss': $CW_BSS, 'batch_size': 256, 'confidence': 0.0},
  'adaptive_pgd_params':       {'lambdas': [float(x) for x in '$ADAPTIVE_LAMBDAS'.split()], 'steps': $ADAPTIVE_STEPS, 'restarts': $ADAPTIVE_RESTARTS, 'eot_samples': 1},
  'fgsm_oversample':           $FGSM_OVERSAMPLE,
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
