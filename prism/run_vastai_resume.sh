#!/bin/bash
# =============================================================================
# PRISM — Vast.ai Resume Script (from Step 4b onward)
# =============================================================================
# Resumes after Steps 0-4 completed successfully. All artifacts locked.
# Usage: bash run_vastai_resume.sh
# =============================================================================

set -euo pipefail
cd /workspace/prism-repo/prism

SEEDS="42 123 456 789 999"
N_TEST=1000

CW_MAX_ITER=40
CW_BSS=5
CW_CHUNK=128
CW_ENGINE=torch

ADAPTIVE_LAMBDAS="0.0 0.5 1.0 2.0 5.0 10.0"
ADAPTIVE_STEPS=100
ADAPTIVE_RESTARTS=10

FGSM_OVERSAMPLE=2.5

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVIDIA_TF32_OVERRIDE=1
export TORCH_CUDNN_V8_API_ENABLED=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

mkdir -p logs models experiments/calibration experiments/evaluation experiments/ablation

echo "============================================================"
echo "PRISM Vast.ai Resume — $(date)"
echo "Instance: $(hostname)"
echo "Resuming from Step 4b (Steps 0-4 already complete)"
echo "============================================================"

# ── Step 4b: Standalone Latency Benchmark ─────────────────────────────────────
echo ""
echo "=== Step 4b: Standalone Latency Benchmark ==="
python experiments/evaluation/run_evaluation_full.py \
  --n-test 200 \
  --latency-only \
  --output experiments/evaluation/results_latency_standalone.json \
  2>&1 | tee logs/step4b_latency.log
echo "Step 4b: DONE"

# ══════════════════════════════════════════════════════════════════════════════
# Steps 5 + 6 + 7: FULL PARALLEL LAUNCH
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo "=== Steps 5+6+7: Full Parallel Launch [n=$N_TEST × 5 seeds] ==="
echo "  Step 5A: CW-L2 (torch engine, max_iter=$CW_MAX_ITER, bss=$CW_BSS)"
echo "  Step 5B: FGSM + PGD + Square + AutoAttack"
echo "  Step 6 : Adaptive PGD × 5 seeds"
echo "  Step 7 : Ablation (FGSM+PGD+Square+CW via torch engine)"
echo ""

# ── Step 5A: CW ──────────────────────────────────────────────────────────────
python experiments/evaluation/run_evaluation_full.py \
  --n-test $N_TEST --attacks CW \
  --multi-seed --seeds $SEEDS \
  --cw-max-iter $CW_MAX_ITER --cw-bss $CW_BSS --cw-chunk $CW_CHUNK \
  --cw-engine $CW_ENGINE \
  --skip-latency \
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
  --skip-latency \
  --checkpoint-interval 100 \
  --output experiments/evaluation/results_fast_n${N_TEST}_ms5.json \
  2>&1 | tee logs/step5_fast_ms5.log &
PID_FAST=$!
echo "  Step 5B FGSM+PGD+Square+AA started (PID=$PID_FAST)"

# ── Step 6: Adaptive PGD — all 5 seeds ───────────────────────────────────────
echo ""
echo "  Launching Step 6 adaptive PGD seeds..."
STEP6_PIDS=""
STEP6_SEEDS=""
for s in $SEEDS; do
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
done

# ── Step 7: Ablation ─────────────────────────────────────────────────────────
echo ""
python experiments/ablation/run_ablation_paper.py \
  --n $N_TEST \
  --multi-seed --seeds $SEEDS \
  --attacks FGSM PGD Square CW \
  --output experiments/ablation/results_ablation_multiseed.json \
  2>&1 | tee logs/step7_ablation.log &
PID_ABLATION=$!
echo "  Step 7 ablation started (PID=$PID_ABLATION)"

# ── Step 6b: L0 threshold calibration ────────────────────────────────────────
echo ""
echo "  Step 6b: L0 threshold calibration launched in parallel"
python scripts/calibrate_l0_thresholds.py \
  --n-clean 500 \
  --n-adv   500 \
  > logs/step6b_l0_calibration.log 2>&1 &
PID_6B=$!
echo "  Step 6b started (PID=$PID_6B, background)"

echo ""
echo "  All processes running. Monitor logs:"
echo "    tail -f logs/step5_cw_ms5.log"
echo "    tail -f logs/step5_fast_ms5.log"
echo "    tail -f logs/step6_adaptive_pgd_seed42.log"
echo "    tail -f logs/step7_ablation.log"
echo "    tail -f logs/step6b_l0_calibration.log"
echo ""

# ── Wait order 1: Step 5 ─────────────────────────────────────────────────────
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
echo ""
echo "  Waiting for Step 6 adaptive PGD seeds..."
STEP6_FAIL=0
set -- $STEP6_PIDS
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

if [ $STEP5_FAIL -ne 0 ]; then
  echo "ERROR: Step 5 evaluation failed. Check logs/step5_cw_ms5.log and logs/step5_fast_ms5.log"
  exit 2
fi

# ── Wait Step 6b ──────────────────────────────────────────────────────────────
echo ""
echo "  Waiting for Step 6b (L0 calibration)..."
STEP6B_EXIT=0
wait $PID_6B || STEP6B_EXIT=$?
[ $STEP6B_EXIT -ne 0 ] && echo "WARNING: Step 6b L0 calibration failed (exit $STEP6B_EXIT)."
echo "Step 6b exit: $STEP6B_EXIT"

# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: Post-evaluation analysis
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo "=== Step 7a: Build Paper Tables ==="
python scripts/build_paper_tables.py \
  --results-dir experiments/evaluation \
  --output-dir experiments/evaluation \
  2>&1 | tee logs/step7a_paper_tables.log
echo "Step 7a exit: $?"

echo ""
echo "=== Step 7b: Analyze Results ==="
python scripts/analyze_results.py \
  --results-dir experiments/evaluation \
  --output-dir experiments/evaluation \
  2>&1 | tee logs/step7b_analyze.log
echo "Step 7b exit: $?"

echo ""
echo "=== Step 7c: Campaign Detection Eval ==="
python experiments/campaign/run_campaign_real.py \
  --n-test $N_TEST \
  --output experiments/evaluation/results_campaign.json \
  2>&1 | tee logs/step7c_campaign.log
echo "Step 7c exit: $?"

echo ""
echo "============================================================"
echo "PRISM Vast.ai Pipeline — COMPLETE"
echo "  $(date)"
echo "============================================================"
echo "Results:"
echo "  experiments/evaluation/results_cw_n${N_TEST}_ms5.json"
echo "  experiments/evaluation/results_fast_n${N_TEST}_ms5.json"
echo "  experiments/evaluation/results_adaptive_pgd_seed*.json"
echo "  experiments/ablation/results_ablation_multiseed.json"
echo "  experiments/evaluation/results_campaign.json"
echo ""
echo "Paper tables:"
echo "  experiments/evaluation/paper_tables/"
echo ""
echo "To download results:"
echo "  scp -P 45417 root@58.224.7.136:/workspace/Prism/prism/experiments/evaluation/*.json ./"
