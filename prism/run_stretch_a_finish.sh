#!/bin/bash
# =============================================================================
# PRISM — Stretch A FINISH script (resume after ThunderCompute A6000 partial)
# =============================================================================
# Picks up where run_stretch_a_thundercompute.sh stopped:
#   - reuses existing models/cifar_wrn28_10.pt, models/wrn/*.pkl, calibrator
#   - trains the missing WRN experts.pkl
#   - reruns Square + AutoAttack for seeds 123/456/789/999 (seed42 already done)
#   - runs the ablation (now that experts.pkl exists)
#   - runs adaptive PGD with a SIZED-DOWN budget that finishes in <12 h on H100
#   - builds the paper tables
#
# Hardware: ANY GPU with >=40 GB. Tested mental-model sizing:
#   H100 80 GB  ~ 2.5x A6000 throughput on PGD-EOT  -> full plan ~10-12 h
#   H200 141 GB ~ 3x A6000                          -> full plan ~7-9 h
#   A100 80 GB  ~ 1.5x A6000                        -> full plan ~16 h
#   RTX 5090 32 GB ~ 2x A6000 but adaptive batch=8  -> full plan ~12 h
#
# Usage:
#   bash run_stretch_a_finish.sh
#   # or trim further:
#   ADAPTIVE_LAMBDAS="0.0 1.0 5.0" ADAPTIVE_N=500 bash run_stretch_a_finish.sh
# =============================================================================
set -euo pipefail

# ── Locate PRISM root ────────────────────────────────────────────────────────
if [ -d /workspace/prism-repo/prism/prism/src ] && [ -f /workspace/prism-repo/prism/prism/requirements.txt ]; then
  PRISM_ROOT=/workspace/prism-repo/prism/prism
elif [ -d /workspace/prism-repo/prism/src ] && [ -f /workspace/prism-repo/prism/requirements.txt ]; then
  PRISM_ROOT=/workspace/prism-repo/prism
elif [ -d "$(pwd)/src" ] && [ -f "$(pwd)/requirements.txt" ]; then
  PRISM_ROOT="$(pwd)"
elif [ -d "$(dirname "$0")/src" ] && [ -f "$(dirname "$0")/requirements.txt" ]; then
  PRISM_ROOT="$(cd "$(dirname "$0")" && pwd)"
else
  echo "ERROR: Could not locate PRISM root." ; exit 1
fi
cd "$PRISM_ROOT"
export PYTHONPATH="$PRISM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PRISM_CONFIG="${PRISM_CONFIG:-configs/wrn_cifar10.yaml}"

mkdir -p logs/stretch_a_finish experiments/wrn/ablation

SEEDS_REMAINING="${SEEDS_REMAINING:-123 456 789 999}"
N_TEST="${N_TEST:-1000}"

# Adaptive PGD: pared down from the original A6000 spec
# Original: 6 lambdas x 5 seeds x 1000 imgs x 100 steps x 10 restarts
# New default: 4 lambdas x 1 seed  x 1000 imgs x 50  steps x 5  restarts
# Override via env: ADAPTIVE_LAMBDAS, ADAPTIVE_SEEDS, ADAPTIVE_N, ADAPTIVE_STEPS, ADAPTIVE_RESTARTS
ADAPTIVE_LAMBDAS="${ADAPTIVE_LAMBDAS:-0.0 1.0 5.0 10.0}"
ADAPTIVE_SEEDS="${ADAPTIVE_SEEDS:-42}"
ADAPTIVE_N="${ADAPTIVE_N:-1000}"
ADAPTIVE_STEPS="${ADAPTIVE_STEPS:-50}"
ADAPTIVE_RESTARTS="${ADAPTIVE_RESTARTS:-5}"

echo "============================================================"
echo "PRISM Stretch A — FINISH script"
echo "$(date)  |  host: $(hostname)"
echo "Config:           $PRISM_CONFIG"
echo "Remaining seeds:  $SEEDS_REMAINING (for Square/AA)"
echo "Adaptive PGD:     lambdas=[$ADAPTIVE_LAMBDAS] seeds=[$ADAPTIVE_SEEDS]"
echo "                  n=$ADAPTIVE_N steps=$ADAPTIVE_STEPS restarts=$ADAPTIVE_RESTARTS"
echo "============================================================"

# ── Step 0: sanity check we have the artifacts we are resuming from ──────────
echo ""
echo "=== Step 0: Verify resumed artifacts ==="
for f in models/cifar_wrn28_10.pt models/wrn/ensemble_scorer.pkl models/wrn/calibrator.pkl \
         models/wrn/reference_profiles.pkl models/wrn/ensemble_no_tda.pkl; do
  if [ ! -f "$f" ]; then
    echo "MISSING: $f"; echo "Re-upload the WRN checkpoints from the prior run before continuing."; exit 1
  fi
done
python scripts/verify_backbone_acc.py \
  --checkpoint models/cifar_wrn28_10.pt \
  --sidecar    models/cifar_wrn28_10.acc.json \
  --min-acc    0.93 --n 1000
echo "Step 0: PASS"

# ── Step A: Train WRN experts (was missing in original script) ───────────────
echo ""
echo "=== Step A: Train WRN MoE experts ==="
if [ -f models/wrn/experts.pkl ]; then
  echo "  experts.pkl already exists — skipping"
else
  python scripts/train_experts.py \
    --output models/wrn/experts.pkl \
    2>&1 | tee logs/stretch_a_finish/stepA_train_experts.log
fi
echo "Step A: DONE"

# ── Step B: Square + AutoAttack for the 4 missing seeds ──────────────────────
echo ""
echo "=== Step B: Square + AutoAttack (seeds: $SEEDS_REMAINING) ==="
for s in $SEEDS_REMAINING; do
  echo "  --- seed=$s ---"
  python experiments/evaluation/run_evaluation_full.py \
    --attacks Square AutoAttack \
    --n_test  $N_TEST \
    --seeds   $s \
    --eval-split test \
    --eval-offset 8000 \
    --output experiments/wrn/evaluation/results_fast_wrn_seed${s}.json \
    --skip-latency \
    2>&1 | tee logs/stretch_a_finish/stepB_fast_seed${s}.log
done

# Aggregate Square+AA across seeds 42+remaining for the paper table
python - <<'PY'
import json, glob, os, statistics
from pathlib import Path
files = sorted(glob.glob('experiments/wrn/evaluation/results_fast_wrn_seed*.json'))
agg = {}
for atk in ['Square', 'AutoAttack']:
    tprs = []; fprs = []
    for fp in files:
        d = json.loads(Path(fp).read_text())
        if atk in d and 'TPR' in d[atk]:
            tprs.append(d[atk]['TPR']); fprs.append(d[atk]['FPR'])
    if tprs:
        agg[atk] = {'TPR_mean': statistics.mean(tprs),
                    'TPR_std':  statistics.pstdev(tprs) if len(tprs) > 1 else 0.0,
                    'FPR_mean': statistics.mean(fprs),
                    'n_seeds':  len(tprs)}
out = 'experiments/wrn/evaluation/results_fast_wrn_aggregate.json'
Path(out).write_text(json.dumps({'aggregate': agg}, indent=2))
print(f'Wrote {out}: {agg}')
PY
echo "Step B: DONE"

# ── Step C: Ablation (now that experts.pkl exists) ───────────────────────────
echo ""
echo "=== Step C: Ablation (multi-seed) ==="
python experiments/ablation/run_ablation_paper.py \
  --n-test $N_TEST \
  --seeds 42 123 456 \
  --output experiments/wrn/ablation/results_ablation_wrn.json \
  2>&1 | tee logs/stretch_a_finish/stepC_ablation.log
echo "Step C: DONE"

# ── Step D: Adaptive PGD with reduced budget ─────────────────────────────────
echo ""
echo "=== Step D: Adaptive PGD (reduced budget) ==="
for s in $ADAPTIVE_SEEDS; do
  echo "  --- seed=$s ---"
  python experiments/evaluation/run_adaptive_pgd.py \
    --seed $s \
    --n $ADAPTIVE_N \
    --steps $ADAPTIVE_STEPS \
    --restarts $ADAPTIVE_RESTARTS \
    --eot-samples 1 \
    --eps 0.0314 \
    --lambdas $ADAPTIVE_LAMBDAS \
    --through-scorer \
    --eval-split test --eval-offset 8000 \
    --output experiments/wrn/evaluation/results_adaptive_pgd_wrn_seed${s}.json \
    2>&1 | tee logs/stretch_a_finish/stepD_adaptive_seed${s}.log
done
echo "Step D: DONE"

# ── Step E: Build paper tables ───────────────────────────────────────────────
echo ""
echo "=== Step E: Build paper tables ==="
python scripts/build_paper_tables.py \
  --arch-tag wrn \
  --results-dir experiments/wrn/evaluation \
  --ablation-dir experiments/wrn/ablation \
  --output-dir paper/tables/wrn \
  2>&1 | tee logs/stretch_a_finish/stepE_tables.log || \
  echo "WARN: build_paper_tables failed — fix flags and re-run locally."

echo ""
echo "============================================================"
echo "Stretch A FINISH complete."
echo "Artifacts:"
ls -la models/wrn/ experiments/wrn/evaluation/results_fast_wrn_aggregate.json \
       experiments/wrn/ablation/*.json experiments/wrn/evaluation/results_adaptive_pgd_wrn_*.json 2>/dev/null || true
echo "============================================================"
