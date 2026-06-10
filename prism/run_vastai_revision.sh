#!/bin/bash
# =============================================================================
# PRISM — Vast.ai Revision Experiments (NeurIPS/ICLR rebuttal)
# =============================================================================
# Runs the TWO drop-in revision experiments that need no new pipeline:
#   Exp 1: Direct zeroth-order (NES/SPSA) attack on the DEPLOYED score S(x)
#          -> kills the "adaptive attack only hits a surrogate" objection.
#   Exp 3: CIFAR-10-C benign-shift FPR audit of the CADG certificate
#          -> turns the exchangeability limitation into a measured curve.
#
# PREREQUISITE: the locked artifacts from run_vastai_full.sh must already exist:
#   models/cifar_resnet18.pt, models/reference_profiles.pkl,
#   models/ensemble_scorer.pkl, models/calibrator.pkl
# This script does NOT retrain/recalibrate — it consumes the frozen artifacts,
# so Exp 1/3 are directly comparable to every table in the paper.
#
# Exp 2 (ImageNet-scale) is intentionally NOT here — it is a multi-stage
# pipeline port, not a drop-in. See REVISION_EXP2_IMAGENET_RUNBOOK.md.
#
# Usage: bash run_vastai_revision.sh
# Exit codes: 0=success, 1=missing artifact/setup, 2=experiment failure
# =============================================================================
set -euo pipefail

# ── Resolve PRISM root (same logic as run_vastai_full.sh) ────────────────────
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

# CIFAR-10 default config — Exp 1 and Exp 3 are CIFAR-10/ResNet-18 only.
PRISM_CONFIG="${PRISM_CONFIG:-configs/default.yaml}"
export PRISM_CONFIG

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
export OMP_NUM_THREADS=4

# Tunables (override via env)
SEED="${SEED:-42}"
DSA_N="${DSA_N:-200}"            # Exp1 images
DSA_STEPS="${DSA_STEPS:-60}"
DSA_QUERIES="${DSA_QUERIES:-20}" # NES antithetic pairs/step
DSA_MODE="${DSA_MODE:-nes}"      # nes | spsa
DSA_C="${DSA_C:-0.0 0.5 1.0 2.0 5.0}"
C10C_DIR="${C10C_DIR:-data/CIFAR-10-C}"
C10C_N="${C10C_N:-1000}"
C10C_CORRUPTIONS="${C10C_CORRUPTIONS:-gaussian_noise shot_noise defocus_blur motion_blur fog frost contrast jpeg_compression}"
C10C_SEVERITIES="${C10C_SEVERITIES:-1 3 5}"

echo "============================================================"
echo "PRISM Revision Experiments — $(date)"
echo "Repo root: $PRISM_ROOT  Config: $PRISM_CONFIG  Seed: $SEED"
echo "============================================================"

# ── Pre-flight: deps ──────────────────────────────────────────────────────────
if ! python -c "import torch" 2>/dev/null; then
  pip install --no-cache-dir -r requirements.txt || { echo "ERROR: pip install failed"; exit 1; }
fi

# ── Pre-flight: locked artifacts must exist ───────────────────────────────────
echo "=== Pre-flight: locked artifacts ==="
MISSING=0
for f in models/cifar_resnet18.pt models/reference_profiles.pkl \
         models/ensemble_scorer.pkl models/calibrator.pkl; do
  if [ ! -f "$f" ]; then echo "  MISSING: $f"; MISSING=1; else echo "  OK: $f"; fi
done
if [ "$MISSING" -ne 0 ]; then
  echo "ERROR: run run_vastai_full.sh first to produce the locked artifacts."
  exit 1
fi
mkdir -p logs experiments/evaluation experiments/stress

# ══════════════════════════════════════════════════════════════════════════════
# Exp 1 — Direct zeroth-order attack on deployed S(x)
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== Exp 1: Direct score attack [$DSA_MODE], n=$DSA_N, steps=$DSA_STEPS, queries=$DSA_QUERIES ==="
python experiments/evaluation/run_direct_score_attack.py \
  --n-test "$DSA_N" --steps "$DSA_STEPS" --nes-queries "$DSA_QUERIES" \
  --mode "$DSA_MODE" --c $DSA_C --seed "$SEED" \
  --output experiments/evaluation/results_direct_score_attack_seed${SEED}.json \
  2>&1 | tee logs/revision_exp1_direct_score_attack.log
EXP1_EXIT=${PIPESTATUS[0]:-$?}
[ "$EXP1_EXIT" -ne 0 ] && { echo "ERROR: Exp 1 failed (exit $EXP1_EXIT)"; exit 2; }
echo "Exp 1: DONE"

# ══════════════════════════════════════════════════════════════════════════════
# Exp 3 — CIFAR-10-C benign-shift FPR audit
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== Exp 3: CIFAR-10-C FPR audit ==="
if [ ! -f "$C10C_DIR/labels.npy" ]; then
  echo "  CIFAR-10-C not found at $C10C_DIR — downloading ..."
  mkdir -p data
  curl -L -o /tmp/CIFAR-10-C.tar https://zenodo.org/record/2535967/files/CIFAR-10-C.tar \
    && tar -xf /tmp/CIFAR-10-C.tar -C data/ \
    || { echo "ERROR: CIFAR-10-C download/extract failed. Fetch manually into $C10C_DIR."; exit 1; }
fi
python experiments/stress/run_cifar10c_fpr_audit.py \
  --data-dir "$C10C_DIR" \
  --corruptions $C10C_CORRUPTIONS \
  --severities $C10C_SEVERITIES \
  --n-per "$C10C_N" --seed "$SEED" \
  --output experiments/stress/results_cifar10c_fpr_audit_seed${SEED}.json \
  2>&1 | tee logs/revision_exp3_cifar10c_fpr_audit.log
EXP3_EXIT=${PIPESTATUS[0]:-$?}
[ "$EXP3_EXIT" -ne 0 ] && { echo "ERROR: Exp 3 failed (exit $EXP3_EXIT)"; exit 2; }
echo "Exp 3: DONE"

echo ""
echo "============================================================"
echo "Revision experiments COMPLETE."
echo "  Exp 1 -> experiments/evaluation/results_direct_score_attack_seed${SEED}.json"
echo "  Exp 3 -> experiments/stress/results_cifar10c_fpr_audit_seed${SEED}.json"
echo "============================================================"
