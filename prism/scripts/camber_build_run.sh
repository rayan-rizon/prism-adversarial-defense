#!/usr/bin/env bash
# =============================================================================
# camber_build_run.sh — Build, upload, and submit PRISM evaluation on CamberCloud
#
# Fixes vs previous runs:
#   - models/ directory (calibrator.pkl, reference_profiles.pkl, scorer.pkl,
#     experts.pkl) is NOW included in the archive — required for valid FPR
#   - Uses run_evaluation_full.py (proper eps=8/255, CIs, n=1000)
#   - Adds CW + AutoAttack attacks
#   - Targets XSMALL GPU node (L4) — only size that reliably runs full eval
#
# Prerequisites:
#   export CAMBER_API_KEY="d3012379fbb81a2b809c5ccdbfd03ee2f2e32c35"
#
# Usage:
#   cd /Users/rayanrizon/Desktop/Research/Prism
#   bash prism/scripts/camber_build_run.sh
# =============================================================================
set -e

CAMBER="${HOME}/.camber/bin/camber"
API_KEY="${CAMBER_API_KEY:?'ERROR: export CAMBER_API_KEY=<your-key>'}"
PRISM_ROOT="$(cd "$(dirname "$0")/.." && pwd)"    # …/Prism/prism
WORKSPACE="$(cd "$PRISM_ROOT/.." && pwd)"          # …/Prism

ARCHIVE_NAME="prism_eval_v3.tar.gz"
STASH_PATH="stash://hexron/${ARCHIVE_NAME}"

echo "======================================================================"
echo "PRISM CamberCloud Build + Run"
echo "  PRISM root : $PRISM_ROOT"
echo "  Archive    : $ARCHIVE_NAME"
echo "  Stash      : $STASH_PATH"
echo "======================================================================"

# ── Step 1: Verify authentication ─────────────────────────────────────────────
echo ""
echo "=== Step 1: Verify authentication ==="
"$CAMBER" --api-key "$API_KEY" me

# ── Step 2: Build tar (include models/, exclude data/ and large dirs) ──────────
echo ""
echo "=== Step 2: Build archive (with models/) ==="
TMPDIR_BUILD=$(mktemp -d)
ARCHIVE_PATH="${TMPDIR_BUILD}/${ARCHIVE_NAME}"

tar czf "$ARCHIVE_PATH" \
    --exclude='prism/data' \
    --exclude='prism/models_10k' \
    --exclude='prism/models/experts.pkl' \
    --exclude='prism/.git' \
    --exclude='prism/.venv' \
    --exclude='prism/tmp_logs' \
    --exclude='prism/experiments/calibration/*.npy' \
    --exclude='prism/paper/figures/*.pdf' \
    --exclude='prism/paper/figures/*.png' \
    --exclude='*/.DS_Store' \
    --exclude='*/__pycache__' \
    -C "$WORKSPACE" prism/

ARCHIVE_SIZE=$(du -sh "$ARCHIVE_PATH" | cut -f1)
echo "Archive built: ${ARCHIVE_PATH}  (${ARCHIVE_SIZE})"

# Verify models are present inside the archive
echo "Verifying models/ in archive:"
tar tzf "$ARCHIVE_PATH" | grep 'prism/models/' | head -10

# ── Step 3: Upload to stash ────────────────────────────────────────────────────
echo ""
echo "=== Step 3: Upload to stash (${STASH_PATH}) ==="
"$CAMBER" --api-key "$API_KEY" stash push "$ARCHIVE_PATH" "$STASH_PATH"
echo "Upload complete."
rm -rf "$TMPDIR_BUILD"

# ── Step 4: Submit evaluation job ─────────────────────────────────────────────
echo ""
echo "=== Step 4: Submit evaluation job ==="

# The remote job command:
#   1. Unpack prism/ from the stash archive
#   2. pip install all requirements (including autoattack from git)
#   3. Run run_evaluation_full.py with n=1000, all 5 attacks
#   4. Print a summary table
JOB_CMD='bash -c "
set -e
echo \"=== Unpacking PRISM (with models/) ===\"
tar xzf prism_eval_v3.tar.gz
cd prism

echo \"=== Python / CUDA info ===\"
python3 -c \"import torch; print(f'"'"'torch={torch.__version__} cuda={torch.cuda.is_available()}'"'"')\"

echo \"=== Installing requirements ===\"
python3 -m pip install -q --upgrade pip
python3 -m pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -3
python3 -m pip install -q \
    adversarial-robustness-toolbox>=1.16.0 \
    git+https://github.com/fra31/auto-attack \
    ripser>=0.6.4 persim>=0.3.1 scikit-learn>=1.3.0 \
    scipy>=1.11.0 numpy>=1.24.0 tqdm>=4.65.0 \
    pyyaml>=6.0 certifi 2>&1 | tail -5

echo \"=== Verifying calibration models ===\"
python3 -c \"
import pickle, os
for f in ['"'"'models/calibrator.pkl'"'"', '"'"'models/reference_profiles.pkl'"'"']:
    size = os.path.getsize(f) if os.path.exists(f) else -1
    print(f'"'"'{f}: {size} bytes'"'"')
\"

echo \"=== Running full evaluation (n=1000, eps=8/255, 5 attacks) ===\"
python3 experiments/evaluation/run_evaluation_full.py \
    --n-test 1000 \
    --seed 42 \
    --attacks FGSM PGD Square CW AutoAttack \
    --output experiments/evaluation/results_camber_v3.json

echo \"=== Final Results ===\"
cat experiments/evaluation/results_camber_v3.json
"'

"$CAMBER" --api-key "$API_KEY" job create \
    --engine base \
    --size xsmall \
    --gpu \
    --path "$STASH_PATH" \
    --cmd "$JOB_CMD"

echo ""
echo "======================================================================"
echo "Job submitted! Use: camber job list  to check status"
echo "              Use: camber job logs <JOB_ID>  to view progress"
echo "======================================================================"
