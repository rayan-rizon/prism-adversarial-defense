#!/usr/bin/env bash
# =============================================================================
# camber_upload.sh — Upload PRISM project to CamberCloud Stash
#
# Prerequisites:
#   ~/.camber/bin/camber or camber on PATH
#   export CAMBER_API_KEY="<your-key>"
#
# Usage:
#   bash scripts/camber_upload.sh
# =============================================================================

set -e
PRISM_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "PRISM root: $PRISM_ROOT"

# 1. Login (no-op if already logged in)
echo ""
echo "=== Step 1: Authenticate with CamberCloud ==="
camber --api-key "${CAMBER_API_KEY}" me

# 2. Upload project code — includes models/ (calibrator, profiles, scorer)
#    Excludes: raw data dirs, large model variants, caches, tmp artifacts
echo ""
echo "=== Step 2: Build and upload eval package (includes models/) ==="
TMPDIR=$(mktemp -d)
ARCHIVE="$TMPDIR/prism_eval_v3.tar.gz"

# Build tar: local dir is prism/ (not PRISM/) so remote unpacks directly as prism/
tar czf "$ARCHIVE" \
    --exclude='prism/data' \
    --exclude='prism/models_10k' \
    --exclude='prism/.git' \
    --exclude='prism/.venv' \
    --exclude='prism/__pycache__' \
    --exclude='prism/tmp_logs' \
    --exclude='prism/experiments/calibration/*.npy' \
    --exclude='prism/paper/figures/*.pdf' \
    --exclude='prism/paper/figures/*.png' \
    --exclude='*/.DS_Store' \
    --exclude='*/__pycache__' \
    -C "$PRISM_ROOT/.." prism/
echo "Archive size: $(du -sh "$ARCHIVE" | cut -f1)"
camber --api-key "${CAMBER_API_KEY}" stash push "$ARCHIVE" stash://hexron/prism_eval_v3.tar.gz
rm -rf "$TMPDIR"
echo "Upload complete: stash://hexron/prism_eval_v3.tar.gz"

echo ""
echo "================================================"
echo "Upload complete."
echo ""
echo "Next: submit job with:"
echo "  bash scripts/camber_build_run.sh"
echo "================================================"
