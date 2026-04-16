#!/usr/bin/env bash
# =============================================================================
# camber_upload.sh — Upload PRISM project to CamberCloud Stash
#
# Prerequisites:
#   pip install camber
#   camber login              (authenticates via browser)
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
camber login

# 2. Upload project code (exclude data, models, __pycache__, .git)
echo ""
echo "=== Step 2: Upload project code ==="
# Create a clean tarball without large binary dirs
# Note: local dir is PRISM/, remote expects prism/ (per SLURM: cd $HOME/prism)
TMPDIR=$(mktemp -d)
ARCHIVE="$TMPDIR/prism-code.tar.gz"
tar czf "$ARCHIVE" \
    --transform 's|^PRISM|prism|' \
    --exclude='PRISM/data' \
    --exclude='PRISM/models' \
    --exclude='PRISM/models_10k' \
    --exclude='PRISM/.git' \
    --exclude='PRISM/__pycache__' \
    --exclude='PRISM/experiments/calibration/*.npy' \
    -C "$PRISM_ROOT/.." PRISM/
echo "Archive size: $(du -sh "$ARCHIVE" | cut -f1)"
camber stash push "$ARCHIVE" camber://prism-code/prism-code.tar.gz
rm -rf "$TMPDIR"

# 3. Upload CIFAR-10 data
echo ""
echo "=== Step 3: Upload CIFAR-10 data ==="
if [[ -d "$PRISM_ROOT/data/cifar-10-batches-py" ]]; then
    camber stash push "$PRISM_ROOT/data/cifar-10-batches-py" camber://prism-data/cifar-10-batches-py/
    echo "CIFAR-10 uploaded."
elif [[ -d "$PRISM_ROOT/../data/cifar-10-batches-py" ]]; then
    camber stash push "$PRISM_ROOT/../data/cifar-10-batches-py" camber://prism-data/cifar-10-batches-py/
    echo "CIFAR-10 (parent dir) uploaded."
else
    echo "WARNING: CIFAR-10 data not found — will download on CamberCloud via torchvision."
fi

# 4. Upload CIFAR-100 data
echo ""
echo "=== Step 4: Upload CIFAR-100 data ==="
if [[ -d "$PRISM_ROOT/data/cifar-100-python" ]]; then
    camber stash push "$PRISM_ROOT/data/cifar-100-python" camber://prism-data/cifar-100-python/
    echo "CIFAR-100 uploaded."
elif [[ -d "$PRISM_ROOT/../data/cifar-100-python" ]]; then
    camber stash push "$PRISM_ROOT/../data/cifar-100-python" camber://prism-data/cifar-100-python/
    echo "CIFAR-100 (parent dir) uploaded."
else
    echo "WARNING: CIFAR-100 data not found — will download on CamberCloud via torchvision."
fi

# 5. Done
echo ""
echo "================================================"
echo "Upload complete."
echo ""
echo "Next: SSH to CamberCloud HPC and run:"
echo "  camber stash pull camber://prism-code/prism-code.tar.gz ~/prism-cloud.tar.gz"
echo "  tar xzf ~/prism-cloud.tar.gz -C ~/"
echo "  cd ~/prism"
echo "  mkdir -p data"
echo "  camber stash pull camber://prism-data/ data/"
echo "  mkdir -p logs"
echo "  sbatch scripts/slurm/run_full_eval.slurm"
echo "================================================"
