#!/bin/bash
# PRISM Paper Evaluation — Thunder.computer Runner
# Runs: smoke test (n=5) → full eval (n=1000, 5 attacks)
# Usage: bash thunder_run.sh
set -euo pipefail

echo "=== PRISM Thunder Evaluation ==="
echo "Started: $(date)"
echo ""

# ── Step 1: Verify GPU ──────────────────────────────────────────────────────
echo ">>> Step 1: GPU Check"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# ── Step 2: Extract project ─────────────────────────────────────────────────
echo ">>> Step 2: Extracting project"
cd ~
if [ ! -d "prism" ]; then
    tar xzf prism_thunder.tar.gz
fi
cd prism
echo "Working dir: $(pwd)"
echo ""

# ── Step 3: Install dependencies ────────────────────────────────────────────
echo ">>> Step 3: Installing dependencies"
pip install -q --upgrade pip
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 2>/dev/null || \
    pip install -q torch torchvision torchaudio
pip install -q adversarial-robustness-toolbox ripser persim scikit-learn scipy numpy tqdm pyyaml certifi
pip install -q "git+https://github.com/fra31/auto-attack" 2>/dev/null || echo "WARN: AutoAttack install failed, will skip"
echo "Deps installed."
echo ""

# ── Step 4: Verify models exist ─────────────────────────────────────────────
echo ">>> Step 4: Verifying models"
for f in models/calibrator.pkl models/reference_profiles.pkl models/scorer.pkl; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found! Aborting."
        exit 1
    fi
    echo "  ✓ $f ($(stat --printf='%s' "$f" 2>/dev/null || stat -f '%z' "$f") bytes)"
done
echo ""

# ── Step 5: Download CIFAR-10 ───────────────────────────────────────────────
echo ">>> Step 5: CIFAR-10 download"
python3 -c "
import torchvision
torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
print('CIFAR-10 test set ready.')
"
echo ""

# ── Step 6: Smoke test (n=5) ────────────────────────────────────────────────
echo ">>> Step 6: SMOKE TEST (n=5, FGSM only)"
echo "This validates the entire pipeline before the big run."
python3 experiments/evaluation/run_evaluation_full.py \
    --n-test 5 \
    --attacks FGSM \
    --output experiments/evaluation/results_smoke_thunder.json \
    --seed 42

echo ""
echo "Smoke test results:"
cat experiments/evaluation/results_smoke_thunder.json
echo ""

# Validate smoke test passed (n_actual should be 5)
python3 -c "
import json
with open('experiments/evaluation/results_smoke_thunder.json') as f:
    r = json.load(f)
meta = r.get('_meta', {})
n = meta.get('n_actual', 0)
dev = meta.get('device', 'unknown')
if n < 5:
    print(f'ERROR: Smoke test only processed {n}/5 samples')
    exit(1)
if 'cuda' not in dev:
    print(f'WARNING: Running on {dev}, not GPU! This will be very slow.')
print(f'Smoke test PASSED: {n} samples on {dev}')
print('Proceeding to full evaluation...')
"

# ── Step 7: Full evaluation (n=1000, all 5 attacks) ─────────────────────────
echo ""
echo ">>> Step 7: FULL EVALUATION (n=1000, FGSM+PGD+CW+Square+AutoAttack)"
echo "Started: $(date)"
python3 experiments/evaluation/run_evaluation_full.py \
    --n-test 1000 \
    --attacks FGSM PGD CW Square AutoAttack \
    --output experiments/evaluation/results_thunder_n1000.json \
    --seed 42

echo ""
echo ">>> DONE!"
echo "Finished: $(date)"
echo ""
echo "=== FINAL RESULTS ==="
cat experiments/evaluation/results_thunder_n1000.json
