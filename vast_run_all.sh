#!/usr/bin/env bash
# Run the 3 paper-grade tests on Vast.ai.
# Output: /root/prism-adversarial-defense/prism/experiments/stress/vastai_*.json
set -e
cd /root/prism-adversarial-defense/prism

mkdir -p experiments/stress
mkdir -p data

# Pre-download CIFAR-10 once.
echo "=== ensure CIFAR-10 downloaded ==="
python3 - <<'PY'
import sys, os
sys.path.insert(0, '.')
from src import bootstrap  # noqa
from src.data_loader import load_test_dataset
ds = load_test_dataset(root='./data')
print('CIFAR-10 test split loaded:', len(ds), 'samples')
PY

echo
echo "=== TEST 1: latency breakdown (RTX 4090, n=200) ==="
python3 experiments/stress/vastai_latency_breakdown.py 2>&1 | tail -30

echo
echo "=== TEST 2: stronger CW (n=1000 x 5 seeds x kappa {0,10,20}) ==="
python3 experiments/stress/vastai_stronger_cw.py 2>&1 | tail -40

echo
echo "=== TEST 3: multi-attack recovery (n=1000 x 5 seeds x FGSM/Square/CW) ==="
python3 experiments/stress/vastai_recovery_multi.py 2>&1 | tail -40

echo
echo "=== ALL DONE ==="
ls -la experiments/stress/vastai_*.json
