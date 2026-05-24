#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/prism"
export PRISM_CONFIG=configs/wrn_cifar10.yaml
mkdir -p logs

echo "[1/3] Adaptive PGD full multi-seed run"
python experiments/evaluation/run_adaptive_pgd.py \
  --n-test 1000 \
  --multi-seed \
  --seeds 42 123 456 789 999 \
  2>&1 | tee logs/adaptive_pgd_new_gpu.log

echo "[2/3] Train experts required for ablation Full PRISM / Ensemble-no-TDA"
python scripts/train_experts.py --device cuda --output models/wrn/experts.pkl \
  2>&1 | tee logs/train_experts_new_gpu.log

echo "[3/3] Rerun ablation"
python experiments/ablation/run_ablation_paper.py \
  2>&1 | tee logs/ablation_new_gpu.log
