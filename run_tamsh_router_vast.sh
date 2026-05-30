#!/bin/bash
# 5-seed learned TAMSH router on Vast (CIFAR-10/ResNet-18). Trains on CAL split,
# evals on disjoint EVAL split; compares uniform/topology/force-best/learned/oracle.
set -uo pipefail
cd ~/prism-adversarial-defense/prism
export SSL_CERT_FILE="$(python3 -c 'import certifi;print(certifi.where())')"
OUT=experiments/evaluation/tamsh_router
mkdir -p "$OUT"
echo "START $(date -u +%FT%TZ)"
for s in 42 123 456 789 999; do
  echo "================ seed $s ================ $(date -u +%TZ)"
  python3 scripts/train_tamsh_router.py --n-train 1500 --n-eval 1000 --seed "$s" \
    --report "$OUT/tamsh_router_report_seed${s}.json" --cache "/tmp/tamsh_cache_seed${s}.npz"
done
echo "ALL DONE $(date -u +%FT%TZ)"
