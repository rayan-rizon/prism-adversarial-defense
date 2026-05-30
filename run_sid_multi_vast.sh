#!/bin/bash
# SID-only multi-setting run for the baselines_multi table: WRN-28-10 + CIFAR-100.
# (CIFAR-10/ResNet-18 SID already obtained from the recent-baselines run.)
# Unsupervised SID, attacks FGSM/PGD/Square, n=1000, 5 seeds, no cross-attack.
set -uo pipefail
cd ~/prism-adversarial-defense/prism
export SSL_CERT_FILE="$(python3 -c 'import certifi;print(certifi.where())')"

SEEDS="${SEEDS:-42 123 456 789 999}"
ATTACKS="FGSM PGD Square"
OUTDIR=experiments/evaluation/recent_baselines_multi
mkdir -p "$OUTDIR"

echo "START $(date -u +%FT%TZ)"
for cfg in wrn_cifar10 cifar100; do
  for s in $SEEDS; do
    echo "================ $cfg seed $s ================ $(date -u +%TZ)"
    python3 experiments/evaluation/run_baselines_recent.py \
      --config "configs/$cfg.yaml" --methods sid --attacks $ATTACKS \
      --n-test 1000 --no-cross --seed "$s" \
      --output "$OUTDIR/results_sid_${cfg}_seed${s}.json"
  done
done
echo "ALL DONE $(date -u +%FT%TZ)"
