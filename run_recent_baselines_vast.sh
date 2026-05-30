#!/bin/bash
# Multi-seed SID + SpectralDefense baseline run on Vast.ai.
# CIFAR-10 / ResNet-18, matched-FPR protocol, full cross-attack matrix.
set -euo pipefail
cd ~/prism-adversarial-defense/prism
export SSL_CERT_FILE="$(python3 -c 'import certifi;print(certifi.where())')"

SEEDS="${SEEDS:-42 123 456 789 999}"
ATTACKS="${ATTACKS:-FGSM PGD Square CW}"
NTEST="${NTEST:-1000}"
NREF="${NREF:-1000}"
OUTDIR="experiments/evaluation/recent_baselines"
mkdir -p "$OUTDIR"

echo "SEEDS=[$SEEDS] ATTACKS=[$ATTACKS] n_test=$NTEST n_ref=$NREF"
echo "START $(date -u +%FT%TZ)"
for s in $SEEDS; do
  echo "================ seed $s ================ $(date -u +%TZ)"
  python3 experiments/evaluation/run_baselines_recent.py \
    --n-test "$NTEST" --n-ref-spectral "$NREF" \
    --attacks $ATTACKS --methods sid spectral --seed "$s" \
    --output "$OUTDIR/results_baselines_recent_seed${s}.json"
done
echo "ALL DONE $(date -u +%FT%TZ)"
