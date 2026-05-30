#!/bin/bash
# Waits for the SID multi-setting run to finish, then pulls the LayerMFS code
# and runs InputMFS + LayerMFS on CIFAR-10/ResNet-18 (FGSM/PGD/Square/CW, 5 seeds)
# for the appendix InputMFS-vs-LayerMFS comparison.
set -uo pipefail
LOG=/tmp/layermfs.log
echo "WAITING for SID run to finish $(date -u +%FT%TZ)" > "$LOG"
while pgrep -f run_sid_multi_vast >/dev/null; do sleep 60; done
sleep 5

cd ~/prism-adversarial-defense
git pull --no-edit >> "$LOG" 2>&1
cd prism
export SSL_CERT_FILE="$(python3 -c 'import certifi;print(certifi.where())')"
OUT=experiments/evaluation/recent_baselines_layermfs
mkdir -p "$OUT"

echo "LAYERMFS START $(date -u +%FT%TZ)" >> "$LOG"
for s in 42 123 456 789 999; do
  echo "================ layermfs seed $s ================ $(date -u +%TZ)" >> "$LOG"
  python3 experiments/evaluation/run_baselines_recent.py \
    --methods spectral layermfs --attacks FGSM PGD Square CW \
    --n-test 1000 --n-ref-spectral 512 --seed "$s" \
    --output "$OUT/results_inputvslayer_seed${s}.json" >> "$LOG" 2>&1
done
echo "LAYERMFS ALL DONE $(date -u +%FT%TZ)" >> "$LOG"
