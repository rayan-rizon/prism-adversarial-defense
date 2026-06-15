#!/bin/bash
# Cert-restoration, parallel by corruption (8 procs) on the box. Each process =
# one corruption x {sev 1,3,5}, deterministic (seed 42) -> parallelism cannot
# change results. Pool with aggregate_cert_restore.py.
set -u
cd /root/prism-adversarial-defense/prism
export PYTHONPATH=/root/prism-adversarial-defense/prism
export PRISM_CONFIG=configs/default.yaml
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2
OUT=experiments/stress/cert_restore
mkdir -p "$OUT"
echo "START $(date)"
for C in gaussian_noise shot_noise defocus_blur motion_blur fog frost contrast jpeg_compression; do
  python3 experiments/stress/run_cifar10c_restore.py \
    --corruptions "$C" --severities 1 3 5 --n-per 1000 --n-recal 300 --seed 42 \
    --output "$OUT/restore_${C}.json" > "$OUT/log_${C}.log" 2>&1 &
done
wait
echo "ALL_CERT_RESTORE_DONE $(date)"
