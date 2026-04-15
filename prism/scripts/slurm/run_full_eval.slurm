#!/bin/bash
# =============================================================================
# PRISM Full Evaluation — CamberCloud HPC (Student Plan)
#
# Plan:    Student plan — 5 GPU hrs, single NVIDIA L4, 50 GB Stash
# Runtime: ~4h 15min total (build 40min + eval 195min + cifar100 20min)
# Budget:  ~4.3hr GPU / 5hr budget — leaves ~45min buffer
# Cost:    Free (Student plan)
#
# Prerequisites on the HPC node:
#   - CIFAR-10 data already in ./data/ (upload separately via Stash)
#   - Python 3.10+, pip available
#
# Submit:
#   sbatch scripts/slurm/run_full_eval.slurm
#
# Monitor:
#   squeue -u $USER
#   scontrol show job <JOBID>
#   tail -f logs/prism-<JOBID>.out
# =============================================================================

#SBATCH --job-name=prism-full-eval
#SBATCH --partition=gpu-queue
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --output=logs/prism-%j.out
#SBATCH --error=logs/prism-%j.err
#SBATCH --mail-type=END,FAIL

set -e   # abort on any error

# ---------------------------------------------------------------------------
# 0. Setup
# ---------------------------------------------------------------------------
cd "$HOME/prism"
mkdir -p logs models experiments/calibration

echo "======================================================"
echo "PRISM Full Evaluation — CamberCloud HPC"
echo "Started: $(date)"
echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "======================================================"

# Install dependencies (first run only; cached in subsequent jobs)
pip install -r requirements.txt --quiet

# ---------------------------------------------------------------------------
# 1. Build topological self-profile  (~35 min on L4)
# ---------------------------------------------------------------------------
echo ""
echo "--- Phase 1: Build profile (10K images, 3 layers) ---"
echo "Started: $(date)"
python scripts/build_profile.py
echo "Finished: $(date)"

# ---------------------------------------------------------------------------
# 2. Calibrate conformal thresholds  (~5 min)
# ---------------------------------------------------------------------------
echo ""
echo "--- Phase 2: Calibrate thresholds (n=5000) ---"
echo "Started: $(date)"
python scripts/calibrate_thresholds.py
echo "Finished: $(date)"

# ---------------------------------------------------------------------------
# 3. Train MoE experts  (~10 min on L4)
# ---------------------------------------------------------------------------
echo ""
echo "--- Phase 3: Train experts ---"
echo "Started: $(date)"
python scripts/train_experts.py
echo "Finished: $(date)"

# ---------------------------------------------------------------------------
# 4. Sanity checks
# ---------------------------------------------------------------------------
echo ""
echo "--- Sanity checks ---"
python sanity_checks.py
echo "Sanity checks passed: $(date)"

# ---------------------------------------------------------------------------
# 5. Full evaluation — seed 42  (~90 min: FGSM + PGD + Square × 1000 imgs)
# ---------------------------------------------------------------------------
echo ""
echo "--- Phase 4a: Full eval seed=42 (n=1000) ---"
echo "Started: $(date)"
python experiments/evaluation/run_evaluation.py \
    --n-test 1000 \
    --seed 42 \
    --output experiments/evaluation/results_seed42.json
echo "Finished: $(date)"

# ---------------------------------------------------------------------------
# 6. Full evaluation — seed 7  (~90 min)
# ---------------------------------------------------------------------------
echo ""
echo "--- Phase 4b: Full eval seed=7 (n=1000) ---"
echo "Started: $(date)"
python experiments/evaluation/run_evaluation.py \
    --n-test 1000 \
    --seed 7 \
    --output experiments/evaluation/results_seed7.json
echo "Finished: $(date)"

# ---------------------------------------------------------------------------
# 7. CIFAR-100 generalization  (~20 min on L4)
# ---------------------------------------------------------------------------
echo ""
echo "--- Phase 5: CIFAR-100 generalization (n=500) ---"
echo "Started: $(date)"
python experiments/generalization/run_cifar100.py --n-clean 500 --n-adv 500
echo "Finished: $(date)"

# ---------------------------------------------------------------------------
# 8. Aggregate results — compute mean ± std across seeds
# ---------------------------------------------------------------------------
echo ""
echo "--- Phase 6: Aggregate multi-seed results ---"
python - <<'PYEOF'
import json, numpy as np

seeds = [42, 7]
files = [f'experiments/evaluation/results_seed{s}.json' for s in seeds]
all_results = [json.load(open(f)) for f in files]

attacks = list(all_results[0].keys())
print("\n=== Multi-seed Results (mean ± std) ===")
print(f"{'Attack':>10} {'TPR mean':>10} {'TPR std':>9} {'FPR mean':>10} {'FPR std':>9}")
print("-" * 55)

aggregated = {}
for attack in attacks:
    tprs = [r[attack]['TPR'] for r in all_results]
    fprs = [r[attack]['FPR'] for r in all_results]
    aggregated[attack] = {
        'TPR_mean': round(float(np.mean(tprs)), 4),
        'TPR_std':  round(float(np.std(tprs)),  4),
        'FPR_mean': round(float(np.mean(fprs)), 4),
        'FPR_std':  round(float(np.std(fprs)),  4),
        'seeds': seeds,
    }
    print(f"{attack:>10} {np.mean(tprs):>10.4f} {np.std(tprs):>9.4f} "
          f"{np.mean(fprs):>10.4f} {np.std(fprs):>9.4f}")

with open('experiments/evaluation/results_aggregated.json', 'w') as f:
    json.dump(aggregated, f, indent=2)
print("\nSaved to experiments/evaluation/results_aggregated.json")
PYEOF

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "======================================================"
echo "All jobs complete: $(date)"
echo "Results:"
echo "  experiments/evaluation/results_seed42.json"
echo "  experiments/evaluation/results_seed7.json"
echo "  experiments/evaluation/results_aggregated.json"
echo "  experiments/generalization/cifar100_results.json"
echo ""
echo "Download with:"
echo "  camber stash push ~/prism/experiments/ camber://prism-results/"
echo "======================================================"
