#!/bin/bash
# =============================================================================
# PRISM Full Evaluation — CamberCloud HPC (Student Plan)
#
# Plan:    Student plan — 5 GPU hrs, single NVIDIA L4, 50 GB Stash
# Runtime: ~2.5h total on L4 (GPU ~10x speedup over CPU)
# Budget:  ~2.5hr GPU / 5hr budget — leaves ~2.5hr buffer
# Cost:    Free (Student plan)
#
# Artifacts produced:
#   models/reference_profiles.pkl  — topological medoid reference
#   models/calibrator.pkl          — conformal thresholds (n_cal=5k, n_val=5k)
#   models/experts.pkl             — k=4 MoE expert sub-networks
#   experiments/evaluation/results_seed42.json
#   experiments/evaluation/results_seed7.json
#   experiments/evaluation/results_aggregated.json   (mean ± std across seeds)
#   experiments/evaluation/results_fgsm_sweep.json   (ε sensitivity curve)
#   experiments/ablation/results.json
#   experiments/campaign/results.json
#   experiments/generalization/cifar100_results.json
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
mkdir -p logs models data experiments/calibration experiments/evaluation \
         experiments/ablation experiments/campaign experiments/generalization
# Note: CIFAR-10 and CIFAR-100 are downloaded automatically by torchvision
# to ./data/ if not present (requires internet access on compute node)

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
#    n_images=10000 → well-calibrated medoid reference
# ---------------------------------------------------------------------------
echo ""
echo "--- Phase 1: Build profile (10K images, layer2/3/4, dim_weights=[0.5,0.5]) ---"
echo "Started: $(date)"
python scripts/build_profile.py   # default: 10000 images (see configs/default.yaml)
echo "Finished: $(date)"

# ---------------------------------------------------------------------------
# 2. Calibrate conformal thresholds  (~5 min)
#    Uses split conformal: n_cal=5000 + n_val=5000
# ---------------------------------------------------------------------------
echo ""
echo "--- Phase 2: Calibrate thresholds (split conformal, n_cal=5k, n_val=5k) ---"
echo "Started: $(date)"
python scripts/calibrate_thresholds.py
echo "Finished: $(date)"

# ---------------------------------------------------------------------------
# 3. Train MoE experts  (~10 min on L4)
#    k=4 topology clusters, n_samples=2000, 20 epochs
# ---------------------------------------------------------------------------
echo ""
echo "--- Phase 3: Train MoE experts (k=4, n=2000, epochs=20) ---"
echo "Started: $(date)"
python scripts/train_experts.py
echo "Finished: $(date)"

# ---------------------------------------------------------------------------
# 4. Sanity checks — abort if any fail
# ---------------------------------------------------------------------------
echo ""
echo "--- Sanity checks ---"
python sanity_checks.py
echo "Sanity checks PASSED: $(date)"

# ---------------------------------------------------------------------------
# 5. Full evaluation seed=42  (~45 min on L4)
#    FGSM ε=0.03, PGD-40, Square ε=0.05 × 1000 samples
# ---------------------------------------------------------------------------
echo ""
echo "--- Phase 4a: Full eval seed=42 (n=1000) ---"
echo "Started: $(date)"
python experiments/evaluation/run_evaluation.py \
    --n-test 1000 \
    --seed 42 \
    --attacks FGSM PGD Square \
    --output experiments/evaluation/results_seed42.json
echo "Finished: $(date)"

# ---------------------------------------------------------------------------
# 6. Full evaluation seed=7  (~45 min on L4)
# ---------------------------------------------------------------------------
echo ""
echo "--- Phase 4b: Full eval seed=7 (n=1000) ---"
echo "Started: $(date)"
python experiments/evaluation/run_evaluation.py \
    --n-test 1000 \
    --seed 7 \
    --attacks FGSM PGD Square \
    --output experiments/evaluation/results_seed7.json
echo "Finished: $(date)"

# ---------------------------------------------------------------------------
# 7. FGSM epsilon sweep  (~20 min on L4)
#    ε ∈ {0.01, 0.02, 0.03, 0.05, 0.10} — detection sensitivity curve
# ---------------------------------------------------------------------------
echo ""
echo "--- Phase 5: FGSM epsilon sweep (n=300 per ε) ---"
echo "Started: $(date)"
python experiments/evaluation/run_evaluation.py \
    --n-test 300 \
    --sweep
echo "Finished: $(date)"

# ---------------------------------------------------------------------------
# 8. Ablation study  (~10 min on L4)
#    FGSM ε=0.03, n=300 clean + 300 adversarial, 4 configs
# ---------------------------------------------------------------------------
echo ""
echo "--- Phase 6: Ablation study (eps=0.03, n=300) ---"
echo "Started: $(date)"
python experiments/ablation/run_ablation.py --n-clean 300 --n-adv 300 --eps 0.03
echo "Finished: $(date)"

# ---------------------------------------------------------------------------
# 9. Campaign detection experiment  (~5 min on L4)
#    50 clean + 100 adversarial FGSM ε=0.05, verify L0 latency
# ---------------------------------------------------------------------------
echo ""
echo "--- Phase 7: Campaign detection (50 clean + 100 adv, eps=0.05) ---"
echo "Started: $(date)"
python experiments/campaign/run_campaign.py
echo "Finished: $(date)"

# ---------------------------------------------------------------------------
# 10. CIFAR-100 generalization  (~15 min on L4)
# ---------------------------------------------------------------------------
echo ""
echo "--- Phase 8: CIFAR-100 generalization (n_clean=500, n_adv=500, eps=0.05) ---"
echo "Started: $(date)"
python experiments/generalization/run_cifar100.py --n-clean 500 --n-adv 500
echo "Finished: $(date)"

# ---------------------------------------------------------------------------
# 11. Aggregate multi-seed results
# ---------------------------------------------------------------------------
echo ""
echo "--- Phase 9: Aggregate results (mean ± std across seeds 42 and 7) ---"
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
print("\nSaved → experiments/evaluation/results_aggregated.json")
PYEOF

# ---------------------------------------------------------------------------
# Done — push all results to Stash for download
# ---------------------------------------------------------------------------
echo ""
echo "======================================================"
echo "All phases complete: $(date)"
echo ""
echo "=== Results Summary ==="
for f in \
    experiments/evaluation/results_seed42.json \
    experiments/evaluation/results_aggregated.json \
    experiments/evaluation/results_fgsm_sweep.json \
    experiments/ablation/results.json \
    experiments/campaign/results.json \
    experiments/generalization/cifar100_results.json; do
    if [ -f "$f" ]; then
        echo "  ✓ $f"
    else
        echo "  ✗ MISSING: $f"
    fi
done
echo ""
echo "Download results with:"
echo "  camber stash push ~/prism/experiments/ camber://prism-results/"
echo "  camber stash push ~/prism/models/ camber://prism-models/"
echo "======================================================"
