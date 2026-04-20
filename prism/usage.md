# PRISM Local Evaluation Guide

This guide describes how to run local evaluations of the PRISM adversarial defense pipeline.

## 1. Prerequisites

Ensure you have installed the required dependencies:
```bash
pip install adversarial-robustness-toolbox
pip install autoattack
```

Make sure the pretrained PRISM models are available in the `models/` directory:
- `models/calibrator.pkl`
- `models/reference_profiles.pkl`
- `models/ensemble_scorer.pkl`

*(If missing, run `scripts/build_profile_testset.py`, then `scripts/calibrate_testset.py`, and `scripts/train_ensemble_scorer.py`)*

## 2. Running the Full Evaluation

The main evaluation script is `experiments/evaluation/run_evaluation_full.py`. It uses `n=1000` by default and targets standard paper metrics (e.g., FGSM, PGD, Square).

```bash
cd prism
python experiments/evaluation/run_evaluation_full.py --n-test 500 --attacks FGSM PGD Square --checkpoint-interval 100
```

### Options:
- `--n-test`: Number of samples to test per attack. Use 500 or 1000 for robust metrics.
- `--attacks`: Space-separated list of attacks. Available: `FGSM`, `PGD`, `CW`, `Square`, `AutoAttack`.
- `--checkpoint-interval`: Prints live metrics (TPR/FPR/F1) during the evaluation.

## 3. Reviewing Results

Results are saved to `experiments/evaluation/results_paper.json` (or the path specified via `--output`).
The JSON file will contain metrics including:
- `TPR` (True Positive Rate)
- `FPR` (False Positive Rate)
- `F1` (F1 Score)
- Confidence intervals and per-tier breakdowns.

### Target Metrics (Paper)
- **Mean TPR**: ≥ 0.88
- **FPR L1+**: ≤ 0.10
- **FPR L2+**: ≤ 0.03
- **FPR L3+**: ≤ 0.005
