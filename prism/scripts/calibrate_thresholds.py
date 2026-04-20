"""
Conformal Threshold Calibration (Phase 2, Week 6-9)

Loads pre-computed clean anomaly scores, splits into calibration/validation,
calibrates conformal thresholds, and verifies coverage guarantees.

DEPRECATION NOTICE
------------------
This script was designed for the legacy pipeline that builds profiles from the
CIFAR-10 TRAINING set (build_profile.py).  The current recommended pipeline is:

  1. python scripts/build_profile_testset.py   -- builds from TEST set
  2. python scripts/calibrate_ensemble.py      -- calibrates with ensemble scorer

This script is retained for ablation/baseline comparisons only.  Using it to
produce the primary results_paper.json metrics is incorrect because the clean
scores it reads (experiments/calibration/clean_scores.npy) were generated from
the training distribution, which causes a distribution shift relative to the
test-set evaluation — and violates the exchangeability assumption of split
conformal prediction.

For publication-quality calibration, use calibrate_ensemble.py.
"""
import numpy as np
import pickle
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.cadg.calibrate import ConformalCalibrator


def calibrate_thresholds(
    scores_path: str = 'experiments/calibration/clean_scores.npy',
    cal_size: int = 5000,
    output_path: str = 'models/calibrator.pkl',
):
    # --- Load scores ---
    if not os.path.exists(scores_path):
        print(f"ERROR: Clean scores not found at {scores_path}")
        print("Run scripts/build_profile.py first to generate clean anomaly scores.")
        sys.exit(1)

    all_scores = np.load(scores_path)
    print(f"Loaded {len(all_scores)} clean anomaly scores")
    print(f"  mean={all_scores.mean():.4f}, std={all_scores.std():.4f}")

    # --- Split: calibration + validation ---
    if len(all_scores) < cal_size * 2:
        cal_size = len(all_scores) // 2
        print(f"  Adjusted cal_size to {cal_size} (limited data)")

    cal_scores = all_scores[:cal_size]
    val_scores = all_scores[cal_size:cal_size * 2]
    print(f"  Calibration: {len(cal_scores)} samples")
    print(f"  Validation:  {len(val_scores)} samples")

    # --- Calibrate ---
    calibrator = ConformalCalibrator()
    thresholds = calibrator.calibrate(cal_scores)

    print("\n=== Conformal Thresholds ===")
    for level, threshold in thresholds.items():
        alpha = calibrator.alphas[level]
        print(f"  {level} (α={alpha}): threshold = {threshold:.6f}")

    # --- Verify coverage on held-out validation set (strict, tolerance=0.0) ---
    # We use tolerance=0.0 here because calibrate_thresholds.py is the final step
    # before publishing metrics.  Finite-sample noise at the recommended n≥5000
    # is small enough that strict verification is appropriate.
    print("\n=== Coverage Verification (strict, tol=0.0) ===")
    report = calibrator.get_coverage_report(val_scores, tolerance=0.0)
    all_passed = True
    for level, info in report.items():
        status = "PASSED ✓" if info['passed'] else "FAILED ✗"
        print(f"  {level}: FPR={info['empirical_fpr']:.4f} "
              f"(cal_α={info['calibration_alpha']:.4f}, "
              f"target≤{info['target_alpha']:.4f}) — {status}")
        if not info['passed']:
            all_passed = False

    if all_passed:
        print("\n✓ All coverage guarantees verified!")
    else:
        print("\n⚠ Some coverage guarantees failed. Consider more calibration data.")

    # Print full summary for paper reporting
    print()
    print(calibrator.summary(validation_scores=val_scores, tolerance=0.0))

    # --- Save ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(calibrator, f)
    print(f"\nCalibrator saved to {output_path}")

    return calibrator


if __name__ == '__main__':
    import warnings
    warnings.warn(
        "\ncalibrate_thresholds.py is DEPRECATED for publication use.\n"
        "Use scripts/calibrate_ensemble.py for paper-quality calibration.\n"
        "See the module docstring for details.",
        DeprecationWarning, stacklevel=1
    )
    calibrate_thresholds()
