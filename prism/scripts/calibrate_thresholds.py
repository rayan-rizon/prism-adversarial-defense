"""
Conformal Threshold Calibration (Phase 2, Week 6-9)

Loads pre-computed clean anomaly scores, splits into calibration/validation,
calibrates conformal thresholds, and verifies coverage guarantees.
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

    # --- Verify coverage on held-out validation set ---
    print("\n=== Coverage Verification ===")
    report = calibrator.get_coverage_report(val_scores)
    all_passed = True
    for level, info in report.items():
        status = "PASSED ✓" if info['passed'] else "FAILED ✗"
        print(f"  {level}: FPR={info['empirical_fpr']:.4f} "
              f"(target ≤ {info['target_alpha']:.3f}) — {status}")
        if not info['passed']:
            all_passed = False

    if all_passed:
        print("\n✓ All coverage guarantees verified!")
    else:
        print("\n⚠ Some coverage guarantees failed. Consider more calibration data.")

    # --- Save ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(calibrator, f)
    print(f"\nCalibrator saved to {output_path}")

    return calibrator


if __name__ == '__main__':
    calibrate_thresholds()
