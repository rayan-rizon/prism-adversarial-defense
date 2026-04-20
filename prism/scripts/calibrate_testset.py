"""
Calibrate Conformal Thresholds from CIFAR-10 Test Calibration Split

Loads pre-computed clean anomaly scores from the test calibration split
(images 5000–6999), fits conformal thresholds, and verifies the guarantee
on the validation split (images 7000–7999).

This fixes the conformal guarantee violation found in the audit:
  - L2 empirical FPR was 3.3% vs target 3.0%
  - L3 empirical FPR was 1.3% vs target 0.5%

Root cause: original calibration used training-set scores while evaluation
was on test images.  This script uses test-set scores throughout, restoring
the exchangeability assumption.

USAGE
-----
  cd prism/
  python scripts/calibrate_testset.py

Run AFTER: python scripts/build_profile_testset.py
"""
import numpy as np
import pickle
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.cadg.calibrate import ConformalCalibrator


def calibrate_testset(
    cal_scores_path: str  = 'experiments/calibration/clean_scores.npy',
    val_scores_path: str  = 'experiments/calibration/val_scores.npy',
    output_path: str      = 'models/calibrator.pkl',
):
    """
    Fit and verify conformal thresholds using test-set splits.

    Args:
        cal_scores_path: Anomaly scores for calibration split (test idx 5000-6999).
        val_scores_path: Anomaly scores for validation split  (test idx 7000-7999).
        output_path: Where to save the fitted ConformalCalibrator.
    """
    # ── Load scores ────────────────────────────────────────────────────────────
    for path in [cal_scores_path, val_scores_path]:
        if not os.path.exists(path):
            print(f"ERROR: {path} not found.")
            print("Run scripts/build_profile_testset.py first.")
            sys.exit(1)

    cal_scores = np.load(cal_scores_path)
    val_scores = np.load(val_scores_path)

    print(f"Calibration scores : n={len(cal_scores)}, "
          f"mean={cal_scores.mean():.4f}, std={cal_scores.std():.4f}")
    print(f"Validation  scores : n={len(val_scores)}, "
          f"mean={val_scores.mean():.4f}, std={val_scores.std():.4f}")

    # ── Fit conformal thresholds ───────────────────────────────────────────────
    calibrator = ConformalCalibrator()
    thresholds = calibrator.calibrate(cal_scores)

    print("\n=== Conformal Thresholds (fitted on test-cal split) ===")
    for level, threshold in thresholds.items():
        alpha = calibrator.alphas[level]
        print(f"  {level:2s} (alpha={alpha:.3f}): threshold = {threshold:.6f}")

    # ── Verify guarantee on held-out validation split ─────────────────────────
    print("\n=== FPR Verification on Validation Split ===")
    report = calibrator.get_coverage_report(val_scores)
    all_passed = True
    for level, info in sorted(report.items()):
        target_alpha = info['target_alpha']
        emp_fpr      = info['empirical_fpr']
        passed       = info['passed']
        status       = "[PASS]" if passed else "[FAIL]"
        delta        = emp_fpr - target_alpha
        print(f"  {level:2s}: empirical FPR = {emp_fpr:.4f}  "
              f"(target <= {target_alpha:.4f}, d={delta:+.4f})  {status}")
        if not passed:
            all_passed = False

    # Per-tier breakdown from val_scores
    print("\n=== Per-Tier FPR Breakdown ===")
    for level in ['L1', 'L2', 'L3']:
        thr = thresholds[level]
        fp  = int(np.sum(val_scores > thr))
        fpr = fp / len(val_scores)
        print(f"  FP at {level}+: {fp}/{len(val_scores)} = {fpr:.4f} "
              f"(target < {calibrator.alphas[level]:.4f})")

    if all_passed:
        print("\n[OK] All conformal coverage guarantees verified!")
    else:
        print("\n[WARN] Some guarantees failed. Increase calibration set size or "
              "check that distribution shift between cal/val is minimal.")

    # ── Save calibrator ────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(calibrator, f)
    print(f"\nCalibrator saved -> {output_path}")

    # Also save as calibrator_base.pkl so TDA-only ablation can load the base
    # Wasserstein calibrator even after calibrate_ensemble.py overwrites calibrator.pkl
    base_path = os.path.join(os.path.dirname(output_path), 'calibrator_base.pkl')
    with open(base_path, 'wb') as f:
        pickle.dump(calibrator, f)
    print(f"Base calibrator copy saved -> {base_path}")

    return calibrator, report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cal-scores', default='experiments/calibration/clean_scores.npy')
    parser.add_argument('--val-scores', default='experiments/calibration/val_scores.npy')
    parser.add_argument('--output',     default='models/calibrator.pkl')
    args = parser.parse_args()
    calibrate_testset(args.cal_scores, args.val_scores, args.output)
