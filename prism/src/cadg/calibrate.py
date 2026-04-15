"""
CADG: Conformal Adversarial Detection Guarantee
Split conformal prediction calibration for adversarial detection
with distribution-free FPR guarantees.

Key correction from plan: conformal quantile uses the standard
formula q = ceil((n+1)*(1-alpha)) / n, clamped to [0, 1].
"""
import numpy as np
from typing import Dict, Optional, Tuple


class ConformalCalibrator:
    """
    Split conformal calibration for tiered adversarial detection.
    Provides distribution-free FPR bounds at each response level.
    """

    def __init__(self, alphas: Optional[Dict[str, float]] = None):
        """
        Args:
            alphas: {level_name: target_fpr}. Defaults to L1/L2/L3 tiers.
        """
        self.alphas = alphas or {
            'L1': 0.10,   # 10% FPR — monitor tier
            'L2': 0.03,   # 3% FPR  — purification tier
            'L3': 0.005,  # 0.5% FPR — rejection/expert tier
        }
        self.calibration_scores: Optional[np.ndarray] = None
        self.thresholds: Dict[str, float] = {}
        self.n_calibration: int = 0

    def calibrate(self, clean_scores: np.ndarray) -> Dict[str, float]:
        """
        Compute conformal thresholds from calibration set of clean scores.

        Args:
            clean_scores: 1D array of anomaly scores from clean inputs.
                          Minimum recommended: 1000 samples.
        Returns:
            Dict of {level: threshold}.
        """
        if len(clean_scores) < 100:
            raise ValueError(
                f"Calibration set too small ({len(clean_scores)}). "
                f"Need at least 100 samples, recommend 1000+."
            )

        self.calibration_scores = np.sort(clean_scores)
        self.n_calibration = len(clean_scores)
        n = self.n_calibration

        for level, alpha in self.alphas.items():
            # Standard split conformal quantile
            q_index = int(np.ceil((n + 1) * (1 - alpha)))
            # Clamp to valid index range
            q_index = min(q_index, n) - 1  # 0-indexed
            q_index = max(q_index, 0)
            self.thresholds[level] = float(self.calibration_scores[q_index])

        return dict(self.thresholds)

    def classify(self, score: float, l0_active: bool = False,
                 l0_factor: float = 0.8) -> str:
        """
        Classify an anomaly score into a response level.

        Thresholds checked from most severe (L3) to least (L1).
        When L0 campaign alert is active, thresholds are lowered by l0_factor.

        Args:
            score: The anomaly score to classify.
            l0_active: Whether the campaign monitor (L0) is active.
            l0_factor: Multiplicative factor to lower thresholds during L0.
        Returns:
            Response level: 'L3', 'L2', 'L1', or 'PASS'.
        """
        if not self.thresholds:
            raise RuntimeError("Calibrator not calibrated. Call calibrate() first.")

        factor = l0_factor if l0_active else 1.0

        # Check from most severe to least
        if score > self.thresholds['L3'] * factor:
            return 'L3'
        elif score > self.thresholds['L2'] * factor:
            return 'L2'
        elif score > self.thresholds['L1'] * factor:
            return 'L1'
        else:
            return 'PASS'

    def verify_coverage(self, validation_scores: np.ndarray,
                        level: str) -> Tuple[bool, float]:
        """
        Verify that empirical FPR <= target alpha on held-out validation set.

        Args:
            validation_scores: Anomaly scores from clean validation data.
            level: Which tier to verify ('L1', 'L2', or 'L3').
        Returns:
            (passed, empirical_fpr).
        """
        if level not in self.thresholds:
            raise KeyError(f"Unknown level '{level}'. Available: {list(self.thresholds)}")

        alpha = self.alphas[level]
        threshold = self.thresholds[level]
        empirical_fpr = float(np.mean(validation_scores > threshold))
        passed = empirical_fpr <= alpha + 0.01  # Small tolerance for finite samples

        return passed, empirical_fpr

    def get_coverage_report(self, validation_scores: np.ndarray) -> Dict:
        """Full verification report across all tiers."""
        report = {}
        for level in self.alphas:
            passed, fpr = self.verify_coverage(validation_scores, level)
            report[level] = {
                'target_alpha': self.alphas[level],
                'empirical_fpr': fpr,
                'threshold': self.thresholds[level],
                'passed': passed,
            }
        return report
