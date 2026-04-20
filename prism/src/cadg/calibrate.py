"""
CADG: Conformal Adversarial Detection Guarantee
Split conformal prediction calibration for adversarial detection
with distribution-free FPR guarantees.

Conformal guarantee (Vovk et al., 2005; Angelopoulos & Bates, 2023):
  Given n clean calibration scores {s_1,...,s_n} drawn i.i.d. from some
  distribution P, the threshold τ_α = s_{⌈(n+1)(1−α)⌉} satisfies:

      P(score_new > τ_α) ≤ α  (exactly, for any finite n)

  where the probability is marginalised over the randomness of the calibration
  set AND the new test point.

Key implementation notes:
  - The quantile index uses the standard ceil((n+1)(1-α)) formula, clamped
    to [1, n] before converting to 0-indexed.
  - calibration_alphas stores the alpha values used to COMPUTE the thresholds.
    These may differ from the declared published_alphas when conservative
    calibration is employed (see calibrate_ensemble.py).
  - verify_coverage uses strict comparison (tolerance=0.0) by default so that
    failures are never masked for publication-quality reporting.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple


class ConformalCalibrator:
    """
    Split conformal calibration for tiered adversarial detection.

    Provides distribution-free FPR bounds at each response level (L1/L2/L3).
    Thresholds are derived from clean calibration scores via the standard
    conformal quantile formula; no model assumptions are required.

    Attributes:
        alphas: Published FPR targets {level: alpha} exposed in the paper.
        calibration_alphas: Alpha values actually used to compute thresholds.
            Equal to alphas unless conservative calibration was applied
            (e.g. calibrate_ensemble.py scales by 0.80 for a slack buffer).
        thresholds: Computed conformal thresholds {level: tau}.
        n_calibration: Number of clean samples used for calibration.
        calibration_scores: Sorted calibration scores (kept for diagnostics).
    """

    def __init__(self, alphas: Optional[Dict[str, float]] = None):
        """
        Args:
            alphas: {level_name: target_fpr}.
                Defaults to the three-tier design:
                  L1 = 0.10  (10 % FPR — monitor / log tier)
                  L2 = 0.03  ( 3 % FPR — purification tier)
                  L3 = 0.005 (0.5% FPR — expert-routing / rejection tier)
        """
        self.alphas: Dict[str, float] = alphas or {
            'L1': 0.10,
            'L2': 0.03,
            'L3': 0.005,
        }
        # Separate record of alphas used to derive thresholds (may differ from
        # self.alphas when conservative calibration is applied externally).
        self.calibration_alphas: Dict[str, float] = dict(self.alphas)

        self.calibration_scores: Optional[np.ndarray] = None
        self.thresholds: Dict[str, float] = {}
        self.n_calibration: int = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Core calibration
    # ─────────────────────────────────────────────────────────────────────────

    def calibrate(
        self,
        clean_scores: np.ndarray,
        alphas: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Compute conformal thresholds from a calibration set of clean scores.

        Uses the standard split-conformal quantile:
            q_idx = min(⌈(n+1)(1−α)⌉, n) − 1  (0-indexed into sorted scores)

        The guarantee holds for any distribution: after calibration on n
        exchangeable clean samples, a future clean sample exceeds the threshold
        with probability at most α (marginalised over calibration randomness).

        Args:
            clean_scores: 1-D array of anomaly scores from clean inputs.
                Minimum: 100 samples. Recommended: ≥ 1 000.
            alphas: Override the per-level FPR targets for this calibration
                call only. If None, uses self.alphas. The supplied alphas are
                stored in self.calibration_alphas for provenance.
        Returns:
            Dict of {level: threshold}.
        Raises:
            ValueError: If fewer than 100 calibration samples are provided.
        """
        if len(clean_scores) < 100:
            raise ValueError(
                f"Calibration set too small ({len(clean_scores)} samples). "
                f"Need at least 100, recommend ≥ 1 000 for reliable guarantees."
            )

        effective_alphas = alphas if alphas is not None else self.alphas
        self.calibration_alphas = dict(effective_alphas)

        self.calibration_scores = np.sort(clean_scores)
        self.n_calibration = len(clean_scores)
        n = self.n_calibration

        for level, alpha in effective_alphas.items():
            # Standard split-conformal quantile index (1-indexed then clamped)
            q_idx_1based = int(np.ceil((n + 1) * (1 - alpha)))
            q_idx_1based = min(q_idx_1based, n)   # clamp to [1, n]
            q_idx_1based = max(q_idx_1based, 1)
            self.thresholds[level] = float(self.calibration_scores[q_idx_1based - 1])

        return dict(self.thresholds)

    # ─────────────────────────────────────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────────────────────────────────────

    def classify(
        self,
        score: float,
        l0_active: bool = False,
        l0_factor: float = 0.8,
    ) -> str:
        """
        Map an anomaly score to a response level.

        Tiers checked from most severe (L3) to least (L1).  When the L0
        campaign alert is active, effective thresholds are multiplied by
        l0_factor (< 1.0), increasing sensitivity at the cost of a higher FPR.

        **Conformal FPR guarantee scope:**
        The distribution-free FPR bound stated in the paper (Vovk et al., 2005)
        applies ONLY when ``l0_active=False`` (i.e., thresholds are unmodified).
        When ``l0_active=True`` the effective thresholds drop to ``τ * l0_factor``,
        which raises the FPR above the calibrated alpha.  The L0 mode is an
        operational response layer — it intentionally trades FPR for sensitivity
        during a detected attack campaign.  Paper claims about guaranteed FPR
        must be stated for the non-L0 case.  Report empirical FPR for both
        modes separately in Section 4.

        Args:
            score: Scalar anomaly score from the TAMM pipeline.
            l0_active: Whether the SACD campaign monitor (L0) is active.
                When True, thresholds are scaled by l0_factor.  The conformal
                FPR guarantee does NOT hold in this mode.
            l0_factor: Threshold scaling factor during L0 alert (default 0.8).
                Must be in (0, 1].  Values below 0.5 risk excessive FPR.
        Returns:
            Response level: 'L3', 'L2', 'L1', or 'PASS'.
        Raises:
            RuntimeError: If calibrate() has not been called yet.
        """
        if not self.thresholds:
            raise RuntimeError(
                "Calibrator not calibrated. Call calibrate() first."
            )

        factor = l0_factor if l0_active else 1.0

        if score > self.thresholds['L3'] * factor:
            return 'L3'
        elif score > self.thresholds['L2'] * factor:
            return 'L2'
        elif score > self.thresholds['L1'] * factor:
            return 'L1'
        else:
            return 'PASS'

    # ─────────────────────────────────────────────────────────────────────────
    # Coverage verification
    # ─────────────────────────────────────────────────────────────────────────

    def verify_coverage(
        self,
        validation_scores: np.ndarray,
        level: str,
        tolerance: float = 0.0,
    ) -> Tuple[bool, float]:
        """
        Verify that the empirical FPR on a held-out clean validation set does
        not exceed the target alpha (plus an optional finite-sample tolerance).

        For publication reporting, use tolerance=0.0 (strict).
        For unit tests with small val sets (n<1 000), a tolerance of up to
        1/n_val is statistically defensible.

        Args:
            validation_scores: 1-D array of anomaly scores from CLEAN inputs
                that were NOT in the calibration set.
            level: Tier to verify ('L1', 'L2', or 'L3').
            tolerance: Additional slack added to the target alpha before
                checking the pass condition (default 0.0 = strict).
        Returns:
            (passed, empirical_fpr)
            - passed: True iff empirical_fpr <= self.alphas[level] + tolerance.
            - empirical_fpr: Fraction of validation_scores exceeding threshold.
        Raises:
            KeyError: If level is not in self.thresholds.
            RuntimeError: If calibrate() has not been called yet.
        """
        if not self.thresholds:
            raise RuntimeError(
                "Calibrator not calibrated. Call calibrate() first."
            )
        if level not in self.thresholds:
            raise KeyError(
                f"Unknown level '{level}'. Available: {list(self.thresholds)}"
            )

        alpha = self.alphas[level]
        threshold = self.thresholds[level]
        empirical_fpr = float(np.mean(validation_scores > threshold))
        passed = empirical_fpr <= alpha + tolerance

        return passed, empirical_fpr

    def get_coverage_report(
        self,
        validation_scores: np.ndarray,
        tolerance: float = 0.0,
    ) -> Dict:
        """
        Full coverage verification report across all calibrated tiers.

        Args:
            validation_scores: 1-D array of clean validation anomaly scores.
            tolerance: Slack added to each alpha before pass/fail decision.
                Use 0.0 for publication reporting.
        Returns:
            Dict keyed by level, each containing:
              'target_alpha'      : declared FPR target from self.alphas
              'calibration_alpha' : alpha actually used to compute the threshold
              'empirical_fpr'     : observed FPR on validation_scores
              'threshold'         : the conformal threshold τ
              'n_cal'             : calibration set size
              'passed'            : bool — empirical_fpr ≤ target + tolerance
        """
        report = {}
        for level in self.alphas:
            passed, fpr = self.verify_coverage(
                validation_scores, level, tolerance=tolerance
            )
            report[level] = {
                'target_alpha':      self.alphas[level],
                'calibration_alpha': self.calibration_alphas.get(level, self.alphas[level]),
                'empirical_fpr':     fpr,
                'threshold':         self.thresholds[level],
                'n_cal':             self.n_calibration,
                'passed':            passed,
            }
        return report

    # ─────────────────────────────────────────────────────────────────────────
    # Reporting helpers
    # ─────────────────────────────────────────────────────────────────────────

    def summary(
        self,
        validation_scores: Optional[np.ndarray] = None,
        tolerance: float = 0.0,
    ) -> str:
        """
        Return a human-readable calibration summary for paper reporting.

        Args:
            validation_scores: Optional held-out clean scores for FPR check.
            tolerance: Tolerance passed to get_coverage_report if val_scores given.
        Returns:
            Formatted multi-line string.
        """
        lines: List[str] = [
            f"ConformalCalibrator Summary",
            f"  n_calibration : {self.n_calibration}",
            f"  Levels        : {list(self.alphas.keys())}",
            "",
            f"  {'Level':<6} {'Cal-α':>8} {'Pub-α':>8} {'Threshold':>12}",
            f"  {'-'*6} {'-'*8} {'-'*8} {'-'*12}",
        ]
        for level in self.alphas:
            cal_a = self.calibration_alphas.get(level, self.alphas[level])
            pub_a = self.alphas[level]
            thr   = self.thresholds.get(level, float('nan'))
            lines.append(f"  {level:<6} {cal_a:>8.4f} {pub_a:>8.4f} {thr:>12.6f}")

        if validation_scores is not None:
            report = self.get_coverage_report(validation_scores, tolerance=tolerance)
            lines += [
                "",
                f"  Coverage verification (n_val={len(validation_scores)}, tol={tolerance}):",
                f"  {'Level':<6} {'Target':>8} {'Empirical':>10} {'Status':>8}",
                f"  {'-'*6} {'-'*8} {'-'*10} {'-'*8}",
            ]
            for level, info in report.items():
                status = "PASS ✓" if info['passed'] else "FAIL ✗"
                lines.append(
                    f"  {level:<6} {info['target_alpha']:>8.4f} "
                    f"{info['empirical_fpr']:>10.4f} {status:>8}"
                )

        return "\n".join(lines)

    def __repr__(self) -> str:
        cal_str = (
            f"n={self.n_calibration}"
            if self.n_calibration > 0
            else "not calibrated"
        )
        thr_str = ", ".join(
            f"{k}={v:.4f}" for k, v in self.thresholds.items()
        ) or "—"
        return (
            f"ConformalCalibrator("
            f"alphas={self.alphas}, "
            f"{cal_str}, "
            f"thresholds={{{thr_str}}})"
        )
