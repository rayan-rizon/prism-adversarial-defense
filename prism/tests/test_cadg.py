"""
CADG unit tests — ConformalCalibrator.

Research-grade test suite for the split conformal calibration module.
Every test uses an isolated, deterministic RNG seed so results are
reproducible regardless of test execution order.

Covers:
  - Conformal quantile formula correctness (mathematical property)
  - FPR guarantee at each tier (L1/L2/L3) — the core paper claim
  - Threshold ordering: L1 < L2 < L3
  - classify() monotonicity and L0-factor sensitivity
  - Edge-case validation (too-small datasets, un-calibrated calls)
  - calibration_alpha provenance tracking
  - summary() output completeness
  - verify_coverage() strict vs toleranced modes
  - get_coverage_report() structure

FPR test methodology:
  Calibrate on n=5 000 clean samples, verify on a separate n=5 000
  clean samples drawn from the SAME distribution (Normal with mean=0.1,
  std=0.02).  At this scale the Monte-Carlo standard error of the FPR
  estimate is ≈ sqrt(α(1-α)/5000) ≈ 0.0042 for L1, giving a 3σ margin
  of ≈ 0.013 — well within the ±0.02 tolerance used in the tests.
  Tests for L2 and L3 use the same n=5 000 to keep the SE small relative
  to their tighter targets.
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.cadg.calibrate import ConformalCalibrator


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_clean_scores(n: int, seed: int, mean: float = 0.1, std: float = 0.02):
    """Independent, seed-isolated clean score generator.  Always non-negative."""
    rng = np.random.RandomState(seed)
    return np.abs(rng.normal(mean, std, n))


# ─────────────────────────────────────────────────────────────────────────────
# 1. FPR Guarantee Tests  (core research claim)
# ─────────────────────────────────────────────────────────────────────────────

class TestFPRGuarantees:
    """
    Verify the conformal FPR guarantee at each response tier.

    Methodology: calibrate on n_cal=5 000, validate on n_val=5 000 from the
    same distribution.  At this scale the Monte-Carlo standard error of the
    FPR estimate is ≤ 0.005, so a tolerance of +0.02 is conservative (≥ 4σ).

    These tests are the primary empirical validation of the paper's Theorem 1
    (distribution-free FPR bound).
    """

    N_CAL = 5000
    N_VAL = 5000
    # Tolerance: per-test finite-sample allowance on top of the target alpha.
    # Set to 0.02 (~4σ for L1) so the test remains stable across seeds without
    # masking genuine calibration failures.
    TOLERANCE = 0.02

    def _run(self, level: str, alpha: float, cal_seed: int, val_seed: int):
        cal = ConformalCalibrator()
        cal.calibrate(make_clean_scores(self.N_CAL, seed=cal_seed))
        val_scores = make_clean_scores(self.N_VAL, seed=val_seed)
        passed, fpr = cal.verify_coverage(val_scores, level, tolerance=self.TOLERANCE)
        assert passed, (
            f"{level} conformal guarantee violated: "
            f"FPR={fpr:.4f} > target={alpha:.4f} + tol={self.TOLERANCE:.4f}"
        )

    def test_fpr_guarantee_l1(self):
        """α=10 %: empirical FPR must be ≤ 0.12 (target + tolerance)."""
        self._run('L1', alpha=0.10, cal_seed=101, val_seed=102)

    def test_fpr_guarantee_l2(self):
        """α=3 %: empirical FPR must be ≤ 0.05."""
        self._run('L2', alpha=0.03, cal_seed=201, val_seed=202)

    def test_fpr_guarantee_l3(self):
        """α=0.5 %: empirical FPR must be ≤ 0.025."""
        self._run('L3', alpha=0.005, cal_seed=301, val_seed=302)

    def test_fpr_strict_l1_large_n(self):
        """
        Strict mode (tolerance=0.0): with n=10 000 calibration samples the
        conformal guarantee should hold exactly with overwhelming probability.
        The theoretical guarantee is ≥ 1-alpha, so we expect near-certain
        success on a single draw.  Only L1 is tested strictly because L3 at
        alpha=0.005 has high variance even at n=10 000 (SE≈0.001).
        """
        cal = ConformalCalibrator()
        cal.calibrate(make_clean_scores(10_000, seed=999))
        val_scores = make_clean_scores(10_000, seed=1000)
        passed, fpr = cal.verify_coverage(val_scores, 'L1', tolerance=0.0)
        assert passed, (
            f"Strict L1 guarantee failed: FPR={fpr:.4f} > 0.10 (n_cal=10 000)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Threshold ordering
# ─────────────────────────────────────────────────────────────────────────────

class TestThresholdOrdering:
    """
    Lower alpha → higher threshold (stricter detection).
    Required for the classify() ladder to be logically consistent.
    """

    def test_thresholds_ordered_l1_l2_l3(self):
        """L1 < L2 < L3 (all tiers)."""
        cal = ConformalCalibrator()
        cal.calibrate(make_clean_scores(1000, seed=10))
        t = cal.thresholds
        assert t['L1'] < t['L2'], (
            f"Threshold ordering violated: L1={t['L1']:.4f} ≥ L2={t['L2']:.4f}"
        )
        assert t['L2'] < t['L3'], (
            f"Threshold ordering violated: L2={t['L2']:.4f} ≥ L3={t['L3']:.4f}"
        )

    def test_thresholds_ordered_custom_alphas(self):
        """Ordering holds for arbitrary custom alpha sets."""
        cal = ConformalCalibrator(alphas={'fast': 0.20, 'medium': 0.05, 'strict': 0.01})
        cal.calibrate(make_clean_scores(2000, seed=11))
        t = cal.thresholds
        assert t['fast'] < t['medium'] < t['strict']


# ─────────────────────────────────────────────────────────────────────────────
# 3. classify() correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestClassify:

    def setup_method(self):
        """Each test method gets a fresh calibrator from an isolated RNG."""
        self.cal = ConformalCalibrator()
        self.cal.calibrate(make_clean_scores(2000, seed=42))

    def test_score_zero_is_pass(self):
        assert self.cal.classify(0.0) == 'PASS'

    def test_very_high_score_is_l3(self):
        assert self.cal.classify(1e6) == 'L3'

    def test_monotonicity(self):
        """Higher score → equal or higher response tier."""
        rng = np.random.RandomState(77)
        scores = np.sort(rng.uniform(0.0, 1.0, 30))
        levels = [self.cal.classify(float(s)) for s in scores]
        order = {'PASS': 0, 'L1': 1, 'L2': 2, 'L3': 3}
        for i in range(len(levels) - 1):
            assert order[levels[i]] <= order[levels[i + 1]], (
                f"Non-monotone at idx {i}: score={scores[i]:.4f} → {levels[i]}, "
                f"score={scores[i+1]:.4f} → {levels[i+1]}"
            )

    def test_l0_factor_raises_tier(self):
        """
        With L0 active (factor=0.8), a score just below the L2 threshold
        should exceed the effective L2 threshold (0.8 × L2) and be classified
        at L2 or above.
        """
        # Score at 95 % of the L2 threshold — slightly below L2, above L1
        border = self.cal.thresholds['L2'] * 0.95
        normal_level = self.cal.classify(border, l0_active=False)
        # Without L0 the score must be below L2 (just above L1 or PASS)
        assert normal_level in ('PASS', 'L1'), (
            f"Expected below L2 without L0, got {normal_level}"
        )
        # With L0 the effective L2 threshold = 0.8 × L2; border (0.95 × L2) > 0.8 × L2
        l0_level = self.cal.classify(border, l0_active=True, l0_factor=0.8)
        assert l0_level in ('L2', 'L3'), (
            f"L0 should elevate to L2 or L3, got {l0_level}"
        )

    def test_exactly_at_threshold_is_pass(self):
        """Score exactly equal to a threshold is NOT above it → PASS or lower tier."""
        # The condition is strictly >, so equality → PASS for L1 threshold
        at_l1 = self.cal.thresholds['L1']
        level = self.cal.classify(at_l1)
        # Should be PASS (not L1) because score > threshold is False when equal
        assert level == 'PASS', (
            f"Score exactly at L1 threshold should be PASS, got {level}"
        )

    def test_uncalibrated_raises(self):
        cal = ConformalCalibrator()
        with pytest.raises(RuntimeError, match="not calibrated"):
            cal.classify(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_too_small_dataset_raises(self):
        cal = ConformalCalibrator()
        with pytest.raises(ValueError, match="too small"):
            cal.calibrate(np.array([0.1, 0.2, 0.3]))

    def test_exact_minimum_size_accepted(self):
        """n=100 is the documented minimum; should NOT raise."""
        cal = ConformalCalibrator()
        cal.calibrate(make_clean_scores(100, seed=5))
        assert cal.n_calibration == 100

    def test_calibrate_returns_threshold_dict(self):
        cal = ConformalCalibrator()
        result = cal.calibrate(make_clean_scores(500, seed=6))
        assert isinstance(result, dict)
        assert set(result.keys()) == {'L1', 'L2', 'L3'}

    def test_calibrate_updates_n_calibration(self):
        cal = ConformalCalibrator()
        cal.calibrate(make_clean_scores(1234, seed=7))
        assert cal.n_calibration == 1234

    def test_verify_coverage_unknown_level_raises(self):
        cal = ConformalCalibrator()
        cal.calibrate(make_clean_scores(500, seed=8))
        with pytest.raises(KeyError):
            cal.verify_coverage(make_clean_scores(100, seed=9), level='L99')

    def test_verify_coverage_before_calibrate_raises(self):
        cal = ConformalCalibrator()
        with pytest.raises(RuntimeError, match="not calibrated"):
            cal.verify_coverage(np.array([0.1, 0.2]), level='L1')

    def test_recalibration_updates_thresholds(self):
        """Calling calibrate() twice must overwrite thresholds, not accumulate."""
        cal = ConformalCalibrator()
        cal.calibrate(make_clean_scores(500, seed=10))
        thresholds_first = dict(cal.thresholds)
        # Use a different distribution
        cal.calibrate(np.abs(np.random.RandomState(99).normal(1.0, 0.5, 500)))
        assert cal.thresholds != thresholds_first
        assert cal.n_calibration == 500  # n_calibration is reset

    def test_custom_alphas_roundtrip(self):
        alphas = {'fast': 0.20, 'strict': 0.01}
        cal = ConformalCalibrator(alphas=alphas)
        result = cal.calibrate(make_clean_scores(300, seed=11))
        assert set(result.keys()) == {'fast', 'strict'}


# ─────────────────────────────────────────────────────────────────────────────
# 5. Conformal quantile formula (mathematical correctness)
# ─────────────────────────────────────────────────────────────────────────────

class TestConformalQuantileFormula:
    """
    Verify the exact quantile index against known closed-form values.

    Standard split conformal: q_idx = ceil((n+1)(1-α)) − 1 (0-indexed).
    With n=100, α=0.10:
      1-indexed: min(ceil(101×0.9), 100) = min(ceil(90.9), 100) = min(91, 100) = 91
      0-indexed: 91 − 1 = 90 → calibration_scores[90] = 90.0
    """

    def test_exact_quantile_n100_alpha01(self):
        cal = ConformalCalibrator(alphas={'L1': 0.10})
        scores = np.arange(100, dtype=float)   # sorted: 0, 1, ..., 99
        cal.calibrate(scores)
        assert cal.thresholds['L1'] == pytest.approx(90.0), (
            f"Expected threshold=90.0, got {cal.thresholds['L1']}"
        )

    def test_exact_quantile_n1000_alpha005(self):
        """n=1000, α=0.005 → q_1indexed = min(ceil(1001×0.995),1000) = min(996,1000) = 996."""
        cal = ConformalCalibrator(alphas={'L3': 0.005})
        scores = np.arange(1000, dtype=float)
        cal.calibrate(scores)
        # 0-indexed: 996 - 1 = 995 → scores[995] = 995.0
        assert cal.thresholds['L3'] == pytest.approx(995.0)

    def test_quantile_clamped_at_n(self):
        """With α extremely close to 0, the index must clamp to n (last element)."""
        cal = ConformalCalibrator(alphas={'L_ultra': 1e-9})
        scores = np.arange(200, dtype=float)
        cal.calibrate(scores)
        # ceil((201)(1-1e-9)) ≈ ceil(201 - 2e-7) = 201 → clamped to 200 → 0-indexed 199
        assert cal.thresholds['L_ultra'] == pytest.approx(199.0)


# ─────────────────────────────────────────────────────────────────────────────
# 6. calibration_alpha provenance
# ─────────────────────────────────────────────────────────────────────────────

class TestCalibrationAlphaProvenance:
    """
    Verify that calibration_alphas correctly tracks what was used to compute
    thresholds, independently of self.alphas (which may be set to published
    targets after conservative calibration — see calibrate_ensemble.py).
    """

    def test_calibration_alpha_equals_alpha_by_default(self):
        cal = ConformalCalibrator()
        cal.calibrate(make_clean_scores(500, seed=20))
        assert cal.calibration_alphas == cal.alphas

    def test_calibration_alpha_set_by_override(self):
        """If calibrate() is called with an alphas override, calibration_alphas records it."""
        cal = ConformalCalibrator(alphas={'L1': 0.10, 'L2': 0.03, 'L3': 0.005})
        conservative = {'L1': 0.08, 'L2': 0.024, 'L3': 0.004}
        cal.calibrate(make_clean_scores(500, seed=21), alphas=conservative)
        # calibration_alphas should reflect the conservative values
        assert cal.calibration_alphas == conservative
        # but self.alphas (published targets) should be unchanged
        assert cal.alphas == {'L1': 0.10, 'L2': 0.03, 'L3': 0.005}

    def test_coverage_report_exposes_both_alphas(self):
        cal = ConformalCalibrator(alphas={'L1': 0.10, 'L2': 0.03, 'L3': 0.005})
        conservative = {'L1': 0.08, 'L2': 0.024, 'L3': 0.004}
        cal.calibrate(make_clean_scores(2000, seed=22), alphas=conservative)
        # After simulating ensemble workflow: overwrite alphas to published targets
        cal.alphas = {'L1': 0.10, 'L2': 0.03, 'L3': 0.005}
        report = cal.get_coverage_report(make_clean_scores(1000, seed=23))
        for level, info in report.items():
            assert 'calibration_alpha' in info
            assert 'target_alpha' in info
            # Calibration alpha was conservative (lower than target)
            assert info['calibration_alpha'] <= info['target_alpha']


# ─────────────────────────────────────────────────────────────────────────────
# 7. get_coverage_report() structure
# ─────────────────────────────────────────────────────────────────────────────

class TestCoverageReport:

    def setup_method(self):
        self.cal = ConformalCalibrator()
        self.cal.calibrate(make_clean_scores(1000, seed=50))
        self.val = make_clean_scores(500, seed=51)

    def test_report_has_all_levels(self):
        report = self.cal.get_coverage_report(self.val)
        assert set(report.keys()) == {'L1', 'L2', 'L3'}

    def test_report_fields_present(self):
        report = self.cal.get_coverage_report(self.val)
        required_fields = {
            'target_alpha', 'calibration_alpha', 'empirical_fpr',
            'threshold', 'n_cal', 'passed',
        }
        for level, info in report.items():
            missing = required_fields - set(info.keys())
            assert not missing, f"Missing fields for {level}: {missing}"

    def test_report_n_cal_matches_calibration(self):
        report = self.cal.get_coverage_report(self.val)
        for info in report.values():
            assert info['n_cal'] == 1000

    def test_report_strict_vs_toleranced(self):
        """Strict (tol=0) report may fail; toleranced may pass. Both are valid."""
        report_strict = self.cal.get_coverage_report(self.val, tolerance=0.0)
        report_tol    = self.cal.get_coverage_report(self.val, tolerance=0.05)
        for level in self.cal.alphas:
            # Toleranced should never be stricter than strict
            if report_strict[level]['passed']:
                assert report_tol[level]['passed'], (
                    f"{level}: passed strict but failed with tolerance=0.05 — impossible"
                )


# ─────────────────────────────────────────────────────────────────────────────
# 8. summary() output
# ─────────────────────────────────────────────────────────────────────────────

class TestSummary:

    def test_summary_without_val_scores(self):
        cal = ConformalCalibrator()
        cal.calibrate(make_clean_scores(500, seed=60))
        text = cal.summary()
        assert 'ConformalCalibrator' in text
        assert 'n_calibration' in text
        assert 'L1' in text and 'L2' in text and 'L3' in text

    def test_summary_with_val_scores(self):
        cal = ConformalCalibrator()
        cal.calibrate(make_clean_scores(1000, seed=61))
        val = make_clean_scores(500, seed=62)
        text = cal.summary(validation_scores=val)
        assert 'Coverage verification' in text
        assert 'PASS' in text or 'FAIL' in text

    def test_repr_contains_key_info(self):
        cal = ConformalCalibrator()
        cal.calibrate(make_clean_scores(300, seed=63))
        r = repr(cal)
        assert 'ConformalCalibrator' in r
        assert 'n=300' in r
