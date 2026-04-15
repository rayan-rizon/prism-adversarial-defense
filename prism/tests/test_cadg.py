"""
CADG unit tests — ConformalCalibrator, TieredThresholdManager.

Cross-check against guide Section 3.1 and Section 2.2:
- Thresholds are ordered L1 < L2 < L3
- Empirical FPR on held-out clean data ≤ alpha + tolerance
- classify() assigns PASS to low scores, L3 to high scores
- L0 factor lowers thresholds (increases sensitivity)
- calibrate() raises on too-small datasets
- verify_coverage() passes with 1000+ clean samples
- get_coverage_report() returns correct structure
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.cadg.calibrate import ConformalCalibrator


RNG = np.random.RandomState(1)


def make_clean_scores(n=1000, mean=0.1, std=0.02):
    return np.abs(RNG.normal(mean, std, n))  # non-negative scores


class TestConformalCalibratorCoverage:
    """Core research requirement: conformal FPR guarantee."""

    def test_fpr_at_l1_satisfies_guarantee(self):
        """Guide Section 2.2 validation check."""
        # Use a per-test local RNG (n=5000 for stable FPR estimate; SE≈0.003)
        rng = np.random.RandomState(42)
        cal = ConformalCalibrator()
        cal.calibrate(np.abs(rng.normal(0.1, 0.02, 5000)))

        val_scores = np.abs(rng.normal(0.1, 0.02, 5000))
        fpr = np.mean(val_scores > cal.thresholds['L1'])
        # Guide allows +0.02 tolerance for finite-sample deviation
        assert fpr <= 0.10 + 0.02, (
            f"L1 conformal guarantee violated: FPR={fpr:.4f} > 0.12"
        )

    def test_fpr_at_l2_satisfies_guarantee(self):
        cal = ConformalCalibrator()
        cal.calibrate(make_clean_scores(1000))
        val_scores = make_clean_scores(1000)
        fpr = np.mean(val_scores > cal.thresholds['L2'])
        assert fpr <= 0.03 + 0.02

    def test_fpr_at_l3_satisfies_guarantee(self):
        cal = ConformalCalibrator()
        cal.calibrate(make_clean_scores(1000))
        val_scores = make_clean_scores(1000)
        fpr = np.mean(val_scores > cal.thresholds['L3'])
        assert fpr <= 0.005 + 0.02

    def test_thresholds_ordered(self):
        """L1 < L2 < L3 — lower alpha = higher threshold."""
        cal = ConformalCalibrator()
        cal.calibrate(make_clean_scores(1000))
        assert cal.thresholds['L1'] < cal.thresholds['L2'], (
            f"L1 threshold ({cal.thresholds['L1']:.4f}) must be < "
            f"L2 ({cal.thresholds['L2']:.4f})"
        )
        assert cal.thresholds['L2'] < cal.thresholds['L3'], (
            f"L2 threshold ({cal.thresholds['L2']:.4f}) must be < "
            f"L3 ({cal.thresholds['L3']:.4f})"
        )


class TestConformalCalibratorClassify:
    def setup_method(self):
        self.cal = ConformalCalibrator()
        self.cal.calibrate(make_clean_scores(1000))

    def test_low_score_is_pass(self):
        # Score very near 0 must always be PASS
        assert self.cal.classify(0.0) == 'PASS'

    def test_high_score_is_l3(self):
        # Score far above any threshold must be L3
        assert self.cal.classify(1e6) == 'L3'

    def test_l0_lowers_thresholds(self):
        """When L0 active, same score hits higher tier."""
        # Find a score just below L2 threshold
        border_score = self.cal.thresholds['L2'] * 0.95
        level_normal = self.cal.classify(border_score, l0_active=False)
        level_l0 = self.cal.classify(border_score, l0_active=True, l0_factor=0.8)
        # With L0 active (factor=0.8), effective L2 threshold drops 20%
        # so border_score at 95% of L2 should now exceed L2*0.8 = 80% of L2
        # and therefore be classified at least L2
        assert level_l0 in ('L2', 'L3'), (
            f"L0 should elevate classification, got {level_l0}"
        )

    def test_classification_monotone(self):
        """Higher score → equal or higher tier."""
        scores = sorted(RNG.uniform(0, 1, 20))
        levels = [self.cal.classify(s) for s in scores]
        order = {'PASS': 0, 'L1': 1, 'L2': 2, 'L3': 3}
        for i in range(len(levels) - 1):
            assert order[levels[i]] <= order[levels[i+1]], (
                f"Non-monotone at scores[{i}]={scores[i]:.4f} ({levels[i]}) "
                f"→ scores[{i+1}]={scores[i+1]:.4f} ({levels[i+1]})"
            )


class TestConformalCalibratorEdgeCases:
    def test_too_small_dataset_raises(self):
        cal = ConformalCalibrator()
        with pytest.raises(ValueError, match="too small"):
            cal.calibrate(np.array([0.1, 0.2, 0.3]))

    def test_classify_before_calibrate_raises(self):
        cal = ConformalCalibrator()
        with pytest.raises(RuntimeError):
            cal.classify(0.5)

    def test_calibrate_returns_threshold_dict(self):
        cal = ConformalCalibrator()
        result = cal.calibrate(make_clean_scores(500))
        assert isinstance(result, dict)
        assert set(result.keys()) == {'L1', 'L2', 'L3'}

    def test_verify_coverage_passes_on_clean_data(self):
        """Guide Section 3.1 test_conformal_coverage."""
        cal = ConformalCalibrator()
        cal.calibrate(make_clean_scores(1000))
        passed, fpr = cal.verify_coverage(make_clean_scores(1000), level='L1')
        assert passed, f"Coverage verification failed: FPR={fpr:.4f}"

    def test_get_coverage_report_structure(self):
        cal = ConformalCalibrator()
        cal.calibrate(make_clean_scores(1000))
        report = cal.get_coverage_report(make_clean_scores(500))
        assert set(report.keys()) == {'L1', 'L2', 'L3'}
        for level, info in report.items():
            assert 'empirical_fpr' in info
            assert 'target_alpha' in info
            assert 'passed' in info


class TestConformalQuantileFormula:
    """Verify the conformal quantile formula is mathematically correct."""

    def test_quantile_index_uses_ceil_n_plus_1(self):
        """
        Standard split conformal: q_{ceil((n+1)(1-α))/n}.
        With n=100, α=0.1:
          q_index = ceil(101 * 0.9) = ceil(90.9) = 91
          0-indexed: min(91, 100) - 1 = 90 → scores[90] = 90.0
        """
        cal = ConformalCalibrator(alphas={'L1': 0.1})
        # Sorted scores: 0, 1, ..., 99
        scores = np.arange(100, dtype=float)
        cal.calibrate(scores)
        # ceil((100+1)*0.9) = ceil(90.9) = 91, clamped → 0-index 90 = score 90.0
        assert cal.thresholds['L1'] == pytest.approx(90.0)
