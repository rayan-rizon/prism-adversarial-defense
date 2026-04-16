"""
Unit tests for PRISM core modules.
Tests each component independently without requiring GPU or large data.
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ============================================================
# Tests for CADG: Conformal Calibrator
# ============================================================

class TestConformalCalibrator:
    def test_calibrate_basic(self):
        from src.cadg.calibrate import ConformalCalibrator

        rng = np.random.RandomState(42)
        scores = rng.normal(0.1, 0.02, 2000)

        cal = ConformalCalibrator()
        thresholds = cal.calibrate(scores)

        assert 'L1' in thresholds
        assert 'L2' in thresholds
        assert 'L3' in thresholds
        # Thresholds should be ordered: L1 < L2 < L3
        assert thresholds['L1'] <= thresholds['L2'] <= thresholds['L3']

    def test_calibrate_small_set_raises(self):
        from src.cadg.calibrate import ConformalCalibrator

        cal = ConformalCalibrator()
        with pytest.raises(ValueError, match="too small"):
            cal.calibrate(np.array([0.1, 0.2]))

    def test_classify(self):
        from src.cadg.calibrate import ConformalCalibrator

        rng = np.random.RandomState(42)
        scores = rng.normal(0.1, 0.02, 2000)

        cal = ConformalCalibrator()
        cal.calibrate(scores)

        # Very low score should pass
        assert cal.classify(0.0) == 'PASS'
        # Very high score should be L3
        assert cal.classify(1.0) == 'L3'

    def test_classify_l0_lowers_thresholds(self):
        from src.cadg.calibrate import ConformalCalibrator

        rng = np.random.RandomState(42)
        scores = rng.normal(0.1, 0.02, 2000)

        cal = ConformalCalibrator()
        cal.calibrate(scores)

        # Find a score that's just below L1 threshold
        threshold_l1 = cal.thresholds['L1']
        score = threshold_l1 * 0.95  # Just below

        assert cal.classify(score, l0_active=False) == 'PASS'
        # With L0 active (factor=0.8), threshold drops so this score may trigger
        # threshold * 0.8 < threshold * 0.95 = score
        assert cal.classify(score, l0_active=True, l0_factor=0.8) != 'PASS'

    def test_verify_coverage(self):
        from src.cadg.calibrate import ConformalCalibrator

        rng = np.random.RandomState(42)
        cal_scores = rng.normal(0.1, 0.02, 2000)
        val_scores = rng.normal(0.1, 0.02, 2000)

        cal = ConformalCalibrator()
        cal.calibrate(cal_scores)

        passed, fpr = cal.verify_coverage(val_scores, level='L1')
        assert passed, f"Coverage failed: FPR={fpr:.4f} > alpha=0.10"


# ============================================================
# Tests for SACD: BOCPD + Campaign Monitor
# ============================================================

class TestBOCPD:
    def test_stable_stream(self):
        from src.sacd.bocpd import BayesianOnlineChangepoint

        bocpd = BayesianOnlineChangepoint(hazard_rate=1/200, mu0=0.1)
        rng = np.random.RandomState(42)

        # All stable — changepoint probs should stay low
        scores = rng.normal(0.1, 0.02, 100)
        cp_probs = [bocpd.update(s) for s in scores]
        # After burn-in, cp probs should be low
        assert np.mean(cp_probs[20:]) < 0.3

    def test_detects_shift(self):
        from src.sacd.bocpd import BayesianOnlineChangepoint

        bocpd = BayesianOnlineChangepoint(hazard_rate=1/50, mu0=0.1)
        rng = np.random.RandomState(42)

        # Stable phase — run_length_probs should concentrate on long runs
        for s in rng.normal(0.1, 0.02, 50):
            bocpd.update(s)

        assert bocpd.get_most_likely_run_length() >= 45, "MAP run length should be near 50"

        # Shift phase — scores jump to 0.5 (far from clean mean of 0.1)
        for s in rng.normal(0.5, 0.02, 5):
            bocpd.update(s)

        # After just a few outlier steps, run_length distribution should collapse
        # to short runs (recent changepoint detected via P(rl <= 5) > 0.5)
        rl_probs = bocpd.run_length_probs
        short_run_prob = float(np.sum(rl_probs[:6]))
        assert short_run_prob > 0.5, (
            f"BOCPD failed to detect shift via run-length collapse. "
            f"P(rl<=5)={short_run_prob:.4f}, MAP_rl={bocpd.get_most_likely_run_length()}"
        )

    def test_bounded_memory(self):
        from src.sacd.bocpd import BayesianOnlineChangepoint

        bocpd = BayesianOnlineChangepoint(max_run_length=100)
        rng = np.random.RandomState(42)

        for s in rng.normal(0.1, 0.02, 500):
            bocpd.update(s)

        assert len(bocpd.run_length_probs) <= 100


class TestCampaignMonitor:
    def test_clean_traffic_no_alert(self):
        from src.sacd.monitor import CampaignMonitor

        monitor = CampaignMonitor(
            cp_threshold=0.5,
            mu0=0.1, kappa0=1.0, alpha0=2.0, beta0=0.02,
            alert_run_length=5, alert_run_prob=0.5,
        )
        rng = np.random.RandomState(42)

        for s in rng.normal(0.1, 0.02, 200):
            state = monitor.process_score(s)

        assert not state['l0_active']
        assert len(monitor.alert_log) == 0

    def test_campaign_detected(self):
        from src.sacd.monitor import CampaignMonitor

        monitor = CampaignMonitor(
            cp_threshold=0.3, hazard_rate=1/50,
            mu0=0.1, kappa0=1.0, alpha0=2.0, beta0=0.02,
            alert_run_length=5, alert_run_prob=0.5,
        )
        rng = np.random.RandomState(42)

        # Clean phase
        for s in rng.normal(0.1, 0.02, 50):
            monitor.process_score(s)

        # Attack phase
        detected = False
        for s in rng.normal(0.5, 0.05, 30):
            state = monitor.process_score(s)
            if state['l0_active']:
                detected = True
                break

        assert detected, "Campaign was not detected"


# ============================================================
# Tests for Immune Memory
# ============================================================

class TestImmuneMemory:
    def test_store_and_match(self):
        from src.memory.immune_memory import ImmuneMemory

        mem = ImmuneMemory(match_threshold=0.5, comparison_dim=0)

        # Create a simple signature
        sig_dgm = [np.array([[0.0, 1.0], [0.5, 2.0]]),
                    np.array([[0.1, 0.5]])]
        mem.store(sig_dgm, attack_type='PGD', response_level='L3')

        # Query with same diagram — should match
        match = mem.match(sig_dgm)
        assert match is not None
        assert match['attack_type'] == 'PGD'

    def test_no_match_different_diagram(self):
        from src.memory.immune_memory import ImmuneMemory

        mem = ImmuneMemory(match_threshold=0.1, comparison_dim=0)

        sig_dgm = [np.array([[0.0, 1.0]]), np.array([[0.1, 0.5]])]
        mem.store(sig_dgm, attack_type='FGSM', response_level='L1')

        # Very different diagram
        query_dgm = [np.array([[10.0, 20.0], [15.0, 30.0]]),
                      np.array([[5.0, 10.0]])]
        match = mem.match(query_dgm)
        assert match is None

    def test_empty_memory_returns_none(self):
        from src.memory.immune_memory import ImmuneMemory

        mem = ImmuneMemory()
        assert mem.match([np.array([[0, 1]])]) is None


# ============================================================
# Tests for TDA Profiler
# ============================================================

class TestTopologicalProfiler:
    def test_compute_diagram_3d(self):
        from src.tamm.tda import TopologicalProfiler

        profiler = TopologicalProfiler(n_subsample=50, max_dim=1)
        # Simulate a (C=16, H=7, W=7) activation
        rng = np.random.RandomState(42)
        act = rng.randn(16, 7, 7).astype(np.float32)

        dgms = profiler.compute_diagram(act)
        assert len(dgms) >= 2  # H0 and H1
        assert dgms[0].shape[1] == 2  # (birth, death) pairs

    def test_compute_diagram_2d(self):
        from src.tamm.tda import TopologicalProfiler

        profiler = TopologicalProfiler(n_subsample=30, max_dim=1)
        rng = np.random.RandomState(42)
        act = rng.randn(50, 10).astype(np.float32)

        dgms = profiler.compute_diagram(act)
        assert len(dgms) >= 2

    def test_wasserstein_identical(self):
        from src.tamm.tda import TopologicalProfiler

        dgm = np.array([[0.0, 1.0], [0.5, 2.0]])
        d = TopologicalProfiler.wasserstein_dist(dgm, dgm)
        assert d == 0.0

    def test_wasserstein_empty(self):
        from src.tamm.tda import TopologicalProfiler

        d = TopologicalProfiler.wasserstein_dist(np.array([]), np.array([]))
        assert d == 0.0


# ============================================================
# Tests for Threshold Manager
# ============================================================

class TestThresholdManager:
    def test_default_actions(self):
        from src.cadg.threshold import TieredThresholdManager

        mgr = TieredThresholdManager()
        action = mgr.get_action('L2')
        assert action.should_purify is True
        assert action.should_log is True

    def test_pass_action(self):
        from src.cadg.threshold import TieredThresholdManager

        mgr = TieredThresholdManager()
        action = mgr.get_action('PASS')
        assert action.should_log is False
        assert action.should_purify is False

    def test_unknown_level_raises(self):
        from src.cadg.threshold import TieredThresholdManager

        mgr = TieredThresholdManager()
        with pytest.raises(KeyError):
            mgr.get_action('L99')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
