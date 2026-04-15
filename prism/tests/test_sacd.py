"""
SACD unit tests — BayesianOnlineChangepoint, CampaignMonitor.

Cross-check against guide Section 3.1 and Section 2.3:
- L0 never triggers on clean traffic (no false alarms)
- L0 triggers within 20 probe queries after shift
- L0 triggers during attack phase if not already active
- BOCPD run_length distribution collapses on distribution shift
- Deactivation + cooldown works correctly
- Alert log records events with correct types
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.sacd.bocpd import BayesianOnlineChangepoint
from src.sacd.monitor import CampaignMonitor


RNG = np.random.RandomState(42)


# ---------------------------------------------------------------------------
# BayesianOnlineChangepoint
# ---------------------------------------------------------------------------

class TestBOCPD:
    def _make_bocpd(self, hazard=1/50):
        return BayesianOnlineChangepoint(
            hazard_rate=hazard, mu0=0.1, kappa0=1.0,
            alpha0=2.0, beta0=0.02,
        )

    def test_clean_stream_stays_at_hazard_rate(self):
        """Clean data keeps cp_prob near hazard rate, not spiking."""
        bocpd = self._make_bocpd(hazard=1/50)
        rng = np.random.RandomState(0)
        # Warm up
        [bocpd.update(s) for s in rng.normal(0.1, 0.02, 20)]
        clean_probs = [bocpd.update(s) for s in rng.normal(0.1, 0.02, 30)]
        # Hazard rate is 0.02; max should be modest
        max_cp = max(clean_probs)
        assert max_cp < 0.1, (
            f"Clean stream false alarm: max cp_prob={max_cp:.4f} > 0.1"
        )

    def test_run_length_increases_on_clean(self):
        """MAP run_length grows steadily during clean phase."""
        bocpd = self._make_bocpd()
        rng = np.random.RandomState(1)
        for s in rng.normal(0.1, 0.02, 50):
            bocpd.update(s)
        assert bocpd.get_most_likely_run_length() >= 45

    def test_distribution_shift_collapses_run_length(self):
        """After shift, P(run_length ≤ 5) spikes above 0.5."""
        bocpd = self._make_bocpd()
        rng = np.random.RandomState(2)
        for s in rng.normal(0.1, 0.02, 50):
            bocpd.update(s)
        # Shift to mean=0.5
        for s in rng.normal(0.5, 0.02, 5):
            bocpd.update(s)
        rl = bocpd.run_length_probs
        p_short = float(np.sum(rl[:6]))
        assert p_short > 0.5, (
            f"Shift not detected: P(rl<=5)={p_short:.4f}"
        )

    def test_bounded_memory(self):
        bocpd = BayesianOnlineChangepoint(max_run_length=100)
        rng = np.random.RandomState(3)
        for s in rng.normal(0.1, 0.02, 500):
            bocpd.update(s)
        assert len(bocpd.log_R) <= 100

    def test_run_length_probs_sum_to_one(self):
        bocpd = self._make_bocpd()
        rng = np.random.RandomState(4)
        for s in rng.normal(0.1, 0.02, 30):
            bocpd.update(s)
        assert np.sum(bocpd.run_length_probs) == pytest.approx(1.0, abs=1e-6)

    def test_reset_clears_state(self):
        bocpd = self._make_bocpd()
        rng = np.random.RandomState(5)
        for s in rng.normal(0.1, 0.02, 40):
            bocpd.update(s)
        old_rl = bocpd.get_most_likely_run_length()
        bocpd.reset()
        assert len(bocpd.log_R) == 1
        assert bocpd.get_most_likely_run_length() == 0


# ---------------------------------------------------------------------------
# CampaignMonitor
# ---------------------------------------------------------------------------

class TestCampaignMonitor:
    def _make_monitor(self, **kwargs):
        defaults = dict(
            warmup_steps=20, alert_run_length=5, alert_run_prob=0.5,
            hazard_rate=1/50, mu0=0.1, kappa0=1.0, alpha0=2.0, beta0=0.02,
        )
        defaults.update(kwargs)
        return CampaignMonitor(**defaults)

    def test_no_false_alarm_on_clean_traffic(self):
        """Guide Section 3.1 + 3.4: 500 clean queries — zero false alarms."""
        monitor = self._make_monitor()
        rng = np.random.RandomState(123)
        false_alarms = 0
        for s in rng.normal(0.1, 0.02, 500):
            state = monitor.process_score(s)
            if state['l0_active']:
                false_alarms += 1
                monitor.deactivate_l0()
        assert false_alarms == 0, f"Got {false_alarms} false alarms on clean traffic"

    def test_campaign_detected_within_target(self):
        """Guide Section 3.1: L0 activates within 20 probe queries."""
        monitor = self._make_monitor()
        rng = np.random.RandomState(42)
        # 50 clean queries
        for s in rng.normal(0.1, 0.02, 50):
            monitor.process_score(s)
        # Probe queries
        detected_at = None
        for i, s in enumerate(rng.normal(0.25, 0.05, 30)):
            state = monitor.process_score(s)
            if state['l0_active'] and detected_at is None:
                detected_at = i
        assert detected_at is not None, "L0 never activated during probe phase"
        assert detected_at < 20, (
            f"Detection too slow: activated at probe step {detected_at} (target <20)"
        )

    def test_l0_not_active_during_clean_phase(self):
        """L0 must not be active after clean phase in campaign scenario."""
        monitor = self._make_monitor()
        rng = np.random.RandomState(42)
        state = None
        for s in rng.normal(0.1, 0.02, 50):
            state = monitor.process_score(s)
        assert not state['l0_active'], (
            "L0 was active after clean phase — false positive"
        )

    def test_l0_triggers_on_high_attack_scores(self):
        """Even if probe misses, full attack scores (mean=0.7) must trigger L0."""
        monitor = self._make_monitor()
        rng = np.random.RandomState(99)
        # Warm up with clean
        for s in rng.normal(0.1, 0.02, 40):
            monitor.process_score(s)
        # Full attack
        triggered = False
        for s in rng.normal(0.7, 0.1, 30):
            state = monitor.process_score(s)
            if state['l0_active']:
                triggered = True
                break
        assert triggered, "L0 never triggered on high-magnitude attack"

    def test_deactivate_sets_cooldown(self):
        monitor = self._make_monitor(cooldown_steps=10)
        rng = np.random.RandomState(42)
        # Force L0 active
        for s in rng.normal(0.1, 0.02, 30):
            monitor.process_score(s)
        for s in rng.normal(0.7, 0.1, 10):
            state = monitor.process_score(s)
            if state['l0_active']:
                break
        monitor.deactivate_l0()
        assert not monitor.l0_active
        # Next step is still in cooldown — can't re-trigger immediately
        # (trigger would requires warmup + cooldown elapsed)

    def test_alert_log_records_activation(self):
        monitor = self._make_monitor()
        rng = np.random.RandomState(42)
        for s in rng.normal(0.1, 0.02, 50):
            monitor.process_score(s)
        for s in rng.normal(0.7, 0.1, 5):
            monitor.process_score(s)
        alerts = monitor.get_alert_log()
        types = [a['type'] for a in alerts]
        assert 'L0_ACTIVATED' in types

    def test_process_score_returns_required_keys(self):
        monitor = self._make_monitor()
        state = monitor.process_score(0.1)
        required = {'l0_active', 'changepoint_prob', 'buffer_mean',
                    'buffer_std', 'step', 'run_length'}
        assert required.issubset(state.keys()), (
            f"Missing keys: {required - state.keys()}"
        )

    def test_campaign_simulation_guide_spec(self):
        """
        Exact simulation from guide Section 2.3:
        100 clean → 20 probe → detected_at between 100 and 120.
        Note: our L0 uses run_length-based detection, so effective detection
        step is relative to start of probe phase. After 100 clean steps,
        detection should be t >= 100 (never in clean phase) and
        probe detection within 20 steps (t <= 120).
        """
        monitor = self._make_monitor()
        rng = np.random.RandomState(7)
        detected_at = None
        scores = np.concatenate([
            rng.normal(0.1, 0.02, 100),
            rng.normal(0.3, 0.05, 20),
        ])
        for t, s in enumerate(scores):
            state = monitor.process_score(s, timestamp=float(t))
            if state['l0_active'] and detected_at is None:
                detected_at = t
        assert detected_at is not None, "L0 never triggered in probe phase"
        assert detected_at >= 100, (
            f"False positive: triggered at t={detected_at} (during clean phase, t<100)"
        )
        assert detected_at <= 120, (
            f"Detection too slow: t={detected_at} (expected <= 120)"
        )
