"""
SACD: Sequential Adversarial Campaign Monitor (L0)
Monitors the stream of anomaly scores for sustained attack campaigns
using BOCPD, with a rolling score buffer for trend analysis.
"""
import os
import pickle
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Any
from .bocpd import BayesianOnlineChangepoint


def _load_thresholds(thresholds_path: Optional[str]) -> Dict[str, Any]:
    """Load a calibrated threshold dict if the file exists; else {}."""
    if not thresholds_path or not os.path.exists(thresholds_path):
        return {}
    with open(thresholds_path, 'rb') as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{thresholds_path}: expected dict, got {type(data)}")
    return data


class CampaignMonitor:
    """
    L0 Sequential Campaign Detector.
    Detects when an adversary transitions from probing to active attack
    by monitoring the distribution of anomaly scores over time.
    """

    def __init__(
        self,
        window_size: int = 100,
        cp_threshold: float = 0.3,
        hazard_rate: float = 1 / 30,
        mu0: float = 7.0,     # matches clean score mean for layer2/3/4 equal dim_weights
        kappa0: float = 5.0,
        alpha0: float = 3.0,
        beta0: float = 15.0,  # alpha0 * clean_var ≈ 3.0 * (2.25)^2, calibrated to clean std
        alert_run_length: int = 10,
        alert_run_prob: float = 0.60,  # raised from 0.45 to reduce clean-phase false alarms
        warmup_steps: int = 35,        # raised from 20 to let BOCPD stabilize on clean data
        l0_factor: float = 0.8,
        cooldown_steps: int = 50,
        thresholds_path: Optional[str] = None,
    ):
        """
        Args:
            window_size: Rolling buffer size for score statistics.
            cp_threshold: (Legacy) raw BOCPD cp_prob threshold. Not used for
                          primary detection — see alert_run_length/alert_run_prob.
            hazard_rate: Prior changepoint probability per step.
            mu0, kappa0, alpha0, beta0: NIG prior for BOCPD Student-t predictive.
            alert_run_length: Trigger L0 when P(run_length ≤ alert_run_length) rises.
                              A sudden collapse to short runs signals a distribution shift.
            alert_run_prob: Trigger when P(run_length ≤ alert_run_length) exceeds this.
            warmup_steps: Ignore triggers during the initial warm-up period.
            l0_factor: Threshold multiplier when L0 is active (lower = more sensitive).
            cooldown_steps: Steps after deactivation before L0 can re-trigger.
        """
        # Overlay calibrated thresholds if provided. `thresholds_path` takes
        # precedence over the constructor defaults for the keys it carries;
        # anything missing falls through to the kwarg values. Persisted fields
        # are populated by scripts/calibrate_l0_thresholds.py.
        tcal = _load_thresholds(thresholds_path)
        hazard_rate      = tcal.get('hazard_rate',      hazard_rate)
        alert_run_length = tcal.get('alert_run_length', alert_run_length)
        alert_run_prob   = tcal.get('alert_run_prob',   alert_run_prob)
        warmup_steps     = tcal.get('warmup_steps',     warmup_steps)
        mu0              = tcal.get('mu0',              mu0)
        kappa0           = tcal.get('kappa0',           kappa0)
        alpha0           = tcal.get('alpha0',           alpha0)
        beta0            = tcal.get('beta0',            beta0)
        self._thresholds_source = thresholds_path if tcal else None
        self._thresholds_calibration = tcal.get('calibration_metrics') if tcal else None

        self.window_size = window_size
        self.cp_threshold = cp_threshold
        self.alert_run_length = alert_run_length
        self.alert_run_prob = alert_run_prob
        self.warmup_steps = warmup_steps
        self.l0_factor = l0_factor
        self.cooldown_steps = cooldown_steps

        self.bocpd = BayesianOnlineChangepoint(
            hazard_rate=hazard_rate,
            mu0=mu0, kappa0=kappa0, alpha0=alpha0, beta0=beta0,
        )
        self.score_buffer: deque = deque(maxlen=window_size)

        self.l0_active: bool = False
        self.l0_start_step: Optional[int] = None
        self._step_counter: int = 0
        self._cooldown_until: int = 0

        # Bounded circular buffer — prevents unbounded growth on long runs.
        # 10000 entries cover any realistic evaluation or deployment window.
        self._alert_log_maxlen: int = 10_000
        self.alert_log: List[Dict[str, Any]] = []

    def reset(self) -> None:
        """Reset BOCPD state and all counters (use between experiments)."""
        self.bocpd.reset()
        self.score_buffer.clear()
        self.l0_active = False
        self.l0_start_step = None
        self._step_counter = 0
        self._cooldown_until = 0
        self.alert_log.clear()

    def process_score(self, score: float, timestamp: Optional[float] = None) -> Dict:
        """
        Process one anomaly score and update L0 state.

        Detection signal: P(run_length ≤ alert_run_length).
        After the BOCPD has accumulated a stable "long run" during clean traffic,
        any distribution shift causes the run-length distribution to suddenly
        collapse to short runs — this probability jumps from ~hazard_rate to ~1
        within a single step of a new distribution.

        Args:
            score: Anomaly score for this inference step.
            timestamp: Optional external timestamp for logging.
        Returns:
            Dict with L0 state info.
        """
        self._step_counter += 1
        self.score_buffer.append(score)

        # Run BOCPD — get cp_prob and run-length distribution
        cp_prob = self.bocpd.update(score)
        rl_probs = self.bocpd.run_length_probs
        # P(run_length <= alert_run_length)
        short_run_prob = float(np.sum(rl_probs[:self.alert_run_length + 1]))

        # Primary detection: short-run probability spike (after warm-up)
        detection_triggered = (
            short_run_prob > self.alert_run_prob
            and self._step_counter > self.warmup_steps
            and self._step_counter > self._cooldown_until
        )

        if detection_triggered and not self.l0_active:
            self.l0_active = True
            self.l0_start_step = self._step_counter
            if len(self.alert_log) >= self._alert_log_maxlen:
                self.alert_log.pop(0)  # drop oldest entry
            self.alert_log.append({
                'type': 'L0_ACTIVATED',
                'step': self._step_counter,
                'timestamp': timestamp,
                'cp_probability': cp_prob,
                'short_run_prob': short_run_prob,
                'map_run_length': self.bocpd.get_most_likely_run_length(),
                'buffer_mean': float(np.mean(self.score_buffer)),
                'buffer_std': float(np.std(self.score_buffer)),
                'recent_scores': list(self.score_buffer)[-20:],
            })

        buf = list(self.score_buffer)
        return {
            'l0_active': self.l0_active,
            'changepoint_prob': cp_prob,
            'short_run_prob': short_run_prob,
            'buffer_mean': float(np.mean(buf)) if buf else 0.0,
            'buffer_std': float(np.std(buf)) if buf else 0.0,
            'step': self._step_counter,
            'run_length': self.bocpd.get_most_likely_run_length(),
        }

    def deactivate_l0(self):
        """Manually deactivate L0 after the threat subsides."""
        if self.l0_active:
            if len(self.alert_log) >= self._alert_log_maxlen:
                self.alert_log.pop(0)
            self.alert_log.append({
                'type': 'L0_DEACTIVATED',
                'step': self._step_counter,
                'duration': self._step_counter - (self.l0_start_step or 0),
            })
        self.l0_active = False
        self.l0_start_step = None
        self._cooldown_until = self._step_counter + self.cooldown_steps
        self.bocpd.reset()

    def get_alert_log(self) -> List[Dict]:
        return list(self.alert_log)


class NoOpCampaignMonitor:
    """
    Disabled campaign monitor for evaluation experiments.
    L0 never activates — measures pure TAMM+CADG detection without L0 interference.
    Use in run_evaluation.py to get clean TPR/FPR metrics.
    The real CampaignMonitor is evaluated separately in campaign experiments.
    """
    alert_log: List[Dict] = []

    def process_score(self, score: float, timestamp=None) -> Dict:
        return {'l0_active': False, 'alert': False,
                'step': 0, 'short_run_prob': 0.0,
                'changepoint_prob': 0.0, 'buffer_mean': score,
                'buffer_std': 0.0, 'run_length': 0}

    def reset(self) -> None:
        pass

    def deactivate_l0(self) -> None:
        pass

    def get_alert_log(self) -> List[Dict]:
        return []
