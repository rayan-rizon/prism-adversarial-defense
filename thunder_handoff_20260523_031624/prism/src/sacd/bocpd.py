"""
SACD: Bayesian Online Changepoint Detection (Adams & MacKay 2007)
Detects distribution shifts in the anomaly score stream.

Implementation maintains the LOG joint distribution log P(r_t, x_{1:t})
and uses logsumexp throughout — this correctly preserves the relative
predictive probabilities across run lengths and avoids the false-alarm
collapse that arises from shifting log-pred before exponentiation.

Predictive model: Normal-Inverse-Gamma conjugate prior, giving a
Student-t posterior predictive that is robust to small-sample variance
estimation (unlike a simple empirical-std Gaussian).

Key fix from plan: bounded observation buffer to prevent unbounded
memory growth on long inference streams.
"""
import numpy as np
from scipy.special import gammaln, logsumexp
from typing import Optional


class BayesianOnlineChangepoint:
    """
    Bayesian Online Changepoint Detection.

    Maintains log_R[r] = log P(run_length = r, x_{1:t}) (log joint, normalised
    at each step for numerical stability). Uses logsumexp throughout so that
    the relative predictive likelihoods across run lengths are never destroyed
    by an ad-hoc shift.

    Predictive: NIG conjugate prior → Student-t posterior predictive.
    """

    def __init__(self,
                 hazard_rate: float = 1 / 200,
                 mu0: float = 0.5,
                 kappa0: float = 1.0,
                 alpha0: float = 2.0,
                 beta0: float = 0.02,
                 max_run_length: int = 500):
        """
        Args:
            hazard_rate: Probability of a changepoint at any step (= 1/expected_run_len).
            mu0:    NIG prior mean.
            kappa0: NIG pseudo-observation count for the mean.
            alpha0: NIG shape for the Inverse-Gamma variance prior.
            beta0:  NIG scale for the Inverse-Gamma variance prior.
                    Prior predictive Student-t scale ≈ sqrt(beta0*(kappa0+1)/(alpha0*kappa0)).
            max_run_length: Truncate run-length vector to bound memory.
        """
        self.hazard = hazard_rate
        self.log_h = np.log(hazard_rate)
        self.log_1mh = np.log(1.0 - hazard_rate)
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.max_run_length = max_run_length

        # log_R[r] = log P(run_length = r, x_{1:t}), normalised log-posterior
        self.log_R = np.array([0.0])

        # Sufficient statistics per run-length slot
        self._sums = np.array([0.0])
        self._sum_sq = np.array([0.0])
        self._counts = np.array([0])

    # ------------------------------------------------------------------
    # Vectorised log posterior-predictive Student-t
    # ------------------------------------------------------------------
    def _log_pred(self,
                  x: float,
                  sums: np.ndarray,
                  sum_sq: np.ndarray,
                  counts: np.ndarray) -> np.ndarray:
        """
        Log p(x | x_{1:r}) under NIG prior for every run-length slot.
        Posterior predictive is Student-t(df=2*alpha_n, loc=mu_n, scale=scale_n).
        """
        n = counts.astype(float)
        kn = self.kappa0 + n
        an = self.alpha0 + 0.5 * n
        mn = (self.kappa0 * self.mu0 + sums) / kn

        bn = (self.beta0
              + 0.5 * sum_sq
              + 0.5 * self.kappa0 * self.mu0 ** 2
              - 0.5 * kn * mn ** 2)
        bn = np.maximum(bn, 1e-30)

        nu = 2.0 * an
        scale_sq = bn * (kn + 1.0) / (an * kn)
        scale_sq = np.maximum(scale_sq, 1e-30)
        z2 = (x - mn) ** 2 / scale_sq

        return (gammaln(0.5 * (nu + 1.0))
                - gammaln(0.5 * nu)
                - 0.5 * np.log(nu * np.pi * scale_sq)
                - 0.5 * (nu + 1.0) * np.log1p(z2 / nu))

    # ------------------------------------------------------------------
    # Core update — full log-space BOCPD
    # ------------------------------------------------------------------
    def update(self, score: float) -> float:
        """
        Process one new anomaly score.

        Returns:
            P(changepoint at this step) — normalised posterior mass at run_length=0.
        """
        T = len(self.log_R)

        # 1. Log predictive for each run-length slot (true log likelihoods, not shifted)
        lp = self._log_pred(score, self._sums, self._sum_sq, self._counts)

        # 2. Log growth: run continues (run_length increments)
        log_growth = self.log_R + lp + self.log_1mh          # shape (T,)

        # 3. Log changepoint: any run resets → new run of length 0
        log_cp = logsumexp(self.log_R + lp) + self.log_h     # scalar

        # 4. Assemble new log joint of shape (T+1,)
        new_log_R = np.empty(T + 1)
        new_log_R[0] = log_cp
        new_log_R[1:] = log_growth

        # 5. Normalise log-posterior for numerical stability
        log_norm = logsumexp(new_log_R)
        cp_prob = float(np.exp(new_log_R[0] - log_norm))
        new_log_R -= log_norm                                  # keep normalised

        # 6. Update sufficient statistics
        new_sums = np.empty(T + 1)
        new_sum_sq = np.empty(T + 1)
        new_counts = np.empty(T + 1, dtype=int)

        new_sums[1:] = self._sums + score
        new_sum_sq[1:] = self._sum_sq + score ** 2
        new_counts[1:] = self._counts + 1
        new_sums[0] = 0.0
        new_sum_sq[0] = 0.0
        new_counts[0] = 0

        # 7. Truncate for bounded memory (fold tail mass via logsumexp)
        if len(new_log_R) > self.max_run_length:
            tail = logsumexp(new_log_R[self.max_run_length - 1:])
            new_log_R = new_log_R[:self.max_run_length]
            new_log_R[-1] = tail
            new_sums = new_sums[:self.max_run_length]
            new_sum_sq = new_sum_sq[:self.max_run_length]
            new_counts = new_counts[:self.max_run_length]

        self.log_R = new_log_R
        self._sums = new_sums
        self._sum_sq = new_sum_sq
        self._counts = new_counts

        return cp_prob

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @property
    def run_length_probs(self) -> np.ndarray:
        """Posterior P(run_length | x_{1:t}) as a probability vector."""
        log_norm = logsumexp(self.log_R)
        return np.exp(self.log_R - log_norm)

    def get_most_likely_run_length(self) -> int:
        """Return MAP estimate of current run length."""
        return int(np.argmax(self.log_R))

    def reset(self):
        """Reset detector state."""
        self.log_R = np.array([0.0])
        self._sums = np.array([0.0])
        self._sum_sq = np.array([0.0])
        self._counts = np.array([0])
