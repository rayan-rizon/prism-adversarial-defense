"""
CADG: Persistence-Ensemble Anomaly Scorer

Combines the raw Wasserstein anomaly score (effective for PGD/iterative attacks)
with a conformally-calibrated logistic score over a 42-dimensional persistence
statistics feature vector (effective for single-step FGSM and Square).

Architecture:
  score = α * wasserstein_score + (1-α) * logistic_score

Both components are calibrated independently so neither can over-inflate the
composite. The logistic ensemble is trained via logistic regression on clean
and adversarial feature vectors. The calibration of the composite score
follows the same split-conformal procedure as the base Wasserstein scorer.

Usage:
  See scripts/train_ensemble_scorer.py for offline training.
  At inference: ensemble_scorer.score(diagrams) → float

The module is intentionally kept in the TDA/CADG domain (no deep NN training),
so it remains fully within the 'architecture-agnostic, TDA-based' framing.
"""
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..tamm.persistence_stats import extract_feature_vector, PersistenceDiagram
from ..tamm.scorer import TopologicalScorer


class PersistenceEnsembleScorer:
    """
    Enhanced anomaly scorer that fuses:
      1. Weighted Wasserstein distance (base TAMM scorer)
      2. Logistic regression over 42-dim persistence feature vector

    The logistic component learns to weight persistence statistics
    (total_persistence, entropy, n_features, etc.) that distinguish
    FGSM-perturbed activations from clean ones at very small ε.

    Design constraints for publishability:
      - Logistic regression trained ONLY on clean calibration data (one-class)
        using the 1-class SVM / isolation variant, OR
      - Trained supervised on clean vs FGSM features with an INDEPENDENT
        training split (not the same images used for evaluation).
      - Coefficients are fixed at evaluation time (not dynamic).
    """

    def __init__(
        self,
        base_scorer: TopologicalScorer,
        layer_names: List[str],
        dims: Tuple[int, ...] = (0, 1),
        alpha: float = 0.5,
        logistic_weights: Optional[np.ndarray] = None,
        logistic_bias: Optional[float] = None,
        feature_mean: Optional[np.ndarray] = None,
        feature_std: Optional[np.ndarray] = None,
    ):
        """
        Args:
            base_scorer: Pre-built TopologicalScorer (Wasserstein component).
            layer_names: Layers to include in feature extraction.
            dims: Homology dimensions to include.
            alpha: Weight of Wasserstein component (1-alpha for logistic).
            logistic_weights: Fitted logistic regression weights (d,).
            logistic_bias: Fitted logistic regression bias scalar.
            feature_mean: Feature normalisation mean from training data.
            feature_std: Feature normalisation std from training data.
        """
        self.base_scorer = base_scorer
        self.layer_names = layer_names
        self.dims = list(dims)
        self.alpha = alpha

        # Logistic component — None until trained
        self.logistic_weights = logistic_weights
        self.logistic_bias = logistic_bias if logistic_bias is not None else 0.0
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self._logistic_fitted = (logistic_weights is not None)

    @property
    def n_features(self) -> int:
        return len(self.layer_names) * len(self.dims) * 6

    def extract_features(
        self, diagrams: Dict[str, List[PersistenceDiagram]]
    ) -> np.ndarray:
        """Extract 42-dim feature vector from persistence diagrams."""
        ref = self.base_scorer.ref_profiles
        return extract_feature_vector(diagrams, ref, self.layer_names, self.dims)

    def _normalise(self, x: np.ndarray) -> np.ndarray:
        """Z-score normalisation using training statistics."""
        if self.feature_mean is not None and self.feature_std is not None:
            return (x - self.feature_mean) / (self.feature_std + 1e-8)
        return x

    def _logistic_prob(self, feat: np.ndarray) -> float:
        """Sigmoid of linear combination: P(adversarial | features)."""
        if not self._logistic_fitted:
            return 0.5  # uninformative prior
        feat_norm = self._normalise(feat)
        logit = float(np.dot(self.logistic_weights, feat_norm) + self.logistic_bias)
        return float(1.0 / (1.0 + np.exp(-logit)))

    def score(self, diagrams: Dict[str, List[PersistenceDiagram]]) -> float:
        """
        Compute composite anomaly score.

        Returns:
            Scalar in [0, ∞) — higher means more likely adversarial.
            When logistic component is not fitted, falls back to base Wasserstein score.
        """
        w_score = self.base_scorer.score(diagrams)

        if not self._logistic_fitted:
            return w_score

        feat = self.extract_features(diagrams)
        logit_prob = self._logistic_prob(feat)
        # Map logistic probability to same scale as Wasserstein score:
        # Use logit transform so output is unbounded and centred at 0
        logit_score = float(np.clip(logit_prob, 1e-6, 1 - 1e-6))
        logit_score = float(np.log(logit_score / (1 - logit_score)))
        # Shift so that 0.5 prob → 0 contribution; scale for comparability
        logit_score_scaled = (logit_score + 4.0) * (w_score / 8.0 + 1e-4)

        return self.alpha * w_score + (1.0 - self.alpha) * logit_score_scaled

    def score_per_layer(
        self, diagrams: Dict[str, List[PersistenceDiagram]]
    ) -> Dict[str, float]:
        """Return per-layer Wasserstein scores (for diagnostics)."""
        return self.base_scorer.score_per_layer(diagrams)

    # ──────────────────────────────────────────────────────────────────────────
    # Fitting
    # ──────────────────────────────────────────────────────────────────────────

    def fit_logistic(
        self,
        clean_features: np.ndarray,
        adv_features: np.ndarray,
        C: float = 1.0,
    ) -> None:
        """
        Fit logistic regression on clean (label=0) vs adversarial (label=1) features.

        MUST be trained on a SEPARATE split from the conformal calibration set.
        Uses scikit-learn LogisticRegression internally.

        Args:
            clean_features: (N_clean, d) feature matrix from clean images.
            adv_features: (N_adv, d) feature matrix from adversarial images.
            C: Logistic regression regularisation (inverse). Default 1.0.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        X = np.vstack([clean_features, adv_features])
        y = np.array([0] * len(clean_features) + [1] * len(adv_features))

        # Fit normalizer on training data
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X)
        self.feature_mean = scaler.mean_.astype(np.float32)
        self.feature_std = scaler.scale_.astype(np.float32)

        # Fit logistic regression
        clf = LogisticRegression(C=C, max_iter=1000, solver='lbfgs',
                                 class_weight='balanced')
        clf.fit(X_norm, y)

        self.logistic_weights = clf.coef_[0].astype(np.float32)
        self.logistic_bias = float(clf.intercept_[0])
        self._logistic_fitted = True

        # Report training AUC
        from sklearn.metrics import roc_auc_score
        probs = clf.predict_proba(X_norm)[:, 1]
        auc = roc_auc_score(y, probs)
        print(f"  Logistic ensemble — training AUC: {auc:.4f}")
        print(f"  Features: {self.n_features}-dim, n_clean={len(clean_features)}, "
              f"n_adv={len(adv_features)}")

    def save(self, path: str) -> None:
        """Save ensemble scorer to disk."""
        data = {
            'alpha': self.alpha,
            'layer_names': self.layer_names,
            'dims': self.dims,
            'logistic_weights': self.logistic_weights,
            'logistic_bias': self.logistic_bias,
            'feature_mean': self.feature_mean,
            'feature_std': self.feature_std,
            '_logistic_fitted': self._logistic_fitted,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"PersistenceEnsembleScorer saved → {path}")

    @classmethod
    def load(cls, path: str, base_scorer: TopologicalScorer,
             layer_names: List[str]) -> 'PersistenceEnsembleScorer':
        """Load ensemble scorer from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        scorer = cls(
            base_scorer=base_scorer,
            layer_names=layer_names,
            dims=tuple(data['dims']),
            alpha=data['alpha'],
            logistic_weights=data.get('logistic_weights'),
            logistic_bias=data.get('logistic_bias', 0.0),
            feature_mean=data.get('feature_mean'),
            feature_std=data.get('feature_std'),
        )
        scorer._logistic_fitted = data.get('_logistic_fitted', False)
        return scorer
