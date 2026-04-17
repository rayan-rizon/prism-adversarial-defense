"""
CADG: Persistence-Ensemble Anomaly Scorer

Combines the raw Wasserstein anomaly score (effective for PGD/iterative attacks)
with a conformally-calibrated logistic score over a 36-dimensional persistence
statistics feature vector (effective for single-step FGSM and Square attacks).

Feature vector: 6 stats × 2 dims (H0, H1) × 3 layers = 36 features.
  Per (layer, dim): [wasserstein_dist, total_persistence, max_persistence,
                     n_features, entropy, mean_persistence]

Architecture:
  score = α * wasserstein_score + (1-α) * logistic_score_scaled

The logistic_score_scaled is derived using data-driven normalisation constants
(logit_shift, w_score_mean) computed from clean training data, making the
composite score robust to dataset/model distribution shifts.

Usage:
  See scripts/train_ensemble_scorer.py for offline training.
  At inference: ensemble_scorer.score(diagrams) -> float

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
      2. Logistic regression over 36-dim persistence feature vector
         (6 statistics x 2 homology dims x 3 layers = 36)

    The logistic component learns to weight persistence statistics
    (total_persistence, entropy, n_features, etc.) that distinguish
    adversarially-perturbed activations from clean ones at small epsilon.

    Composite score formula (data-derived, no magic numbers):
      logit_centered = raw_logit - logit_shift   (centred on clean mean)
      w_norm         = w_score / w_score_mean     (normalised by clean mean)
      logit_scaled   = logit_centered * (w_norm + 1e-4)
      score          = alpha * w_score + (1-alpha) * logit_scaled

    logit_shift and w_score_mean are computed from clean TRAINING data
    and stored in the pkl, ensuring reproducibility.

    Design constraints for publishability:
      - Trained supervised on clean vs adversarial features with an INDEPENDENT
        training split (CIFAR-10 training set, not the test evaluation split).
      - Coefficients are fixed at evaluation time (not dynamic).
      - Regularisation C=1.0 selected by held-out 20% AUC on training data.
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
        logit_shift: float = 0.0,
        w_score_mean: float = 1.0,
        training_eps: Optional[float] = None,
        training_attacks: Optional[List[str]] = None,
        training_n: Optional[int] = None,
    ):
        """
        Args:
            base_scorer: Pre-built TopologicalScorer (Wasserstein component).
            layer_names: Layers to include in feature extraction.
            dims: Homology dimensions to include.
            alpha: Weight of Wasserstein component (1-alpha for logistic).
            logistic_weights: Fitted logistic regression weights (36,).
            logistic_bias: Fitted logistic regression bias scalar.
            feature_mean: Feature normalisation mean from training data (36,).
            feature_std: Feature normalisation std from training data (36,).
            logit_shift: Mean raw logit value on clean training data (data-derived).
            w_score_mean: Mean Wasserstein score on clean training data (data-derived).
            training_eps: L-inf epsilon used to generate adversarials during training.
            training_attacks: List of attack names used during training.
            training_n: Total number of adversarial samples used in training.
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

        # Data-derived normalisation constants (set during fit_logistic)
        self.logit_shift = logit_shift    # mean clean logit; centers logit_centered at 0
        self.w_score_mean = w_score_mean  # mean clean Wasserstein score for normalisation

        # Training provenance metadata
        self.training_eps = training_eps
        self.training_attacks = training_attacks or []
        self.training_n = training_n

    @property
    def n_features(self) -> int:
        """Feature vector dimension: 6 stats x len(dims) x len(layer_names)."""
        return len(self.layer_names) * len(self.dims) * 6

    def extract_features(
        self, diagrams: Dict[str, List[PersistenceDiagram]]
    ) -> np.ndarray:
        """Extract 36-dim feature vector from persistence diagrams."""
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

        Uses data-derived normalisation (logit_shift, w_score_mean) instead of
        hard-coded magic numbers, making the formula robust and reproducible.

        Returns:
            Scalar — higher means more likely adversarial.
            Falls back to base Wasserstein score if logistic is not fitted.
        """
        w_score = self.base_scorer.score(diagrams)

        if not self._logistic_fitted:
            return w_score

        feat = self.extract_features(diagrams)
        logit_prob = self._logistic_prob(feat)

        # Convert probability to raw logit (unbounded)
        logit_score = float(np.clip(logit_prob, 1e-6, 1 - 1e-6))
        logit_score = float(np.log(logit_score / (1 - logit_score)))

        # Centre logit by its mean on clean training data
        # For clean inputs: logit_score ≈ logit_shift → logit_centered ≈ 0
        # For adversarial inputs: logit_score >> logit_shift → logit_centered > 0
        logit_centered = logit_score - self.logit_shift

        # Normalise Wasserstein by its mean on clean training data
        w_norm = w_score / max(self.w_score_mean, 1e-4)

        # Multiplicative coupling: both components reinforce each other
        # when the input is adversarial (both logit and Wasserstein increase)
        logit_score_scaled = logit_centered * (w_norm + 1e-4)

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
        clean_w_scores: Optional[np.ndarray] = None,
    ) -> None:
        """
        Fit logistic regression on clean (label=0) vs adversarial (label=1) features.

        Also computes data-derived normalisation constants:
          - logit_shift: mean logit on clean training features (to centre the logit)
          - w_score_mean: mean Wasserstein score on clean training data

        MUST be trained on a SEPARATE split from the conformal calibration set.
        Uses scikit-learn LogisticRegression with L2 regularisation (C=1.0).

        Args:
            clean_features: (N_clean, 36) feature matrix from clean images.
            adv_features:   (N_adv, 36) feature matrix from adversarial images.
            C: Logistic regression regularisation (inverse). Default 1.0.
               Selected by held-out 20% AUC on training data.
            clean_w_scores: (N_clean,) base Wasserstein scores for clean images.
                            Used to compute w_score_mean. If None, estimated
                            from feature vector column 0 of each 6-feature block.
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

        # ── Compute data-derived normalisation constants from clean training data ──
        clean_norm = scaler.transform(clean_features)
        # Raw logits for clean samples
        logits_clean = clean_norm @ clf.coef_[0] + clf.intercept_[0]
        self.logit_shift = float(np.mean(logits_clean))

        # Mean Wasserstein score on clean training data
        if clean_w_scores is not None and len(clean_w_scores) > 0:
            self.w_score_mean = float(np.mean(clean_w_scores))
        else:
            # Fallback: Wasserstein is feature index 0 in each 6-feat block
            n_blocks = len(self.layer_names) * len(self.dims)
            w_indices = [i * 6 for i in range(n_blocks)]
            self.w_score_mean = float(np.mean(clean_features[:, w_indices]))

        print(f"  Data-derived constants: logit_shift={self.logit_shift:.4f}, "
              f"w_score_mean={self.w_score_mean:.4f}")

        # Report training AUC
        from sklearn.metrics import roc_auc_score
        probs = clf.predict_proba(X_norm)[:, 1]
        auc = roc_auc_score(y, probs)
        print(f"  Logistic ensemble — training AUC: {auc:.4f}")
        print(f"  Features: {self.n_features}-dim, n_clean={len(clean_features)}, "
              f"n_adv={len(adv_features)}")

    def save(self, path: str) -> None:
        """Save ensemble scorer to disk (includes provenance metadata)."""
        data = {
            'alpha': self.alpha,
            'layer_names': self.layer_names,
            'dims': self.dims,
            'logistic_weights': self.logistic_weights,
            'logistic_bias': self.logistic_bias,
            'feature_mean': self.feature_mean,
            'feature_std': self.feature_std,
            '_logistic_fitted': self._logistic_fitted,
            # Data-derived normalisation constants
            'logit_shift': self.logit_shift,
            'w_score_mean': self.w_score_mean,
            # Training provenance
            'training_eps': self.training_eps,
            'training_attacks': self.training_attacks,
            'training_n': self.training_n,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"PersistenceEnsembleScorer saved -> {path}")

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
            logit_shift=data.get('logit_shift', 0.0),
            w_score_mean=data.get('w_score_mean', 1.0),
            training_eps=data.get('training_eps'),
            training_attacks=data.get('training_attacks'),
            training_n=data.get('training_n'),
        )
        scorer._logistic_fitted = data.get('_logistic_fitted', False)
        return scorer
