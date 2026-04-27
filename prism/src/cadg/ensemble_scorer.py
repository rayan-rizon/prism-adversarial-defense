"""
CADG: Persistence-Ensemble Anomaly Scorer

Combines the raw Wasserstein anomaly score (effective for PGD/iterative attacks)
with a conformally-calibrated logistic score over a 36-dimensional persistence
statistics feature vector (effective for single-step FGSM and Square attacks).

Feature vector: 6 stats × 2 dims (H0, H1) × 3 layers = 36 features.
  Per (layer, dim): [wasserstein_dist, total_persistence, max_persistence,
                     n_features, entropy, mean_persistence]
Optional 37th feature: log high-frequency DCT energy (use_dct=True).
Optional 38th feature: softmax entropy of model logits (use_softmax_entropy=True).
Optional 39th feature: input-gradient L2 norm (use_grad_norm=True).

Architecture:
  score = α * wasserstein_score + (1-α) * logistic_score_centered

The logistic_score_centered is derived using a data-driven shift constant
(logit_shift) computed from clean training data.  Additive fusion ensures
that the logistic channel contributes independently even when the
Wasserstein channel is near-zero (e.g. CW-L2 attacks produce minimal
activation-space distortion).

Usage:
  See scripts/train_ensemble_scorer.py for offline training.
  At inference: ensemble_scorer.score(diagrams) -> float

The module is intentionally kept in the TDA/CADG domain (no deep NN training),
so it remains fully within the 'architecture-agnostic, TDA-based' framing.
"""
import logging
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..tamm.persistence_stats import extract_feature_vector, PersistenceDiagram
from ..tamm.scorer import TopologicalScorer

logger = logging.getLogger(__name__)


class PersistenceEnsembleScorer:
    """
    Enhanced anomaly scorer that fuses:
      1. Weighted Wasserstein distance (base TAMM scorer)
      2. Logistic regression over 36-dim persistence feature vector
         (6 statistics x 2 homology dims x 3 layers = 36)

    The logistic component learns to weight persistence statistics
    (total_persistence, entropy, n_features, etc.) that distinguish
    adversarially-perturbed activations from clean ones at small epsilon.

    Composite score formula (additive, data-derived):
      logit_centered = raw_logit - logit_shift   (centred on clean mean)
      score          = alpha * w_score + (1-alpha) * logit_centered

    Additive fusion keeps both channels independent: when the Wasserstein
    channel is near-zero (CW-L2 minimal perturbation), the logistic channel
    still contributes its full signal. The previous multiplicative coupling
    (logit_centered * w_norm) would crush the logistic to zero in that
    regime — the root cause of the CW TPR regression.

    logit_shift is computed from clean TRAINING data and stored in the pkl,
    ensuring reproducibility.

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
        use_dct: bool = False,
        use_softmax_entropy: bool = False,
        use_grad_norm: bool = False,
        use_tda: bool = True,
    ):
        """
        Args:
            base_scorer: Pre-built TopologicalScorer (Wasserstein component).
            layer_names: Layers to include in feature extraction.
            dims: Homology dimensions to include.
            alpha: Weight of Wasserstein component (1-alpha for logistic).
            logistic_weights: Fitted logistic regression weights.
            logistic_bias: Fitted logistic regression bias scalar.
            feature_mean: Feature normalisation mean from training data.
            feature_std: Feature normalisation std from training data.
            logit_shift: Mean raw logit value on clean training data (data-derived).
            w_score_mean: Mean Wasserstein score on clean training data (data-derived).
            training_eps: L-inf epsilon used to generate adversarials during training.
            training_attacks: List of attack names used during training.
            training_n: Total number of adversarial samples used in training.
            use_dct: If True, append DCT high-frequency energy as a feature.
            use_softmax_entropy: If True, append softmax entropy of model
                    logits as a feature.  Captures CW-L2 decision-boundary
                    proximity that TDA features cannot detect.  Adds < 1ms
                    latency (just a softmax + entropy over existing logits).
            use_grad_norm: If True, input-gradient L2 norm is appended as the final
                           feature. Adds 1 to n_features; grad_norm must be passed to
                           score() and extract_features() at inference.
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
        self.use_dct = use_dct
        self.use_softmax_entropy = use_softmax_entropy
        self.use_grad_norm = use_grad_norm
        # P0.6 ablation flag: when False, the 36-dim persistence-statistics block
        # is dropped from extract_features() and alpha is expected to be 0 so
        # the Wasserstein component does not contribute. score() returns the
        # logistic prob directly in that case.
        self.use_tda = use_tda

    @property
    def n_features(self) -> int:
        """Feature vector dimension: 6 stats × len(dims) × len(layer_names) [+1 DCT] [+1 grad_norm].
        When use_tda=False (P0.6 ablation), the persistence-stats block is
        dropped and only DCT/grad-norm features remain."""
        base = len(self.layer_names) * len(self.dims) * 6 if self.use_tda else 0
        return (base
                + (1 if self.use_dct else 0)
                + (1 if self.use_softmax_entropy else 0)
                + (1 if self.use_grad_norm else 0))

    def extract_features(
        self,
        diagrams: Dict[str, List[PersistenceDiagram]],
        image: Optional[np.ndarray] = None,
        grad_norm: Optional[float] = None,
        logits: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Extract feature vector. Length depends on use_tda/use_dct/use_softmax_entropy/use_grad_norm flags."""
        ref = self.base_scorer.ref_profiles
        full = extract_feature_vector(
            diagrams, ref, self.layer_names, self.dims,
            image=image,
            grad_norm=grad_norm if self.use_grad_norm else None,
            logits=logits if self.use_softmax_entropy else None,
        )
        if not self.use_tda:
            # P0.6: strip the 36 leading persistence-statistics features.
            n_tda = len(self.layer_names) * len(self.dims) * 6
            return full[n_tda:]
        return full

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

    def score(
        self,
        diagrams: Dict[str, List[PersistenceDiagram]],
        image: Optional[np.ndarray] = None,
        grad_norm: Optional[float] = None,
        logits: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute composite anomaly score.

        Uses additive fusion of Wasserstein and logistic channels with a
        data-derived logit_shift to centre the logistic on clean data.

        Additive fusion ensures the logistic channel contributes its full
        signal even when the Wasserstein channel is near-zero (CW-L2).

        Args:
            diagrams: Persistence diagrams keyed by layer name.
            image: Optional (C, H, W) float32 image (normalised). Required when
                   use_dct=True; if omitted with use_dct=True, falls back to
                   base Wasserstein score to avoid feature dimension mismatch.
            grad_norm: Optional pre-computed input-gradient L2 norm. Required when
                       use_grad_norm=True; if omitted, falls back to base score.
            logits: Optional 1-D array of raw model logits (pre-softmax).
                    Required when use_softmax_entropy=True; if omitted, falls
                    back to base Wasserstein score.
        Returns:
            Scalar — higher means more likely adversarial.
            Falls back to base Wasserstein score if logistic is not fitted.
        """
        w_score = self.base_scorer.score(diagrams) if self.use_tda else 0.0

        if not self._logistic_fitted:
            return w_score

        if self.use_dct and image is None:
            return w_score

        if self.use_grad_norm and grad_norm is None:
            return w_score

        if self.use_softmax_entropy and logits is None:
            return w_score

        feat = self.extract_features(diagrams, image=image, grad_norm=grad_norm, logits=logits)
        logit_prob = self._logistic_prob(feat)

        if not self.use_tda:
            # P0.6: alpha=0 and Wasserstein disabled. Return the centred logit
            # directly so downstream calibration can thresh on a raw score.
            logit_score = float(np.clip(logit_prob, 1e-6, 1 - 1e-6))
            logit_score = float(np.log(logit_score / (1 - logit_score)))
            return logit_score - self.logit_shift

        # Convert probability to raw logit (unbounded)
        logit_score = float(np.clip(logit_prob, 1e-6, 1 - 1e-6))
        logit_score = float(np.log(logit_score / (1 - logit_score)))

        # Centre logit by its mean on clean training data
        # For clean inputs: logit_score ≈ logit_shift → logit_centered ≈ 0
        # For adversarial inputs: logit_score >> logit_shift → logit_centered > 0
        logit_centered = logit_score - self.logit_shift

        # Additive fusion: both channels contribute independently.
        # When both are elevated (adversarial), the sum is high.
        # When Wasserstein is near-zero (CW-L2), the logistic still
        # contributes its full signal — fixing the CW detection gap.
        return self.alpha * w_score + (1.0 - self.alpha) * logit_centered

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
            clean_features: (N_clean, d) feature matrix from clean images.
            adv_features:   (N_adv, d) feature matrix from adversarial images.
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

        logger.info("Data-derived constants: logit_shift=%.4f, w_score_mean=%.4f",
                    self.logit_shift, self.w_score_mean)

        from sklearn.metrics import roc_auc_score
        probs = clf.predict_proba(X_norm)[:, 1]
        auc = roc_auc_score(y, probs)
        logger.info("Logistic ensemble — training AUC: %.4f", auc)
        logger.info("Features: %d-dim, n_clean=%d, n_adv=%d",
                    self.n_features, len(clean_features), len(adv_features))

    def composite_score_from_features(
        self,
        features: np.ndarray,
        w_scores: np.ndarray,
        alpha: Optional[float] = None,
    ) -> np.ndarray:
        """
        Batched recomputation of the inference-time composite score.

        Replays the same formula used at inference (see `score()`) for a matrix
        of pre-extracted feature vectors and their matching aggregate Wasserstein
        scores. Intended for alpha grid-search on a held-out training slice —
        avoids re-extracting persistence diagrams during hyperparameter tuning.

        Args:
            features: (N, d) pre-extracted feature vectors (same layout as
                      `extract_features` produces for this scorer).
            w_scores: (N,) aggregate Wasserstein composite scores (same layout
                      as `base_scorer.score(diagrams)` produces).
            alpha:    Optional override for self.alpha during scoring.
                      When None, uses self.alpha.

        Returns:
            (N,) float32 array of composite scores, one per input.
        """
        if not self._logistic_fitted:
            raise RuntimeError("composite_score_from_features: logistic not fitted.")
        a = self.alpha if alpha is None else alpha
        feats_norm = (features - self.feature_mean) / (self.feature_std + 1e-8)
        logits = feats_norm @ self.logistic_weights + self.logistic_bias
        logit_prob = 1.0 / (1.0 + np.exp(-logits))
        logit_prob = np.clip(logit_prob, 1e-6, 1 - 1e-6)
        logit_score = np.log(logit_prob / (1 - logit_prob))
        logit_centered = logit_score - self.logit_shift
        if not self.use_tda:
            # P0.6 ablation: pure logistic, alpha must be 0 at inference.
            return logit_centered.astype(np.float32)
        # Additive fusion — matches score() formula.
        return (a * w_scores + (1.0 - a) * logit_centered).astype(np.float32)

    def tune_alpha(
        self,
        clean_features: np.ndarray,
        adv_features: np.ndarray,
        clean_w_scores: np.ndarray,
        adv_w_scores: np.ndarray,
        grid: Tuple[float, ...] = (0.2, 0.35, 0.5, 0.65, 0.8),
    ) -> Dict[str, float]:
        """
        Pick alpha that maximises held-out composite-score AUC.

        Must be called *after* `fit_logistic` (the logistic weights and the
        data-derived normalisation constants need to already exist). Skips
        and returns a no-op summary when use_tda=False (P0.6 ablation), since
        alpha is meaningless when the Wasserstein head is disabled.

        Args:
            clean_features, adv_features: held-out feature matrices, MUST be
                disjoint from those passed to fit_logistic() to get an
                honest AUC estimate.
            clean_w_scores, adv_w_scores: aggregate Wasserstein scores for the
                same held-out rows.
            grid: alpha values to evaluate.

        Returns:
            {'selected_alpha', 'grid', 'aucs'} — selected alpha is also
            written to self.alpha as a side effect.
        """
        from sklearn.metrics import roc_auc_score

        if not self.use_tda:
            logger.info("tune_alpha: skipped (use_tda=False, alpha remains %.3f)", self.alpha)
            return {'selected_alpha': self.alpha, 'grid': list(grid), 'aucs': [], 'skipped': True}

        X = np.vstack([clean_features, adv_features])
        w = np.concatenate([clean_w_scores, adv_w_scores])
        y = np.array([0] * len(clean_features) + [1] * len(adv_features))

        aucs = []
        for a in grid:
            s = self.composite_score_from_features(X, w, alpha=a)
            aucs.append(float(roc_auc_score(y, s)))

        best_idx = int(np.argmax(aucs))
        self.alpha = float(grid[best_idx])
        logger.info("tune_alpha: selected α=%.3f (AUC=%.4f); grid AUCs: %s",
                    self.alpha, aucs[best_idx],
                    ", ".join(f"α={a:.2f}:{auc:.4f}" for a, auc in zip(grid, aucs)))
        return {
            'selected_alpha': self.alpha,
            'grid': list(grid),
            'aucs': aucs,
            'best_auc': aucs[best_idx],
            'skipped': False,
        }

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
            # P0.3 regression gate: sanity_checks.py::Check 6 asserts >= 2.5.
            'fgsm_oversample': getattr(self, 'fgsm_oversample', None),
            'no_tda_features': getattr(self, 'no_tda_features', False),
            # P0.6 lever: held-out AUC grid search picks α; summary records the
            # grid and per-α AUCs so downstream verification can reconstruct why
            # this α was selected.
            'alpha_tune_summary': getattr(self, 'alpha_tune_summary', None),
            # Feature engineering flags
            'use_dct': self.use_dct,
            'use_softmax_entropy': self.use_softmax_entropy,
            'use_grad_norm': self.use_grad_norm,
            'use_tda': self.use_tda,   # P0.6 ablation flag
            'n_features': self.n_features,   # @property value; serialised for verification
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        logger.info("PersistenceEnsembleScorer saved -> %s", path)

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
            use_dct=data.get('use_dct', False),
            use_softmax_entropy=data.get('use_softmax_entropy', False),
            use_grad_norm=data.get('use_grad_norm', False),
            use_tda=data.get('use_tda', True),
        )
        scorer._logistic_fitted = data.get('_logistic_fitted', False)
        return scorer
