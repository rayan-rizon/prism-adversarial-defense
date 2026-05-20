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
Optional logit-profile block: eight label-free confidence and margin summary
    features for FGSM/Square lower-tail recovery.
Optional stability block: deterministic logit consistency under cheap pixel
    transforms (use_stability_features=True). Legacy artifacts use 4 features;
    pixel-stability-v2 uses 8 features for FGSM/Square lower-tail recovery.
Optional side-channel quadratic expansion: appends pairwise products of the
    non-TDA side channels inside the stored linear heads. The raw feature
    contract remains unchanged; only the scorer's fitted model input expands.
Optional final feature: input-gradient L2 norm (use_grad_norm=True) for the
    promoted 55-feature arm. Keep it opt-in so legacy no-grad-norm artifacts
    can still be reproduced.

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
from typing import Any, Dict, List, Optional, Tuple

from ..tamm.logit_stability import DEFAULT_STABILITY_FEATURE_COUNT
from ..tamm.persistence_stats import (
    LOGIT_PROFILE_FEATURE_COUNT,
    extract_feature_vector,
    PersistenceDiagram,
)
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
        active-dataset training split, not the test evaluation split.
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
        use_logit_profile_features: bool = False,
        logit_profile_feature_count: int = LOGIT_PROFILE_FEATURE_COUNT,
        use_stability_features: bool = False,
        stability_feature_count: int = DEFAULT_STABILITY_FEATURE_COUNT,
        use_grad_norm: bool = False,
        use_tda: bool = True,
        feature_space_version: str = 'pixel-v1',
        selection_objective: str = 'auc',
        per_attack_validation_metrics: Optional[Dict[str, Any]] = None,
        training_attack_counts: Optional[Dict[str, int]] = None,
        attack_heads: Optional[Dict[str, Dict[str, Any]]] = None,
        attack_head_mode: str = 'off',
        score_channel_calibration: Optional[Dict[str, Any]] = None,
        score_channel_aggregation: str = 'max',
        use_side_quadratic_features: bool = False,
        quadratic_feature_start: int = 36,
        logistic_input_dim: Optional[int] = None,
        balanced_attacks: bool = False,
        pgd_train_steps: Optional[int] = None,
        aa_train_mode: Optional[str] = None,
        gradient_head_enabled: Optional[bool] = None,
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
            use_logit_profile_features: If True, append the 8-feature
                    confidence/margin/energy block used by the current
                    logitprofile+sidequad candidate.
            use_stability_features: If True, append deterministic
                    transform-consistency features computed from logits before
                    and after fixed pixel transforms.
            stability_feature_count: Number of stability features expected by
                    this artifact. Legacy pixel-v1 artifacts use 4; the current
                    pixel-stability-v2 candidate uses 8.
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
        self.use_logit_profile_features = bool(use_logit_profile_features)
        self.logit_profile_feature_count = int(logit_profile_feature_count or 0)
        self.use_stability_features = use_stability_features
        self.stability_feature_count = int(stability_feature_count or 0)
        self.use_grad_norm = use_grad_norm
        self.feature_space_version = feature_space_version
        self.selection_objective = selection_objective
        self.per_attack_validation_metrics = per_attack_validation_metrics or {}
        self.training_attack_counts = training_attack_counts or {}
        self.attack_heads = attack_heads or {}
        self.attack_head_mode = attack_head_mode
        self.score_channel_calibration = score_channel_calibration or {}
        self.score_channel_aggregation = score_channel_aggregation
        self.use_side_quadratic_features = bool(use_side_quadratic_features)
        self.quadratic_feature_start = int(quadratic_feature_start)
        self.logistic_input_dim = logistic_input_dim
        self.balanced_attacks = balanced_attacks
        self.pgd_train_steps = pgd_train_steps
        self.aa_train_mode = aa_train_mode
        self.gradient_head_enabled = (
            use_grad_norm if gradient_head_enabled is None else gradient_head_enabled
        )
        # P0.6 ablation flag: when False, the 36-dim persistence-statistics block
        # is dropped from extract_features() and alpha is expected to be 0 so
        # the Wasserstein component does not contribute. score() returns the
        # logistic prob directly in that case.
        self.use_tda = use_tda

    @property
    def n_features(self) -> int:
        """Feature vector dimension: 6 stats x dims x layers plus enabled side-channel features.
        When use_tda=False (P0.6 ablation), the persistence-stats block is
        dropped and only DCT/logit/stability/grad-norm features remain."""
        base = len(self.layer_names) * len(self.dims) * 6 if self.use_tda else 0
        return (base
                + (1 if self.use_dct else 0)
                + (1 if self.use_softmax_entropy else 0)
                + (self.logit_profile_feature_count if self.use_logit_profile_features else 0)
                + (self.stability_feature_count if self.use_stability_features else 0)
                + (1 if self.use_grad_norm else 0))

    def extract_features(
        self,
        diagrams: Dict[str, List[PersistenceDiagram]],
        image: Optional[np.ndarray] = None,
        grad_norm: Optional[float] = None,
        logits: Optional[np.ndarray] = None,
        logit_profile_features: Optional[np.ndarray] = None,
        stability_features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Extract feature vector. Length depends on the configured feature flags."""
        ref = self.base_scorer.ref_profiles
        full = extract_feature_vector(
            diagrams, ref, self.layer_names, self.dims,
            image=image,
            grad_norm=grad_norm if self.use_grad_norm else None,
            logits=logits if self.use_softmax_entropy else None,
            logit_profile_features=(
                logit_profile_features if self.use_logit_profile_features else None
            ),
            stability_features=(
                stability_features if self.use_stability_features else None
            ),
        )
        if not self.use_tda:
            # P0.6: strip the 36 leading persistence-statistics features.
            n_tda = len(self.layer_names) * len(self.dims) * 6
            return full[n_tda:]
        return full

    def _normalise(self, x: np.ndarray) -> np.ndarray:
        """Z-score normalisation using training statistics."""
        if self.feature_mean is not None and self.feature_std is not None:
            model_x = self._model_feature_matrix(x)
            return (model_x - self.feature_mean) / (self.feature_std + 1e-8)
        return self._model_feature_matrix(x)

    @staticmethod
    def _as_feature_matrix(features: np.ndarray) -> Tuple[np.ndarray, bool]:
        feats = np.asarray(features, dtype=np.float32)
        was_1d = feats.ndim == 1
        if was_1d:
            feats = feats.reshape(1, -1)
        if feats.ndim != 2:
            raise ValueError(
                f"Expected a 1-D or 2-D feature array, got shape {feats.shape}"
            )
        return feats, was_1d

    def _model_feature_matrix(
        self,
        features: np.ndarray,
        use_side_quadratic_features: Optional[bool] = None,
        quadratic_feature_start: Optional[int] = None,
    ) -> np.ndarray:
        """
        Return the stored model input matrix for raw PRISM features.

        The external feature contract remains `n_features`. When enabled, this
        appends pairwise products of the non-TDA side channels after raw
        extraction, giving the linear global and specialist heads an explicit
        interaction basis without changing persistence extraction or
        conformal calibration.
        """
        feats, was_1d = self._as_feature_matrix(features)
        enabled = (
            self.use_side_quadratic_features
            if use_side_quadratic_features is None
            else bool(use_side_quadratic_features)
        )
        if not enabled:
            return feats[0] if was_1d else feats

        start = (
            self.quadratic_feature_start
            if quadratic_feature_start is None
            else int(quadratic_feature_start)
        )
        start = min(max(start, 0), feats.shape[1])
        side = feats[:, start:]
        if side.shape[1] == 0:
            return feats[0] if was_1d else feats

        tri = np.triu_indices(side.shape[1])
        quad = (side[:, tri[0]] * side[:, tri[1]]).astype(np.float32)
        expanded = np.concatenate([feats, quad], axis=1).astype(np.float32)
        return expanded[0] if was_1d else expanded

    def _logistic_prob(self, feat: np.ndarray) -> float:
        """Sigmoid of linear combination: P(adversarial | features)."""
        logit = self._logistic_raw_logit(feat)
        return float(1.0 / (1.0 + np.exp(-logit)))

    def _logistic_raw_logit(self, feat: np.ndarray) -> float:
        """Raw linear logit before sigmoid."""
        if not self._logistic_fitted:
            return 0.0
        feat_norm = self._normalise(feat)
        return float(np.dot(self.logistic_weights, feat_norm) + self.logistic_bias)

    def score_components(
        self,
        diagrams: Dict[str, List[PersistenceDiagram]],
        image: Optional[np.ndarray] = None,
        grad_norm: Optional[float] = None,
        logits: Optional[np.ndarray] = None,
        logit_profile_features: Optional[np.ndarray] = None,
        stability_features: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Return score internals for audits without changing inference behavior.
        """
        w_score = self.base_scorer.score(diagrams) if self.use_tda else 0.0
        components: Dict[str, Any] = {
            'score': float(w_score),
            'w_score': float(w_score),
            'feature_space_version': self.feature_space_version,
            'fallback': False,
        }

        if not self._logistic_fitted:
            components['fallback'] = True
            components['fallback_reason'] = 'logistic_not_fitted'
            return components
        if self.use_dct and image is None:
            components['fallback'] = True
            components['fallback_reason'] = 'missing_dct_image'
            return components
        if self.use_grad_norm and grad_norm is None:
            components['fallback'] = True
            components['fallback_reason'] = 'missing_grad_norm'
            return components
        if self.use_softmax_entropy and logits is None:
            components['fallback'] = True
            components['fallback_reason'] = 'missing_logits'
            return components
        if self.use_logit_profile_features and logit_profile_features is None:
            components['fallback'] = True
            components['fallback_reason'] = 'missing_logit_profile_features'
            return components
        if self.use_stability_features and stability_features is None:
            components['fallback'] = True
            components['fallback_reason'] = 'missing_stability_features'
            return components

        feat = self.extract_features(
            diagrams, image=image, grad_norm=grad_norm, logits=logits,
            logit_profile_features=logit_profile_features,
            stability_features=stability_features,
        )
        raw_logit = self._logistic_raw_logit(feat)
        logit_prob = float(1.0 / (1.0 + np.exp(-raw_logit)))
        logit_centered = raw_logit - self.logit_shift
        if self.use_tda:
            score = self.alpha * w_score + (1.0 - self.alpha) * logit_centered
        else:
            score = logit_centered

        legacy_score = float(score)
        attack_head_scores: Dict[str, float] = {}
        if (
            self.attack_heads
            and self.attack_head_mode == 'calibrated_max'
            and self.score_channel_calibration
        ):
            raw_channels: Dict[str, float] = {'global': legacy_score}
            for attack, head in self.attack_heads.items():
                raw_channels[attack] = float(self._attack_head_score(head, feat, w_score))
            raw_channels['raw_max'] = max(raw_channels.values())
            evidence_scores = self._calibrated_channel_evidence(raw_channels)
            if evidence_scores:
                best_attack, _best_score = max(evidence_scores.items(), key=lambda kv: kv[1])
                score = self._aggregate_channel_evidence(evidence_scores)
                components['attack_head_winner'] = best_attack
                components['score_channel_aggregation'] = self.score_channel_aggregation
            attack_head_scores = {
                attack: raw_channels[attack]
                for attack in self.attack_heads.keys()
                if attack in raw_channels
            }
            components['raw_channel_scores'] = raw_channels
            components['channel_evidence_scores'] = evidence_scores
        elif self.attack_heads and self.attack_head_mode == 'max':
            for attack, head in self.attack_heads.items():
                attack_head_scores[attack] = float(
                    self._attack_head_score(head, feat, w_score)
                )
            if attack_head_scores:
                best_attack, best_score = max(
                    attack_head_scores.items(), key=lambda kv: kv[1]
                )
                if best_score > score:
                    score = best_score
                    components['attack_head_winner'] = best_attack

        components.update({
            'score': float(score),
            'legacy_score': legacy_score,
            'features': feat,
            'raw_logit': float(raw_logit),
            'logit_prob': float(logit_prob),
            'logit_centered': float(logit_centered),
            'attack_head_scores': attack_head_scores,
        })
        return components

    def score(
        self,
        diagrams: Dict[str, List[PersistenceDiagram]],
        image: Optional[np.ndarray] = None,
        grad_norm: Optional[float] = None,
        logits: Optional[np.ndarray] = None,
        logit_profile_features: Optional[np.ndarray] = None,
        stability_features: Optional[np.ndarray] = None,
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
        return float(self.score_components(
            diagrams, image=image, grad_norm=grad_norm, logits=logits,
            logit_profile_features=logit_profile_features,
            stability_features=stability_features,
        )['score'])

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

        X_raw = np.vstack([clean_features, adv_features])
        X = self._model_feature_matrix(X_raw)
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
        self.logistic_input_dim = int(self.logistic_weights.shape[0])
        self._logistic_fitted = True

        # ── Compute data-derived normalisation constants from clean training data ──
        clean_norm = scaler.transform(self._model_feature_matrix(clean_features))
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
        logger.info("Features: raw=%d-dim, model=%d-dim, n_clean=%d, n_adv=%d",
                    self.n_features, self.logistic_input_dim,
                    len(clean_features), len(adv_features))

    def _head_logits(self, head: Dict[str, Any], features: np.ndarray) -> np.ndarray:
        feats = self._model_feature_matrix(
            features,
            use_side_quadratic_features=head.get(
                'use_side_quadratic_features',
                self.use_side_quadratic_features,
            ),
            quadratic_feature_start=head.get(
                'quadratic_feature_start',
                self.quadratic_feature_start,
            ),
        )
        if feats.ndim == 1:
            feats = feats.reshape(1, -1)
        mean = np.asarray(head['feature_mean'], dtype=np.float32)
        std = np.asarray(head['feature_std'], dtype=np.float32)
        weights = np.asarray(head['weights'], dtype=np.float32)
        feats_norm = (feats - mean) / (std + 1e-8)
        return feats_norm @ weights + float(head['bias'])

    def _attack_head_score(
        self,
        head: Dict[str, Any],
        feat: np.ndarray,
        w_score: float,
    ) -> float:
        raw = float(self._head_logits(head, feat)[0])
        centered = raw - float(head['logit_shift'])
        if not self.use_tda:
            return centered
        alpha = float(head.get('alpha', self.alpha))
        return float(alpha * w_score + (1.0 - alpha) * centered)

    def attack_head_scores_from_features(
        self,
        features: np.ndarray,
        w_scores: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Vectorised per-head scores for diagnostics and validation."""
        scores = {}
        for attack, head in self.attack_heads.items():
            logits = self._head_logits(head, features)
            centered = logits - float(head['logit_shift'])
            if not self.use_tda:
                scores[attack] = centered.astype(np.float32)
            else:
                alpha = float(head.get('alpha', self.alpha))
                scores[attack] = (
                    alpha * w_scores + (1.0 - alpha) * centered
                ).astype(np.float32)
        return scores

    @staticmethod
    def _upper_tail_threshold(scores: np.ndarray, clean_fpr_target: float) -> float:
        arr = np.sort(np.asarray(scores, dtype=np.float32))
        if arr.size == 0:
            return 0.0
        q_idx = int(np.ceil((arr.size + 1) * (1.0 - clean_fpr_target)))
        q_idx = min(max(q_idx, 1), arr.size) - 1
        return float(arr[q_idx])

    @staticmethod
    def _robust_tail_scale(scores: np.ndarray) -> float:
        arr = np.asarray(scores, dtype=np.float32)
        if arr.size == 0:
            return 1.0
        median = float(np.percentile(arr, 50))
        p95 = float(np.percentile(arr, 95))
        std = float(np.std(arr))
        return max(p95 - median, 0.5 * std, 1e-3)

    def _calibrated_channel_evidence(
        self,
        raw_channels: Dict[str, float],
    ) -> Dict[str, float]:
        channels = (self.score_channel_calibration or {}).get('channels', {})
        evidence: Dict[str, float] = {}
        for name, raw in raw_channels.items():
            cal = channels.get(name)
            if cal is None:
                continue
            threshold = float(cal.get('threshold', 0.0))
            scale = max(float(cal.get('scale', 1.0)), 1e-8)
            evidence[name] = float((float(raw) - threshold) / scale)
        return evidence

    def _aggregate_channel_evidence(self, evidence: Dict[str, float]) -> float:
        """Aggregate calibrated evidence channels into the scalar conformal score."""
        if not evidence:
            return 0.0
        vals = np.asarray(list(evidence.values()), dtype=np.float32)
        mode = getattr(self, 'score_channel_aggregation', 'max') or 'max'
        if mode == 'max':
            return float(np.max(vals))
        if mode == 'positive_sum':
            pos = vals[vals > 0.0]
            return float(np.sum(pos)) if pos.size else float(np.max(vals))
        if mode == 'top2_positive':
            pos = np.sort(vals[vals > 0.0])
            return float(np.sum(pos[-2:])) if pos.size else float(np.max(vals))
        raise ValueError(
            f"Unknown score_channel_aggregation={mode!r}; "
            "expected 'max', 'positive_sum', or 'top2_positive'."
        )

    def _aggregate_channel_evidence_arrays(
        self,
        channels: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Vectorised equivalent of _aggregate_channel_evidence()."""
        if not channels:
            return np.zeros(0, dtype=np.float32)
        stacked = np.vstack([v for _, v in sorted(channels.items())]).astype(np.float32)
        max_vals = np.max(stacked, axis=0)
        mode = getattr(self, 'score_channel_aggregation', 'max') or 'max'
        if mode == 'max':
            return max_vals.astype(np.float32)
        if mode == 'positive_sum':
            pos_sum = np.maximum(stacked, 0.0).sum(axis=0)
            return np.where(pos_sum > 0.0, pos_sum, max_vals).astype(np.float32)
        if mode == 'top2_positive':
            pos = np.maximum(stacked, 0.0)
            sorted_pos = np.sort(pos, axis=0)
            top2 = sorted_pos[-2:, :].sum(axis=0)
            return np.where(top2 > 0.0, top2, max_vals).astype(np.float32)
        raise ValueError(
            f"Unknown score_channel_aggregation={mode!r}; "
            "expected 'max', 'positive_sum', or 'top2_positive'."
        )

    def calibrated_channel_scores_from_features(
        self,
        features: np.ndarray,
        w_scores: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Return clean-tail-normalised evidence scores for each score channel."""
        if not self.score_channel_calibration:
            raise RuntimeError(
                "calibrated_channel_scores_from_features: score channels are not calibrated."
            )
        raw_channels: Dict[str, np.ndarray] = {
            'global': self.composite_score_from_features(features, w_scores)
        }
        raw_channels.update(self.attack_head_scores_from_features(features, w_scores))
        raw_channels['raw_max'] = np.max(
            np.vstack([v for _, v in sorted(raw_channels.items())]), axis=0
        ).astype(np.float32)

        calibrated: Dict[str, np.ndarray] = {}
        channels = self.score_channel_calibration.get('channels', {})
        for name, raw in raw_channels.items():
            cal = channels.get(name)
            if cal is None:
                continue
            threshold = float(cal.get('threshold', 0.0))
            scale = max(float(cal.get('scale', 1.0)), 1e-8)
            calibrated[name] = ((raw - threshold) / scale).astype(np.float32)
        return calibrated

    def calibrated_max_score_from_features(
        self,
        features: np.ndarray,
        w_scores: np.ndarray,
    ) -> np.ndarray:
        channels = self.calibrated_channel_scores_from_features(features, w_scores)
        if not channels:
            return self.composite_score_from_features(features, w_scores)
        return self._aggregate_channel_evidence_arrays(channels)

    def calibrate_score_channels(
        self,
        clean_val_features: np.ndarray,
        clean_val_w_scores: np.ndarray,
        adv_val_features: Optional[np.ndarray] = None,
        adv_val_w_scores: Optional[np.ndarray] = None,
        adv_val_labels: Optional[np.ndarray] = None,
        clean_fpr_target: float = 0.10,
    ) -> Dict[str, Any]:
        """
        Put the global head and attack-specialist heads on a common evidence scale.

        Each raw channel is converted into clean-tail evidence:
            evidence = (raw_score - q_(1-FPR)(clean_validation)) / robust_tail_scale

        This fixes the raw max-aggregation failure mode where one broad clean
        channel raises the conformal threshold and suppresses useful specialist
        channels. The evidence transform is fitted only on the scorer's internal
        validation split; the independent conformal calibration split still owns
        the final FPR guarantee.
        """
        raw_clean: Dict[str, np.ndarray] = {
            'global': self.composite_score_from_features(
                clean_val_features, clean_val_w_scores
            )
        }
        if self.attack_heads:
            raw_clean.update(
                self.attack_head_scores_from_features(clean_val_features, clean_val_w_scores)
            )
        raw_clean['raw_max'] = np.max(
            np.vstack([v for _, v in sorted(raw_clean.items())]), axis=0
        ).astype(np.float32)

        channels: Dict[str, Dict[str, float]] = {}
        for name, scores in sorted(raw_clean.items()):
            threshold = self._upper_tail_threshold(scores, clean_fpr_target)
            scale = self._robust_tail_scale(scores)
            evidence = (scores - threshold) / scale
            channels[name] = {
                'threshold': float(threshold),
                'scale': float(scale),
                'clean_fpr_at_zero': float(np.mean(evidence > 0.0)),
                'clean_median_raw': float(np.percentile(scores, 50)),
                'clean_p95_raw': float(np.percentile(scores, 95)),
            }

        self.score_channel_calibration = {
            'mode': 'clean_quantile_excess',
            'aggregation': getattr(self, 'score_channel_aggregation', 'max'),
            'clean_fpr_target': float(clean_fpr_target),
            'channels': channels,
        }
        self.attack_head_mode = 'calibrated_max' if self.attack_heads else 'off'

        summary: Dict[str, Any] = {
            'mode': self.attack_head_mode,
            'aggregation': getattr(self, 'score_channel_aggregation', 'max'),
            'clean_fpr_target': float(clean_fpr_target),
            'channels': channels,
        }

        if (
            adv_val_features is not None
            and adv_val_w_scores is not None
            and adv_val_labels is not None
            and len(adv_val_features) > 0
        ):
            labels = np.asarray(adv_val_labels, dtype=object)
            clean_scores = self.calibrated_max_score_from_features(
                clean_val_features, clean_val_w_scores
            )
            adv_scores = self.calibrated_max_score_from_features(
                adv_val_features, adv_val_w_scores
            )
            threshold = self._upper_tail_threshold(clean_scores, clean_fpr_target)
            per_attack = {}
            for attack in sorted(set(labels.tolist())):
                mask = labels == attack
                per_attack[attack] = float(np.mean(adv_scores[mask] > threshold))
            summary['aggregate_validation_metrics'] = {
                'clean_threshold': float(threshold),
                'clean_fpr': float(np.mean(clean_scores > threshold)),
                'per_attack_tpr': per_attack,
                'worst_case_tpr': float(min(per_attack.values())) if per_attack else None,
                'mean_attack_tpr': float(np.mean(list(per_attack.values()))) if per_attack else None,
            }
            self.score_channel_calibration['validation_metrics'] = summary[
                'aggregate_validation_metrics'
            ]
        return summary

    def fit_attack_heads(
        self,
        clean_train_features: np.ndarray,
        adv_train_features: np.ndarray,
        adv_train_labels: np.ndarray,
        clean_val_features: np.ndarray,
        adv_val_features: np.ndarray,
        adv_val_labels: np.ndarray,
        clean_val_w_scores: np.ndarray,
        adv_val_w_scores: np.ndarray,
        grid: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8),
        clean_fpr_target: float = 0.10,
    ) -> Dict[str, Any]:
        """
        Fit one one-vs-clean specialist head per attack family.

        Inference uses the max over the legacy global head and these specialist
        heads. Conformal thresholds are then fitted on clean calibration scores
        for that exact aggregate, so FPR remains an empirical gate.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler

        labels_train = np.asarray(adv_train_labels, dtype=object)
        labels_val = np.asarray(adv_val_labels, dtype=object)
        self.attack_heads = {}
        summary: Dict[str, Any] = {
            'mode': 'max',
            'clean_fpr_target': clean_fpr_target,
            'grid': [float(x) for x in grid],
            'heads': {},
        }

        for attack in sorted(set(labels_train.tolist())):
            tr_mask = labels_train == attack
            val_mask = labels_val == attack
            n_train_adv = int(np.sum(tr_mask))
            n_val_adv = int(np.sum(val_mask))
            if n_train_adv < 20 or n_val_adv < 10:
                summary['heads'][attack] = {
                    'skipped': True,
                    'reason': 'insufficient train/validation examples',
                    'n_train_adv': n_train_adv,
                    'n_val_adv': n_val_adv,
                }
                continue

            X = self._model_feature_matrix(
                np.vstack([clean_train_features, adv_train_features[tr_mask]])
            )
            y = np.array([0] * len(clean_train_features) + [1] * n_train_adv)

            scaler = StandardScaler()
            X_norm = scaler.fit_transform(X)
            clf = LogisticRegression(
                C=1.0, max_iter=1000, solver='lbfgs', class_weight='balanced'
            )
            clf.fit(X_norm, y)

            clean_train_norm = scaler.transform(
                self._model_feature_matrix(clean_train_features)
            )
            clean_train_logits = clean_train_norm @ clf.coef_[0] + clf.intercept_[0]
            logit_shift = float(np.mean(clean_train_logits))

            head = {
                'attack': attack,
                'weights': clf.coef_[0].astype(np.float32),
                'bias': float(clf.intercept_[0]),
                'feature_mean': scaler.mean_.astype(np.float32),
                'feature_std': scaler.scale_.astype(np.float32),
                'logit_shift': logit_shift,
                'alpha': float(self.alpha),
                'raw_n_features': int(self.n_features),
                'model_input_dim': int(clf.coef_[0].shape[0]),
                'use_side_quadratic_features': self.use_side_quadratic_features,
                'quadratic_feature_start': int(self.quadratic_feature_start),
                'n_train_clean': int(len(clean_train_features)),
                'n_train_adv': n_train_adv,
                'n_val_adv': n_val_adv,
            }

            clean_logits = self._head_logits(head, clean_val_features)
            adv_logits = self._head_logits(head, adv_val_features[val_mask])
            clean_centered = clean_logits - logit_shift
            adv_centered = adv_logits - logit_shift
            clean_w = clean_val_w_scores
            adv_w = adv_val_w_scores[val_mask]
            y_auc = np.array([0] * len(clean_centered) + [1] * len(adv_centered))

            rows = []
            for alpha in grid:
                if self.use_tda:
                    s_clean = alpha * clean_w + (1.0 - alpha) * clean_centered
                    s_adv = alpha * adv_w + (1.0 - alpha) * adv_centered
                else:
                    s_clean = clean_centered
                    s_adv = adv_centered
                q_idx = int(np.ceil((len(s_clean) + 1) * (1 - clean_fpr_target)))
                q_idx = min(max(q_idx, 1), len(s_clean)) - 1
                threshold = float(np.sort(s_clean)[q_idx])
                scores = np.concatenate([s_clean, s_adv])
                rows.append({
                    'alpha': float(alpha),
                    'auc': float(roc_auc_score(y_auc, scores)),
                    'clean_threshold': threshold,
                    'clean_fpr': float(np.mean(s_clean > threshold)),
                    'attack_tpr': float(np.mean(s_adv > threshold)),
                })

            best = max(rows, key=lambda r: (r['attack_tpr'], r['auc']))
            head['alpha'] = float(best['alpha'])
            head['validation_metrics'] = best
            self.attack_heads[attack] = head
            summary['heads'][attack] = {
                'skipped': False,
                'n_train_adv': n_train_adv,
                'n_val_adv': n_val_adv,
                'selected_alpha': head['alpha'],
                'best_auc': best['auc'],
                'best_tpr': best['attack_tpr'],
                'rows': rows,
            }

        self.attack_head_mode = 'max' if self.attack_heads else 'off'
        summary['active_heads'] = sorted(self.attack_heads.keys())
        summary['n_active_heads'] = len(self.attack_heads)
        return summary

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
        feats_norm = self._normalise(features)
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
        adv_attack_labels: Optional[List[str]] = None,
        selection_objective: str = 'auc',
        clean_fpr_target: float = 0.10,
    ) -> Dict[str, float]:
        """
        Pick alpha on a held-out slice.

        selection_objective='auc' preserves the historical aggregate AUC
        behavior. selection_objective='worst_case_tpr' chooses the alpha with
        the best minimum per-attack TPR at a fixed clean FPR threshold, using
        aggregate AUC only as a tie-breaker.

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
        rows = []
        labels = (
            np.asarray(adv_attack_labels)
            if adv_attack_labels is not None and len(adv_attack_labels) == len(adv_features)
            else None
        )
        for a in grid:
            s_clean = self.composite_score_from_features(clean_features, clean_w_scores, alpha=a)
            s_adv = self.composite_score_from_features(adv_features, adv_w_scores, alpha=a)
            s = np.concatenate([s_clean, s_adv])
            auc = float(roc_auc_score(y, s))
            aucs.append(auc)

            row: Dict[str, Any] = {'alpha': float(a), 'auc': auc}
            if labels is not None:
                thresh_idx = int(np.ceil((len(s_clean) + 1) * (1 - clean_fpr_target)))
                thresh_idx = min(max(thresh_idx, 1), len(s_clean)) - 1
                threshold = float(np.sort(s_clean)[thresh_idx])
                per_attack = {}
                for atk in sorted(set(labels.tolist())):
                    mask = labels == atk
                    per_attack[atk] = float(np.mean(s_adv[mask] > threshold))
                row.update({
                    'clean_threshold': threshold,
                    'clean_fpr': float(np.mean(s_clean > threshold)),
                    'per_attack_tpr': per_attack,
                    'worst_case_tpr': float(min(per_attack.values())) if per_attack else None,
                    'mean_attack_tpr': float(np.mean(list(per_attack.values()))) if per_attack else None,
                })
            rows.append(row)

        objective = selection_objective.lower()
        if objective == 'worst_case_tpr' and labels is not None:
            best_idx = max(
                range(len(rows)),
                key=lambda i: (
                    rows[i].get('worst_case_tpr') or -1.0,
                    rows[i].get('mean_attack_tpr') or -1.0,
                    rows[i]['auc'],
                ),
            )
        else:
            best_idx = int(np.argmax(aucs))
            objective = 'auc'

        self.alpha = float(grid[best_idx])
        self.selection_objective = objective
        self.per_attack_validation_metrics = rows[best_idx]
        logger.info("tune_alpha: selected α=%.3f objective=%s AUC=%.4f; grid AUCs: %s",
                    self.alpha, objective, aucs[best_idx],
                    ", ".join(f"α={a:.2f}:{auc:.4f}" for a, auc in zip(grid, aucs)))
        return {
            'selected_alpha': self.alpha,
            'grid': list(grid),
            'aucs': aucs,
            'rows': rows,
            'best_auc': aucs[best_idx],
            'best_row': rows[best_idx],
            'selection_objective': objective,
            'clean_fpr_target': clean_fpr_target,
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
            'training_attack_counts': self.training_attack_counts,
            'attack_heads': self.attack_heads,
            'attack_head_mode': self.attack_head_mode,
            'attack_head_summary': getattr(self, 'attack_head_summary', None),
            'score_channel_calibration': self.score_channel_calibration,
            'score_channel_aggregation': getattr(self, 'score_channel_aggregation', 'max'),
            'use_side_quadratic_features': self.use_side_quadratic_features,
            'quadratic_feature_start': self.quadratic_feature_start,
            'logistic_input_dim': (
                self.logistic_input_dim
                if self.logistic_input_dim is not None
                else (
                    int(len(self.logistic_weights))
                    if self.logistic_weights is not None else None
                )
            ),
            'score_channel_summary': getattr(self, 'score_channel_summary', None),
            'balanced_attacks': self.balanced_attacks,
            'pgd_train_steps': self.pgd_train_steps,
            'square_train_max_iter': getattr(self, 'square_train_max_iter', None),
            'aa_train_mode': self.aa_train_mode,
            'training_source_split': getattr(self, 'training_source_split', None),
            'training_source_description': getattr(self, 'training_source_description', None),
            # P0.3 regression gate: sanity_checks.py::Check 6 asserts >= 2.5.
            'fgsm_oversample': getattr(self, 'fgsm_oversample', None),
            'pgd_oversample': getattr(self, 'pgd_oversample', None),
            'square_oversample': getattr(self, 'square_oversample', None),
            'cw_oversample': getattr(self, 'cw_oversample', None),
            'autoattack_oversample': getattr(self, 'autoattack_oversample', None),
            'requested_oversample_weights': getattr(self, 'requested_oversample_weights', None),
            'no_tda_features': getattr(self, 'no_tda_features', False),
            # P0.6 lever: held-out AUC grid search picks α; summary records the
            # grid and per-α AUCs so downstream verification can reconstruct why
            # this α was selected.
            'alpha_tune_summary': getattr(self, 'alpha_tune_summary', None),
            # Feature engineering flags
            'use_dct': self.use_dct,
            'use_softmax_entropy': self.use_softmax_entropy,
            'use_logit_profile_features': self.use_logit_profile_features,
            'logit_profile_feature_count': self.logit_profile_feature_count,
            'use_stability_features': self.use_stability_features,
            'stability_feature_count': self.stability_feature_count,
            'use_grad_norm': self.use_grad_norm,
            'use_tda': self.use_tda,   # P0.6 ablation flag
            'n_features': self.n_features,   # @property value; serialised for verification
            'feature_space_version': self.feature_space_version,
            'selection_objective': self.selection_objective,
            'per_attack_validation_metrics': self.per_attack_validation_metrics,
            'gradient_head_enabled': self.gradient_head_enabled,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        logger.info("PersistenceEnsembleScorer saved -> %s", path)

    @classmethod
    def load(cls, path: str, base_scorer: TopologicalScorer,
             layer_names: List[str]) -> 'PersistenceEnsembleScorer':
        """Load a local, dict-format ensemble scorer artifact from disk."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Ensemble scorer not found: {path}")
        resolved = p.resolve()
        project_root = Path(__file__).resolve().parents[2]
        cwd = Path.cwd().resolve()
        if not (resolved.is_relative_to(project_root) or resolved.is_relative_to(cwd)):
            raise ValueError(
                f"Refusing to load ensemble pickle outside trusted project paths: {resolved}"
            )
        with open(resolved, 'rb') as f:
            data = pickle.load(f)
        if not isinstance(data, dict):
            raise TypeError(
                f"Expected dict-format PersistenceEnsembleScorer artifact, got {type(data).__name__}"
            )
        stability_feature_count = data.get('stability_feature_count')
        if stability_feature_count is None:
            # Backward compatibility for artifacts saved before the explicit
            # stability_feature_count field. Infer it from n_features when
            # possible so 42-feature pixel-v1 models keep their 4-feature block.
            base_n = len(layer_names) * len(tuple(data['dims'])) * 6 if data.get('use_tda', True) else 0
            side_n = (
                (1 if data.get('use_dct', False) else 0)
                + (1 if data.get('use_softmax_entropy', False) else 0)
                + (1 if data.get('use_grad_norm', False) else 0)
            )
            saved_n = data.get('n_features')
            if data.get('use_stability_features', False) and saved_n is not None:
                stability_feature_count = max(0, int(saved_n) - base_n - side_n)
            else:
                stability_feature_count = (
                    DEFAULT_STABILITY_FEATURE_COUNT
                    if data.get('use_stability_features', False) else 0
                )
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
            use_logit_profile_features=data.get('use_logit_profile_features', False),
            logit_profile_feature_count=data.get(
                'logit_profile_feature_count',
                LOGIT_PROFILE_FEATURE_COUNT if data.get('use_logit_profile_features', False) else 0,
            ),
            use_stability_features=data.get('use_stability_features', False),
            stability_feature_count=stability_feature_count,
            use_grad_norm=data.get('use_grad_norm', False),
            use_tda=data.get('use_tda', True),
            feature_space_version=data.get('feature_space_version', 'pixel-v1'),
            selection_objective=data.get('selection_objective', 'auc'),
            per_attack_validation_metrics=data.get('per_attack_validation_metrics', {}),
            training_attack_counts=data.get('training_attack_counts', {}),
            attack_heads=data.get('attack_heads'),
            attack_head_mode=data.get('attack_head_mode', 'off'),
            score_channel_calibration=data.get('score_channel_calibration'),
            score_channel_aggregation=data.get(
                'score_channel_aggregation',
                (data.get('score_channel_calibration') or {}).get('aggregation', 'max'),
            ),
            use_side_quadratic_features=data.get('use_side_quadratic_features', False),
            quadratic_feature_start=data.get('quadratic_feature_start', 36),
            logistic_input_dim=data.get('logistic_input_dim'),
            balanced_attacks=data.get('balanced_attacks', False),
            pgd_train_steps=data.get('pgd_train_steps'),
            aa_train_mode=data.get('aa_train_mode'),
            gradient_head_enabled=data.get('gradient_head_enabled'),
        )
        scorer.square_train_max_iter = data.get('square_train_max_iter')
        scorer.training_source_split = data.get('training_source_split')
        scorer.training_source_description = data.get('training_source_description')
        scorer.fgsm_oversample = data.get('fgsm_oversample')
        scorer.pgd_oversample = data.get('pgd_oversample')
        scorer.square_oversample = data.get('square_oversample')
        scorer.cw_oversample = data.get('cw_oversample')
        scorer.autoattack_oversample = data.get('autoattack_oversample')
        scorer.requested_oversample_weights = data.get('requested_oversample_weights')
        scorer.no_tda_features = data.get('no_tda_features', False)
        scorer.alpha_tune_summary = data.get('alpha_tune_summary')
        scorer.attack_head_summary = data.get('attack_head_summary')
        scorer.score_channel_summary = data.get('score_channel_summary')
        scorer._logistic_fitted = data.get('_logistic_fitted', False)
        return scorer
