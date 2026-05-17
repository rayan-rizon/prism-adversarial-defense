"""
TAMM: Persistence Statistics Feature Extractor

Augments raw Wasserstein distance with richer persistence diagram statistics
that are more sensitive to subtle single-step perturbations (FGSM at ε≈0.03).

Raw Wasserstein distance measures 'shape similarity' but discards distributional
information. Single-step FGSM at ε=8/255 creates many very-short-lived topological
micro-features that individually fall below the Wasserstein matching threshold yet
collectively shift total persistence, feature count, and entropy.

Features per (layer, dimension):
  - wasserstein_dist  : raw W2 distance to reference medoid
  - total_persistence : Sum(death - birth)  — sensitive to micro-feature density
  - max_persistence   : max(death - birth) — sensitive to structural destruction
  - n_features        : count of features above birth_threshold
  - entropy           : -Sum p_i log p_i, p_i = (d_i-b_i)/total_pers — shape complexity
  - mean_persistence  : total / n_features (or 0 if empty)

These 6 scalars x 2 dims (H0, H1) x 3 layers = 36-dim feature vector.
Note: wasserstein_dist IS one of the 6 features per (layer,dim), not separate.
A logistic regression trained conformally on this 36-dim vector achieves meaningful
FGSM separation (validated in the audit: Wasserstein alone cannot).
"""
import numpy as np
from typing import Dict, List, Optional, Tuple

try:
    from scipy.fft import dctn as _dctn
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

from .tda import TopologicalProfiler, PersistenceDiagram


LOGIT_PROFILE_FEATURE_COUNT = 8
# Small floor to avoid log(0); persistence values << EPS are treated as zero
_ENTROPY_EPS = 1e-10
# Features with persistence below this are considered noise (ripser artefacts)
_BIRTH_THRESHOLD = 1e-6


def compute_softmax_entropy(logits: np.ndarray) -> float:
    """
    Compute entropy of the softmax distribution over model logits.

    CW-L2 adversarials push inputs toward the decision boundary, producing
    higher softmax entropy (less confident predictions) compared to clean
    inputs.  This feature captures the signal that TDA features cannot see:
    CW's minimal perturbation is invisible to persistence diagrams but
    measurably changes the model's confidence distribution.

    L-inf attacks (FGSM, PGD) also tend to elevate entropy, so this feature
    is monotonically related to adversariality and will not hurt existing
    attack detection.

    Args:
        logits: 1-D array of raw model logits (pre-softmax), any length.
    Returns:
        H(softmax(logits)) — scalar float >= 0.
        Returns 0.0 if logits is None or empty.
    """
    if logits is None or len(logits) == 0:
        return 0.0
    logits = np.asarray(logits, dtype=np.float64)
    # Numerically stable softmax
    shifted = logits - np.max(logits)
    exp_x = np.exp(shifted)
    probs = exp_x / np.sum(exp_x)
    # Entropy: -sum(p * log(p)), skipping zero entries
    ent = -np.sum(probs * np.log(probs + 1e-12))
    return float(ent)


def compute_logit_profile_features(logits: np.ndarray) -> np.ndarray:
    """
    Compute deterministic classifier-confidence summary features.

    These features are intentionally label-free and use only the frozen
    backbone logits at evaluation time. They target the remaining FGSM/Square
    lower tail where topological and transform-stability evidence overlaps the
    clean distribution under the L1 FPR constraint.

    Returned order:
      [top_confidence, top2_probability_gap, top3_probability_mass,
       logit_margin, logit_range, centered_logit_l2, logsumexp_energy,
       probability_l2]
    """
    if logits is None or len(logits) == 0:
        return np.zeros(LOGIT_PROFILE_FEATURE_COUNT, dtype=np.float32)

    z = np.asarray(logits, dtype=np.float64).reshape(-1)
    if z.size == 0 or not np.all(np.isfinite(z)):
        return np.zeros(LOGIT_PROFILE_FEATURE_COUNT, dtype=np.float32)

    shifted = z - np.max(z)
    exp_z = np.exp(shifted)
    probs = exp_z / np.sum(exp_z)
    p_sorted = np.sort(probs)[::-1]
    z_sorted = np.sort(z)[::-1]

    top_conf = float(p_sorted[0])
    top2_gap = float(p_sorted[0] - (p_sorted[1] if p_sorted.size > 1 else 0.0))
    top3_mass = float(np.sum(p_sorted[: min(3, p_sorted.size)]))
    margin = float(z_sorted[0] - (z_sorted[1] if z_sorted.size > 1 else 0.0))
    logit_range = float(z_sorted[0] - z_sorted[-1])
    centered_l2 = float(np.linalg.norm(z - np.mean(z)))
    energy = float(np.log(np.sum(np.exp(shifted)) + 1e-12) + np.max(z))
    prob_l2 = float(np.linalg.norm(probs))

    return np.array([
        top_conf,
        top2_gap,
        top3_mass,
        margin,
        logit_range,
        centered_l2,
        energy,
        prob_l2,
    ], dtype=np.float32)


def compute_logit_stability_features(
    logits: np.ndarray,
    smoothed_logits: np.ndarray,
) -> np.ndarray:
    """
    Compare model logits before and after a deterministic 3x3 pixel smoothing.

    This is a lightweight transform-consistency signal. Iterative L-infinity
    attacks often preserve the classifier decision while making the logit margin
    brittle under benign smoothing; that failure mode is weakly represented in
    TDA/DCT alone.

    Returns four scalar features:
      [JS(prob, prob_smooth), top1_changed, abs_confidence_delta,
       abs_margin_delta]
    """
    if logits is None or smoothed_logits is None:
        return np.zeros(4, dtype=np.float32)
    a = np.asarray(logits, dtype=np.float64).reshape(-1)
    b = np.asarray(smoothed_logits, dtype=np.float64).reshape(-1)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return np.zeros(4, dtype=np.float32)

    def _prob(x: np.ndarray) -> np.ndarray:
        shifted = x - np.max(x)
        exp_x = np.exp(shifted)
        return exp_x / np.sum(exp_x)

    pa = _prob(a)
    pb = _prob(b)
    m = 0.5 * (pa + pb)
    js = 0.5 * np.sum(pa * (np.log(pa + 1e-12) - np.log(m + 1e-12)))
    js += 0.5 * np.sum(pb * (np.log(pb + 1e-12) - np.log(m + 1e-12)))

    top1_changed = float(int(np.argmax(pa) != np.argmax(pb)))
    conf_delta = abs(float(np.max(pa) - np.max(pb)))
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    if a_sorted.size >= 2:
        margin_a = float(a_sorted[-1] - a_sorted[-2])
        margin_b = float(b_sorted[-1] - b_sorted[-2])
        margin_delta = abs(margin_a - margin_b)
    else:
        margin_delta = 0.0
    return np.array([js, top1_changed, conf_delta, margin_delta], dtype=np.float32)


def compute_logit_stability_summary(
    logits: np.ndarray,
    transformed_logits: List[np.ndarray],
) -> np.ndarray:
    """
    Aggregate deterministic transform-consistency evidence across transforms.

    The original stability block used one 3x3 smoothing transform and returned
    four values. The v2 block keeps that signal but summarizes several cheap
    deterministic transforms so a single benign transform cannot dominate the
    feature. Returned order:

      [max_js, mean_js, max_top1_changed, mean_top1_changed,
       max_confidence_delta, mean_confidence_delta,
       max_margin_delta, mean_margin_delta]
    """
    rows = []
    if transformed_logits is None:
        transformed_logits = []
    for item in transformed_logits:
        rows.append(compute_logit_stability_features(logits, item))
    if not rows:
        return np.zeros(8, dtype=np.float32)
    arr = np.vstack(rows).astype(np.float32)
    return np.array([
        float(np.max(arr[:, 0])),
        float(np.mean(arr[:, 0])),
        float(np.max(arr[:, 1])),
        float(np.mean(arr[:, 1])),
        float(np.max(arr[:, 2])),
        float(np.mean(arr[:, 2])),
        float(np.max(arr[:, 3])),
        float(np.mean(arr[:, 3])),
    ], dtype=np.float32)


def compute_dct_energy(image: np.ndarray) -> float:
    """
    Compute log high-frequency DCT energy of an image.

    Square attacks perturb pixel statistics more than latent topology; this
    feature captures the high-frequency noise signature they leave in pixel
    space, supplementing the 36 persistence statistics.

    Args:
        image: (C, H, W) float32 array in canonical pixel space [0, 1].
    Returns:
        log(\u03a3 high_freq_coeff\u00b2 + 1e-8) \u2014 scalar float.
        Returns 0.0 if scipy is unavailable or image is None.
    """
    if not _SCIPY_AVAILABLE or image is None:
        return 0.0
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[np.newaxis]  # treat as single-channel
    energy = 0.0
    for c in range(arr.shape[0]):
        coeffs = _dctn(arr[c], norm='ortho')
        H, W = coeffs.shape
        mask = np.ones_like(coeffs, dtype=bool)
        mask[: H // 4, : W // 4] = False  # exclude low-frequency quadrant
        energy += float(np.sum(coeffs[mask] ** 2))
    return float(np.log(energy + 1e-8))


def _persistence_stats(dgm: PersistenceDiagram,
                       birth_threshold: float = _BIRTH_THRESHOLD
                       ) -> Dict[str, float]:
    """
    Compute summary statistics for a single persistence diagram.

    Args:
        dgm: (N, 2) array of (birth, death) pairs. Infinite death values
             are filtered out before computing stats.
        birth_threshold: Features with persistence <= this are ignored
                         (removes ripser artefacts).
    Returns:
        Dict with keys: total_persistence, max_persistence, n_features,
                        entropy, mean_persistence.
    """
    if dgm is None or len(dgm) == 0:
        return {
            'total_persistence': 0.0,
            'max_persistence': 0.0,
            'n_features': 0.0,
            'entropy': 0.0,
            'mean_persistence': 0.0,
        }

    # Filter out infinite death values (present in H0 as the global component)
    finite_mask = np.isfinite(dgm[:, 1])
    dgm_finite = dgm[finite_mask]

    if len(dgm_finite) == 0:
        return {
            'total_persistence': 0.0,
            'max_persistence': 0.0,
            'n_features': 0.0,
            'entropy': 0.0,
            'mean_persistence': 0.0,
        }

    pers = dgm_finite[:, 1] - dgm_finite[:, 0]  # death - birth
    pers = np.maximum(pers, 0.0)                 # clip numerical negatives

    # Apply birth threshold
    sig_mask = pers > birth_threshold
    if not np.any(sig_mask):
        return {
            'total_persistence': float(np.sum(pers)),
            'max_persistence': float(np.max(pers)) if len(pers) > 0 else 0.0,
            'n_features': 0.0,
            'entropy': 0.0,
            'mean_persistence': 0.0,
        }

    sig_pers = pers[sig_mask]
    total = float(np.sum(sig_pers))
    max_p = float(np.max(sig_pers))
    n = float(len(sig_pers))

    # Persistence entropy (normalised)
    if total > _ENTROPY_EPS:
        probs = sig_pers / total
        entropy = float(-np.sum(probs * np.log(probs + _ENTROPY_EPS)))
    else:
        entropy = 0.0

    mean_p = total / n if n > 0 else 0.0

    return {
        'total_persistence': total,
        'max_persistence': max_p,
        'n_features': n,
        'entropy': entropy,
        'mean_persistence': mean_p,
    }


def extract_feature_vector(
    diagrams: Dict[str, List[PersistenceDiagram]],
    ref_profiles: Dict[str, List[PersistenceDiagram]],
    layer_names: List[str],
    dims: List[int] = (0, 1),
    image: Optional[np.ndarray] = None,
    grad_norm: Optional[float] = None,
    logits: Optional[np.ndarray] = None,
    logit_profile_features: Optional[np.ndarray] = None,
    stability_features: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Build the feature vector for one input.
    Base: 6 stats × 2 dims × 3 layers = 36 features.
    Optional: log high-frequency DCT energy when *image* is provided.
    Optional: softmax entropy when *logits* is provided.
    Optional: classifier-confidence profile when *logit_profile_features* is provided.
    Optional: transform-stability feature vector when *stability_features* is provided.
    Optional: input-gradient L2 norm when *grad_norm* is provided.

    Features ordered: for each layer, for each dim:
      [wasserstein_dist, total_persistence, max_persistence,
       n_features, entropy, mean_persistence]
    Then, if image is not None: [dct_energy].
    Then, if logits is not None: [softmax_entropy].
    Then, if logit_profile_features is not None: [logit_profile_features...].
    Then, if stability_features is not None: [stability_features...].
    Then, if grad_norm is not None: [grad_norm].

    Args:
        diagrams: {layer: [H0, H1, ...]} for the current input.
        ref_profiles: {layer: [H0, H1, ...]} medoid reference profiles.
        layer_names: Ordered list of layers to include.
        dims: Which homology dimensions to include (default [0, 1]).
        image: Optional (C, H, W) float32 pixel-space [0, 1] image. When
               provided, appends the DCT high-frequency energy as a feature.
        grad_norm: Optional pre-computed input-gradient L2 norm. When
                   provided, appended as the final feature.
        logits: Optional 1-D array of raw model logits (pre-softmax).
                When provided, appends the softmax entropy as a feature.
        logit_profile_features: Optional fixed-length classifier-confidence vector.
        stability_features: Optional fixed-length transform-consistency vector.
    Returns:
        1-D float32 array.
    """
    features = []
    for layer in layer_names:
        inp_dgms = diagrams.get(layer, [])
        ref_dgms = ref_profiles.get(layer, [])

        for dim in dims:
            inp = inp_dgms[dim] if dim < len(inp_dgms) else np.array([]).reshape(0, 2)
            ref = ref_dgms[dim] if dim < len(ref_dgms) else np.array([]).reshape(0, 2)

            w_dist = TopologicalProfiler.wasserstein_dist(inp, ref)
            stats = _persistence_stats(inp)

            features.extend([
                w_dist,
                stats['total_persistence'],
                stats['max_persistence'],
                stats['n_features'],
                stats['entropy'],
                stats['mean_persistence'],
            ])

    if image is not None:
        features.append(compute_dct_energy(image))

    if logits is not None:
        features.append(compute_softmax_entropy(logits))

    if logit_profile_features is not None:
        features.extend(
            np.asarray(logit_profile_features, dtype=np.float32).reshape(-1).tolist()
        )

    if stability_features is not None:
        features.extend(np.asarray(stability_features, dtype=np.float32).reshape(-1).tolist())

    if grad_norm is not None:
        features.append(float(grad_norm))

    return np.array(features, dtype=np.float32)


def compute_clean_feature_matrix(
    diagrams_list: List[Dict[str, List[PersistenceDiagram]]],
    ref_profiles: Dict[str, List[PersistenceDiagram]],
    layer_names: List[str],
    dims: Tuple[int, ...] = (0, 1),
) -> np.ndarray:
    """
    Compute feature matrix for a list of clean inputs.

    Args:
        diagrams_list: list of N dicts, each {layer: [H0, H1, ...]}.
        ref_profiles: reference medoid profiles.
        layer_names: layers to include.
        dims: homology dimensions to include.
    Returns:
        (N, d) float32 feature matrix.
    """
    rows = []
    for dgms in diagrams_list:
        vec = extract_feature_vector(dgms, ref_profiles, layer_names, list(dims))
        rows.append(vec)
    return np.stack(rows, axis=0) if rows else np.zeros((0, len(layer_names) * len(dims) * 6))
