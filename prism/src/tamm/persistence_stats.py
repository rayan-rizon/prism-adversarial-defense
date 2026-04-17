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

from .tda import TopologicalProfiler, PersistenceDiagram


# Small floor to avoid log(0); persistence values << EPS are treated as zero
_ENTROPY_EPS = 1e-10
# Features with persistence below this are considered noise (ripser artefacts)
_BIRTH_THRESHOLD = 1e-6


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
) -> np.ndarray:
    """
    Build the full 36-dimensional feature vector for one input:
      6 stats x 2 dims x 3 layers = 36 features
    Wasserstein distance is the first of the 6 stats per (layer, dim) block.

    Features are ordered: for each layer, for each dim:
      [wasserstein_dist, total_persistence, max_persistence,
       n_features, entropy, mean_persistence]

    Args:
        diagrams: {layer: [H0, H1, ...]} for the current input.
        ref_profiles: {layer: [H0, H1, ...]} medoid reference profiles.
        layer_names: Ordered list of layers to include.
        dims: Which homology dimensions to include (default [0, 1]).
    Returns:
        1-D float32 array of length len(layer_names) * len(dims) * 6.
    """
    features = []
    for layer in layer_names:
        inp_dgms = diagrams.get(layer, [])
        ref_dgms = ref_profiles.get(layer, [])

        for dim in dims:
            inp = inp_dgms[dim] if dim < len(inp_dgms) else np.array([]).reshape(0, 2)
            ref = ref_dgms[dim] if dim < len(ref_dgms) else np.array([]).reshape(0, 2)

            # Wasserstein distance to reference
            w_dist = TopologicalProfiler.wasserstein_dist(inp, ref)

            # Persistence statistics of the INPUT diagram
            stats = _persistence_stats(inp)

            features.extend([
                w_dist,
                stats['total_persistence'],
                stats['max_persistence'],
                stats['n_features'],
                stats['entropy'],
                stats['mean_persistence'],
            ])

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
