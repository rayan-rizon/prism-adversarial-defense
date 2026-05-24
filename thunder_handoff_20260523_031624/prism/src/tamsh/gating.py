"""
TAMSH: Wasserstein-Based Expert Gating
Provides clustering utilities for building expert reference diagrams
from clean activation topology, and soft gating weights.
"""
import numpy as np
from typing import List, Tuple
from gudhi.wasserstein import wasserstein_distance


def cluster_diagrams_by_topology(
    diagrams_list: List[list],
    k: int = 4,
    dim: int = 1,
    max_iter: int = 50,
    random_state: int = 42,
) -> Tuple[List[int], List[list]]:
    """
    K-medoids clustering on persistence diagrams using Wasserstein distance.
    Used to partition clean data into K topological regimes,
    one per expert sub-network.

    Args:
        diagrams_list: List of N diagram sets, each [H0, H1, ...].
        k: Number of clusters (experts).
        dim: Homology dimension for clustering.
        max_iter: Maximum PAM iterations.
        random_state: For reproducibility.
    Returns:
        (labels, medoids) — cluster assignment per diagram, and medoid diagrams.
    """
    rng = np.random.RandomState(random_state)
    n = len(diagrams_list)

    if n < k:
        raise ValueError(f"Need at least {k} diagrams, got {n}")

    # Extract target-dimension diagrams
    target_dgms = []
    for dgm_set in diagrams_list:
        if dim < len(dgm_set):
            target_dgms.append(dgm_set[dim])
        else:
            target_dgms.append(np.array([]).reshape(0, 2))

    # Compute pairwise distance matrix
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _safe_wasserstein(target_dgms[i], target_dgms[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    # K-medoids via PAM (Partitioning Around Medoids)
    # Initialize: pick k random medoids
    medoid_indices = list(rng.choice(n, k, replace=False))

    for _ in range(max_iter):
        # Assign each point to nearest medoid
        labels = np.argmin(dist_matrix[:, medoid_indices], axis=1)

        # Update medoids
        new_medoids = []
        for c in range(k):
            cluster_members = np.where(labels == c)[0]
            if len(cluster_members) == 0:
                new_medoids.append(medoid_indices[c])
                continue
            # Pick member with smallest total distance to others in cluster
            sub_dist = dist_matrix[np.ix_(cluster_members, cluster_members)]
            best = cluster_members[np.argmin(sub_dist.sum(axis=1))]
            new_medoids.append(best)

        if set(new_medoids) == set(medoid_indices):
            break
        medoid_indices = new_medoids

    # Final label assignment
    labels = np.argmin(dist_matrix[:, medoid_indices], axis=1)
    medoid_diagrams = [diagrams_list[i] for i in medoid_indices]

    return labels.tolist(), medoid_diagrams


def compute_soft_gating_weights(
    input_diagram: np.ndarray,
    ref_diagrams: List[np.ndarray],
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Compute soft gating weights based on Wasserstein distances.
    Uses softmin: w_k = exp(-d_k / T) / sum(exp(-d_j / T)).

    Args:
        input_diagram: Single persistence diagram for the input.
        ref_diagrams: List of K reference diagrams (one per expert).
        temperature: Softmin temperature (lower = harder selection).
    Returns:
        Array of K gating weights summing to 1.
    """
    distances = np.array([
        _safe_wasserstein(input_diagram, ref) for ref in ref_diagrams
    ])

    # Softmin with numerical stability
    neg_d = -distances / max(temperature, 1e-8)
    neg_d -= np.max(neg_d)  # Shift for stability
    weights = np.exp(neg_d)
    total = np.sum(weights)
    if total > 0:
        weights /= total
    else:
        weights = np.ones(len(ref_diagrams)) / len(ref_diagrams)

    return weights


def _safe_wasserstein(dgm_a: np.ndarray, dgm_b: np.ndarray) -> float:
    """Wasserstein distance with empty diagram handling."""
    if len(dgm_a) == 0 and len(dgm_b) == 0:
        return 0.0
    if len(dgm_a) == 0 or len(dgm_b) == 0:
        non_empty = dgm_a if len(dgm_a) > 0 else dgm_b
        return float(np.sum(np.abs(non_empty[:, 1] - non_empty[:, 0])))
    return float(wasserstein_distance(dgm_a, dgm_b, order=2))
