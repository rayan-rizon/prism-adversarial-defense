"""
TAMM: Topological Data Analysis Engine
Computes persistent homology from activation point clouds and provides
Wasserstein-based comparison between persistence diagrams.

Key fix from plan: reference profiles are stored as medoid diagrams
(not raw lists of 10K diagrams) to enable direct scoring.
"""
import hashlib
import numpy as np
import warnings
from ripser import ripser
from gudhi.wasserstein import wasserstein_distance
from typing import List, Optional, Tuple


# Type alias: a persistence diagram is a (N, 2) array of (birth, death) pairs
PersistenceDiagram = np.ndarray


class TopologicalProfiler:
    """Computes persistence diagrams from activation point clouds."""

    def __init__(self, n_subsample: int = 200, max_dim: int = 1,
                 random_state: Optional[int] = 42):
        self.n_subsample = n_subsample
        self.max_dim = max_dim
        # random_state kept for backward compatibility; primary subsampling
        # now uses deterministic hashing so this seed only affects medoid
        # candidate subsampling in compute_reference_medoid().
        self.rng = np.random.RandomState(random_state)

    def compute_diagram(self, activation: np.ndarray) -> List[PersistenceDiagram]:
        """
        Compute persistence diagrams from an activation tensor.

        Args:
            activation: shape (C, H, W) for conv layers or (N, D) for point cloud.
        Returns:
            List of persistence diagrams [H0, H1, ...] up to max_dim.
        """
        points = self._to_point_cloud(activation)
        points = self._subsample(points)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*more columns than rows.*")
            result = ripser(points, maxdim=self.max_dim)
        return result['dgms']

    def _to_point_cloud(self, activation: np.ndarray) -> np.ndarray:
        """Convert activation tensor to (N, D) point cloud."""
        if activation.ndim == 3:
            # (C, H, W) -> treat each spatial location as a point in C-dim space
            C, H, W = activation.shape
            return activation.reshape(C, -1).T  # (H*W, C)
        elif activation.ndim == 2:
            return activation
        elif activation.ndim == 1:
            return activation.reshape(-1, 1)
        else:
            raise ValueError(f"Unsupported activation shape: {activation.shape}")

    def _subsample(self, points: np.ndarray) -> np.ndarray:
        """Deterministic subsample based on activation content hash.

        Uses a per-image deterministic seed derived from the activation values
        so the same activation always produces the same subsample, regardless
        of how many other images have been processed before it.

        This replaces the previous stateful RNG approach that caused score
        variance when images were interleaved across evaluation batches —
        a requirement for the conformal iid assumption to hold.
        """
        if points.shape[0] <= self.n_subsample:
            return points
        # Derive a deterministic seed from the activation content.
        # xxh / md5 over a compact fixed-size fingerprint of the array:
        # we quantise to fp16 first to ignore rounding noise.
        fingerprint = points.astype(np.float16).tobytes()
        seed = int(hashlib.md5(fingerprint).hexdigest(), 16) % (2 ** 31)
        rng = np.random.RandomState(seed)
        idx = rng.choice(points.shape[0], self.n_subsample, replace=False)
        return points[idx]

    @staticmethod
    def wasserstein_dist(dgm_a: PersistenceDiagram, dgm_b: PersistenceDiagram,
                         order: int = 2) -> float:
        """Wasserstein distance between two persistence diagrams."""
        # Handle empty diagrams
        if len(dgm_a) == 0 and len(dgm_b) == 0:
            return 0.0
        if len(dgm_a) == 0 or len(dgm_b) == 0:
            # Distance = total persistence of the non-empty diagram
            non_empty = dgm_a if len(dgm_a) > 0 else dgm_b
            return float(np.sum(np.abs(non_empty[:, 1] - non_empty[:, 0])))
        return float(wasserstein_distance(dgm_a, dgm_b, order=order))

    def compute_reference_medoid(
        self,
        diagrams_list: List[List[PersistenceDiagram]],
        dims: Optional[List[int]] = None,
        dim_weights: Optional[List[float]] = None,
    ) -> List[PersistenceDiagram]:
        """
        Select the medoid diagram from a collection — the diagram whose
        mean weighted Wasserstein distance to all others is minimized.

        Medoid selection uses the SAME weighted multi-dimension criterion as
        TopologicalScorer.score(), so the reference diagram is optimal with
        respect to the actual scoring metric used at inference time.

        Previously this function accepted a single ``dim: int = 1`` argument,
        which caused a mismatch: the medoid was chosen to minimise H1 distance
        only, while scoring aggregates H0 and H1 with equal weight.  This
        inflated anomaly scores on clean inputs whose H0 topology deviated
        from the H1-optimal medoid, degrading conformal calibration quality.

        Args:
            diagrams_list: List of N diagram sets, each being [H0, H1, ...].
            dims: Homology dimensions to include in the distance criterion.
                Defaults to all dimensions present in the first diagram.
            dim_weights: Per-dimension weights (must sum to 1.0 if provided).
                Defaults to uniform weighting across dims.
        Returns:
            The medoid diagram set [H0, H1, ...].
        Raises:
            ValueError: If diagrams_list is empty.
        """
        n = len(diagrams_list)
        if n == 0:
            raise ValueError("Cannot compute medoid of empty list")
        if n == 1:
            return diagrams_list[0]

        # Resolve dims and weights from first diagram if not supplied
        n_dims_available = len(diagrams_list[0])
        if dims is None:
            dims = list(range(n_dims_available))
        if dim_weights is None:
            dim_weights = [1.0 / len(dims)] * len(dims)
        if len(dim_weights) != len(dims):
            raise ValueError(
                f"len(dim_weights)={len(dim_weights)} must match len(dims)={len(dims)}"
            )
        # Normalise weights to sum to 1
        w_sum = sum(dim_weights)
        if w_sum <= 0:
            raise ValueError("dim_weights must sum to a positive value")
        dim_weights = [w / w_sum for w in dim_weights]

        # For tractability, subsample candidates when collection is large
        max_medoid_candidates = 200
        if n > max_medoid_candidates:
            indices = self.rng.choice(n, max_medoid_candidates, replace=False)
            candidates = [diagrams_list[i] for i in indices]
        else:
            candidates = diagrams_list

        nc = len(candidates)

        # Compute pairwise weighted Wasserstein distances across all dims
        dist_matrix = np.zeros((nc, nc))
        for i in range(nc):
            for j in range(i + 1, nc):
                d = 0.0
                for dim, w in zip(dims, dim_weights):
                    if dim < len(candidates[i]) and dim < len(candidates[j]):
                        d += w * self.wasserstein_dist(
                            candidates[i][dim], candidates[j][dim]
                        )
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        # Medoid = argmin of mean distance row
        mean_dists = dist_matrix.mean(axis=1)
        medoid_idx = int(np.argmin(mean_dists))

        return candidates[medoid_idx]
