"""
TAMM: Topological Anomaly Scorer
Computes aggregated anomaly scores by comparing per-layer persistence
diagrams against reference profiles using Wasserstein distance.

This module was listed in the project structure but omitted from the plan's code.
"""
import numpy as np
from typing import Dict, List, Optional
from .tda import TopologicalProfiler, PersistenceDiagram


class TopologicalScorer:
    """
    Scores an input's topological divergence from reference profiles.
    Higher score = more topologically anomalous = more likely adversarial.
    """

    def __init__(
        self,
        ref_profiles: Dict[str, List[PersistenceDiagram]],
        layer_names: List[str],
        layer_weights: Optional[Dict[str, float]] = None,
        dims: Optional[List[int]] = None,
    ):
        """
        Args:
            ref_profiles: {layer_name: medoid_diagram_set} from profiling phase.
            layer_names: Ordered list of layers to score.
            layer_weights: Optional per-layer weights. Defaults to uniform.
            dims: Which homology dimensions to use (default [0, 1]).
        """
        self.ref_profiles = ref_profiles
        self.layer_names = layer_names
        self.dims = dims or [0, 1]

        if layer_weights is None:
            # Uniform weights, normalized
            w = 1.0 / len(layer_names)
            self.layer_weights = {name: w for name in layer_names}
        else:
            # Normalize provided weights
            total = sum(layer_weights.values())
            self.layer_weights = {k: v / total for k, v in layer_weights.items()}

    def score(
        self,
        diagrams: Dict[str, List[PersistenceDiagram]],
    ) -> float:
        """
        Compute aggregated anomaly score across layers and dimensions.

        Args:
            diagrams: {layer_name: [H0, H1, ...]} for the input being scored.
        Returns:
            Scalar anomaly score (higher = more anomalous).
        """
        total_score = 0.0

        for layer in self.layer_names:
            if layer not in diagrams or layer not in self.ref_profiles:
                continue

            input_dgms = diagrams[layer]
            ref_dgms = self.ref_profiles[layer]
            w = self.layer_weights[layer]

            layer_score = 0.0
            n_dims = 0
            for dim in self.dims:
                if dim >= len(input_dgms) or dim >= len(ref_dgms):
                    continue
                d = TopologicalProfiler.wasserstein_dist(
                    input_dgms[dim], ref_dgms[dim]
                )
                layer_score += d
                n_dims += 1

            if n_dims > 0:
                layer_score /= n_dims

            total_score += w * layer_score

        return total_score

    def score_per_layer(
        self,
        diagrams: Dict[str, List[PersistenceDiagram]],
    ) -> Dict[str, float]:
        """Return per-layer anomaly scores for diagnostics."""
        scores = {}
        for layer in self.layer_names:
            if layer not in diagrams or layer not in self.ref_profiles:
                scores[layer] = 0.0
                continue

            input_dgms = diagrams[layer]
            ref_dgms = self.ref_profiles[layer]
            s = 0.0
            n = 0
            for dim in self.dims:
                if dim >= len(input_dgms) or dim >= len(ref_dgms):
                    continue
                s += TopologicalProfiler.wasserstein_dist(
                    input_dgms[dim], ref_dgms[dim]
                )
                n += 1
            scores[layer] = s / max(n, 1)
        return scores
