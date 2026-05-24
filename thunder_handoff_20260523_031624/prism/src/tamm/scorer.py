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
        dim_weights: Optional[List[float]] = None,
    ):
        """
        Args:
            ref_profiles: {layer_name: medoid_diagram_set} from profiling phase.
            layer_names: Ordered list of layers to score.
            layer_weights: Optional per-layer weights. Defaults to uniform.
            dims: Which homology dimensions to use (default [0, 1]).
            dim_weights: Per-dimension weights aligned with dims.
                         Recommended [0.7, 0.3]: H0 connectivity disruption
                         dominates for single-step attacks (FGSM, Square);
                         H1 loops relevant for iterative attacks (PGD).
        """
        self.ref_profiles = ref_profiles
        self.layer_names = layer_names
        self.dims = dims or [0, 1]

        # Per-dimension weights (H0 vs H1)
        if dim_weights is not None:
            self.dim_weights = dim_weights
        else:
            # Equal weighting as safe default; pass explicit dim_weights for tuned behavior
            self.dim_weights = [1.0 / len(self.dims)] * len(self.dims)

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
        **kwargs,   # Accept image=/grad_norm= from PRISM.defend() dispatch; not used here
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
            total_weight = 0.0
            for i, dim in enumerate(self.dims):
                if dim >= len(input_dgms) or dim >= len(ref_dgms):
                    continue
                dw = self.dim_weights[i] if i < len(self.dim_weights) else 1.0
                d = TopologicalProfiler.wasserstein_dist(
                    input_dgms[dim], ref_dgms[dim]
                )
                layer_score += dw * d
                total_weight += dw

            if total_weight > 0:
                layer_score /= total_weight

            total_score += w * layer_score

        return total_score

    def score_per_layer(
        self,
        diagrams: Dict[str, List[PersistenceDiagram]],
        **kwargs,   # API compatibility with PersistenceEnsembleScorer
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
            tw = 0.0
            for i, dim in enumerate(self.dims):
                if dim >= len(input_dgms) or dim >= len(ref_dgms):
                    continue
                dw = self.dim_weights[i] if i < len(self.dim_weights) else 1.0
                s += dw * TopologicalProfiler.wasserstein_dist(
                    input_dgms[dim], ref_dgms[dim]
                )
                tw += dw
            scores[layer] = s / max(tw, 1e-8)
        return scores
