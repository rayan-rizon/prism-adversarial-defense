"""
TAMSH: Expert Sub-Networks
Small expert MLPs that can replace spans of potentially compromised layers.

Key fixes from plan:
- Added numpy import for TopologyAwareMoE (was missing)
- ExpertSubNetwork uses eval() mode check to handle BatchNorm1d
  with batch_size=1 at inference
- Separated expert definition from MoE routing (gating.py handles routing)
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple
from gudhi.wasserstein import wasserstein_distance


class ExpertSubNetwork(nn.Module):
    """Small expert that replaces a span of compromised layers."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),   # LayerNorm works with batch_size=1 (inference + small clusters)
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim > 2:
            x = x.view(x.size(0), -1)
        # Truncate or pad to match expected input_dim
        if x.size(-1) > self.input_dim:
            x = x[..., :self.input_dim]
        elif x.size(-1) < self.input_dim:
            pad = torch.zeros(*x.shape[:-1], self.input_dim - x.size(-1),
                              device=x.device)
            x = torch.cat([x, pad], dim=-1)
        return self.net(x)


class TopologyAwareMoE:
    """
    Mixture of Experts with Wasserstein-based topology-aware gating.
    Selects the most topologically compatible expert for a given input
    by comparing persistence diagrams.

    Diagnostic (2026-05-21, P0.5): on ResNet18 layer4 activations, H1
    diagrams are very sparse (1-3 points per medoid). Empty input-H1
    diagrams force the empty-fallback path which makes expert 0 win
    ~94% of the time. Two new options were added without changing the
    default to preserve backward compat with the F3 ablation pipeline:

      - comparison_mode='H1'        — original behavior (default)
      - comparison_mode='H0'        — route on connected-components instead
      - comparison_mode='combined'  — weighted sum of H0 + H1 distances,
                                       skipping empty dims at either end
    """

    def __init__(
        self,
        experts: List[ExpertSubNetwork],
        expert_ref_diagrams: List[list],
        comparison_dim: int = 1,
        comparison_mode: str = 'H1',
        h0_weight: float = 1.0,
        h1_weight: float = 1.0,
    ):
        """
        Args:
            experts: List of K trained expert sub-networks.
            expert_ref_diagrams: List of K reference diagram sets,
                one per expert (each is [H0, H1, ...]).
            comparison_dim: Which homology dimension to use for gating
                when comparison_mode='H1' or 'H0' (kept for backwards compat).
            comparison_mode: 'H1' (default, original), 'H0', or 'combined'.
            h0_weight, h1_weight: weights for combined mode.
        """
        if len(experts) != len(expert_ref_diagrams):
            raise ValueError(
                f"Mismatch: {len(experts)} experts vs "
                f"{len(expert_ref_diagrams)} reference diagrams"
            )
        if comparison_mode not in ('H1', 'H0', 'combined'):
            raise ValueError(
                f"comparison_mode must be one of 'H1', 'H0', 'combined'; "
                f"got {comparison_mode!r}"
            )
        self.experts = experts
        self.ref_diagrams = expert_ref_diagrams
        self.comparison_dim = comparison_dim
        self.comparison_mode = comparison_mode
        self.h0_weight = float(h0_weight)
        self.h1_weight = float(h1_weight)

    @staticmethod
    def _safe_wass(a: np.ndarray, b: np.ndarray) -> float:
        """Wasserstein distance with empty-diagram fallback."""
        if len(a) == 0 and len(b) == 0:
            return 0.0
        if len(a) == 0 or len(b) == 0:
            non_empty = a if len(a) > 0 else b
            return float(np.sum(np.abs(non_empty[:, 1] - non_empty[:, 0])))
        return float(wasserstein_distance(a, b, order=2))

    def _per_expert_distances(self, input_diagrams: list) -> List[float]:
        """Distances input → each expert reference, honoring comparison_mode."""
        def _slot(dgm_set, k):
            return dgm_set[k] if k < len(dgm_set) else np.array([]).reshape(0, 2)

        if self.comparison_mode == 'H1':
            d_h1 = []
            in_dgm = _slot(input_diagrams, self.comparison_dim)
            for ref in self.ref_diagrams:
                ref_dgm = _slot(ref, self.comparison_dim)
                d_h1.append(self._safe_wass(in_dgm, ref_dgm))
            return d_h1

        if self.comparison_mode == 'H0':
            in_dgm = _slot(input_diagrams, 0)
            return [self._safe_wass(in_dgm, _slot(ref, 0)) for ref in self.ref_diagrams]

        # combined
        in_h0 = _slot(input_diagrams, 0)
        in_h1 = _slot(input_diagrams, 1)
        out = []
        for ref in self.ref_diagrams:
            d0 = self._safe_wass(in_h0, _slot(ref, 0))
            d1 = self._safe_wass(in_h1, _slot(ref, 1))
            out.append(self.h0_weight * d0 + self.h1_weight * d1)
        return out

    def select_expert(
        self, input_diagrams: list
    ) -> Tuple[int, ExpertSubNetwork, float]:
        """
        Select the most topologically compatible expert.

        Args:
            input_diagrams: Persistence diagrams [H0, H1, ...] of the input.
        Returns:
            (expert_index, expert_module, distance)
        """
        distances = self._per_expert_distances(input_diagrams)
        best_idx = int(np.argmin(distances))
        return best_idx, self.experts[best_idx], distances[best_idx]

    def forward_through_expert(
        self, input_diagrams: list, activation: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        """
        Route an activation through the selected expert.

        Args:
            input_diagrams: Persistence diagrams for expert selection.
            activation: The activation tensor to route through the expert.
        Returns:
            (expert_output, expert_index)
        """
        idx, expert, _ = self.select_expert(input_diagrams)
        expert.eval()
        with torch.no_grad():
            output = expert(activation)
        return output, idx

    def forward_uniform(
        self, activation: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        """
        Uniform ensemble — each expert weighted 1/K, no topology gating.
        Isolates the router contribution: if recovery_acc(uniform) is close to
        recovery_acc(topology), the topology gating adds little.
        """
        mixed = None
        n = len(self.experts)
        for k, expert in enumerate(self.experts):
            expert.eval()
            with torch.no_grad():
                logits = expert(activation)
            contribution = logits / float(n)
            mixed = contribution if mixed is None else mixed + contribution
        hard_idx = int(mixed.argmax(1).item()) if mixed is not None else 0  # not used; surface for telemetry
        return mixed, 0

    def forward_max_confidence(
        self, activation: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        """
        Bypass topology routing; return the expert with highest softmax max.
        Useful when topology routing is unreliable (e.g. very sparse H1
        diagrams on a final-layer activation).
        """
        best_logits = None
        best_idx = 0
        best_conf = -float('inf')
        for k, expert in enumerate(self.experts):
            expert.eval()
            with torch.no_grad():
                logits = expert(activation)
            conf = float(torch.softmax(logits, dim=-1).max().item())
            if conf > best_conf:
                best_conf = conf
                best_idx = k
                best_logits = logits
        return best_logits, best_idx

    def forward_through_expert_ensemble(
        self,
        input_diagrams: list,
        activation: torch.Tensor,
        temperature: float = 0.1,
    ) -> Tuple[torch.Tensor, int, List[float]]:
        """
        Soft-gate ensemble: average expert logits weighted by softmin of
        topology distances. Selected `expert_idx` is the argmin (for
        downstream telemetry); the returned logits are the soft average.

        Lower temperature → harder selection. temperature=0.1 puts ~99%
        weight on the closest expert when distances are reasonably spread.

        Args:
            input_diagrams: Persistence diagrams [H0, H1, ...] for routing.
            activation: Pooled activation tensor for the expert input.
            temperature: Softmin temperature (>0).
        Returns:
            (mixed_logits, hard_expert_idx, gating_weights)
        """
        distances = self._per_expert_distances(input_diagrams)
        d_arr = np.array(distances, dtype=float)
        neg = -d_arr / max(float(temperature), 1e-8)
        neg -= float(np.max(neg))  # numerical stability
        w = np.exp(neg)
        s = float(np.sum(w))
        weights = (w / s) if s > 0 else np.ones_like(w) / len(w)
        hard_idx = int(np.argmin(d_arr))

        mixed = None
        for k, expert in enumerate(self.experts):
            expert.eval()
            with torch.no_grad():
                logits = expert(activation)
            contribution = float(weights[k]) * logits
            mixed = contribution if mixed is None else mixed + contribution
        return mixed, hard_idx, weights.tolist()
