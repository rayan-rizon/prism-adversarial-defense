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
    """

    def __init__(
        self,
        experts: List[ExpertSubNetwork],
        expert_ref_diagrams: List[list],
        comparison_dim: int = 1,
    ):
        """
        Args:
            experts: List of K trained expert sub-networks.
            expert_ref_diagrams: List of K reference diagram sets,
                one per expert (each is [H0, H1, ...]).
            comparison_dim: Which homology dimension to use for gating.
        """
        if len(experts) != len(expert_ref_diagrams):
            raise ValueError(
                f"Mismatch: {len(experts)} experts vs "
                f"{len(expert_ref_diagrams)} reference diagrams"
            )
        self.experts = experts
        self.ref_diagrams = expert_ref_diagrams
        self.comparison_dim = comparison_dim

    def select_expert(
        self, input_diagrams: list
    ) -> Tuple[int, ExpertSubNetwork, float]:
        """
        Select the most topologically compatible expert.

        Args:
            input_diagrams: Persistence diagrams [H0, H1, ...] of the input.
        Returns:
            (expert_index, expert_module, wasserstein_distance)
        """
        dim = self.comparison_dim
        input_dgm = input_diagrams[dim] if dim < len(input_diagrams) else np.array([])

        distances = []
        for ref_dgm_set in self.ref_diagrams:
            ref_dgm = ref_dgm_set[dim] if dim < len(ref_dgm_set) else np.array([])
            if len(input_dgm) == 0 and len(ref_dgm) == 0:
                distances.append(0.0)
            elif len(input_dgm) == 0 or len(ref_dgm) == 0:
                non_empty = input_dgm if len(input_dgm) > 0 else ref_dgm
                distances.append(float(np.sum(np.abs(non_empty[:, 1] - non_empty[:, 0]))))
            else:
                d = wasserstein_distance(input_dgm, ref_dgm, order=2)
                distances.append(float(d))

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
