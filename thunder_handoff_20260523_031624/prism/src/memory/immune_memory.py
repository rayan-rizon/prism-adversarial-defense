"""
Immune Memory: Persistence Diagram Attack Signature Store
Stores known attack signatures as persistence diagrams for fast recall.
When a new input's topology matches a known attack, enables fast-path escalation
without full TDA scoring.
"""
import numpy as np
from typing import Dict, List, Optional, Any
from gudhi.wasserstein import wasserstein_distance
from dataclasses import dataclass, field
import time


@dataclass
class AttackSignature:
    """A stored attack signature."""
    diagram: list            # Persistence diagram set [H0, H1, ...]
    attack_type: str         # E.g., 'PGD', 'CW', 'FGSM'
    response_level: str      # The level this attack warrants
    confidence: float = 1.0  # How confident we are in this signature
    timestamp: float = 0.0   # When this signature was stored
    match_count: int = 0     # How many times this signature was matched


class ImmuneMemory:
    """
    Stores attack signatures as persistence diagrams for fast recall.
    Uses Wasserstein nearest-neighbor matching.
    """

    def __init__(self, match_threshold: float = 0.5,
                 comparison_dim: int = 1,
                 max_signatures: int = 1000):
        """
        Args:
            match_threshold: Max Wasserstein distance to consider a match.
            comparison_dim: Which homology dimension to compare.
            max_signatures: Maximum stored signatures (FIFO eviction).
        """
        self.threshold = match_threshold
        self.comparison_dim = comparison_dim
        self.max_signatures = max_signatures
        self.signatures: List[AttackSignature] = []

    def store(self, diagram: list, attack_type: str,
              response_level: str, confidence: float = 1.0):
        """Store a new attack signature."""
        sig = AttackSignature(
            diagram=diagram,
            attack_type=attack_type,
            response_level=response_level,
            confidence=confidence,
            timestamp=time.time(),
        )
        self.signatures.append(sig)

        # Evict oldest if over capacity
        if len(self.signatures) > self.max_signatures:
            # Sort by match_count (keep frequently matched) and recency
            self.signatures.sort(
                key=lambda s: (s.match_count, s.timestamp), reverse=True
            )
            self.signatures = self.signatures[:self.max_signatures]

    def match(self, input_diagrams: list) -> Optional[Dict[str, Any]]:
        """
        Check if input matches any known attack signature.

        Args:
            input_diagrams: Persistence diagram set [H0, H1, ...].
        Returns:
            Dict with match info, or None if no match.
        """
        if not self.signatures:
            return None

        dim = self.comparison_dim
        if dim >= len(input_diagrams):
            return None

        input_dgm = input_diagrams[dim]
        if len(input_dgm) == 0:
            return None

        best_sig: Optional[AttackSignature] = None
        best_dist = float('inf')

        for sig in self.signatures:
            if dim >= len(sig.diagram):
                continue
            ref_dgm = sig.diagram[dim]
            if len(ref_dgm) == 0:
                continue

            d = float(wasserstein_distance(input_dgm, ref_dgm, order=2))
            if d < best_dist:
                best_dist = d
                best_sig = sig

        if best_sig is not None and best_dist < self.threshold:
            best_sig.match_count += 1
            return {
                'attack_type': best_sig.attack_type,
                'level': best_sig.response_level,
                'distance': best_dist,
                'confidence': best_sig.confidence,
                'times_matched': best_sig.match_count,
            }

        return None

    def get_statistics(self) -> Dict:
        """Return summary statistics of the memory store."""
        if not self.signatures:
            return {'n_signatures': 0}

        attack_types = {}
        for sig in self.signatures:
            attack_types[sig.attack_type] = attack_types.get(sig.attack_type, 0) + 1

        return {
            'n_signatures': len(self.signatures),
            'attack_types': attack_types,
            'total_matches': sum(s.match_count for s in self.signatures),
        }
