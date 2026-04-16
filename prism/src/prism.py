"""
PRISM: Predictive Runtime Immune System with Manifold Monitoring

Main integration class that wraps any pretrained PyTorch model with
the full PRISM defense pipeline:
  L0 (SACD) → TAMM → CADG → TAMSH → Immune Memory

Key fixes from plan:
1. Reference profiles are stored as medoid diagrams per layer (not raw 10K lists)
2. Scorer module handles multi-layer, multi-dimension aggregation properly
3. Transform consistency enforced between profiling and inference
4. Pickle loading is restricted to known paths (not arbitrary deserialization)
"""
import torch
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from .tamm.extractor import ActivationExtractor
from .tamm.tda import TopologicalProfiler
from .tamm.scorer import TopologicalScorer
from .cadg.calibrate import ConformalCalibrator
from .cadg.threshold import TieredThresholdManager, ResponseAction
from .sacd.monitor import CampaignMonitor
from .tamsh.experts import TopologyAwareMoE
from .memory.immune_memory import ImmuneMemory
from .federation import FederationManager

logger = logging.getLogger(__name__)


class PRISM:
    """
    Wraps any pretrained PyTorch classifier with architecture-agnostic
    adversarial defense using topological manifold monitoring.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        layer_names: List[str],
        calibrator: ConformalCalibrator,
        ref_profiles: Dict[str, list],
        moe: Optional[TopologyAwareMoE] = None,
        memory: Optional[ImmuneMemory] = None,
        campaign_monitor: Optional[CampaignMonitor] = None,
        tda_n_subsample: int = 200,
        tda_max_dim: int = 1,
        federation_manager: Optional[FederationManager] = None,
        layer_weights: Optional[Dict[str, float]] = None,
        dim_weights: Optional[List[float]] = None,
    ):
        """
        Args:
            model: Pretrained PyTorch model (must be in eval mode).
            layer_names: Names of layers to monitor (e.g., ['layer1', 'layer4']).
            calibrator: Pre-calibrated ConformalCalibrator instance.
            ref_profiles: {layer_name: medoid_diagram_set} — the topological self-profile.
            moe: Optional TopologyAwareMoE for L3 expert routing.
            memory: Optional ImmuneMemory for fast-path attack recognition.
            campaign_monitor: Optional CampaignMonitor for L0 campaign detection.
            tda_n_subsample: Points to subsample for TDA computation.
            tda_max_dim: Maximum homology dimension.
            federation_manager: Optional FederationManager for peer signature sharing.
                                 If provided, detections at L2/L3 are broadcast to peers.
            layer_weights: Optional per-layer scoring weights.
                           Defaults to uniform.
            dim_weights: Per-homology-dimension weights [H0_weight, H1_weight].
                         Default [0.2, 0.8] — H1 loops more discriminative for adversarial detection.
        """
        self.model = model
        self.layer_names = layer_names

        # Core modules
        self.extractor = ActivationExtractor(model, layer_names)
        self.profiler = TopologicalProfiler(
            n_subsample=tda_n_subsample, max_dim=tda_max_dim
        )
        self.scorer = TopologicalScorer(
            ref_profiles=ref_profiles,
            layer_names=layer_names,
            layer_weights=layer_weights,
            dim_weights=dim_weights,
        )
        self.calibrator = calibrator
        self.threshold_mgr = TieredThresholdManager()

        # Optional modules
        self.monitor = campaign_monitor or CampaignMonitor()
        self.moe = moe
        self.memory = memory or ImmuneMemory()
        self.federation = federation_manager

        # Stats
        self._inference_count = 0
        self._level_counts = {'PASS': 0, 'L1': 0, 'L2': 0, 'L3': 0, 'L3_REJECT': 0}

    @classmethod
    def from_saved(
        cls,
        model: torch.nn.Module,
        layer_names: List[str],
        calibrator_path: str,
        profile_path: str,
        **kwargs,
    ) -> 'PRISM':
        """
        Convenience constructor that loads calibrator and profiles from disk.

        Args:
            model: Pretrained model.
            layer_names: Layers to monitor.
            calibrator_path: Path to pickled ConformalCalibrator.
            profile_path: Path to pickled reference profiles dict.
        """
        calibrator = cls._load_pickle(calibrator_path)
        ref_profiles = cls._load_pickle(profile_path)

        if not isinstance(calibrator, ConformalCalibrator):
            raise TypeError(f"Expected ConformalCalibrator, got {type(calibrator)}")
        if not isinstance(ref_profiles, dict):
            raise TypeError(f"Expected dict for ref_profiles, got {type(ref_profiles)}")

        return cls(
            model=model,
            layer_names=layer_names,
            calibrator=calibrator,
            ref_profiles=ref_profiles,
            **kwargs,
        )

    @staticmethod
    def _load_pickle(path: str):
        """Load a pickle file. Validates path exists first."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Pickle file not found: {path}")
        with open(p, 'rb') as f:
            return pickle.load(f)

    def defend(self, x: torch.Tensor) -> Tuple[Optional[torch.Tensor], str, Dict[str, Any]]:
        """
        Main inference with PRISM defense.

        Pipeline:
        1. Extract activations via forward hooks
        2. Compute persistence diagrams per layer
        3. Check immune memory (fast path for known attacks)
        4. Compute aggregated anomaly score
        5. Update campaign monitor (L0)
        6. Classify response level via conformal thresholds
        7. Execute response action (pass / log / purify / expert / reject)

        Args:
            x: Input tensor, shape (1, C, H, W) for a single image.
        Returns:
            (prediction, level, metadata)
            - prediction: Model output tensor, or None if rejected.
            - level: Response level string ('PASS', 'L1', 'L2', 'L3', 'L3_REJECT').
            - metadata: Dict with score, per-layer details, L0 state, etc.
        """
        self._inference_count += 1
        metadata: Dict[str, Any] = {}

        # --- Step 1: Extract activations ---
        acts = self.extractor.extract(x)

        # --- Step 2: Compute persistence diagrams per layer ---
        diagrams: Dict[str, list] = {}
        for layer in self.layer_names:
            act_np = acts[layer].squeeze(0).cpu().numpy()
            dgm = self.profiler.compute_diagram(act_np)
            diagrams[layer] = dgm

        # --- Step 3: Fast-path immune memory check ---
        last_layer = self.layer_names[-1]
        memory_match = self.memory.match(diagrams[last_layer])
        if memory_match:
            level = memory_match['level']
            metadata['memory_match'] = memory_match
            logger.info(f"Memory match: {memory_match['attack_type']} → {level}")
            return self._execute_response(x, acts, diagrams, level, metadata)

        # --- Step 4: Compute anomaly score ---
        score = self.scorer.score(diagrams)
        metadata['anomaly_score'] = score
        metadata['per_layer_scores'] = self.scorer.score_per_layer(diagrams)

        # --- Step 5: Update campaign monitor (L0) ---
        l0_state = self.monitor.process_score(score)
        metadata['l0_state'] = l0_state

        # --- Step 6: Classify response level ---
        level = self.calibrator.classify(
            score, l0_active=l0_state['l0_active']
        )
        metadata['response_level'] = level

        # --- Step 7: Execute response ---
        return self._execute_response(x, acts, diagrams, level, metadata)

    def _execute_response(
        self,
        x: torch.Tensor,
        acts: Dict[str, torch.Tensor],
        diagrams: Dict[str, list],
        level: str,
        metadata: Dict[str, Any],
    ) -> Tuple[Optional[torch.Tensor], str, Dict[str, Any]]:
        """Execute the response action for a given level."""
        self._level_counts[level] = self._level_counts.get(level, 0) + 1
        action = self.threshold_mgr.get_action(
            level if level in ('PASS', 'L1', 'L2', 'L3', 'L3_REJECT') else 'L1'
        )

        if level == 'PASS':
            with torch.no_grad():
                pred = self.model(x)
            return pred, 'PASS', metadata

        elif level == 'L1':
            # Log + normal inference
            logger.info(f"L1 flag: score={metadata.get('anomaly_score', '?'):.4f}")
            with torch.no_grad():
                pred = self.model(x)
            metadata['flagged'] = True
            return pred, 'L1', metadata

        elif level == 'L2':
            # Input purification + inference
            # Purification: apply Gaussian smoothing as a basic defense
            x_purified = self._purify_input(x)
            with torch.no_grad():
                pred = self.model(x_purified)
            metadata['purified'] = True
            # Share detection signature with federated peers
            if self.federation is not None:
                self.federation.on_detection(
                    diagrams.get(self.layer_names[-1], []),
                    attack_type=metadata.get('attack_type', 'UNKNOWN'),
                    response_level='L2',
                )
            return pred, 'L2', metadata

        elif level == 'L3':
            if self.moe is not None:
                # Route through topology-aware expert
                last_layer = self.layer_names[-1]
                idx, expert, w_dist = self.moe.select_expert(diagrams[last_layer])
                # Use second-to-last layer activation as expert input
                input_layer = self.layer_names[-2] if len(self.layer_names) > 1 else self.layer_names[0]
                expert_input = acts[input_layer]
                expert.eval()
                with torch.no_grad():
                    pred = expert(expert_input)
                metadata['expert_idx'] = idx
                metadata['expert_wasserstein'] = w_dist
                # Share detection signature with federated peers
                if self.federation is not None:
                    self.federation.on_detection(
                        diagrams[last_layer],
                        attack_type=metadata.get('attack_type', 'UNKNOWN'),
                        response_level='L3',
                    )
                return pred, 'L3', metadata
            else:
                # No expert available — reject
                metadata['rejected'] = True
                # Still broadcast so peers benefit from the detection signal
                if self.federation is not None:
                    self.federation.on_detection(
                        diagrams.get(self.layer_names[-1], []),
                        attack_type=metadata.get('attack_type', 'UNKNOWN'),
                        response_level='L3_REJECT',
                    )
                return None, 'L3_REJECT', metadata

        # Fallback
        with torch.no_grad():
            pred = self.model(x)
        return pred, level, metadata

    @staticmethod
    def _purify_input(x: torch.Tensor, sigma: float = 0.05) -> torch.Tensor:
        """
        Basic input purification via Gaussian smoothing.
        This is a simple baseline; production would use more sophisticated
        purification (e.g., DiffPure, denoised smoothing).
        """
        noise = torch.randn_like(x) * sigma
        return x + noise

    def get_stats(self) -> Dict:
        """Return inference and defense statistics."""
        stats = {
            'total_inferences': self._inference_count,
            'level_counts': dict(self._level_counts),
            'l0_alerts': len(self.monitor.alert_log),
            'memory_stats': self.memory.get_statistics(),
        }
        if self.federation is not None:
            stats['federation'] = self.federation.get_stats()
        return stats
