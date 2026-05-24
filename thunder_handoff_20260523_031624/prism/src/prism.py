"""
PRISM: Predictive Runtime Immune System with Manifold Monitoring

Main integration class that wraps any pretrained PyTorch model with
the full PRISM defense pipeline:
  L0 (SACD) → TAMM → CADG → TAMSH → Immune Memory

Key fixes from plan:
1. Reference profiles are stored as medoid diagrams per layer (not raw 10K lists)
2. Scorer module handles multi-layer, multi-dimension aggregation properly
3. Transform consistency enforced between profiling and inference
4. Runtime artifact loading is restricted to local project artifacts rather
   than arbitrary filesystem paths.
"""
import torch
import torch.nn.functional as F
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
from .config import N_SUBSAMPLE, MAX_DIM, BACKBONE_MEAN, BACKBONE_STD
from .tamm.logit_stability import compute_input_stability_features
from .tamm.persistence_stats import compute_logit_profile_features

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
        ensemble_scorer: Optional[Any] = None,
        moe: Optional[TopologyAwareMoE] = None,
        memory: Optional[ImmuneMemory] = None,
        campaign_monitor: Optional[CampaignMonitor] = None,
        tda_n_subsample: int = N_SUBSAMPLE,
        tda_max_dim: int = MAX_DIM,
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
            ensemble_scorer: Optional PersistenceEnsembleScorer for improved detection.
            moe: Optional TopologyAwareMoE for L3 expert routing.
            memory: Optional ImmuneMemory for fast-path attack recognition.
            campaign_monitor: Optional CampaignMonitor for L0 campaign detection.
            tda_n_subsample: Points to subsample for TDA computation.
            tda_max_dim: Maximum homology dimension.
            federation_manager: Optional FederationManager for peer signature sharing.
            layer_weights: Optional per-layer scoring weights.
            dim_weights: Per-homology-dimension weights [H0_weight, H1_weight].
        """
        self.model = model
        self.layer_names = layer_names

        # Core modules
        self.extractor = ActivationExtractor(model, layer_names)
        self.profiler = TopologicalProfiler(
            n_subsample=tda_n_subsample, max_dim=tda_max_dim
        )
        if ensemble_scorer is not None:
            self.scorer = ensemble_scorer
        else:
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
        ensemble_path: Optional[str] = None,
        **kwargs,
    ) -> 'PRISM':
        """
        Convenience constructor that loads calibrator and profiles from disk.

        Args:
            model: Pretrained model.
            layer_names: Layers to monitor.
            calibrator_path: Path to pickled ConformalCalibrator.
            profile_path: Path to pickled reference profiles dict.
            ensemble_path: Optional path to pickled PersistenceEnsembleScorer.
        """
        calibrator = cls._load_pickle(calibrator_path)
        ref_profiles = cls._load_pickle(profile_path)

        if not isinstance(calibrator, ConformalCalibrator):
            raise TypeError(f"Expected ConformalCalibrator, got {type(calibrator)}")
        if not isinstance(ref_profiles, dict):
            raise TypeError(f"Expected dict for ref_profiles, got {type(ref_profiles)}")

        ensemble_scorer = None
        if ensemble_path is not None:
            # When the caller explicitly requests an ensemble, missing file is a
            # hard error — silent fallback to the baseline TopologicalScorer
            # would invisibly degrade detection (no logistic component, no α
            # fusion) and falsify any "PRISM full" claim downstream. Callers
            # that genuinely want the optional path can pass ensemble_path=None.
            if not Path(ensemble_path).exists():
                raise FileNotFoundError(
                    f"PRISM.from_saved: ensemble_path='{ensemble_path}' does not "
                    "exist. Run scripts/train_ensemble_scorer.py to produce it, "
                    "or pass ensemble_path=None to run with the baseline "
                    "TopologicalScorer (Wasserstein-only, no logistic fusion)."
                )
            # PersistenceEnsembleScorer requires loading with base_scorer
            from .cadg.ensemble_scorer import PersistenceEnsembleScorer
            from .tamm.scorer import TopologicalScorer
            base_scorer = TopologicalScorer(
                ref_profiles=ref_profiles,
                layer_names=layer_names,
                layer_weights=kwargs.get('layer_weights'),
                dim_weights=kwargs.get('dim_weights'),
            )
            ensemble_scorer = PersistenceEnsembleScorer.load(
                ensemble_path, base_scorer, layer_names
            )

        return cls(
            model=model,
            layer_names=layer_names,
            calibrator=calibrator,
            ref_profiles=ref_profiles,
            ensemble_scorer=ensemble_scorer,
            **kwargs,
        )

    @staticmethod
    def _load_pickle(path: str):
        """Load a local PRISM artifact pickle after path validation."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Pickle file not found: {path}")
        resolved = p.resolve()
        project_root = Path(__file__).resolve().parents[1]
        cwd = Path.cwd().resolve()
        if not (
            resolved.is_relative_to(project_root)
            or resolved.is_relative_to(cwd)
        ):
            raise ValueError(
                f"Refusing to load pickle outside trusted project paths: {resolved}"
            )
        with open(resolved, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def _normalised_to_pixel_numpy(
        x: torch.Tensor,
        pixel_image: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        Return a (C,H,W) pixel-space [0,1] numpy image for feature extractors.

        PRISM's backbone path receives normalised tensors. DCT features are
        trained on pixel-space tensors, so runtime scoring must denormalise
        before computing DCT energy. Callers that already have the exact pixel
        tensor can pass it through pixel_image.
        """
        if pixel_image is not None:
            pix = pixel_image.detach()
            if pix.dim() == 4:
                pix = pix.squeeze(0)
            return pix.clamp(0.0, 1.0).cpu().numpy().astype(np.float32)

        mean = torch.tensor(BACKBONE_MEAN, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        std = torch.tensor(BACKBONE_STD, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        pix = (x.detach() * std + mean).clamp(0.0, 1.0)
        return pix.squeeze(0).cpu().numpy().astype(np.float32)

    @staticmethod
    def _pixel_tensor(
        x: torch.Tensor,
        pixel_image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return a 4D pixel-space tensor on x.device matching x."""
        if pixel_image is not None:
            pix = pixel_image.detach().to(device=x.device, dtype=x.dtype)
            if pix.dim() == 3:
                pix = pix.unsqueeze(0)
            return pix.clamp(0.0, 1.0)

        mean = torch.tensor(BACKBONE_MEAN, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        std = torch.tensor(BACKBONE_STD, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        return (x.detach() * std + mean).clamp(0.0, 1.0)

    def _stability_features(
        self,
        x: torch.Tensor,
        pixel_image: Optional[torch.Tensor],
        logits_np: Optional[np.ndarray],
    ) -> np.ndarray:
        """Compute the scorer's configured deterministic stability block."""
        feature_count = int(getattr(self.scorer, 'stability_feature_count', 4))
        return compute_input_stability_features(
            model=self.model,
            x_norm=x,
            img_pixel=pixel_image,
            mean=BACKBONE_MEAN,
            std=BACKBONE_STD,
            logits_np=logits_np,
            feature_count=feature_count,
        )

    def defend(
        self,
        x: torch.Tensor,
        pixel_image: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], str, Dict[str, Any]]:
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
            x: Normalised input tensor, shape (1, C, H, W), for a single image.
            pixel_image: Optional matching pixel-space [0,1] tensor used only
                for pixel-domain features such as DCT energy.
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
        use_dct      = getattr(self.scorer, 'use_dct', False)
        use_grad_norm = getattr(self.scorer, 'use_grad_norm', False)
        use_softmax_entropy = getattr(self.scorer, 'use_softmax_entropy', False)
        use_logit_profile_features = getattr(self.scorer, 'use_logit_profile_features', False)
        use_stability_features = getattr(self.scorer, 'use_stability_features', False)

        img_np = (
            self._normalised_to_pixel_numpy(x, pixel_image=pixel_image)
            if use_dct else None
        )
        grad_norm = None
        if use_grad_norm:
            x_g = x.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                logits_g = self.model(x_g)
                pred_idx = int(logits_g.argmax(1).item())
                # autograd.grad avoids accumulating model parameter gradients
                (grad_x,) = torch.autograd.grad(logits_g[0, pred_idx], x_g)
            grad_norm = float(grad_x.norm().item())

        # Compute model logits for softmax-entropy feature (CW-L2 detection).
        # This forward pass is cheap (~5ms) and captures decision-boundary
        # proximity that TDA features cannot detect.
        logits_np = None
        if use_softmax_entropy or use_logit_profile_features or use_stability_features:
            with torch.no_grad():
                logits_out = self.model(x)
            logits_np = logits_out.squeeze(0).cpu().numpy()

        logit_profile_features = None
        if use_logit_profile_features:
            logit_profile_features = compute_logit_profile_features(logits_np)

        stability_features = None
        if use_stability_features:
            stability_features = self._stability_features(x, pixel_image, logits_np)

        score = self.scorer.score(
            diagrams,
            image=img_np,
            grad_norm=grad_norm,
            logits=logits_np,
            logit_profile_features=logit_profile_features,
            stability_features=stability_features,
        )
        metadata['anomaly_score'] = score
        if use_dct:
            metadata['feature_space_version'] = getattr(
                self.scorer, 'feature_space_version', 'pixel-v1'
            )
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
                # Experts are trained on the final monitored layer activation.
                expert_input = acts[last_layer]
                if expert_input.dim() > 2:
                    expert_input = F.adaptive_avg_pool2d(expert_input, 1).view(
                        expert_input.size(0), -1
                    )
                expert.eval()
                # Ensure expert is on the same device as input
                device = expert_input.device
                expert = expert.to(device)
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
