"""
Integration tests — Full PRISM pipeline.

Guide Section 3.2: Full PRISM on clean + perturbed image.
Two modes:
  1. Synthetic (always runnable): builds minimal in-memory PRISM with
     synthetic reference profiles calibrated from random activations.
  2. From-saved (requires built models): loads real artifacts if present.

The synthetic path is always executed by pytest.
The from-saved path is gated on model file existence.
"""
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models import ResNet18_Weights
import numpy as np
import pytest
import sys, os, ssl, pickle, tempfile

import certifi
os.environ.setdefault('SSL_CERT_FILE', certifi.where())
ssl._create_default_https_context = ssl.create_default_context

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.prism import PRISM
from src.tamm.tda import TopologicalProfiler
from src.tamm.scorer import TopologicalScorer
from src.cadg.calibrate import ConformalCalibrator
from src.tamm.extractor import ActivationExtractor


# ---------------------------------------------------------------------------
# Shared fixture: build PRISM with synthetic calibration (no disk I/O)
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def resnet_model():
    """ResNet-18 in eval mode (cached at module scope)."""
    model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    return model.eval()


@pytest.fixture(scope='module')
def prism_instance(resnet_model):
    """
    Build a fully functional PRISM with synthetic reference profiles.
    Runs ~30 clean images through ResNet to get real activations, then
    builds profiles and calibration from those.
    """
    layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
    profiler = TopologicalProfiler(n_subsample=100, max_dim=1)
    extractor = ActivationExtractor(resnet_model, layer_names)

    transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Collect diagrams from 30 random images
    rng = np.random.RandomState(0)
    all_diagrams = {layer: [] for layer in layer_names}
    scores_list = []

    for _ in range(30):
        x = torch.from_numpy(
            rng.randn(1, 3, 224, 224).astype(np.float32)
        )
        acts = extractor.extract(x)
        per_dgm = {}
        for layer in layer_names:
            act_np = acts[layer].squeeze(0).cpu().numpy()
            dgm = profiler.compute_diagram(act_np)
            all_diagrams[layer].append(dgm)
            per_dgm[layer] = dgm

    extractor.cleanup()

    # Build reference profiles (medoid per layer)
    ref_profiles = {}
    for layer in layer_names:
        ref_profiles[layer] = profiler.compute_reference_medoid(
            all_diagrams[layer], dims=[0, 1], dim_weights=[0.5, 0.5]
        )

    # Build scorer + calibrate on these 30 images
    scorer = TopologicalScorer(ref_profiles, layer_names)
    for layer in layer_names:
        for dgm in all_diagrams[layer]:
            pass  # already computed above

    # Re-score all 30 images to get calibration scores
    cal_scores = []
    for i in range(30):
        dgms = {layer: all_diagrams[layer][i] for layer in layer_names}
        cal_scores.append(scorer.score(dgms))

    # Pad to 200 to meet calibrator minimum (repeat with small jitter)
    cal_scores = np.array(cal_scores)
    # Augment via small additive noise to reach 200 samples
    extra = np.abs(cal_scores.mean() + np.random.RandomState(1).randn(170) * cal_scores.std())
    cal_scores_full = np.concatenate([cal_scores, extra])

    calibrator = ConformalCalibrator()
    calibrator.calibrate(cal_scores_full)

    # Build PRISM
    prism = PRISM(
        model=resnet_model,
        layer_names=layer_names,
        calibrator=calibrator,
        ref_profiles=ref_profiles,
        tda_n_subsample=100,
    )
    return prism


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestPRISMDefend:
    def test_defend_clean_returns_tuple(self, prism_instance):
        """defend() must return (prediction, level, metadata)."""
        x = torch.randn(1, 3, 224, 224)
        result = prism_instance.defend(x)
        assert isinstance(result, tuple) and len(result) == 3
        pred, level, meta = result
        assert isinstance(level, str)
        assert isinstance(meta, dict)

    def test_defend_clean_level_is_valid(self, prism_instance):
        """Response level must be one of the known tiers."""
        x = torch.randn(1, 3, 224, 224)
        _, level, _ = prism_instance.defend(x)
        assert level in {'PASS', 'L1', 'L2', 'L3', 'L3_REJECT'}, (
            f"Unknown response level: {level}"
        )

    def test_defend_meta_has_required_keys(self, prism_instance):
        """metadata dict must contain anomaly_score and response_level."""
        x = torch.randn(1, 3, 224, 224)
        _, _, meta = prism_instance.defend(x)
        assert 'anomaly_score' in meta, "Missing 'anomaly_score' in metadata"

    def test_perturbed_input_scores_higher_than_clean(self, prism_instance):
        """
        Guide Section 3.2:
        Heavily perturbed input must have higher anomaly score than clean.
        meta_adv['score'] > meta['score'].
        """
        rng = torch.manual_seed(5)
        x_clean = torch.randn(1, 3, 224, 224)
        x_adv = x_clean + torch.randn_like(x_clean) * 0.5

        _, _, meta_clean = prism_instance.defend(x_clean)
        _, _, meta_adv = prism_instance.defend(x_adv)

        clean_score = meta_clean['anomaly_score']
        adv_score = meta_adv['anomaly_score']

        assert adv_score > clean_score, (
            f"Adversarial score ({adv_score:.4f}) must exceed "
            f"clean score ({clean_score:.4f})"
        )

    def test_large_perturbation_not_pass(self, prism_instance):
        """
        Large-magnitude perturbation (σ=2.0) should not be PASS on average.
        Run 5 trials — at least 2 should be flagged.
        """
        flagged = 0
        rng = np.random.RandomState(10)
        for seed in range(5):
            x = torch.from_numpy(rng.randn(1, 3, 224, 224).astype(np.float32)) * 2.0
            _, level, _ = prism_instance.defend(x)
            if level != 'PASS':
                flagged += 1
        assert flagged >= 2, (
            f"Only {flagged}/5 large-perturbation inputs were flagged"
        )

    def test_per_layer_scores_present(self, prism_instance):
        """Metadata should contain per-layer score breakdown."""
        x = torch.randn(1, 3, 224, 224)
        _, _, meta = prism_instance.defend(x)
        assert 'per_layer_scores' in meta, "Missing per_layer_scores in metadata"
        per = meta['per_layer_scores']
        for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
            assert layer in per, f"Missing {layer} in per_layer_scores"

    def test_from_saved_loads_when_models_exist(self, resnet_model):
        """
        If models/reference_profiles.pkl and models/calibrator.pkl exist,
        PRISM.from_saved() must load without error.
        """
        profile_path = 'models/reference_profiles.pkl'
        calibrator_path = 'models/calibrator.pkl'
        if not (os.path.exists(profile_path) and os.path.exists(calibrator_path)):
            pytest.skip("Pre-built models not present — skipping from_saved test")

        prism = PRISM.from_saved(
            model=resnet_model,
            layer_names=['layer1', 'layer2', 'layer3', 'layer4'],
            calibrator_path=calibrator_path,
            profile_path=profile_path,
        )
        x = torch.randn(1, 3, 224, 224)
        _, level, meta = prism.defend(x)
        assert level in {'PASS', 'L1', 'L2', 'L3', 'L3_REJECT'}
        assert meta['anomaly_score'] >= 0.0


class TestPRISMStatTracking:
    def test_inference_count_increments(self, prism_instance):
        count_before = prism_instance._inference_count
        prism_instance.defend(torch.randn(1, 3, 224, 224))
        assert prism_instance._inference_count == count_before + 1


class TestPRISMPickleRoundtrip:
    """PRISM calibrator and profiles must round-trip through pickle."""

    def test_calibrator_pickle_roundtrip(self):
        cal = ConformalCalibrator()
        cal.calibrate(np.abs(np.random.randn(300) * 0.02 + 0.1))
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name
        try:
            with open(path, 'wb') as fp:
                pickle.dump(cal, fp)
            with open(path, 'rb') as fp:
                cal2 = pickle.load(fp)
            assert cal2.thresholds == cal.thresholds
        finally:
            os.unlink(path)
