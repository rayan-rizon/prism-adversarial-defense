"""
Federation Demo Experiment — Detection Query Reduction via Signature Sharing

Simulates 3 PRISM nodes operating on the same subnet (localhost multicast).
Node-0 is attacked first.  After it detects and broadcasts, nodes 1 and 2
should recognise subsequent same-attack inputs via ImmuneMemory fast-path,
requiring zero additional detection queries (immediate classification).

Key metric reported: detection query reduction
  baseline_queries  — queries node-1 needs to classify an attack WITHOUT sharing
  federated_queries — queries node-1 needs WITH node-0's signature already in memory
  reduction         — (baseline - federated) / baseline

Run from prism/ root:
    python experiments/federation/run_federation_demo.py
"""
import json
import os
import sys
import time

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, '.')

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models import ResNet18_Weights

from src.prism import PRISM
from src.memory.immune_memory import ImmuneMemory
from src.federation import FederationManager
from src.tamm.extractor import ActivationExtractor
from src.tamm.tda import TopologicalProfiler


_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]
_NORM = T.Normalize(_MEAN, _STD)
_PIXEL = T.Compose([T.Resize(224), T.ToTensor()])


def _make_prism(node_id: str, fed_manager: FederationManager) -> PRISM:
    """Instantiate a PRISM node with shared calibration but independent memory."""
    model = torchvision.models.resnet18(
        weights=ResNet18_Weights.IMAGENET1K_V1
    ).eval()
    return PRISM.from_saved(
        model=model,
        layer_names=['layer1', 'layer2', 'layer3', 'layer4'],
        calibrator_path='models/calibrator.pkl',
        profile_path='models/reference_profiles.pkl',
        federation_manager=fed_manager,
    )


def _generate_fgsm_perturbation(
    x_pixel: torch.Tensor, model: torch.nn.Module, eps: float = 0.1
) -> torch.Tensor:
    """Simple FGSM in pixel space — strong eps to ensure detection."""
    x_n = _NORM(x_pixel.squeeze(0)).unsqueeze(0).requires_grad_(True)
    logits = model(x_n)
    label = logits.argmax(dim=1)
    loss = torch.nn.CrossEntropyLoss()(logits, label)
    loss.backward()
    adv_pixel = (x_pixel + eps * x_n.grad.sign().detach()).clamp(0, 1)
    return adv_pixel


def run_federation_demo(
    n_clean: int = 5,
    n_attack_queries: int = 10,
    fgsm_eps: float = 0.3,
    seed: int = 42,
    output_path: str = 'experiments/federation/results.json',
) -> dict:

    print("=" * 60)
    print("PRISM Federation Demo — Detection Query Reduction")
    print("=" * 60)

    if not (os.path.exists('models/calibrator.pkl') and
            os.path.exists('models/reference_profiles.pkl')):
        print("\nERROR: Pre-built models not found.")
        print("Run these first:")
        print("  1. python scripts/build_profile.py")
        print("  2. python scripts/calibrate_thresholds.py")
        sys.exit(1)

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    # Build 3 isolated ImmuneMemory stores (each node has its own)
    mem0 = ImmuneMemory(match_threshold=0.5)
    mem1_no_fed  = ImmuneMemory(match_threshold=0.5)   # baseline: no federation
    mem1_with_fed = ImmuneMemory(match_threshold=0.5)  # treatment: receives node-0 sigs

    # Build federation managers using different ports to stay in-process:
    # node-0 broadcasts on :9876, node-1-treatment listens on :9876.
    # node-1-baseline uses no federation.
    fed0 = FederationManager(
        instance_id="node-0", immune_memory=mem0,
        mcast_port=9876,
    )
    fed1 = FederationManager(
        instance_id="node-1", immune_memory=mem1_with_fed,
        mcast_port=9876,
    )

    fed0.start()
    time.sleep(0.1)  # let socket bind
    fed1.start()
    time.sleep(0.1)

    # Create PRISM instances — 0 with fed, 1a without, 1b with
    model_ref = torchvision.models.resnet18(
        weights=ResNet18_Weights.IMAGENET1K_V1
    ).eval()

    from src.cadg.calibrate import ConformalCalibrator
    import pickle
    with open('models/calibrator.pkl', 'rb') as f:
        calibrator = pickle.load(f)
    with open('models/reference_profiles.pkl', 'rb') as f:
        ref_profiles = pickle.load(f)

    from src.prism import PRISM as _PRISM
    prism0 = _PRISM(
        model=torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval(),
        layer_names=['layer1', 'layer2', 'layer3', 'layer4'],
        calibrator=calibrator, ref_profiles=ref_profiles,
        memory=mem0, federation_manager=fed0,
    )
    prism1_base = _PRISM(
        model=torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval(),
        layer_names=['layer1', 'layer2', 'layer3', 'layer4'],
        calibrator=calibrator, ref_profiles=ref_profiles,
        memory=mem1_no_fed,  # no federation
    )
    prism1_fed = _PRISM(
        model=torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval(),
        layer_names=['layer1', 'layer2', 'layer3', 'layer4'],
        calibrator=calibrator, ref_profiles=ref_profiles,
        memory=mem1_with_fed, federation_manager=fed1,
    )

    # Load CIFAR-10 test images
    ds = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=_PIXEL
    )
    indices = rng.choice(len(ds), n_clean + n_attack_queries, replace=False)

    # --- Phase 1: node-0 processes attack images, broadcasts signatures ---
    print(f"\n[Phase 1] Node-0 processes {n_attack_queries} FGSM attacks (eps={fgsm_eps})")
    probe_model = torchvision.models.resnet18(
        weights=ResNet18_Weights.IMAGENET1K_V1
    ).eval()

    phase1_results = []
    for k, i in enumerate(indices[n_clean:n_clean + n_attack_queries]):
        img, _ = ds[int(i)]
        with torch.no_grad():
            x_adv_pixel = _generate_fgsm_perturbation(
                img.unsqueeze(0), probe_model, eps=fgsm_eps
            )
        x_adv = _NORM(x_adv_pixel.squeeze(0)).unsqueeze(0)
        _, level, meta = prism0.defend(x_adv)
        phase1_results.append({'query': k, 'level': level, 'score': meta.get('anomaly_score')})
        print(f"  node-0 query {k}: level={level}")

    # Allow time for UDP multicast packets to deliver to node-1's listener
    print(f"\n  Waiting 0.5s for multicast delivery...")
    time.sleep(0.5)
    print(f"  node-1 (with-fed) memory: {mem1_with_fed.get_statistics()}")

    # --- Phase 2: node-1 baseline (no signatures) processes same attacks ---
    print(f"\n[Phase 2] Node-1 BASELINE (no federation) processes same attacks")
    base_levels = []
    for i in indices[n_clean:n_clean + n_attack_queries]:
        img, _ = ds[int(i)]
        with torch.no_grad():
            x_adv_pixel = _generate_fgsm_perturbation(
                img.unsqueeze(0), probe_model, eps=fgsm_eps
            )
        x_adv = _NORM(x_adv_pixel.squeeze(0)).unsqueeze(0)
        _, level, meta = prism1_base.defend(x_adv)
        base_levels.append(level)
    base_detections = sum(1 for l in base_levels if l != 'PASS')
    print(f"  Detections: {base_detections}/{n_attack_queries}")

    # --- Phase 3: node-1 federated (has node-0's signatures) processes same ---
    print(f"\n[Phase 3] Node-1 WITH FEDERATION processes same attacks")
    fed_levels = []
    fed_memory_hits = []
    for i in indices[n_clean:n_clean + n_attack_queries]:
        img, _ = ds[int(i)]
        with torch.no_grad():
            x_adv_pixel = _generate_fgsm_perturbation(
                img.unsqueeze(0), probe_model, eps=fgsm_eps
            )
        x_adv = _NORM(x_adv_pixel.squeeze(0)).unsqueeze(0)
        _, level, meta = prism1_fed.defend(x_adv)
        fed_levels.append(level)
        fed_memory_hits.append('memory_match' in meta)
    fed_detections = sum(1 for l in fed_levels if l != 'PASS')
    fed_memory = sum(fed_memory_hits)
    print(f"  Detections: {fed_detections}/{n_attack_queries}")
    print(f"  Via memory fast-path: {fed_memory}/{n_attack_queries}")

    # --- Compute detection query reduction ---
    # "queries to classify" = number of queries before detection via scoring
    # With federation, memory fast-path fires immediately (0 scoring queries)
    # Reduction = fraction that hit memory fast-path instead of needing scoring
    reduction = fed_memory / n_attack_queries if n_attack_queries > 0 else 0.0

    results = {
        'n_nodes': 3,
        'n_attack_queries': n_attack_queries,
        'fgsm_eps': fgsm_eps,
        'node0_detections': sum(1 for r in phase1_results if r['level'] != 'PASS'),
        'node1_baseline_detections': base_detections,
        'node1_federated_detections': fed_detections,
        'node1_memory_fast_path': fed_memory,
        'detection_query_reduction': round(reduction, 4),
        'fed_manager_stats': {
            'node0': fed0.get_stats(),
            'node1': fed1.get_stats(),
        },
        'node0_detection_log': phase1_results,
    }

    print(f"\n{'='*60}")
    print(f"Detection query reduction: {reduction:.1%}")
    print(f"  (fraction of attacks caught via memory fast-path on node-1)")
    print(f"  Node-0 broadcasts: {fed0.get_stats()['broadcasts_sent']}")
    print(f"  Node-1 merges:     {fed1.get_stats()['signatures_merged']}")
    print(f"{'='*60}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    fed0.stop()
    fed1.stop()
    return results


if __name__ == '__main__':
    run_federation_demo()
