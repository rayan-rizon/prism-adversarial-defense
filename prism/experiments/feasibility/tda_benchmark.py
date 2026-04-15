"""
TDA Feasibility Benchmark — THE CRITICAL GATE (Phase 1, Week 2-3)

Measures TDA computation time at various subsample sizes and layers.
This determines whether PRISM is viable at inference time.

Decision matrix:
  < 10ms per layer  → Proceed as planned
  10-50ms           → Proceed with subsampling strategy
  50-200ms          → Switch to cubical complexes or landmark-based approx
  > 200ms           → STOP. Consider ripser++ (GPU-accelerated)
"""
import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import numpy as np
import time
import sys
import os
import ssl
import certifi

# Fix SSL certificate verification on macOS Python 3.11
os.environ.setdefault('SSL_CERT_FILE', certifi.where())
os.environ.setdefault('REQUESTS_CA_BUNDLE', certifi.where())
ssl._create_default_https_context = ssl.create_default_context

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ripser import ripser
from gudhi.wasserstein import wasserstein_distance


def run_benchmark():
    # 1. Load pretrained ResNet-18 (fixed: use weights= instead of pretrained=)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model = model.to(device).eval()

    # 2. Register forward hooks
    activations = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            activations[name] = output.detach().cpu().numpy()
        return hook_fn

    target_layers = ['layer1', 'layer2', 'layer3', 'layer4']
    module_dict = dict(model.named_modules())
    for name in target_layers:
        hook = module_dict[name].register_forward_hook(make_hook(name))
        hooks.append(hook)

    # 3. Forward pass with dummy input
    x = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        _ = model(x)

    # Print activation shapes
    print("\nActivation shapes:")
    for name in target_layers:
        print(f"  {name}: {activations[name].shape}")

    # 4. Benchmark TDA at different subsample sizes
    print(f"\n{'n_points':>10} {'layer':>8} {'spatial':>10} "
          f"{'TDA_ms':>10} {'Wass_ms':>10} {'H0_pts':>8} {'H1_pts':>8}")
    print("-" * 76)

    results = []

    for n_points in [50, 100, 200, 500, 1000]:
        for layer_name in target_layers:
            act = activations[layer_name]
            # Shape: (1, C, H, W) -> squeeze batch dim
            act = act.squeeze(0)  # (C, H, W)
            C, H, W = act.shape

            # Convert to point cloud: each spatial location is a point in C-dim space
            flat = act.reshape(C, -1).T  # (H*W, C)

            # Random subsample
            if flat.shape[0] > n_points:
                idx = np.random.choice(flat.shape[0], n_points, replace=False)
                pts = flat[idx]
            else:
                pts = flat

            # Measure TDA time (suppress expected transpose warning for low-spatial layers)
            start = time.perf_counter()
            with __import__('warnings').catch_warnings():
                __import__('warnings').filterwarnings("ignore", message=".*more columns than rows.*")
                result = ripser(pts, maxdim=1)
            tda_time = (time.perf_counter() - start) * 1000  # ms

            dgm_h0 = result['dgms'][0]
            dgm_h1 = result['dgms'][1]

            # Measure Wasserstein distance time (vs. slightly perturbed diagram)
            wass_time = 0.0
            if len(dgm_h1) > 0:
                ref_dgm = dgm_h1 + np.random.normal(0, 0.01, dgm_h1.shape)
                start = time.perf_counter()
                _ = wasserstein_distance(dgm_h1, ref_dgm, order=2)
                wass_time = (time.perf_counter() - start) * 1000

            print(f"{n_points:>10} {layer_name:>8} {H*W:>10} "
                  f"{tda_time:>10.1f} {wass_time:>10.1f} "
                  f"{len(dgm_h0):>8} {len(dgm_h1):>8}")

            results.append({
                'n_points': n_points,
                'layer': layer_name,
                'spatial_dim': H * W,
                'tda_ms': tda_time,
                'wasserstein_ms': wass_time,
                'h0_features': len(dgm_h0),
                'h1_features': len(dgm_h1),
            })

    # Cleanup hooks
    for h in hooks:
        h.remove()

    # Print decision
    target_result = [r for r in results if r['n_points'] == 200 and r['layer'] == 'layer3']
    if target_result:
        t = target_result[0]['tda_ms']
        print(f"\n=== DECISION POINT ===")
        print(f"TDA time at n=200, layer3: {t:.1f}ms")
        if t < 10:
            print("✓ PROCEED as planned — TDA is fast enough for real-time")
        elif t < 50:
            print("✓ PROCEED with subsampling strategy")
        elif t < 200:
            print("⚠ Consider cubical complexes or landmark-based approximation")
        else:
            print("✗ STOP — TDA too slow. Consider ripser++ (GPU-accelerated)")

    return results


if __name__ == '__main__':
    run_benchmark()
