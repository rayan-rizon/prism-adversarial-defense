"""
Build Topological Self-Profile (Phase 1, Week 3-5)

Passes clean images through the model, collects per-layer persistence diagrams,
computes medoid reference diagrams, and saves them as the "topological self-profile."

Key fixes from plan:
1. Uses torchvision weights= API instead of deprecated pretrained=True
2. Actually computes medoid reference diagrams (plan only saved raw lists)
3. Computes and saves anomaly scores for the calibration phase
"""
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models import ResNet18_Weights
import numpy as np
import pickle
import os
import sys
import ssl
import certifi
from pathlib import Path
from tqdm import tqdm

# Fix SSL certificate verification on macOS Python 3.11
os.environ.setdefault('SSL_CERT_FILE', certifi.where())
os.environ.setdefault('REQUESTS_CA_BUNDLE', certifi.where())
ssl._create_default_https_context = ssl.create_default_context

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.tamm.extractor import ActivationExtractor
from src.tamm.tda import TopologicalProfiler
from src.tamm.scorer import TopologicalScorer


def build_profile(
    n_images: int = 10000,
    n_subsample: int = 200,
    data_root: str = './data',
    output_dir: str = './models',
    device: str = None,
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # --- Setup model ---
    model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model = model.to(device).eval()

    layer_names = ['layer2', 'layer3', 'layer4']
    # Deeper layers encode higher-level features with stronger adversarial signal.
    # layer2: 0.15, layer3: 0.30, layer4: 0.55 — validated against ablation results.
    layer_weights = {'layer2': 0.15, 'layer3': 0.30, 'layer4': 0.55}
    extractor = ActivationExtractor(model, layer_names)
    profiler = TopologicalProfiler(n_subsample=n_subsample, max_dim=1)

    # --- Setup data (CIFAR-10 resized to 224 for ResNet) ---
    # CRITICAL: these transforms must match exactly at inference time
    transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=2
    )

    # --- Collect persistence diagrams ---
    all_diagrams = {layer: [] for layer in layer_names}
    print(f"Collecting diagrams from {n_images} clean images...")

    for i, (img, _) in enumerate(tqdm(loader, total=n_images)):
        if i >= n_images:
            break

        acts = extractor.extract(img.to(device))

        for layer_name in layer_names:
            act_np = acts[layer_name].squeeze(0).cpu().numpy()
            dgms = profiler.compute_diagram(act_np)
            all_diagrams[layer_name].append(dgms)

    # --- Compute medoid reference diagrams per layer ---
    print("\nComputing medoid reference diagrams...")
    ref_profiles = {}
    for layer_name in layer_names:
        print(f"  {layer_name}: computing medoid from {len(all_diagrams[layer_name])} diagrams...")
        medoid = profiler.compute_reference_medoid(
            all_diagrams[layer_name], dim=1
        )
        ref_profiles[layer_name] = medoid
        n_h0 = len(medoid[0]) if len(medoid) > 0 else 0
        n_h1 = len(medoid[1]) if len(medoid) > 1 else 0
        print(f"    Medoid: {n_h0} H0 features, {n_h1} H1 features")

    # --- Save reference profiles ---
    os.makedirs(output_dir, exist_ok=True)
    profile_path = os.path.join(output_dir, 'reference_profiles.pkl')
    with open(profile_path, 'wb') as f:
        pickle.dump(ref_profiles, f)
    print(f"\nReference profiles saved to {profile_path}")

    # --- Compute anomaly scores for calibration phase ---
    print("\nComputing anomaly scores for calibration...")
    scorer = TopologicalScorer(
        ref_profiles=ref_profiles,
        layer_names=layer_names,
        layer_weights=layer_weights,
    )

    scores = []
    for i in tqdm(range(min(n_images, len(all_diagrams[layer_names[0]]))),
                  desc="Scoring"):
        input_dgms = {layer: all_diagrams[layer][i] for layer in layer_names}
        s = scorer.score(input_dgms)
        scores.append(s)

    scores = np.array(scores)
    scores_path = os.path.join(output_dir, '..', 'experiments', 'calibration')
    os.makedirs(scores_path, exist_ok=True)
    np.save(os.path.join(scores_path, 'clean_scores.npy'), scores)
    print(f"Clean scores saved: mean={scores.mean():.4f}, std={scores.std():.4f}")
    print(f"  min={scores.min():.4f}, max={scores.max():.4f}")

    extractor.cleanup()
    print("\nTopological self-profile built successfully!")
    return ref_profiles, scores


if __name__ == '__main__':
    build_profile()
