"""
Expert Sub-Network Training (Phase 4, Weeks 14-17)

1. Loads clean persistence diagrams from the profiling phase
2. Clusters them into K topological regimes using K-medoids
3. Trains one expert sub-network per cluster
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torchvision.models import ResNet18_Weights
import numpy as np
import pickle
import os
import sys
import ssl
import certifi
from tqdm import tqdm

# Fix SSL certificate verification on macOS Python 3.11
os.environ.setdefault('SSL_CERT_FILE', certifi.where())
os.environ.setdefault('REQUESTS_CA_BUNDLE', certifi.where())
ssl._create_default_https_context = ssl.create_default_context

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.tamm.extractor import ActivationExtractor
from src.tamm.tda import TopologicalProfiler
from src.tamsh.experts import ExpertSubNetwork
from src.tamsh.gating import cluster_diagrams_by_topology


def train_experts(
    k: int = 4,
    hidden_dim: int = 256,
    n_samples: int = 2000,
    n_epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    data_root: str = './data',
    output_dir: str = './models',
    device: str = None,
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Training {k} expert sub-networks...")

    # --- Setup model ---
    model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model = model.to(device).eval()

    layer_names = ['layer2', 'layer3', 'layer4']
    extractor = ActivationExtractor(model, layer_names)
    profiler = TopologicalProfiler(n_subsample=200, max_dim=1)

    # --- Load data ---
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

    # --- Collect activations and diagrams ---
    print(f"\nCollecting activations from {n_samples} images...")
    all_diagrams = []
    layer3_acts = []
    layer4_acts = []

    for i, (img, _) in enumerate(tqdm(loader, total=n_samples)):
        if i >= n_samples:
            break

        acts = extractor.extract(img.to(device))

        # Compute persistence diagram from last layer
        act_np = acts['layer4'].squeeze(0).cpu().numpy()
        dgms = profiler.compute_diagram(act_np)
        all_diagrams.append(dgms)

        # Store activation pairs for training (layer3 → layer4)
        layer3_acts.append(acts['layer3'].cpu())
        layer4_acts.append(acts['layer4'].cpu())

    extractor.cleanup()

    # --- Cluster by topology ---
    print(f"\nClustering {len(all_diagrams)} diagrams into {k} groups...")
    # Use a random subsample for clustering (pairwise distances are O(n^2)).
    # Random subsampling (fixed seed=42) avoids bias from sequential ordering.
    max_cluster_samples = min(500, len(all_diagrams))
    cluster_rng = np.random.RandomState(42)
    cluster_idx = cluster_rng.choice(len(all_diagrams), max_cluster_samples, replace=False)
    cluster_subset = [all_diagrams[i] for i in cluster_idx]

    labels, medoid_diagrams = cluster_diagrams_by_topology(
        cluster_subset, k=k, dim=1
    )

    # Extend labels to full dataset using nearest medoid
    print("Assigning clusters to all samples...")
    full_labels = []
    for dgms in tqdm(all_diagrams):
        dists = []
        for med in medoid_diagrams:
            d = profiler.wasserstein_dist(
                dgms[1] if len(dgms) > 1 else np.array([]).reshape(0, 2),
                med[1] if len(med) > 1 else np.array([]).reshape(0, 2),
            )
            dists.append(d)
        full_labels.append(int(np.argmin(dists)))

    # Print cluster sizes
    for c in range(k):
        count = sum(1 for l in full_labels if l == c)
        print(f"  Cluster {c}: {count} samples")

    # --- Train expert per cluster ---
    # Expert: layer3_flat → layer4_flat
    l3_shape = layer3_acts[0].squeeze(0).shape  # (C3, H3, W3)
    l4_shape = layer4_acts[0].squeeze(0).shape  # (C4, H4, W4)
    input_dim = int(np.prod(l3_shape))
    output_dim = int(np.prod(l4_shape))
    print(f"\nExpert dimensions: {input_dim} → {output_dim}")

    experts = []
    for c in range(k):
        print(f"\n--- Training Expert {c} ---")
        cluster_indices = [i for i, l in enumerate(full_labels) if l == c]
        if len(cluster_indices) < 10:
            print(f"  WARNING: Only {len(cluster_indices)} samples. Skipping.")
            # Create untrained expert as placeholder
            expert = ExpertSubNetwork(input_dim, output_dim, hidden_dim)
            experts.append(expert)
            continue

        # Build training data for this cluster
        X = torch.cat([layer3_acts[i].view(1, -1) for i in cluster_indices])
        Y = torch.cat([layer4_acts[i].view(1, -1) for i in cluster_indices])

        # Create and train expert
        expert = ExpertSubNetwork(input_dim, output_dim, hidden_dim).to(device)
        optimizer = torch.optim.Adam(expert.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        train_dataset = torch.utils.data.TensorDataset(X, Y)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        expert.train()
        for epoch in range(n_epochs):
            total_loss = 0
            n_batches = 0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                pred = expert(x_batch)
                loss = loss_fn(pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 5 == 0 or epoch == 0:
                avg_loss = total_loss / max(n_batches, 1)
                print(f"  Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.6f}")

        expert.eval()
        experts.append(expert.cpu())

    # --- Save experts and medoid diagrams ---
    os.makedirs(output_dir, exist_ok=True)
    save_data = {
        'experts': [e.state_dict() for e in experts],
        'medoid_diagrams': medoid_diagrams,
        'input_dim': input_dim,
        'output_dim': output_dim,
        'hidden_dim': hidden_dim,
        'k': k,
    }
    save_path = os.path.join(output_dir, 'experts.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\nExperts saved to {save_path}")

    return experts, medoid_diagrams


if __name__ == '__main__':
    train_experts()
