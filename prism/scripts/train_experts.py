"""
Train Differentiated MoE Experts for TAMSH (P0.5)

Each expert is a small MLP that maps the *final monitored layer*'s activation
→ ImageNet-1000 logits, trained on a specific attack distribution so the
four experts are genuinely differentiated (not just independent clones). The
Wasserstein router can then route e.g. PGD-shaped perturbations to the
PGD-specialised expert.

Per-expert training mix (attack → target logits from the clean backbone on
the original clean image; this is "distillation of the clean prediction"
under the attack's perturbation — the expert learns to see through the attack):

    Expert 0: Clean only (availability baseline)
    Expert 1: FGSM (ε=8/255)  — single-step
    Expert 2: PGD-10 (ε=8/255) — iterative
    Expert 3: Mixed (clean + FGSM + PGD + Square at 25% each)

Outputs:
    models/experts.pkl — dict with keys:
        'experts': list of 4 state_dicts
        'input_dim', 'output_dim', 'hidden_dim'
        'medoid_diagrams': list of 4 reference diagram sets (one per expert's
                           training distribution, used by the Wasserstein router)

USAGE
-----
    cd prism/
    python scripts/train_experts.py --n-train 3000 --epochs 3
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.models import ResNet18_Weights
import numpy as np
import os, sys, ssl, certifi, pickle, argparse
from tqdm import tqdm

os.environ.setdefault('SSL_CERT_FILE', certifi.where())
os.environ.setdefault('REQUESTS_CA_BUNDLE', certifi.where())
ssl._create_default_https_context = ssl.create_default_context

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Route --config CLI flag to PRISM_CONFIG env var BEFORE importing src.config.
from src import bootstrap  # noqa: F401

from art.attacks.evasion import (
    FastGradientMethod,
    ProjectedGradientDescent,
    SquareAttack,
)
from art.estimators.classification import PyTorchClassifier

from src.tamm.extractor import ActivationExtractor
from src.tamm.tda import TopologicalProfiler
from src.tamsh.experts import ExpertSubNetwork, TopologyAwareMoE
from src.config import (
    LAYER_NAMES, IMAGENET_MEAN, IMAGENET_STD, EPS_LINF_STANDARD,
    CAL_IDX, N_SUBSAMPLE, MAX_DIM, DATASET, PATHS,
)
from src.data_loader import load_test_dataset

_MEAN = IMAGENET_MEAN
_STD  = IMAGENET_STD
_PIXEL_TRANSFORM = T.Compose([T.Resize(224), T.ToTensor()])
_NORMALIZE = T.Normalize(mean=_MEAN, std=_STD)


class _NormalizedResNet(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self._model = model
        self.register_buffer('_mean', torch.tensor(_MEAN).view(3, 1, 1))
        self.register_buffer('_std',  torch.tensor(_STD).view(3, 1, 1))

    def forward(self, x):
        return self._model((x - self._mean) / self._std)


def _gen_perturbation(attack_name, pixel_imgs_np, classifier, eps):
    if attack_name == 'clean':
        return pixel_imgs_np.copy()
    if attack_name == 'fgsm':
        return FastGradientMethod(classifier, eps=eps).generate(pixel_imgs_np)
    if attack_name == 'pgd':
        return ProjectedGradientDescent(
            classifier, eps=eps, eps_step=eps / 4, max_iter=10, num_random_init=1
        ).generate(pixel_imgs_np)
    if attack_name == 'square':
        return SquareAttack(classifier, eps=eps, max_iter=1000, nb_restarts=1
                            ).generate(pixel_imgs_np)
    raise ValueError(attack_name)


def train_experts(n_train=3000, epochs=3, batch_size=32, hidden_dim=256,
                  lr=1e-3, seed=42, device_str=None, data_root='./data',
                  output='models/experts.pkl'):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    device = torch.device(device_str) if device_str else \
             torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Model + dataset ──
    model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model = model.to(device).eval()
    norm_model = _NormalizedResNet(model).to(device).eval()
    device_type = 'gpu' if device.type == 'cuda' else 'cpu'
    classifier = PyTorchClassifier(
        model=norm_model, loss=nn.CrossEntropyLoss(),
        input_shape=(3, 224, 224), nb_classes=1000,
        clip_values=(0.0, 1.0), device_type=device_type,
    )

    ds = load_test_dataset(root=data_root, download=True, transform=_PIXEL_TRANSFORM)
    # Train on the CAL split so the eval split stays clean.
    cal_start, cal_end = CAL_IDX
    train_idx = rng.choice(range(cal_start, cal_end), min(n_train, cal_end - cal_start),
                           replace=False)
    print(f"Training on {len(train_idx)} images from {DATASET.upper()} CAL split")

    pixel_imgs = [ds[int(i)][0] for i in train_idx]
    X_pixel = torch.stack(pixel_imgs).numpy()

    # ── Activation extractor + profiler (for medoid diagrams per expert) ──
    extractor = ActivationExtractor(model, LAYER_NAMES)
    profiler = TopologicalProfiler(n_subsample=N_SUBSAMPLE, max_dim=MAX_DIM)

    # ── Compute clean backbone targets (soft distillation) ──
    print("Computing clean-backbone target logits...")
    X_pixel_t = torch.tensor(X_pixel).to(device)
    target_logits_list = []
    with torch.no_grad():
        for i in range(0, len(X_pixel_t), 64):
            target_logits_list.append(norm_model(X_pixel_t[i:i+64]).cpu())
    target_logits_all = torch.cat(target_logits_list, dim=0)
    target_labels = target_logits_all.argmax(dim=1)

    eps = EPS_LINF_STANDARD
    last_layer = LAYER_NAMES[-1]

    # ── Define per-expert attack distributions ──
    expert_specs = [
        ('expert0_clean',  ['clean']),
        ('expert1_fgsm',   ['fgsm']),
        ('expert2_pgd',    ['pgd']),
        ('expert3_mixed',  ['clean', 'fgsm', 'pgd', 'square']),
    ]

    # ── Pre-generate adversarials per attack type (shared across experts) ──
    print("Pre-generating adversarial training pools...")
    pools = {}
    for atk in ['clean', 'fgsm', 'pgd', 'square']:
        if any(atk in spec for _, spec in expert_specs):
            print(f"  {atk}...")
            pools[atk] = _gen_perturbation(atk, X_pixel, classifier, eps)

    # Helper: extract activation vector for a batch of pixel images
    def _extract_last_layer(X_np_batch):
        X_norm = torch.tensor(X_np_batch, device=device)
        X_norm = (X_norm - torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)) / \
                 torch.tensor(_STD, device=device).view(1, 3, 1, 1)
        with torch.no_grad():
            acts = extractor.extract(X_norm)
        a = acts[last_layer]
        if a.dim() > 2:
            a = F.adaptive_avg_pool2d(a, 1).view(a.size(0), -1)
        return a

    # Probe dimensions
    probe = _extract_last_layer(X_pixel[:1])
    input_dim = probe.shape[1]
    output_dim = 1000
    print(f"Expert input_dim={input_dim}, output_dim={output_dim}, hidden_dim={hidden_dim}")

    # ── Train each expert ──
    expert_state_dicts = []
    medoid_diagrams = []
    for (name, attack_mix) in expert_specs:
        print(f"\n{'='*60}\nTraining {name} on mix {attack_mix}\n{'='*60}")

        expert = ExpertSubNetwork(input_dim=input_dim, output_dim=output_dim,
                                  hidden_dim=hidden_dim).to(device)
        opt = torch.optim.Adam(expert.parameters(), lr=lr)

        # Build training tensor (X_np, targets) by mixing pools
        n_per = n_train // len(attack_mix)
        mix_X_list, mix_tgt_list = [], []
        mix_atk_labels = []
        for atk in attack_mix:
            idx = rng.choice(len(X_pixel), min(n_per, len(X_pixel)), replace=False)
            mix_X_list.append(pools[atk][idx])
            mix_tgt_list.append(target_labels[idx])
            mix_atk_labels.extend([atk] * len(idx))
        mix_X = np.concatenate(mix_X_list, axis=0)
        mix_tgt = torch.cat(mix_tgt_list, dim=0)

        # Train loop
        n = len(mix_X)
        for ep in range(epochs):
            perm = rng.permutation(n)
            total_loss = 0.0
            correct = 0
            for b in tqdm(range(0, n, batch_size), desc=f"  epoch {ep+1}/{epochs}"):
                bidx = perm[b:b+batch_size]
                X_b = mix_X[bidx]
                y_b = mix_tgt[bidx].to(device)
                with torch.no_grad():
                    feats = _extract_last_layer(X_b)
                logits = expert(feats)
                loss = F.cross_entropy(logits, y_b)
                opt.zero_grad(); loss.backward(); opt.step()
                total_loss += loss.item() * len(bidx)
                correct += int((logits.argmax(1) == y_b).sum().item())
            print(f"    loss={total_loss/n:.4f}  acc={correct/n:.4f}")

        expert_state_dicts.append({k: v.detach().cpu() for k, v in expert.state_dict().items()})

        # Compute medoid persistence diagram for this expert's training pool.
        # Sample 16 random images from the expert's attack mix and average
        # diagrams by taking the medoid (lowest mean pairwise L1-persistence).
        pool_key = attack_mix[0]  # representative attack
        sample_idx = rng.choice(len(pools[pool_key]), min(16, len(pools[pool_key])),
                                replace=False)
        diags = []
        for si in tqdm(sample_idx, desc=f"  medoid({pool_key})"):
            X_one = pools[pool_key][si:si+1]
            X_norm = torch.tensor(X_one, device=device)
            X_norm = (X_norm - torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)) / \
                     torch.tensor(_STD, device=device).view(1, 3, 1, 1)
            with torch.no_grad():
                acts = extractor.extract(X_norm)
            a_np = acts[last_layer].squeeze(0).cpu().numpy()
            diags.append(profiler.compute_diagram(a_np))

        # Pick medoid: diagram minimising sum of total-persistence distances to others.
        # Persistence diagrams from gudhi can contain `inf` for features that
        # are born but never die (e.g. dim-0 connected component covering the
        # whole filtration). `inf - finite = inf` and `inf - inf = NaN`, which
        # poisoned the argmin and silently selected index 0 for every expert.
        # Filter to finite lifetimes before summing.
        def _total_persistence(dgm_list):
            s = 0.0
            for dgm in dgm_list:
                if len(dgm) > 0:
                    arr = np.asarray(dgm, dtype=float)
                    if arr.ndim == 2:
                        lifetimes = arr[:, 1] - arr[:, 0]
                        finite = lifetimes[np.isfinite(lifetimes)]
                        s += float(np.sum(finite))
            return s

        tps = np.array([_total_persistence(d) for d in diags], dtype=float)
        if not np.any(np.isfinite(tps)):
            # Pathological — fall through with index 0 but warn loudly.
            print(f"  WARN: all total-persistence values non-finite; medoid=0")
            medoid_idx = 0
        else:
            mean_tp = float(np.mean(tps[np.isfinite(tps)]))
            distances = np.abs(tps - mean_tp)
            # Replace any residual NaN with +inf so they lose argmin.
            distances = np.where(np.isfinite(distances), distances, np.inf)
            medoid_idx = int(np.argmin(distances))
        medoid_diagrams.append(diags[medoid_idx])
        print(f"  medoid diagram index: {medoid_idx}")

    # ── Save ──
    out_data = {
        'experts': expert_state_dicts,
        'input_dim': input_dim,
        'output_dim': output_dim,
        'hidden_dim': hidden_dim,
        'medoid_diagrams': medoid_diagrams,
        'expert_names': [name for name, _ in expert_specs],
        'attack_mixes': [mix for _, mix in expert_specs],
        'training_seed': seed,
    }
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'wb') as f:
        pickle.dump(out_data, f)
    print(f"\n✅ Experts saved → {output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train differentiated TAMSH experts (P0.5)")
    parser.add_argument('--n-train', type=int, default=3000)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default=None)
    parser.add_argument('--config', default=None,
                        help='YAML config path (routes via PRISM_CONFIG env var).')
    parser.add_argument('--output', default=PATHS['experts'])
    args = parser.parse_args()

    train_experts(
        n_train=args.n_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        seed=args.seed,
        device_str=args.device,
        output=args.output,
    )
