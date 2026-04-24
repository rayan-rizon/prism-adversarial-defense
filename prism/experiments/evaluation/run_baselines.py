"""
Reproduced Baseline Detectors: LID + Mahalanobis + ODIN + Energy

Implements and evaluates four established adversarial detection / OOD baselines
on the same CIFAR-10 eval split used by PRISM, providing a fair comparison.

LID (Local Intrinsic Dimensionality)
  Ma et al., "Characterizing Adversarial Subspaces Using Local Intrinsic
  Dimensionality", ICLR 2018.

Mahalanobis Distance
  Lee et al., "A Simple Unified Framework for Detecting Out-of-Distribution
  Samples and Adversarial Attacks", NeurIPS 2018.

ODIN (Out-of-DIstribution detector for Neural networks)
  Liang et al., "Enhancing The Reliability of Out-of-distribution Image
  Detection in Neural Networks", ICLR 2018.
  - Temperature-scaled max softmax probability; input-preprocessing gradient
    perturbation to widen the clean/OOD gap.
  - Score: 1 - max_c softmax(f(x + eps * sign(grad)) / T)_c.
  - Higher score = more adversarial/OOD.

Energy (Liu 2020)
  Liu et al., "Energy-based Out-of-distribution Detection", NeurIPS 2020.
  - Free energy of the logits: E(x) = -T * logsumexp(f(x) / T).
  - Adversarial / OOD inputs have higher E(x) (less negative).
  - No training, no input preprocessing; single forward pass.

All four detectors are calibrated at matched FPR tiers (10% / 3% / 0.5%)
on a disjoint threshold-fit split so the comparison table vs. PRISM's
three-tier output is apples-to-apples (P0.2).

USAGE
-----
  cd prism/
  python experiments/evaluation/run_baselines.py --n-test 1000 --attacks FGSM PGD Square
  python experiments/evaluation/run_baselines.py --methods lid mahalanobis odin energy

EVAL SPLIT:   CIFAR-10 test indices 8000-9999 (same held-out split as PRISM evaluation)
REF SPLIT:    CIFAR-10 test indices 5000-5999 (k-NN reference / Gaussian fitting)
THRESH SPLIT: CIFAR-10 test indices 6000-6999 (threshold calibration; disjoint from reference)
"""
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models import ResNet18_Weights
import numpy as np
import json, os, sys, ssl, certifi, time, argparse
from tqdm import tqdm
from collections import defaultdict

os.environ.setdefault('SSL_CERT_FILE', certifi.where())
os.environ.setdefault('REQUESTS_CA_BUNDLE', certifi.where())
ssl._create_default_https_context = ssl.create_default_context

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Route --config CLI flag to PRISM_CONFIG env var BEFORE importing src.config.
from src import bootstrap  # noqa: F401

try:
    from art.attacks.evasion import (
        FastGradientMethod,
        ProjectedGradientDescent,
        SquareAttack,
    )
    from art.estimators.classification import PyTorchClassifier
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False
    print("WARNING: ART not installed. pip install adversarial-robustness-toolbox")

from src.config import (
    LAYER_NAMES, IMAGENET_MEAN, IMAGENET_STD, EPS_LINF_STANDARD,
    EVAL_IDX, CAL_IDX, DATASET, PATHS,
)
from src.data_loader import load_test_dataset

_MEAN = IMAGENET_MEAN
_STD  = IMAGENET_STD
_PIXEL_TRANSFORM = T.Compose([T.Resize(224), T.ToTensor()])
_NORMALIZE       = T.Normalize(mean=_MEAN, std=_STD)


class _NormalizedResNet(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self._model = model
        self.register_buffer('_mean', torch.tensor(_MEAN).view(3, 1, 1))
        self.register_buffer('_std',  torch.tensor(_STD).view(3, 1, 1))

    def forward(self, x):
        return self._model((x - self._mean) / self._std)


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


# ═════════════════════════════════════════════════════════════════════════════
# LID Detector (Ma et al. 2018)
# ═════════════════════════════════════════════════════════════════════════════

def _extract_activations(model, x, layer_names, device):
    """Extract flattened activations from specified layers via hooks."""
    acts = {}
    handles = []

    def make_hook(name):
        def hook_fn(module, inp, out):
            acts[name] = out.detach()
        return hook_fn

    module_dict = dict(model.named_modules())
    for name in layer_names:
        h = module_dict[name].register_forward_hook(make_hook(name))
        handles.append(h)

    mean_t = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std_t  = torch.tensor(_STD,  device=device).view(1, 3, 1, 1)
    x_norm = (x - mean_t) / std_t

    with torch.no_grad():
        model(x_norm)

    for h in handles:
        h.remove()

    result = {}
    for name in layer_names:
        a = acts[name]
        # Global average pool if spatial dims exist
        if a.dim() == 4:
            a = torch.nn.functional.adaptive_avg_pool2d(a, 1).flatten(1)
        elif a.dim() == 3:
            a = a.mean(dim=-1)
        result[name] = a.cpu().numpy()
    return result


def compute_lid(x_acts, ref_acts, k=20):
    """
    Compute Local Intrinsic Dimensionality for a single sample.

    LID(x) = -k / Σ_{i=1}^{k} log(d_i / d_k)

    where d_1 ... d_k are the k smallest distances from x to ref_acts.
    """
    from scipy.spatial.distance import cdist
    dists = cdist(x_acts.reshape(1, -1), ref_acts, metric='euclidean')[0]
    dists = np.sort(dists)

    # Use k nearest (excluding self if present)
    d_k = dists[k - 1]
    if d_k < 1e-10:
        return 0.0

    # Clamp to avoid log(0)
    d_arr = np.maximum(dists[:k], 1e-10)
    log_ratios = np.log(d_arr / d_k)
    lid = -float(k) / np.sum(log_ratios)
    return max(lid, 0.0)


def compute_lid_multi_layer(model, x_pixel, cal_acts_per_layer, layer_names, device, k=20):
    """Compute average LID across all monitored layers."""
    acts = _extract_activations(model, x_pixel, layer_names, device)
    lid_scores = []
    for name in layer_names:
        lid = compute_lid(acts[name], cal_acts_per_layer[name], k=k)
        lid_scores.append(lid)
    return float(np.mean(lid_scores))


# ═════════════════════════════════════════════════════════════════════════════
# Mahalanobis Distance Detector (Lee et al. 2018)
# ═════════════════════════════════════════════════════════════════════════════

def fit_mahalanobis_params(model, dataset, cal_indices, layer_names, device,
                           n_classes=10, use_true_labels=True):
    """
    Fit class-conditional Gaussians per layer on calibration split.

    Uses dataset ground-truth labels (CIFAR-10: 0-9; CIFAR-100: 0-99) by
    default for stable per-class estimates. Set use_true_labels=False to
    fall back to ImageNet backbone argmax pseudo-labels.
    """
    mean_t = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std_t  = torch.tensor(_STD, device=device).view(1, 3, 1, 1)

    # Collect activations and labels
    all_acts = defaultdict(list)
    all_labels = []

    label_src = 'true CIFAR-10 labels' if use_true_labels else 'backbone pseudo-labels'
    print(f"  Fitting Mahalanobis on {len(cal_indices)} images ({label_src})...")
    for i in tqdm(cal_indices, desc="  Maha fit"):
        img, true_label = dataset[int(i)]
        x = img.unsqueeze(0).to(device)
        acts = _extract_activations(model, x, layer_names, device)

        if use_true_labels:
            label = int(true_label)
        else:
            x_norm = (x - mean_t) / std_t
            with torch.no_grad():
                logits = model(x_norm)
            label = int(logits.argmax(dim=1).item())
        all_labels.append(label)

        for name in layer_names:
            all_acts[name].append(acts[name].squeeze(0))

    # Stack activations per layer
    acts_arrays = {}
    for name in layer_names:
        acts_arrays[name] = np.stack(all_acts[name])

    labels = np.array(all_labels)
    unique_labels = np.unique(labels)

    # Compute per-class means and shared covariance per layer
    params = {}
    for name in layer_names:
        X = acts_arrays[name]  # (N, D)
        D = X.shape[1]

        class_means = {}
        cov_sum = np.zeros((D, D))
        n_total = 0

        for c in unique_labels:
            mask = labels == c
            X_c = X[mask]
            class_means[int(c)] = X_c.mean(axis=0)
            X_centered = X_c - class_means[int(c)]
            cov_sum += X_centered.T @ X_centered
            n_total += len(X_c)

        # Shared covariance with regularisation
        cov = cov_sum / max(n_total, 1)
        cov += np.eye(D) * 1e-5  # regularisation for invertibility
        cov_inv = np.linalg.inv(cov)

        params[name] = {
            'class_means': class_means,
            'cov_inv': cov_inv,
        }

    return params


def compute_odin_score(norm_model, x_pixel, device, T=1000.0, eps=0.0014):
    """
    ODIN score (Liang et al., ICLR 2018).

    Implements temperature scaling plus input-gradient preprocessing:
      1. Forward pass: softmax(f(x)/T), take predicted class c*.
      2. Gradient step: x' = x - eps * sign(grad_x log softmax(f(x)/T)_{c*}).
      3. Recompute softmax(f(x')/T); score = 1 - max_c softmax.

    Higher score ⇒ more OOD/adversarial. Uses norm_model (normalisation
    baked in) so x_pixel stays in [0,1] and the preprocessing gradient
    is well-defined in pixel space. T and eps follow the ODIN paper's
    CIFAR-10/ResNet defaults.
    """
    x = x_pixel.clone().detach().to(device).requires_grad_(True)
    logits = norm_model(x) / T
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    pred = int(logits.argmax(1).item())
    loss = -log_probs[0, pred]
    (grad_x,) = torch.autograd.grad(loss, x)
    x_pert = (x - eps * grad_x.sign()).detach().clamp(0.0, 1.0)
    with torch.no_grad():
        logits_pert = norm_model(x_pert) / T
        msp = float(torch.softmax(logits_pert, dim=1).max().item())
    return 1.0 - msp


def compute_energy_score(norm_model, x_pixel, device, T=1.0):
    """
    Free-energy OOD score (Liu et al., NeurIPS 2020).

        E(x) = -T * logsumexp(f(x) / T)

    Higher E(x) ⇒ more OOD/adversarial. We return +E (i.e. a score where
    larger means more anomalous) so thresholding is consistent with
    LID/Mahalanobis/ODIN. T=1.0 matches the paper default; T can be
    tuned on the threshold-fit split but we keep the literature default
    to avoid hyperparameter tuning on the target distribution.
    """
    x = x_pixel.to(device)
    with torch.no_grad():
        logits = norm_model(x) / T
        lse = torch.logsumexp(logits, dim=1).item()
    return float(-T * lse)


def compute_mahalanobis_score(model, x_pixel, maha_params, layer_names, device):
    """
    Compute max Mahalanobis distance across layers (simplified Lee et al.).

    For each layer: min over classes of (x - μ_c)^T Σ^{-1} (x - μ_c).
    Return max across layers (the layer with the strongest signal).
    """
    acts = _extract_activations(model, x_pixel, layer_names, device)
    layer_scores = []

    for name in layer_names:
        a = acts[name].squeeze(0)  # (D,)
        p = maha_params[name]
        cov_inv = p['cov_inv']

        # Min Mahalanobis across classes
        min_dist = float('inf')
        for c, mu_c in p['class_means'].items():
            diff = a - mu_c
            dist = float(diff @ cov_inv @ diff)
            min_dist = min(min_dist, dist)

        layer_scores.append(min_dist)

    return float(np.max(layer_scores))


# ═════════════════════════════════════════════════════════════════════════════
# Main Evaluation
# ═════════════════════════════════════════════════════════════════════════════

_DEFAULT_METHODS = ['lid', 'mahalanobis', 'odin', 'energy']
_METHOD_DISPLAY = {
    'lid': 'LID',
    'mahalanobis': 'Mahalanobis',
    'odin': 'ODIN',
    'energy': 'Energy',
}
# Match PRISM's three-tier conformal output (L1/L2/L3) so the comparison table
# is apples-to-apples. Percentiles correspond to FPR targets 10% / 3% / 0.5%.
_FPR_TIERS = [
    ('L1', 10.0, 90.0),
    ('L2',  3.0, 97.0),
    ('L3',  0.5, 99.5),
]


def run_baselines(
    n_test=500,
    attacks_to_run=None,
    seed=42,
    output_path='experiments/evaluation/results_baselines.json',
    device_str=None,
    data_root='./data',
    lid_k=20,
    methods=None,
    odin_T=1000.0,
    odin_eps=0.0014,
    energy_T=1.0,
):
    if not ART_AVAILABLE:
        print("ERROR: ART not installed."); sys.exit(1)

    eps = EPS_LINF_STANDARD
    attacks_to_run = attacks_to_run or ['FGSM', 'PGD', 'Square']
    methods = [m.lower() for m in (methods or _DEFAULT_METHODS)]
    for m in methods:
        if m not in _METHOD_DISPLAY:
            raise ValueError(f"Unknown baseline method: {m} (expected one of {list(_METHOD_DISPLAY)})")
    method_displays = [_METHOD_DISPLAY[m] for m in methods]

    device = torch.device(device_str) if device_str else \
             torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Baselines: {', '.join(method_displays)} (LID k={lid_k}, ODIN T={odin_T} eps={odin_eps}, Energy T={energy_T})")
    print(f"Attacks: {attacks_to_run}")
    cal_start, cal_end = CAL_IDX
    cal_mid = cal_start + (cal_end - cal_start) // 2  # 6000
    print(f"n_test={n_test}, eval_split=test[{EVAL_IDX[0]}-{EVAL_IDX[1]-1}]")
    print(f"ref_split=test[{cal_start}-{cal_mid-1}], thresh_split=test[{cal_mid}-{cal_end-1}]\n")

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    # ── Model ──
    model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model = model.to(device).eval()

    layer_names = LAYER_NAMES

    # ── Scorer registry: per-method callables on (x_pixel: (1,3,H,W) tensor) → float ──
    # Built after model/maha_params are ready; populated inside Phase 1/2.
    scorer_fns = {}

    # ── ART classifier ──
    norm_model = _NormalizedResNet(model).to(device).eval()
    device_type = 'gpu' if device.type == 'cuda' else 'cpu'
    classifier = PyTorchClassifier(
        model=norm_model,
        loss=torch.nn.CrossEntropyLoss(),
        input_shape=(3, 224, 224),
        nb_classes=1000,
        clip_values=(0.0, 1.0),
        device_type=device_type,
    )

    all_attacks = {
        'FGSM': lambda: FastGradientMethod(classifier, eps=eps),
        'PGD': lambda: ProjectedGradientDescent(
            classifier, eps=eps, eps_step=eps / 4, max_iter=40, num_random_init=1),
        'Square': lambda: SquareAttack(
            classifier, eps=eps, max_iter=5000, nb_restarts=1),
    }

    # ── Dataset (dispatches on DATASET: cifar10 / cifar100) ──
    ds = load_test_dataset(root=data_root, download=True, transform=_PIXEL_TRANSFORM)

    # ref_indices   — k-NN reference database and Gaussian fitting (never used to set threshold)
    # thresh_indices — threshold calibration only (disjoint from ref, prevents self-reference bias)
    ref_indices    = list(range(cal_start, cal_mid))   # test[5000-5999]
    thresh_indices = list(range(cal_mid, cal_end))     # test[6000-6999]
    eval_indices   = list(range(*EVAL_IDX))
    sample_idx = rng.choice(eval_indices, min(n_test, len(eval_indices)), replace=False)

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 1: Fit detectors on calibration split
    # ═══════════════════════════════════════════════════════════════════════
    print("="*60)
    print("Phase 1: Fitting detectors on calibration split")
    print("="*60)

    # LID: collect reference activations (ref_indices only — threshold images are kept separate)
    cal_acts_per_layer = None
    if 'lid' in methods:
        print("\n  [LID] Collecting reference activations...")
        cal_acts_per_layer = defaultdict(list)
        for i in tqdm(ref_indices, desc="  LID ref"):
            img, _ = ds[int(i)]
            x = img.unsqueeze(0).to(device)
            acts = _extract_activations(model, x, layer_names, device)
            for name in layer_names:
                cal_acts_per_layer[name].append(acts[name].squeeze(0))
        for name in layer_names:
            cal_acts_per_layer[name] = np.stack(cal_acts_per_layer[name])
        print(f"  LID reference: {cal_acts_per_layer[layer_names[0]].shape[0]} samples × "
              f"{len(layer_names)} layers")
        scorer_fns['lid'] = lambda x: compute_lid_multi_layer(
            model, x, cal_acts_per_layer, layer_names, device, k=lid_k)

    # Mahalanobis: fit class-conditional Gaussians on ref_indices with true CIFAR-10 labels
    maha_params = None
    if 'mahalanobis' in methods:
        print("\n  [Mahalanobis] Fitting class-conditional Gaussians...")
        _n_classes = 100 if DATASET == 'cifar100' else 10
        maha_params = fit_mahalanobis_params(
            model, ds, ref_indices, layer_names, device,
            n_classes=_n_classes, use_true_labels=True
        )
        scorer_fns['mahalanobis'] = lambda x: compute_mahalanobis_score(
            model, x, maha_params, layer_names, device)

    # ODIN and Energy: no fitting phase, just bind hyperparameters.
    if 'odin' in methods:
        scorer_fns['odin'] = lambda x: compute_odin_score(
            norm_model, x, device, T=odin_T, eps=odin_eps)
    if 'energy' in methods:
        scorer_fns['energy'] = lambda x: compute_energy_score(
            norm_model, x, device, T=energy_T)

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 2: Compute clean calibration scores for threshold fitting
    # ═══════════════════════════════════════════════════════════════════════
    # Threshold fitting on thresh_indices — disjoint from ref so no self-reference bias.
    # Fit thresholds at three FPR tiers (10% / 3% / 0.5%) to match PRISM's
    # conformal L1/L2/L3 output for apples-to-apples comparison.
    print(f"\n  Computing clean detection scores on threshold-fit split ({len(thresh_indices)} images)...")
    cal_scores = {m: [] for m in methods}
    for i in tqdm(thresh_indices, desc="  Threshold fit"):
        img, _ = ds[int(i)]
        x = img.unsqueeze(0).to(device)
        for m in methods:
            cal_scores[m].append(float(scorer_fns[m](x)))

    thresholds = {}  # thresholds[method][tier] = float
    for m in methods:
        arr = np.asarray(cal_scores[m], dtype=float)
        thresholds[m] = {
            tier: float(np.percentile(arr, pct)) for (tier, _fpr, pct) in _FPR_TIERS
        }
        tier_str = ', '.join(
            f"{tier}(FPR≤{fpr}%)={thresholds[m][tier]:.4f}"
            for (tier, fpr, _pct) in _FPR_TIERS
        )
        print(f"  {_METHOD_DISPLAY[m]:>12}: {tier_str}")

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 3: Evaluate on held-out eval split per attack
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\nPre-loading {len(sample_idx)} eval images...")
    imgs_pixel = []
    for i in sample_idx:
        img, _ = ds[int(i)]
        imgs_pixel.append(img)
    X_pixel_np = torch.stack(imgs_pixel).numpy()
    print(f"Pre-loaded {len(imgs_pixel)} images\n")

    results = {_METHOD_DISPLAY[m]: {} for m in methods}
    t_start = time.time()

    for attack_name in attacks_to_run:
        if attack_name not in all_attacks:
            print(f"Unknown attack: {attack_name}. Skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"Attack: {attack_name}")
        print(f"{'='*60}")

        attack = all_attacks[attack_name]()

        # Batch generate adversarials
        print(f"  Generating {len(sample_idx)} adversarial examples...")
        try:
            X_adv_np = attack.generate(X_pixel_np)
        except Exception as e:
            print(f"  Batch generation failed ({e}), falling back to per-sample...")
            X_adv_np = np.zeros_like(X_pixel_np)
            for idx_i, x_np_i in enumerate(tqdm(X_pixel_np, desc="  fallback")):
                try:
                    X_adv_np[idx_i] = attack.generate(x_np_i[np.newaxis])[0]
                except Exception:
                    X_adv_np[idx_i] = x_np_i

        # Cache clean + adv scores per method so each image only runs the scorer twice.
        scores_clean = {m: np.empty(len(imgs_pixel), dtype=float) for m in methods}
        scores_adv   = {m: np.empty(len(imgs_pixel), dtype=float) for m in methods}

        for j, img_pixel in enumerate(tqdm(imgs_pixel, desc="  scoring")):
            x_clean = img_pixel.unsqueeze(0).to(device)
            x_adv = torch.tensor(X_adv_np[j]).unsqueeze(0).to(device)
            for m in methods:
                scores_clean[m][j] = float(scorer_fns[m](x_clean))
                scores_adv[m][j]   = float(scorer_fns[m](x_adv))

        for m in methods:
            display = _METHOD_DISPLAY[m]
            per_tier = {}
            for (tier, fpr_target, _pct) in _FPR_TIERS:
                thr = thresholds[m][tier]
                det_adv   = scores_adv[m]   > thr
                det_clean = scores_clean[m] > thr
                tp = int(det_adv.sum()); fn = int((~det_adv).sum())
                fp = int(det_clean.sum()); tn = int((~det_clean).sum())
                n_adv = tp + fn; n_clean = fp + tn
                tpr = tp / max(n_adv, 1)
                fpr_emp = fp / max(n_clean, 1)
                prec = tp / max(tp + fp, 1)
                f1 = 2 * prec * tpr / max(prec + tpr, 1e-8)
                tpr_ci = wilson_ci(tp, n_adv)
                fpr_ci = wilson_ci(fp, n_clean)
                per_tier[tier] = {
                    'TPR': round(tpr, 4),
                    'TPR_CI_95': [round(tpr_ci[0], 4), round(tpr_ci[1], 4)],
                    'FPR': round(fpr_emp, 4),
                    'FPR_CI_95': [round(fpr_ci[0], 4), round(fpr_ci[1], 4)],
                    'FPR_target': fpr_target / 100.0,
                    'Precision': round(prec, 4),
                    'F1': round(f1, 4),
                    'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
                    'n_adv': n_adv, 'n_clean': n_clean,
                    'threshold': round(thr, 6),
                }

            # Top-level keys mirror PRISM's L1 default so existing consumers keep working.
            l1 = per_tier['L1']
            results[display][attack_name] = {
                **l1,
                'tiers': per_tier,
            }

            status = '✅' if l1['TPR'] >= 0.85 else ('⚠' if l1['TPR'] >= 0.70 else '❌')
            print(f"  {display}: L1 TPR={l1['TPR']:.4f} FPR={l1['FPR']:.4f} | "
                  f"L2 TPR={per_tier['L2']['TPR']:.4f} FPR={per_tier['L2']['FPR']:.4f} | "
                  f"L3 TPR={per_tier['L3']['TPR']:.4f} FPR={per_tier['L3']['FPR']:.4f} {status}")

    elapsed = time.time() - t_start

    # ── Summary table (L1 tier, matches PRISM default) ──
    print(f"\n{'='*70}")
    print(f"{'Detector':>12} {'Attack':>8} {'TPR':>8} {'FPR':>8} {'F1':>8}")
    print(f"{'-'*70}")
    for m in methods:
        display = _METHOD_DISPLAY[m]
        for atk in attacks_to_run:
            if atk in results[display]:
                r = results[display][atk]
                print(f"{display:>12} {atk:>8} {r['TPR']:>8.4f} {r['FPR']:>8.4f} {r['F1']:>8.4f}")

    thresholds_serializable = {
        _METHOD_DISPLAY[m]: {tier: round(thr, 6) for tier, thr in thresholds[m].items()}
        for m in methods
    }

    all_refs = {
        'LID': 'Ma et al., 2018. Characterizing Adversarial Subspaces Using Local '
               'Intrinsic Dimensionality. ICLR 2018.',
        'Mahalanobis': 'Lee et al., 2018. A Simple Unified Framework for Detecting '
                       'Out-of-Distribution Samples and Adversarial Attacks. NeurIPS 2018.',
        'ODIN': 'Liang et al., 2018. Enhancing The Reliability of Out-of-distribution '
                'Image Detection in Neural Networks. ICLR 2018.',
        'Energy': 'Liu et al., 2020. Energy-based Out-of-distribution Detection. '
                  'NeurIPS 2020.',
    }

    results['_meta'] = {
        'n_test': n_test,
        'n_actual': int(len(sample_idx)),
        'n_ref': len(ref_indices),
        'n_thresh': len(thresh_indices),
        'dataset': DATASET,
        'eval_split': f'{DATASET.upper()} test idx {EVAL_IDX[0]}-{EVAL_IDX[1]-1}',
        'ref_split': f'{DATASET.upper()} test idx {cal_start}-{cal_mid-1}',
        'thresh_split': f'{DATASET.upper()} test idx {cal_mid}-{cal_end-1}',
        'seed': seed,
        'device': str(device),
        'attacks': attacks_to_run,
        'methods': methods,
        'eps': round(eps, 6),
        'lid_k': lid_k,
        'odin_T': odin_T,
        'odin_eps': odin_eps,
        'energy_T': energy_T,
        'fpr_tiers': [
            {'name': name, 'target_fpr': fpr / 100.0, 'percentile': pct}
            for (name, fpr, pct) in _FPR_TIERS
        ],
        'thresholds': thresholds_serializable,
        'maha_label_source': (
            f'{DATASET.upper()} ground-truth labels, '
            f'n_classes={100 if DATASET == "cifar100" else 10}'
        ),
        'layer_names': LAYER_NAMES,
        'elapsed_s': round(elapsed, 1),
        'references': {_METHOD_DISPLAY[m]: all_refs[_METHOD_DISPLAY[m]] for m in methods},
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {output_path}")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Baseline adversarial detectors: LID, Mahalanobis, ODIN, Energy")
    parser.add_argument('--config', default=None,
                        help='YAML config path (routes via PRISM_CONFIG env var).')
    parser.add_argument('--n-test', type=int, default=500)
    parser.add_argument('--attacks', nargs='+', default=['FGSM', 'PGD', 'Square'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lid-k', type=int, default=20)
    parser.add_argument('--methods', nargs='+', default=_DEFAULT_METHODS,
                        choices=list(_METHOD_DISPLAY),
                        help='Which baselines to run (default: all four)')
    parser.add_argument('--odin-T', type=float, default=1000.0)
    parser.add_argument('--odin-eps', type=float, default=0.0014)
    parser.add_argument('--energy-T', type=float, default=1.0)
    _default_out = os.path.join(
        os.path.dirname(PATHS['clean_scores']).replace('calibration', 'evaluation')
            or 'experiments/evaluation',
        (os.path.basename(PATHS['clean_scores']).replace('clean_scores.npy', '')
         + 'results_baselines.json')
    )
    parser.add_argument('--output', default=_default_out)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    run_baselines(
        n_test=args.n_test,
        attacks_to_run=args.attacks,
        seed=args.seed,
        output_path=args.output,
        device_str=args.device,
        lid_k=args.lid_k,
        methods=args.methods,
        odin_T=args.odin_T,
        odin_eps=args.odin_eps,
        energy_T=args.energy_T,
    )
