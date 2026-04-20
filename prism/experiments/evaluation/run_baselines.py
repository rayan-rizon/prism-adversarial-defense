"""
Reproduced Baseline Detectors: LID + Mahalanobis

Implements and evaluates two established adversarial detection baselines
on the same CIFAR-10 eval split used by PRISM, providing a fair comparison.

LID (Local Intrinsic Dimensionality)
  Ma et al., "Characterizing Adversarial Subspaces Using Local Intrinsic
  Dimensionality", ICLR 2018.
  - Estimates the local intrinsic dimensionality of activation representations
  - Uses k-nearest-neighbor distances at multiple layers
  - Threshold fitted on calibration split (test 5000-6999) at FPR ≤ 10%

Mahalanobis Distance
  Lee et al., "A Simple Unified Framework for Detecting Out-of-Distribution
  Samples and Adversarial Attacks", NeurIPS 2018.
  - Computes Mahalanobis distance of activations to class-conditional Gaussians
  - Per-layer distances combined via logistic regression (simplified: max across layers)
  - Class means and covariance fitted on calibration split
  - Threshold fitted at FPR ≤ 10% on calibration split

USAGE
-----
  cd prism/
  python experiments/evaluation/run_baselines.py --n-test 500 --attacks FGSM PGD Square
  python experiments/evaluation/run_baselines.py --n-test 500 --attacks FGSM

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
    EVAL_IDX, CAL_IDX,
)

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

    Uses CIFAR-10 ground-truth labels (0-9) by default for stable per-class
    estimates (~100 samples per class at n=1000).  Set use_true_labels=False
    to fall back to ImageNet backbone argmax pseudo-labels.
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

def run_baselines(
    n_test=500,
    attacks_to_run=None,
    seed=42,
    output_path='experiments/evaluation/results_baselines.json',
    device_str=None,
    data_root='./data',
    lid_k=20,
):
    if not ART_AVAILABLE:
        print("ERROR: ART not installed."); sys.exit(1)

    eps = EPS_LINF_STANDARD
    attacks_to_run = attacks_to_run or ['FGSM', 'PGD', 'Square']

    device = torch.device(device_str) if device_str else \
             torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Baselines: LID (k={lid_k}), Mahalanobis")
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

    # ── Dataset ──
    ds = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=_PIXEL_TRANSFORM)

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

    # Mahalanobis: fit class-conditional Gaussians on ref_indices with true CIFAR-10 labels
    print("\n  [Mahalanobis] Fitting class-conditional Gaussians...")
    maha_params = fit_mahalanobis_params(
        model, ds, ref_indices, layer_names, device, n_classes=10, use_true_labels=True
    )

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 2: Compute clean calibration scores for threshold fitting
    # ═══════════════════════════════════════════════════════════════════════
    # Threshold fitting on thresh_indices — disjoint from ref so no self-reference bias
    print(f"\n  Computing clean detection scores on threshold-fit split ({len(thresh_indices)} images)...")

    lid_cal_scores = []
    maha_cal_scores = []
    for i in tqdm(thresh_indices, desc="  Threshold fit"):
        img, _ = ds[int(i)]
        x = img.unsqueeze(0).to(device)
        lid_s = compute_lid_multi_layer(model, x, cal_acts_per_layer, layer_names, device, k=lid_k)
        maha_s = compute_mahalanobis_score(model, x, maha_params, layer_names, device)
        lid_cal_scores.append(lid_s)
        maha_cal_scores.append(maha_s)

    lid_cal_scores = np.array(lid_cal_scores)
    maha_cal_scores = np.array(maha_cal_scores)

    # Threshold at 90th percentile of clean scores (≤10% FPR)
    lid_threshold = float(np.percentile(lid_cal_scores, 90))
    maha_threshold = float(np.percentile(maha_cal_scores, 90))

    print(f"  LID threshold (FPR≤10%):  {lid_threshold:.4f}")
    print(f"  Maha threshold (FPR≤10%): {maha_threshold:.4f}")

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

    results = {'LID': {}, 'Mahalanobis': {}}
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

        # Evaluate both detectors
        for detector_name in ['LID', 'Mahalanobis']:
            tp, fp, fn, tn = 0, 0, 0, 0

            for j, img_pixel in enumerate(tqdm(imgs_pixel, desc=f"  {detector_name}")):
                x = img_pixel.unsqueeze(0).to(device)

                # Clean score
                if detector_name == 'LID':
                    s_clean = compute_lid_multi_layer(
                        model, x, cal_acts_per_layer, layer_names, device, k=lid_k)
                    detected_clean = s_clean > lid_threshold
                else:
                    s_clean = compute_mahalanobis_score(
                        model, x, maha_params, layer_names, device)
                    detected_clean = s_clean > maha_threshold

                if detected_clean:
                    fp += 1
                else:
                    tn += 1

                # Adversarial score
                x_adv = torch.tensor(X_adv_np[j]).unsqueeze(0).to(device)
                if detector_name == 'LID':
                    s_adv = compute_lid_multi_layer(
                        model, x_adv, cal_acts_per_layer, layer_names, device, k=lid_k)
                    detected_adv = s_adv > lid_threshold
                else:
                    s_adv = compute_mahalanobis_score(
                        model, x_adv, maha_params, layer_names, device)
                    detected_adv = s_adv > maha_threshold

                if detected_adv:
                    tp += 1
                else:
                    fn += 1

            n_adv = tp + fn
            n_clean = fp + tn
            tpr = tp / max(n_adv, 1)
            fpr = fp / max(n_clean, 1)
            prec = tp / max(tp + fp, 1)
            f1 = 2 * prec * tpr / max(prec + tpr, 1e-8)
            tpr_ci = wilson_ci(tp, n_adv)
            fpr_ci = wilson_ci(fp, n_clean)

            results[detector_name][attack_name] = {
                'TPR': round(tpr, 4),
                'TPR_CI_95': [round(tpr_ci[0], 4), round(tpr_ci[1], 4)],
                'FPR': round(fpr, 4),
                'FPR_CI_95': [round(fpr_ci[0], 4), round(fpr_ci[1], 4)],
                'Precision': round(prec, 4),
                'F1': round(f1, 4),
                'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
                'n_adv': n_adv, 'n_clean': n_clean,
            }

            status = '✅' if tpr >= 0.85 else ('⚠' if tpr >= 0.70 else '❌')
            print(f"  {detector_name}: TPR={tpr:.4f} CI[{tpr_ci[0]:.4f}, {tpr_ci[1]:.4f}] "
                  f"FPR={fpr:.4f} {status}")

    elapsed = time.time() - t_start

    # ── Summary table ──
    print(f"\n{'='*70}")
    print(f"{'Detector':>12} {'Attack':>8} {'TPR':>8} {'FPR':>8} {'F1':>8}")
    print(f"{'-'*70}")
    for det in ['LID', 'Mahalanobis']:
        for atk in attacks_to_run:
            if atk in results[det]:
                r = results[det][atk]
                print(f"{det:>12} {atk:>8} {r['TPR']:>8.4f} {r['FPR']:>8.4f} {r['F1']:>8.4f}")

    results['_meta'] = {
        'n_test': n_test,
        'n_actual': int(len(sample_idx)),
        'n_ref': len(ref_indices),
        'n_thresh': len(thresh_indices),
        'eval_split': f'CIFAR-10 test idx {EVAL_IDX[0]}-{EVAL_IDX[1]-1}',
        'ref_split': f'CIFAR-10 test idx {cal_start}-{cal_mid-1}',
        'thresh_split': f'CIFAR-10 test idx {cal_mid}-{cal_end-1}',
        'seed': seed,
        'device': str(device),
        'attacks': attacks_to_run,
        'eps': round(eps, 6),
        'lid_k': lid_k,
        'lid_threshold': round(lid_threshold, 6),
        'maha_threshold': round(maha_threshold, 6),
        'maha_label_source': 'CIFAR-10 ground-truth labels (0-9), n_classes=10',
        'layer_names': LAYER_NAMES,
        'elapsed_s': round(elapsed, 1),
        'references': {
            'LID': 'Ma et al., 2018. Characterizing Adversarial Subspaces Using Local '
                   'Intrinsic Dimensionality. ICLR 2018.',
            'Mahalanobis': 'Lee et al., 2018. A Simple Unified Framework for Detecting '
                           'Out-of-Distribution Samples and Adversarial Attacks. NeurIPS 2018.',
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {output_path}")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LID + Mahalanobis baseline evaluation")
    parser.add_argument('--n-test', type=int, default=500)
    parser.add_argument('--attacks', nargs='+', default=['FGSM', 'PGD', 'Square'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lid-k', type=int, default=20)
    parser.add_argument('--output', default='experiments/evaluation/results_baselines.json')
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    run_baselines(
        n_test=args.n_test,
        attacks_to_run=args.attacks,
        seed=args.seed,
        output_path=args.output,
        device_str=args.device,
        lid_k=args.lid_k,
    )
