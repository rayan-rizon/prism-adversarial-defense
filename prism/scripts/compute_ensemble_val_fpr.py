"""
Compute Ensemble Val FPR — High-Statistical-Power FPR Report

Scores the validation split (CIFAR-10 test idx 7000-7999, n=1000) through
the FULL PRISM pipeline (ensemble scorer + ensemble calibrator) to produce
FPR estimates with narrow Wilson CIs (~+/-1.5% at n=1000 vs +/-3.3% at n=300).

IMPORTANT: Must be run AFTER calibrate_ensemble.py so models/calibrator.pkl
contains the ensemble-calibrated thresholds. The val split is held-out from
both training (CIFAR-10 train set) and calibration (test idx 5000-6999).

Output: experiments/calibration/ensemble_fpr_report.json

USAGE
-----
  cd prism/
  python scripts/compute_ensemble_val_fpr.py
"""
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models import ResNet18_Weights
import numpy as np
import json, os, sys, ssl, certifi, math
from tqdm import tqdm

os.environ.setdefault('SSL_CERT_FILE', certifi.where())
os.environ.setdefault('REQUESTS_CA_BUNDLE', certifi.where())
ssl._create_default_https_context = ssl.create_default_context

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.prism import PRISM
from src.sacd.monitor import NoOpCampaignMonitor
from src.config import IMAGENET_MEAN, IMAGENET_STD, VAL_IDX, LAYER_WEIGHTS, DIM_WEIGHTS

_MEAN = IMAGENET_MEAN
_STD  = IMAGENET_STD
_PIXEL_TRANSFORM = T.Compose([T.Resize(224), T.ToTensor()])
_NORMALIZE       = T.Normalize(mean=_MEAN, std=_STD)
# VAL_IDX imported from src.config (configs/default.yaml splits.val_idx)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score 95% CI for proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def compute_ensemble_val_fpr(
    data_root: str = './data',
    output_path: str = 'experiments/calibration/ensemble_fpr_report.json',
):
    """Score val split through full PRISM ensemble pipeline, report per-tier FPR."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Validation split: test idx {VAL_IDX[0]}-{VAL_IDX[1]-1} (n={VAL_IDX[1]-VAL_IDX[0]})")

    # ── Load PRISM with ensemble scorer ───────────────────────────────────────
    layer_names   = ['layer2', 'layer3', 'layer4']
    layer_weights = LAYER_WEIGHTS
    dim_weights   = DIM_WEIGHTS

    model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model = model.to(device).eval()

    for p in ['models/calibrator.pkl', 'models/reference_profiles.pkl']:
        if not os.path.exists(p):
            print(f"ERROR: {p} not found. Run pipeline phases 2+3.5 first.")
            sys.exit(1)

    prism = PRISM.from_saved(
        model=model,
        layer_names=layer_names,
        calibrator_path='models/calibrator.pkl',
        profile_path='models/reference_profiles.pkl',
        ensemble_path='models/ensemble_scorer.pkl' if os.path.exists('models/ensemble_scorer.pkl') else None,
        layer_weights=layer_weights,
        dim_weights=dim_weights,
        campaign_monitor=NoOpCampaignMonitor(),
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    pixel_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=_PIXEL_TRANSFORM
    )
    val_indices = list(range(*VAL_IDX))

    # ── Score all validation images ───────────────────────────────────────────
    level_counts = {}
    all_scores   = []

    print(f"\nScoring {len(val_indices)} validation images through full PRISM pipeline...")
    for i in tqdm(val_indices):
        img, _ = pixel_dataset[i]
        x = _NORMALIZE(img).unsqueeze(0).to(device)
        _, level, meta = prism.defend(x)
        level_counts[level] = level_counts.get(level, 0) + 1
        all_scores.append(meta.get('anomaly_score', 0.0))

    n = len(val_indices)

    # ── Per-tier FPR ──────────────────────────────────────────────────────────
    # L1+ = any flag (not PASS)
    # L2+ = L2, L3, L3_REJECT
    # L3+ = L3, L3_REJECT
    fp_l1 = n - level_counts.get('PASS', 0)
    fp_l2 = (level_counts.get('L2', 0) + level_counts.get('L3', 0)
             + level_counts.get('L3_REJECT', 0))
    fp_l3 = level_counts.get('L3', 0) + level_counts.get('L3_REJECT', 0)

    targets = {'L1': 0.10, 'L2': 0.03, 'L3': 0.005}
    fps     = {'L1': fp_l1, 'L2': fp_l2, 'L3': fp_l3}

    results = {
        'n_val': n,
        'val_split': f'CIFAR-10 test idx {VAL_IDX[0]}-{VAL_IDX[1]-1}',
        'level_distribution': level_counts,
        'tiers': {},
    }

    print(f"\n{'='*60}")
    print(f"  Per-Tier FPR on Validation Split (n={n})")
    print(f"{'='*60}")
    print(f"  Level distribution: {level_counts}")
    print()

    all_pass = True
    for tier in ['L1', 'L2', 'L3']:
        fp  = fps[tier]
        fpr = fp / n
        ci  = wilson_ci(fp, n)
        tgt = targets[tier]
        passed = fpr <= tgt
        if not passed:
            all_pass = False
        status = 'PASS' if passed else 'FAIL'
        print(f"  {tier}+: FPR = {fpr:.4f}  CI=[{ci[0]:.4f}, {ci[1]:.4f}]  "
              f"target<={tgt:.3f}  [{status}]")
        results['tiers'][tier] = {
            'FP': fp, 'n': n, 'FPR': round(fpr, 6),
            'CI_95': [round(ci[0], 6), round(ci[1], 6)],
            'target': tgt, 'passed': passed,
        }

    print()
    if all_pass:
        print("  [OK] All ensemble FPR guarantees verified on val split.")
    else:
        print("  [WARN] Some FPR targets exceeded. Re-calibrate with lower alpha.")

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nFPR report saved -> {output_path}")
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='./data')
    parser.add_argument('--output',    default='experiments/calibration/ensemble_fpr_report.json')
    args = parser.parse_args()
    compute_ensemble_val_fpr(data_root=args.data_root, output_path=args.output)
