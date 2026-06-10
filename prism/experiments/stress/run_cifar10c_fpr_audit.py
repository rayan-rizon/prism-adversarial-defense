"""
CIFAR-10-C Certificate FPR Audit (benign covariate shift)
=========================================================

Why this exists (turns a stated limitation into a measured result)
------------------------------------------------------------------
CADG's split-conformal FPR certificate (Prop. 1) holds only under
exchangeability: the deployed clean test distribution must match the clean
calibration distribution. The paper acknowledges that benign distribution shift
"may inflate FPR" but never quantifies it.

This script measures it. We feed *benign* (non-adversarial) CIFAR-10-C corrupted
images through the FROZEN detector and report the empirical L1/L2/L3 alert rate
(= realised FPR under shift) against the certified targets alpha = 0.10 / 0.03 /
0.005. Every flagged image here is a FALSE positive: the input is clean, just
corrupted. This is forward-passes only (no attack, no gradients) — the cheapest
experiment in the revision set — and the resulting "certified alpha vs. realised
FPR under benign shift" curve directly stress-tests the headline guarantee.

CIFAR-10 / ResNet-18 ONLY (CIFAR-10-C is a CIFAR-10 artifact).

Data
----
Download CIFAR-10-C (Hendrycks & Dietterich, ICLR 2019) once on the box:
  mkdir -p data/CIFAR-10-C
  curl -L -o /tmp/CIFAR-10-C.tar https://zenodo.org/record/2535967/files/CIFAR-10-C.tar
  tar -xf /tmp/CIFAR-10-C.tar -C data/        # -> data/CIFAR-10-C/<corruption>.npy + labels.npy
Each <corruption>.npy is (50000, 32, 32, 3) uint8: 5 severity blocks of 10000,
severity s in 1..5 occupies rows [(s-1)*10000 : s*10000].

Usage
-----
  cd prism/
  python experiments/stress/run_cifar10c_fpr_audit.py \
      --data-dir data/CIFAR-10-C \
      --corruptions gaussian_noise defocus_blur fog jpeg_compression \
      --severities 1 3 5 --n-per 1000 \
      --output experiments/stress/results_cifar10c_fpr_audit.json
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src import bootstrap  # noqa: F401
from src.prism import PRISM
from src.sacd.monitor import NoOpCampaignMonitor
from src.models import load_backbone
from src.config import (
    LAYER_NAMES, LAYER_WEIGHTS, DIM_WEIGHTS,
    BACKBONE_MEAN, BACKBONE_STD, BACKBONE_INPUT_SIZE,
    DATASET, PATHS,
)

_MEAN = BACKBONE_MEAN
_STD = BACKBONE_STD

# Representative subset across the four CIFAR-10-C corruption families
# (noise / blur / weather / digital). Override with --corruptions.
DEFAULT_CORRUPTIONS = [
    'gaussian_noise', 'shot_noise',          # noise
    'defocus_blur', 'motion_blur',           # blur
    'fog', 'frost',                          # weather
    'contrast', 'jpeg_compression',          # digital
]


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    margin = (z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _pixel_from_uint8(arr_hwc):
    """(H,W,3) uint8 [0,255] -> (3,H,W) float tensor in [0,1]."""
    t = torch.from_numpy(arr_hwc.astype(np.float32) / 255.0)  # H,W,3
    return t.permute(2, 0, 1).contiguous()


def _to_norm(x_pixel, device):
    mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)
    return (x_pixel - mean) / std


@torch.no_grad()
def audit_block(prism, images_hwc, device, n_per, seed):
    """
    Run a block of benign corrupted images through the frozen detector.
    Returns level counts. Every non-PASS is a false positive (input is clean).
    """
    rng = np.random.RandomState(seed)
    n_total = images_hwc.shape[0]
    idx = rng.choice(n_total, min(n_per, n_total), replace=False)
    levels = {}
    for i in idx:
        x_pixel = _pixel_from_uint8(images_hwc[int(i)]).unsqueeze(0).to(device)
        x_norm = _to_norm(x_pixel, device)
        _, level, _ = prism.defend(x_norm, pixel_image=x_pixel.squeeze(0))
        levels[level] = levels.get(level, 0) + 1
    return levels, int(len(idx))


def tier_fprs(levels, n):
    fp_l1 = n - levels.get('PASS', 0)
    fp_l2 = levels.get('L2', 0) + levels.get('L3', 0) + levels.get('L3_REJECT', 0)
    fp_l3 = levels.get('L3', 0) + levels.get('L3_REJECT', 0)
    out = {}
    for name, k, tgt in (('L1', fp_l1, 0.10), ('L2', fp_l2, 0.03), ('L3', fp_l3, 0.005)):
        lo, hi = wilson_ci(k, n)
        out[name] = {
            'FPR': round(k / max(n, 1), 4),
            'CI_95': [round(lo, 4), round(hi, 4)],
            'target_alpha': tgt,
            'within_target': (k / max(n, 1)) <= tgt,
        }
    return out


def run(data_dir, corruptions, severities, n_per, seed, output_path,
        device_str=None):
    assert DATASET == 'cifar10', (
        f"CIFAR-10-C audit is CIFAR-10 only; active DATASET={DATASET}. Run under the default config."
    )
    assert BACKBONE_INPUT_SIZE == 32, "CIFAR-10-C is 32x32; backbone input size must be 32."

    device = torch.device(device_str) if device_str else \
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"CIFAR-10-C FPR audit: corruptions={corruptions}, severities={severities}, "
          f"n_per={n_per}, seed={seed}")

    labels_path = os.path.join(data_dir, 'labels.npy')
    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"CIFAR-10-C not found at {data_dir}. Download per the module docstring."
        )

    model = load_backbone(device)
    prism = PRISM.from_saved(
        model=model,
        layer_names=LAYER_NAMES,
        calibrator_path=PATHS['calibrator'],
        profile_path=PATHS['reference_profiles'],
        ensemble_path=PATHS['ensemble_scorer'],
        layer_weights=LAYER_WEIGHTS,
        dim_weights=DIM_WEIGHTS,
        campaign_monitor=NoOpCampaignMonitor(),
    )

    t_start = time.time()
    results = {}
    for corruption in corruptions:
        npy = os.path.join(data_dir, f'{corruption}.npy')
        if not os.path.exists(npy):
            print(f"  SKIP {corruption}: {npy} missing")
            continue
        arr = np.load(npy)  # (50000,32,32,3) uint8 — 5 severity blocks of 10000
        for sev in severities:
            block = arr[(sev - 1) * 10000: sev * 10000]
            levels, n = audit_block(prism, block, device, n_per, seed)
            tiers = tier_fprs(levels, n)
            key = f'{corruption}_sev{sev}'
            results[key] = {
                'corruption': corruption, 'severity': sev, 'n': n,
                'tiers': tiers, 'level_distribution': levels,
            }
            l1 = tiers['L1']['FPR']
            flag = 'OK ' if tiers['L1']['within_target'] else 'OVER'
            print(f"  {key:>28}: L1 FPR={l1:.4f} (alpha=0.10) [{flag}]  "
                  f"L2={tiers['L2']['FPR']:.4f} L3={tiers['L3']['FPR']:.4f}")

    # Aggregate: mean realised FPR per tier across all corruption x severity cells.
    agg = {}
    for tier in ('L1', 'L2', 'L3'):
        vals = [v['tiers'][tier]['FPR'] for v in results.values()]
        agg[tier] = {
            'mean_realised_FPR': round(float(np.mean(vals)), 4) if vals else None,
            'max_realised_FPR': round(float(np.max(vals)), 4) if vals else None,
            'target_alpha': {'L1': 0.10, 'L2': 0.03, 'L3': 0.005}[tier],
            'cells_within_target': int(sum(v['tiers'][tier]['within_target'] for v in results.values())),
            'cells_total': len(results),
        }

    payload = {
        'experiment': 'cifar10c_fpr_audit',
        'dataset': 'cifar10', 'backbone': 'resnet18',
        'seed': seed, 'n_per': n_per,
        'note': ('Benign covariate shift (CIFAR-10-C). Every non-PASS is a false '
                 'positive; realised FPR vs certified alpha quantifies certificate '
                 'degradation under non-adversarial distribution shift.'),
        'per_cell': results,
        'aggregate': agg,
        'elapsed_sec': round(time.time() - t_start, 1),
    }
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"\nAggregate mean realised FPR — "
          f"L1 {agg['L1']['mean_realised_FPR']} / L2 {agg['L2']['mean_realised_FPR']} "
          f"/ L3 {agg['L3']['mean_realised_FPR']}")
    print(f"Wrote {output_path}  ({payload['elapsed_sec']}s)")
    return payload


def main():
    ap = argparse.ArgumentParser(description="CIFAR-10-C benign-shift FPR audit of the CADG certificate")
    ap.add_argument('--data-dir', type=str, default='data/CIFAR-10-C')
    ap.add_argument('--corruptions', type=str, nargs='+', default=DEFAULT_CORRUPTIONS)
    ap.add_argument('--severities', type=int, nargs='+', default=[1, 3, 5])
    ap.add_argument('--n-per', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', type=str, default=None)
    ap.add_argument('--config', type=str, default=None)
    ap.add_argument('--output', type=str,
                    default='experiments/stress/results_cifar10c_fpr_audit.json')
    args = ap.parse_args()
    run(data_dir=args.data_dir, corruptions=args.corruptions, severities=args.severities,
        n_per=args.n_per, seed=args.seed, output_path=args.output, device_str=args.device)


if __name__ == '__main__':
    main()
