"""
CIFAR-10-C Certificate RESTORATION under benign covariate shift
================================================================
Companion to run_cifar10c_fpr_audit.py. The audit showed the split-conformal
FPR certificate degrades 3.9x-28x under benign CIFAR-10-C corruption (the
exchangeability assumption of Prop. 1 is broken). This script shows the
degradation is *recoverable* by shift-aware recalibration: re-fitting the
conformal quantile on a small held-out slice of the SAME corrupted-but-benign
distribution restores realised FPR to the certified target alpha.

Per corruption x severity cell:
  1. score every benign corrupted image with the FROZEN detector -> S(x), level.
  2. split into a recalibration slice (n_recal) and an evaluation slice (rest).
  3. FROZEN FPR  = non-PASS rate on the eval slice (= the audit degradation).
  4. RESTORED FPR = re-fit the split-conformal quantile q_alpha on the recal
     slice's scores, then measure P(S > q_alpha) on the eval slice. Now
     calibration and test are exchangeable (same corruption), so FPR -> alpha.

This is the certificate analogue of the SACD session-level fix: a measured
negative (cert breaks under shift) turned into a measured positive (online
recalibration restores it). Forward-passes only. CIFAR-10/ResNet-18 only.
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

_MEAN, _STD = BACKBONE_MEAN, BACKBONE_STD
DEFAULT_CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'defocus_blur', 'motion_blur',
    'fog', 'frost', 'contrast', 'jpeg_compression',
]
ALPHAS = {'L1': 0.10, 'L2': 0.03, 'L3': 0.005}


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    m = (z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))) / d
    return (max(0.0, c - m), min(1.0, c + m))


def _pixel_from_uint8(arr_hwc):
    t = torch.from_numpy(arr_hwc.astype(np.float32) / 255.0)
    return t.permute(2, 0, 1).contiguous()


def _to_norm(x_pixel, device):
    mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)
    return (x_pixel - mean) / std


def conformal_threshold(scores, alpha):
    """Split-conformal upper quantile: smallest value with <= alpha mass above it."""
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return s[k - 1]


@torch.no_grad()
def score_block(prism, images_hwc, device, n_per, seed):
    rng = np.random.RandomState(seed)
    idx = rng.choice(images_hwc.shape[0], min(n_per, images_hwc.shape[0]), replace=False)
    scores, levels = [], []
    for i in idx:
        x_pixel = _pixel_from_uint8(images_hwc[int(i)]).unsqueeze(0).to(device)
        _, level, info = prism.defend(_to_norm(x_pixel, device), pixel_image=x_pixel.squeeze(0))
        scores.append(float(info.get('anomaly_score', 0.0)))
        levels.append(level)
    return np.array(scores), levels


def frozen_fpr(levels):
    n = len(levels)
    non_pass = sum(1 for l in levels if l != 'PASS')
    l2 = sum(1 for l in levels if l in ('L2', 'L3', 'L3_REJECT'))
    l3 = sum(1 for l in levels if l in ('L3', 'L3_REJECT'))
    return {'L1': non_pass / max(n, 1), 'L2': l2 / max(n, 1), 'L3': l3 / max(n, 1)}


def run(data_dir, corruptions, severities, n_per, n_recal, seed, output_path, device_str=None):
    assert DATASET == 'cifar10', f"CIFAR-10-C is CIFAR-10 only; DATASET={DATASET}"
    assert BACKBONE_INPUT_SIZE == 32
    device = torch.device(device_str) if device_str else \
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  n_per={n_per}  n_recal={n_recal}  seed={seed}", flush=True)

    if not os.path.exists(os.path.join(data_dir, 'labels.npy')):
        raise FileNotFoundError(f"CIFAR-10-C not at {data_dir}")

    model = load_backbone(device)
    prism = PRISM.from_saved(
        model=model, layer_names=LAYER_NAMES, calibrator_path=PATHS['calibrator'],
        profile_path=PATHS['reference_profiles'], ensemble_path=PATHS['ensemble_scorer'],
        layer_weights=LAYER_WEIGHTS, dim_weights=DIM_WEIGHTS,
        campaign_monitor=NoOpCampaignMonitor())

    t0 = time.time()
    results = {}
    for corruption in corruptions:
        npy = os.path.join(data_dir, f'{corruption}.npy')
        if not os.path.exists(npy):
            print(f"  SKIP {corruption}", flush=True)
            continue
        arr = np.load(npy)
        for sev in severities:
            block = arr[(sev - 1) * 10000: sev * 10000]
            scores, levels = score_block(prism, block, device, n_per, seed)
            n = len(scores)
            nr = min(n_recal, n // 2)
            recal_s, eval_s = scores[:nr], scores[nr:]
            eval_levels = levels[nr:]
            fr = frozen_fpr(eval_levels)                 # degradation on eval slice
            rest = {}
            for tier, a in ALPHAS.items():
                thr = conformal_threshold(recal_s, a)
                k = int(np.sum(eval_s > thr))
                lo, hi = wilson_ci(k, len(eval_s))
                rest[tier] = {'FPR': round(k / max(len(eval_s), 1), 4),
                              'CI_95': [round(lo, 4), round(hi, 4)], 'target_alpha': a}
            key = f'{corruption}_sev{sev}'
            results[key] = {
                'corruption': corruption, 'severity': sev,
                'n_eval': len(eval_s), 'n_recal': nr,
                'frozen_FPR': {t: round(fr[t], 4) for t in fr},
                'restored_FPR': {t: rest[t]['FPR'] for t in rest},
                'restored_detail': rest,
            }
            print(f"  {key:>28}: L1 frozen {fr['L1']:.3f} -> restored {rest['L1']['FPR']:.3f} "
                  f"(a=0.10) | L3 frozen {fr['L3']:.3f} -> {rest['L3']['FPR']:.3f} (a=0.005)",
                  flush=True)

    agg = {}
    for tier, a in ALPHAS.items():
        fz = [v['frozen_FPR'][tier] for v in results.values()]
        rs = [v['restored_FPR'][tier] for v in results.values()]
        agg[tier] = {
            'target_alpha': a,
            'mean_frozen_FPR': round(float(np.mean(fz)), 4) if fz else None,
            'mean_restored_FPR': round(float(np.mean(rs)), 4) if rs else None,
            'max_frozen_FPR': round(float(np.max(fz)), 4) if fz else None,
            'max_restored_FPR': round(float(np.max(rs)), 4) if rs else None,
            'cells_restored_within_target': int(sum(r <= a for r in rs)),
            'cells_total': len(rs),
        }

    payload = {
        'experiment': 'cifar10c_certificate_restoration', 'dataset': 'cifar10',
        'backbone': 'resnet18', 'seed': seed, 'n_per': n_per, 'n_recal': n_recal,
        'method': ('shift-aware split-conformal recalibration: re-fit q_alpha on a '
                   'held-out slice of the same corruption, measure FPR on the rest.'),
        'per_cell': results, 'aggregate': agg, 'elapsed_sec': round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print("\n=== AGGREGATE (mean over cells) ===", flush=True)
    for t, a in ALPHAS.items():
        print(f"  {t} (a={a}): frozen {agg[t]['mean_frozen_FPR']} -> restored "
              f"{agg[t]['mean_restored_FPR']}  ({agg[t]['cells_restored_within_target']}/"
              f"{agg[t]['cells_total']} cells within target)", flush=True)
    print(f"Wrote {output_path}  ({payload['elapsed_sec']}s)", flush=True)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=str, default='data/CIFAR-10-C')
    ap.add_argument('--corruptions', type=str, nargs='+', default=DEFAULT_CORRUPTIONS)
    ap.add_argument('--severities', type=int, nargs='+', default=[1, 3, 5])
    ap.add_argument('--n-per', type=int, default=1000)
    ap.add_argument('--n-recal', type=int, default=300)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', type=str, default=None)
    ap.add_argument('--config', type=str, default=None)
    ap.add_argument('--output', type=str,
                    default='experiments/stress/results_cifar10c_restore.json')
    args = ap.parse_args()
    run(data_dir=args.data_dir, corruptions=args.corruptions, severities=args.severities,
        n_per=args.n_per, n_recal=args.n_recal, seed=args.seed,
        output_path=args.output, device_str=args.device)


if __name__ == '__main__':
    main()
