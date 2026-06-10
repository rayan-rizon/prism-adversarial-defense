"""
Direct Zeroth-Order Attack on the DEPLOYED PRISM score S(x)
===========================================================

Why this exists (reviewer rebuttal)
-----------------------------------
The ensemble-complete adaptive PGD in `run_adaptive_pgd.py` attacks
*differentiable surrogates* of the non-smooth detector channels (StabilityV2,
the side-quadratic logistic head, TDA-via-activation-matching). A NeurIPS/ICLR
reviewer in the Tramer-et-al-2020 tradition will object that any residual
robustness could be an artifact of the surrogate gap, not of the detector.

This script removes that objection. PRISM's forward pass is fully deterministic
(MD5-hash subsampling, no RNG), so the deployed fused score
S(x) = metadata['anomaly_score'] is a queryable black-box scalar. We attack it
DIRECTLY with a zeroth-order (gradient-free) optimiser — NES (Ilyas et al. 2018)
or SPSA (Uesato et al. 2018) — never differentiating through a proxy.

Threat model (strongest standard one for a detector)
----------------------------------------------------
  * Classifier f: WHITE-BOX. Attacker has exact CE gradient (worst case).
  * Detector S(x): BLACK-BOX but fully queryable. Attacker estimates dS/dx by
    finite differences on the EXACT deployed score — no surrogate.
Combined ascent objective (maximise):
      J(x') = CE(f(x'), y_clean)  -  c * S(x')
  - first term  -> misclassify
  - second term -> push the deployed detector score down (evade)
We sweep c so the trade-off curve (model ASR vs. detector evasion) is explicit,
exactly mirroring the lambda-sweep framing of the adaptive-PGD table.

This is a CIFAR-10 / ResNet-18 ONLY stress test by design (one setting, small n).
We do NOT claim it for WRN / CIFAR-100 / ViT.

Usage
-----
  cd prism/
  # quick smoke
  python experiments/evaluation/run_direct_score_attack.py \
      --n-test 50 --steps 30 --nes-queries 10 --c 1.0

  # paper stress run (matches run_vastai_revision.sh)
  python experiments/evaluation/run_direct_score_attack.py \
      --n-test 200 --steps 60 --nes-queries 20 --mode nes \
      --c 0.0 0.5 1.0 2.0 5.0 --seed 42 \
      --output experiments/evaluation/results_direct_score_attack_seed42.json

EVAL SPLIT: src.config.EVAL_IDX (same held-out window as every other table).
"""
import os
import sys
import ssl
import json
import time
import argparse

import numpy as np
import torch
import torchvision.transforms as T

os.environ.setdefault('SSL_CERT_FILE', __import__('certifi').where())
os.environ.setdefault('REQUESTS_CA_BUNDLE', __import__('certifi').where())
ssl._create_default_https_context = ssl.create_default_context

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src import bootstrap  # noqa: F401  routes --config -> PRISM_CONFIG
from src.prism import PRISM
from src.sacd.monitor import NoOpCampaignMonitor
from src.models import load_backbone
from src.data_loader import load_test_dataset
from src.config import (
    LAYER_NAMES, LAYER_WEIGHTS, DIM_WEIGHTS,
    BACKBONE_MEAN, BACKBONE_STD, BACKBONE_INPUT_SIZE,
    EPS_LINF_STANDARD, EVAL_IDX, DATASET, PATHS,
)

_MEAN = BACKBONE_MEAN
_STD = BACKBONE_STD
if BACKBONE_INPUT_SIZE == 32:
    _PIXEL_TRANSFORM = T.Compose([T.ToTensor()])
else:
    _PIXEL_TRANSFORM = T.Compose([T.Resize(BACKBONE_INPUT_SIZE), T.ToTensor()])
_NORMALIZE = T.Normalize(mean=_MEAN, std=_STD)


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    margin = (z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def per_tier_fpr(clean_levels, n_clean):
    fp_l1 = n_clean - clean_levels.get('PASS', 0)
    fp_l2 = clean_levels.get('L2', 0) + clean_levels.get('L3', 0) + clean_levels.get('L3_REJECT', 0)
    fp_l3 = clean_levels.get('L3', 0) + clean_levels.get('L3_REJECT', 0)
    return {
        'FPR_L1_plus': round(fp_l1 / max(n_clean, 1), 4),
        'FPR_L2_plus': round(fp_l2 / max(n_clean, 1), 4),
        'FPR_L3_plus': round(fp_l3 / max(n_clean, 1), 4),
        'target_L1': 0.10, 'target_L2': 0.03, 'target_L3': 0.005,
    }


def _to_norm(x_pixel, device):
    """(1,3,H,W) pixel [0,1] -> normalised tensor on device."""
    mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)
    return (x_pixel - mean) / std


@torch.no_grad()
def query_score(prism, x_pixel, device):
    """Query the EXACT deployed fused score S(x) for one (1,3,H,W) pixel input."""
    x_norm = _to_norm(x_pixel, device)
    pix = x_pixel.squeeze(0).detach()
    _, level, info = prism.defend(x_norm, pixel_image=pix)
    score = float(info.get('anomaly_score', 0.0))
    return score, level


def ce_grad_pixel(model, x_pixel, y, device):
    """Exact white-box CE gradient w.r.t. PIXEL space (chain rule through (x-mean)/std)."""
    mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)
    x = x_pixel.detach().clone().requires_grad_(True)
    x_norm = (x - mean) / std
    logits = model(x_norm)
    loss = torch.nn.functional.cross_entropy(logits, y)
    (g,) = torch.autograd.grad(loss, x)
    return g.detach()  # d CE / d pixel


def nes_score_grad(prism, x_pixel, device, sigma, n_queries, mode='nes'):
    """
    Estimate dS/dx of the DEPLOYED detector score by antithetic zeroth-order
    sampling. Pure black-box: each evaluation is one real prism.defend() call.

      NES   : u ~ N(0, I)
      SPSA  : u ~ Rademacher(+/-1)
    g ~= (1 / (2 sigma Q)) * sum_i [S(x + sigma u_i) - S(x - sigma u_i)] u_i
    """
    g = torch.zeros_like(x_pixel)
    for _ in range(n_queries):
        if mode == 'spsa':
            u = torch.randint(0, 2, x_pixel.shape, device=device, dtype=x_pixel.dtype) * 2 - 1
        else:
            u = torch.randn_like(x_pixel)
        s_plus, _ = query_score(prism, (x_pixel + sigma * u).clamp(0.0, 1.0), device)
        s_minus, _ = query_score(prism, (x_pixel - sigma * u).clamp(0.0, 1.0), device)
        g += (s_plus - s_minus) * u
    return g / (2.0 * sigma * max(n_queries, 1))


def direct_score_attack(prism, model, x_pixel, y, device, eps, steps, step_size,
                        c, sigma, n_queries, mode):
    """
    Maximise  J = CE(f(x'), y) - c * S(x')  via sign-ascent in the L_inf ball.
    CE term uses the exact white-box gradient; S term uses the black-box NES/SPSA
    estimate on the EXACT deployed score. No surrogate anywhere on the S path.
    """
    x0 = x_pixel.detach().clone()
    delta = torch.empty_like(x0).uniform_(-eps, eps)
    x = (x0 + delta).clamp(0.0, 1.0)
    for _ in range(steps):
        g_ce = ce_grad_pixel(model, x, y, device)          # ascend CE
        if c != 0.0:
            g_s = nes_score_grad(prism, x, device, sigma, n_queries, mode)  # descend S
        else:
            g_s = torch.zeros_like(x)
        ascent = g_ce - c * g_s
        x = x + step_size * ascent.sign()
        x = torch.min(torch.max(x, x0 - eps), x0 + eps).clamp(0.0, 1.0)
    return x.detach()


def run(n_test, c_values, steps, sigma, n_queries, mode, seed,
        output_path, device_str=None, data_root='./data'):
    eps = EPS_LINF_STANDARD
    step_size = eps / 4.0
    device = torch.device(device_str) if device_str else \
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    assert DATASET == 'cifar10', (
        f"Direct-score attack is scoped to CIFAR-10/ResNet-18 only; active DATASET={DATASET}. "
        "Run under the default CIFAR-10 config."
    )

    print(f"Device: {device}")
    print(f"Direct score attack [{mode.upper()}]: n={n_test}, steps={steps}, "
          f"nes_queries={n_queries}, sigma={sigma}, eps={eps:.4f}, c={c_values}")
    print(f"Eval split: {DATASET.upper()} test[{EVAL_IDX[0]}-{EVAL_IDX[1]-1}], seed={seed}\n")

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    model = load_backbone(device)
    ds = load_test_dataset(root=data_root, download=True, transform=_PIXEL_TRANSFORM)
    eval_indices = list(range(*EVAL_IDX))
    sample_idx = rng.choice(eval_indices, min(n_test, len(eval_indices)), replace=False)

    imgs = []
    for i in sample_idx:
        img, _ = ds[int(i)]
        imgs.append(img)
    print(f"Pre-loaded {len(imgs)} images\n")

    results = {}
    t_start = time.time()

    for c in c_values:
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

        tp = fp = fn = tn = 0
        attack_success = 0
        detected_success = 0
        evaded_success = 0
        level_clean, level_adv = {}, {}

        print(f"{'='*60}\nc={c}\n{'='*60}")
        for j, img in enumerate(imgs):
            x_pixel = img.unsqueeze(0).to(device)
            with torch.no_grad():
                clean_pred = int(model(_to_norm(x_pixel, device)).argmax(1).item())
            y = torch.tensor([clean_pred], device=device)

            # clean side (benign FPR)
            _, lv_c = query_score(prism, x_pixel, device)
            level_clean[lv_c] = level_clean.get(lv_c, 0) + 1
            tn += (lv_c == 'PASS')
            fp += (lv_c != 'PASS')

            x_adv = direct_score_attack(prism, model, x_pixel, y, device, eps,
                                        steps, step_size, c, sigma, n_queries, mode)
            with torch.no_grad():
                adv_pred = int(model(_to_norm(x_adv, device)).argmax(1).item())
            is_success = (adv_pred != clean_pred)
            attack_success += is_success

            _, lv_a = query_score(prism, x_adv, device)
            level_adv[lv_a] = level_adv.get(lv_a, 0) + 1
            if lv_a != 'PASS':
                tp += 1
                detected_success += is_success
            else:
                fn += 1
                evaded_success += is_success

            if (j + 1) % 25 == 0:
                _tpr = tp / max(tp + fn, 1)
                print(f"  [{j+1}/{len(imgs)}] TPR={_tpr:.4f} ASR={attack_success/(j+1):.4f}")

        n_adv = tp + fn
        n_clean = fp + tn
        tpr = tp / max(n_adv, 1)
        fpr = fp / max(n_clean, 1)
        tpr_ci = wilson_ci(tp, n_adv)
        key = f'DirectScoreAttack_{mode}_c_{c}'
        results[key] = {
            'mode': mode, 'c': c,
            'TPR': round(tpr, 4),
            'TPR_CI_95': [round(tpr_ci[0], 4), round(tpr_ci[1], 4)],
            'FPR': round(fpr, 4),
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'n_adv': n_adv, 'n_clean': n_clean,
            'model_ASR': round(attack_success / max(n_adv, 1), 4),
            'TPR_on_successful_attacks': round(detected_success / max(attack_success, 1), 4),
            'undetected_success_rate': round(evaded_success / max(n_adv, 1), 4),
            'detected_successful_adv': int(detected_success),
            'evaded_successful_adv': int(evaded_success),
            'n_successful_adv': int(attack_success),
            'per_tier_fpr': per_tier_fpr(level_clean, n_clean),
            'clean_level_distribution': level_clean,
            'adversarial_level_distribution': level_adv,
            'eps': round(eps, 6),
            'steps': steps, 'nes_queries': n_queries, 'sigma': sigma,
            'attacker': 'whitebox-classifier + blackbox-deployed-score (no surrogate)',
        }
        print(f"  c={c}: TPR={tpr:.4f} CI[{tpr_ci[0]:.4f},{tpr_ci[1]:.4f}] "
              f"ASR={results[key]['model_ASR']:.4f} "
              f"TPR|success={results[key]['TPR_on_successful_attacks']:.4f} "
              f"undetected_success={results[key]['undetected_success_rate']:.4f}\n")

    payload = {
        'experiment': 'direct_score_attack',
        'dataset': DATASET, 'backbone': 'resnet18',
        'seed': seed, 'mode': mode,
        'eval_idx': list(EVAL_IDX),
        'elapsed_sec': round(time.time() - t_start, 1),
        'results': results,
    }
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"Wrote {output_path}  ({payload['elapsed_sec']}s)")
    return payload


def main():
    ap = argparse.ArgumentParser(description="Direct zeroth-order attack on deployed PRISM score")
    ap.add_argument('--n-test', type=int, default=200)
    ap.add_argument('--c', type=float, nargs='+', default=[0.0, 0.5, 1.0, 2.0, 5.0],
                    help="Detector-evasion weights to sweep (0.0 = NES off = plain whitebox PGD ref).")
    ap.add_argument('--steps', type=int, default=60)
    ap.add_argument('--sigma', type=float, default=0.001, help="NES/SPSA finite-difference radius (pixel space).")
    ap.add_argument('--nes-queries', type=int, default=20, help="Antithetic sample pairs per step.")
    ap.add_argument('--mode', choices=['nes', 'spsa'], default='nes')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', type=str, default=None)
    ap.add_argument('--data-root', type=str, default='./data')
    ap.add_argument('--config', type=str, default=None, help="Routed to PRISM_CONFIG via src.bootstrap.")
    ap.add_argument('--output', type=str,
                    default='experiments/evaluation/results_direct_score_attack.json')
    args = ap.parse_args()

    run(n_test=args.n_test, c_values=args.c, steps=args.steps, sigma=args.sigma,
        n_queries=args.nes_queries, mode=args.mode, seed=args.seed,
        output_path=args.output, device_str=args.device, data_root=args.data_root)


if __name__ == '__main__':
    main()
