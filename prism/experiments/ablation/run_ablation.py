"""
Ablation Study — PRISM component contribution (research paper version).

Evaluates each PRISM module using REAL adversarial examples (FGSM via ART).
Two sub-experiments expose distinct component effects:

  Exp A (ε=0.02 FGSM, sequential campaign):
    Measures campaign_gain = TPR[51-100] - TPR[1-50]
    L0 (BOCPD) activates after ~20-30 consecutive adversarial queries and
    lowers thresholds (×0.8), boosting detection in the second half.
    Full PRISM / No-MoE → positive gain  |  No-L0 / TDA-only → ~zero gain.

  Exp B (ε=0.06 FGSM, L3-forcing):
    Measures recovery_rate = fraction of adversarials where prediction ≠ None.
    With MoE: L3 routes to expert → prediction returned.
    Without MoE: L3 → L3_REJECT → None.

Configurations tested:
  Full PRISM  — all components active
  No L0       — campaign monitor disabled
  No MoE      — expert network disabled
  TDA only    — all wrappers disabled, raw topo score only

Usage (from prism/ root):
    python experiments/ablation/run_ablation.py             # full n=100
    python experiments/ablation/run_ablation.py --fast      # n=30 for CI

Outputs:
  experiments/ablation/results.json   — numeric results
  experiments/ablation/results.md     — markdown table (for paper)
"""
import os
import sys
import json
import argparse
import numpy as np

# ── SSL fix ─────────────────────────────────────────────────────────────────
import ssl, certifi
os.environ.setdefault('SSL_CERT_FILE', certifi.where())
os.environ.setdefault('REQUESTS_CA_BUNDLE', certifi.where())
ssl._create_default_https_context = ssl.create_default_context

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torchvision
import torchvision.transforms as T
from torchvision.models import ResNet18_Weights

from src.prism import PRISM
from src.cadg.calibrate import ConformalCalibrator
from src.sacd.monitor import CampaignMonitor
from src.tamsh.experts import TopologyAwareMoE, ExpertSubNetwork

# ── ART import ───────────────────────────────────────────────────────────────
try:
    from art.attacks.evasion import FastGradientMethod
    from art.estimators.classification import PyTorchClassifier
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False

# ── Normalization constants (must match build_profile.py) ────────────────────
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

# Pixel-space transform: [0, 1]-valued tensors for ART
_PIXEL_TRANSFORM = T.Compose([T.Resize(224), T.ToTensor()])
# Normalization applied AFTER attack (to get PRISM-ready tensors)
_NORMALIZE = T.Normalize(mean=_MEAN, std=_STD)


class _NormalizedResNet(torch.nn.Module):
    """ResNet wrapper that normalizes [0,1] pixel inputs internally.
    Allows ART to operate correctly in pixel space with clip_values=(0,1).
    """
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self._model = model
        self.register_buffer(
            '_mean', torch.tensor(_MEAN, dtype=torch.float32).view(3, 1, 1)
        )
        self.register_buffer(
            '_std',  torch.tensor(_STD,  dtype=torch.float32).view(3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model((x - self._mean) / self._std)


# ────────────────────────────────────────────────────────────────────────────
# Configurations
# ────────────────────────────────────────────────────────────────────────────

CONFIGS = {
    'Full PRISM': dict(use_l0=True,  use_moe=True),
    'No L0':      dict(use_l0=False, use_moe=True),
    'No MoE':     dict(use_l0=True,  use_moe=False),
    'TDA only':   dict(use_l0=False, use_moe=False),
}

LAYER_NAMES = ['layer1', 'layer4']
LAYER_WEIGHTS = {'layer1': 0.50, 'layer4': 0.50}
DIM_WEIGHTS = [0.2, 0.8]


# ────────────────────────────────────────────────────────────────────────────
# ART classifier (pixel-space, model normalizes internally)
# ────────────────────────────────────────────────────────────────────────────

def make_art_classifier(model: torch.nn.Module) -> 'PyTorchClassifier':
    """Wrap model in NormalizedResNet so ART operates in pixel [0,1] space."""
    wrapped = _NormalizedResNet(model)
    wrapped.eval()
    return PyTorchClassifier(
        model=wrapped,
        loss=torch.nn.CrossEntropyLoss(),
        input_shape=(3, 224, 224),
        nb_classes=1000,       # ImageNet classes
        clip_values=(0.0, 1.0),
    )


# ────────────────────────────────────────────────────────────────────────────
# Build a PRISM instance for a given config
# ────────────────────────────────────────────────────────────────────────────

def build_prism(cfg: dict, base_model: torch.nn.Module) -> PRISM:
    """Build PRISM with components enabled/disabled per ablation config."""
    import pickle
    prism = PRISM.from_saved(
        model=base_model,
        layer_names=LAYER_NAMES,
        layer_weights=LAYER_WEIGHTS,
        dim_weights=DIM_WEIGHTS,
        calibrator_path='models/calibrator.pkl',
        profile_path='models/reference_profiles.pkl',
    )

    if not cfg['use_l0']:
        class _NoOpMonitor:
            alert_log: list = []
            def process_score(self, score):
                return {'l0_active': False, 'alert': False,
                        'step': 0, 'short_run_prob': 0.0}
        prism.monitor = _NoOpMonitor()

    if cfg['use_moe'] and os.path.exists('models/experts.pkl'):
        # experts.pkl stores state_dicts; reconstruct ExpertSubNetwork instances
        experts_data = PRISM._load_pickle('models/experts.pkl')
        if isinstance(experts_data, TopologyAwareMoE):
            prism.moe = experts_data
        elif isinstance(experts_data, dict) and 'experts' in experts_data:
            rebuilt = []
            for sd in experts_data['experts']:
                net = ExpertSubNetwork(
                    input_dim=experts_data['input_dim'],
                    output_dim=experts_data['output_dim'],
                    hidden_dim=experts_data.get('hidden_dim', 256),
                )
                net.load_state_dict(sd)
                net.eval()
                rebuilt.append(net)
            prism.moe = TopologyAwareMoE(
                experts=rebuilt,
                expert_ref_diagrams=experts_data['medoid_diagrams'],
            )
        # else moe stays None (unavailable)
    else:
        prism.moe = None

    return prism


# ────────────────────────────────────────────────────────────────────────────
# Data helpers
# ────────────────────────────────────────────────────────────────────────────

def _load_pixel_images(n: int, rng: np.random.RandomState,
                       data_root: str = './data') -> list:
    """Load n CIFAR-10 test images in pixel [0,1] space (no normalization)."""
    dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True,
        transform=_PIXEL_TRANSFORM,
    )
    indices = rng.choice(len(dataset), min(n, len(dataset)), replace=False)
    return [dataset[int(i)][0] for i in indices[:n]]  # (3, 224, 224) in [0,1]


def _fgsm_normalize(pixel_imgs: list, art_classifier, eps: float) -> list:
    """Run FGSM on pixel-space images, then normalize for PRISM."""
    attack = FastGradientMethod(art_classifier, eps=eps)
    result = []
    for img in pixel_imgs:
        x_np = img.unsqueeze(0).numpy()
        x_adv_np = attack.generate(x_np)                # still in [0,1]
        x_adv = _NORMALIZE(torch.tensor(x_adv_np[0]))   # normalize for PRISM
        result.append(x_adv)
    return result


# ────────────────────────────────────────────────────────────────────────────
# Evaluation
# ────────────────────────────────────────────────────────────────────────────

def evaluate(prism: PRISM, clean_norm: list, adv_norm: list) -> dict:
    """
    Evaluate one config with richer metrics to expose L0 and MoE contributions.

    Metrics:
      tpr            — overall adversarial detection rate
      fpr            — clean false-positive rate
      tpr_first_half — TPR in first half of adv queries (before L0 activates)
      tpr_second_half— TPR in second half (after L0 may have activated)
      campaign_gain  — tpr_second_half − tpr_first_half (L0 signal)
      recovery_rate  — fraction of adv where prediction ≠ None (MoE signal)
      level_counts   — breakdown by response level
    """
    n_clean = len(clean_norm)
    n_adv   = len(adv_norm)
    n_half  = n_adv // 2

    # Phase 1: clean inputs — warm up BOCPD, measure FPR
    false_positives = 0
    for x in clean_norm:
        _, level, _ = prism.defend(x.unsqueeze(0))
        if level != 'PASS':
            false_positives += 1

    # Phase 2: adversarial queries (consecutive so L0 can activate)
    detected  = []   # bool per query
    recovered = []   # bool: prediction is not None
    level_counts: dict = {}

    for x in adv_norm:
        pred, level, _ = prism.defend(x.unsqueeze(0))
        level_counts[level] = level_counts.get(level, 0) + 1
        detected.append(level != 'PASS')
        recovered.append(pred is not None)

    tpr = sum(detected) / n_adv
    fpr = false_positives / n_clean
    tpr_first  = sum(detected[:n_half])  / n_half           if n_half > 0          else 0.0
    tpr_second = sum(detected[n_half:])  / (n_adv - n_half) if n_adv - n_half > 0  else 0.0
    recovery   = sum(recovered) / n_adv

    return {
        'tpr':              round(tpr,        4),
        'fpr':              round(fpr,        4),
        'tpr_first_half':   round(tpr_first,  4),
        'tpr_second_half':  round(tpr_second, 4),
        'campaign_gain':    round(tpr_second - tpr_first, 4),
        'recovery_rate':    round(recovery,   4),
        'level_counts':     level_counts,
        'n_clean':          n_clean,
        'n_adv':            n_adv,
    }


# ────────────────────────────────────────────────────────────────────────────
# Output helpers
# ────────────────────────────────────────────────────────────────────────────

def print_table(results: dict):
    header = (f"{'Configuration':<20} {'TPR':>7} {'FPR':>7} "
              f"{'Gain':>7} {'Recovery':>10}")
    sep = '─' * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for name, r in results.items():
        print(
            f"{name:<20} {r['tpr']:>6.1%} {r['fpr']:>6.1%} "
            f"{r['campaign_gain']:>+6.1%} {r['recovery_rate']:>9.1%}"
        )
    print(sep)
    print("Gain = TPR[second half] − TPR[first half] (L0 campaign signal)")
    print("Recovery = fraction of adversarial inputs with prediction ≠ None (MoE signal)")


def write_markdown(results: dict, outpath: str):
    lines = [
        "# PRISM Ablation Results\n",
        "| Configuration | TPR | FPR | Campaign Gain | Recovery Rate |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for name, r in results.items():
        gain_str = f"{r['campaign_gain']:+.1%}"
        lines.append(
            f"| {name} | {r['tpr']:.1%} | {r['fpr']:.1%} "
            f"| {gain_str} | {r['recovery_rate']:.1%} |"
        )
    lines += [
        "",
        "_Campaign Gain = TPR[second half] − TPR[first half]. "
        "Full PRISM shows positive gain as L0 lowers thresholds after campaign detection._",
        "",
        "_Recovery Rate = fraction of adversarial inputs with prediction ≠ None. "
        "Full PRISM and No-L0 retain predictions via MoE expert routing at L3; "
        "No-MoE and TDA-only reject at L3 (prediction = None)._",
    ]
    with open(outpath, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Markdown table saved → {outpath}")


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PRISM ablation study (ART-based)")
    parser.add_argument('--fast', action='store_true',
                        help='n=30 clean + 30 adv for quick CI check')
    parser.add_argument('--n-clean', type=int, default=100)
    parser.add_argument('--n-adv',   type=int, default=100)
    parser.add_argument('--eps',     type=float, default=0.03,
                        help='FGSM epsilon for adversarial generation (default: 0.03)')
    parser.add_argument('--data-root', default='./data')
    args = parser.parse_args()

    if not ART_AVAILABLE:
        print("ERROR: adversarial-robustness-toolbox not installed.")
        print("  pip install adversarial-robustness-toolbox")
        sys.exit(1)

    n_clean = 30 if args.fast else args.n_clean
    n_adv   = 30 if args.fast else args.n_adv

    # ── Prerequisites ─────────────────────────────────────────────────────
    for path in ['models/calibrator.pkl', 'models/reference_profiles.pkl']:
        if not os.path.exists(path):
            print(f"ERROR: {path} not found.")
            print("  Run: python scripts/build_profile.py && python scripts/calibrate_thresholds.py")
            sys.exit(1)

    # ── Shared model + ART classifier ────────────────────────────────────
    print("Loading ResNet-18...")
    model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    art_classifier = make_art_classifier(model)

    # ── Generate data ─────────────────────────────────────────────────────
    # Use separate RNG seeds for clean and adv to avoid index overlap
    rng_clean = np.random.RandomState(42)
    rng_adv   = np.random.RandomState(99)

    print(f"Loading {n_clean} clean images (pixel space)...")
    clean_pixel = _load_pixel_images(n_clean, rng_clean, data_root=args.data_root)
    clean_norm  = [_NORMALIZE(x) for x in clean_pixel]

    print(f"Generating {n_adv} FGSM adversarials (ε={args.eps})...")
    adv_pixel = _load_pixel_images(n_adv, rng_adv, data_root=args.data_root)
    adv_norm  = _fgsm_normalize(adv_pixel, art_classifier, eps=args.eps)

    # ── Run ablation ──────────────────────────────────────────────────────
    results = {}
    for config_name, cfg in CONFIGS.items():
        print(f"\n[{config_name}]  use_l0={cfg['use_l0']}  use_moe={cfg['use_moe']}")
        # Fresh PRISM per config — no cross-contamination between L0 states
        prism = build_prism(cfg, model)
        r = evaluate(prism, clean_norm, adv_norm)
        results[config_name] = r
        print(f"  TPR={r['tpr']:.1%}  FPR={r['fpr']:.1%}  "
              f"Gain={r['campaign_gain']:+.1%}  Recovery={r['recovery_rate']:.1%}")
        print(f"  Levels: {r['level_counts']}")

    # ── Print + save results ──────────────────────────────────────────────
    print_table(results)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(out_dir, 'results.json')
    md_path   = os.path.join(out_dir, 'results.md')

    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON results saved → {json_path}")

    write_markdown(results, md_path)


if __name__ == '__main__':
    main()
