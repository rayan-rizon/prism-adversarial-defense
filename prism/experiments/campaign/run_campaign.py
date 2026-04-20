"""
Real PRISM Campaign Detection Experiment
Uses actual PRISM inference to generate anomaly scores.
Scenario: 50 clean CIFAR-10 images, then 100 FGSM adversarials.
Measures: step at which SACD/L0 fires (campaign detection latency).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import json

from torchvision.models import ResNet18_Weights
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier

from src.prism import PRISM
from src.tamsh.experts import TopologyAwareMoE
from src.sacd.monitor import CampaignMonitor
from src.config import (
    LAYER_NAMES, LAYER_WEIGHTS, DIM_WEIGHTS,
    IMAGENET_MEAN, IMAGENET_STD, EPS_LINF_STANDARD,
)

# ── Transforms ─────────────────────────────────────────────────────────────────
_MEAN = IMAGENET_MEAN
_STD  = IMAGENET_STD

class _NormalizedResNet(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.register_buffer('mean', torch.tensor(_MEAN).view(1,3,1,1))
        self.register_buffer('std',  torch.tensor(_STD).view(1,3,1,1))
    def forward(self, x):
        return self.backbone((x - self.mean) / self.std)

_PIXEL = T.Compose([T.Resize(224), T.ToTensor()])
_NORM  = T.Normalize(_MEAN, _STD)


def run_campaign_experiment(n_clean=50, n_adv=100, eps=EPS_LINF_STANDARD, seed=42):
    print(f"=== Real PRISM Campaign Detection ===")
    print(f"  n_clean={n_clean}, n_adv={n_adv}, eps={eps}\n")

    rng = np.random.RandomState(seed)

    # ── Load PRISM ────────────────────────────────────────────────────────────
    backbone = torchvision.models.resnet18(
        weights=ResNet18_Weights.IMAGENET1K_V1
    ).eval()
    wrapped = _NormalizedResNet(backbone).eval()

    prism = PRISM.from_saved(
        model=backbone,
        layer_names=LAYER_NAMES,
        layer_weights=LAYER_WEIGHTS,
        dim_weights=DIM_WEIGHTS,
        calibrator_path='models/calibrator.pkl',
        profile_path='models/reference_profiles.pkl',
        # Calibrated BOCPD prior for CIFAR-10 clean scores (mean~4.6, std~1.1)
        # alert_run_prob=0.3 → fire when P(run_length≤alert_run_length)>30%
        # hazard_rate=1/30 → prior expects changepoint every ~30 steps
        campaign_monitor=CampaignMonitor(
            mu0=7.0, kappa0=5.0, alpha0=3.0, beta0=15.0,
            hazard_rate=1/30, alert_run_length=10, alert_run_prob=0.60,
            warmup_steps=35, l0_factor=0.8, cooldown_steps=30,
        ),
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = backbone.to(device)
    wrapped  = wrapped.to(device)
    prism.model = backbone
    prism.extractor.model = backbone

    # ── Load dataset ──────────────────────────────────────────────────────────
    ds = torchvision.datasets.CIFAR10(
        './data', train=False, download=False, transform=_PIXEL
    )

    # ── ART classifier ────────────────────────────────────────────────────────
    art_clf = PyTorchClassifier(
        model=wrapped,
        loss=torch.nn.CrossEntropyLoss(),
        input_shape=(3, 224, 224),
        nb_classes=1000,  # ResNet-18 ImageNet backbone has 1000 output classes
        clip_values=(0.0, 1.0),
    )
    fgsm = FastGradientMethod(art_clf, eps=eps)

    # ── Build query stream ────────────────────────────────────────────────────
    # Fresh PRISM monitor (reset L0 state)
    prism.monitor.reset()

    indices = rng.choice(len(ds), n_clean + n_adv, replace=False)
    clean_idx = indices[:n_clean]
    adv_idx   = indices[n_clean:]

    scores = []
    levels = []
    l0_fired_at = None

    print("Phase 1: Clean queries")
    for t, idx in enumerate(clean_idx):
        pixel_img, _ = ds[int(idx)]
        norm_img = _NORM(pixel_img).unsqueeze(0).to(device)
        _, level, meta = prism.defend(norm_img)
        s = meta.get('anomaly_score', 0.0)
        scores.append(s)
        levels.append(level)
        l0 = meta.get('l0_state', {}).get('l0_active', False)
        if l0 and l0_fired_at is None:
            l0_fired_at = t
            print(f"  !! False alarm at clean step {t} (score={s:.3f})")

    print(f"  Done. Mean score={np.mean(scores[:n_clean]):.3f}  "
          f"Std={np.std(scores[:n_clean]):.3f}")
    print(f"  L0 so far: {'FIRED (false alarm)'  if l0_fired_at is not None else 'quiet'}\n")

    print("Phase 2: Adversarial queries (FGSM)")
    adv_l0_at = None
    for t, idx in enumerate(adv_idx):
        pixel_img, _ = ds[int(idx)]
        x_np = pixel_img.unsqueeze(0).numpy()
        x_adv_np = fgsm.generate(x_np)
        x_adv_norm = _NORM(torch.tensor(x_adv_np[0])).unsqueeze(0).to(device)

        _, level, meta = prism.defend(x_adv_norm)
        s = meta.get('anomaly_score', 0.0)
        scores.append(s)
        levels.append(level)

        l0 = meta.get('l0_state', {}).get('l0_active', False)
        if l0 and adv_l0_at is None:
            adv_l0_at = t
            step_global = n_clean + t
            print(f"  *** L0 fired at adversarial step {t} "
                  f"(global step {step_global}, score={s:.3f})")

        if (t + 1) % 20 == 0:
            print(f"  Processed {t+1}/{n_adv} adversarials...")

    if adv_l0_at is None:
        print("  L0 never fired during adversarial phase.")

    # ── Results ───────────────────────────────────────────────────────────────
    adv_scores = np.array(scores[n_clean:])
    clean_scores_arr = np.array(scores[:n_clean])

    print(f"\n=== Campaign Detection Results ===")
    print(f"  Clean scores:  mean={clean_scores_arr.mean():.3f} std={clean_scores_arr.std():.3f}")
    print(f"  Adv scores:    mean={adv_scores.mean():.3f}  std={adv_scores.std():.3f}")
    print(f"  L0 detection:  {'step ' + str(adv_l0_at) + ' after campaign onset' if adv_l0_at is not None else 'NOT DETECTED'}")

    if adv_l0_at is not None:
        print(f"  Lead time:     {adv_l0_at} adversarial queries before L0 fired")
        if adv_l0_at < 20:
            print(f"  ✓ Within target (<20 queries)")
        else:
            print(f"  ⚠ Slower than target (>20 queries)")

    # ── Save ──────────────────────────────────────────────────────────────────
    result = {
        'n_clean': n_clean,
        'n_adv': n_adv,
        'eps': eps,
        'l0_detected_at_adv_step': int(adv_l0_at) if adv_l0_at is not None else None,
        'l0_global_step': int(n_clean + adv_l0_at) if adv_l0_at is not None else None,
        'false_alarm': l0_fired_at is not None,
        'clean_score_mean': float(clean_scores_arr.mean()),
        'clean_score_std': float(clean_scores_arr.std()),
        'adv_score_mean': float(adv_scores.mean()),
        'adv_score_std': float(adv_scores.std()),
        'all_levels': levels,
    }
    os.makedirs('experiments/campaign', exist_ok=True)
    with open('experiments/campaign/results.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved → experiments/campaign/results.json")
    return result


def run_campaign_multitrial(
    n_trials: int = 10,
    seeds: list = None,
    n_clean: int = 50,
    n_adv: int = 100,
    eps: float = EPS_LINF_STANDARD,
    output_path: str = 'experiments/campaign/results_multitrial.json',
):
    """
    Run run_campaign_experiment() over ≥10 independent random seeds and
    report aggregated detection statistics suitable for publication.

    Each trial is a fresh PRISM monitor seeing the same scenario
    (n_clean clean images → n_adv FGSM adversarials) drawn from
    independently-seeded random index subsets of the CIFAR-10 test set.

    Reported statistics
    -------------------
    - detection_rate : fraction of trials where L0 fired during adversarial phase
    - false_alarm_rate: fraction of trials where L0 fired during clean phase
    - latency_mean_steps : mean adversarial steps before detection (detected trials only)
    - latency_std_steps  : std dev (detected trials only)
    - latency_CI_95      : [lo, hi] via t-distribution (n≥2) or [mean, mean] (n=1)
    - P_detect_lt_20     : P(detection < 20 adversarial queries) across trials

    Output format
    -------------
    {
      "n_trials": ...,
      "per_trial": { "0": {...}, ... },    # keyed by trial index
      "aggregate": { ... },
      "metadata": { ... }
    }
    """
    from scipy import stats as _stats

    if seeds is None:
        seeds = list(range(n_trials))

    print(f"\n{'='*65}")
    print(f"Multi-trial campaign detection: n_trials={n_trials}")
    print(f"n_clean={n_clean}, n_adv={n_adv}, eps={eps:.4f} ({eps*255:.1f}/255)")
    print(f"{'='*65}\n")

    per_trial = {}
    detected_latencies = []   # steps before detection (detected trials only)
    false_alarms = 0
    detections = 0

    for trial_idx, seed in enumerate(seeds[:n_trials]):
        print(f"\n--- Trial {trial_idx+1}/{n_trials}  (seed={seed}) ---")
        result = run_campaign_experiment(
            n_clean=n_clean, n_adv=n_adv, eps=eps, seed=seed
        )
        per_trial[str(trial_idx)] = result

        if result['false_alarm']:
            false_alarms += 1

        if result['l0_detected_at_adv_step'] is not None:
            detections += 1
            detected_latencies.append(result['l0_detected_at_adv_step'])

    n_completed = len(seeds[:n_trials])
    detection_rate = detections / max(n_completed, 1)
    false_alarm_rate = false_alarms / max(n_completed, 1)

    # ── Latency statistics (detected trials only) ─────────────────────────────
    latency_stats = {}
    if detected_latencies:
        arr = np.array(detected_latencies)
        lat_mean = float(arr.mean())
        lat_std  = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        n_det = len(arr)
        if n_det >= 2:
            se = lat_std / np.sqrt(n_det)
            t_crit = _stats.t.ppf(0.975, df=n_det - 1)
            ci_lo = lat_mean - t_crit * se
            ci_hi = lat_mean + t_crit * se
        else:
            ci_lo = ci_hi = lat_mean
        p_lt_20 = float(np.mean(arr < 20))
        latency_stats = {
            'mean_steps': round(lat_mean, 2),
            'std_steps':  round(lat_std, 2),
            'CI_95':      [round(ci_lo, 2), round(ci_hi, 2)],
            'median':     round(float(np.median(arr)), 2),
            'min':        int(arr.min()),
            'max':        int(arr.max()),
            'P_lt_20_queries': round(p_lt_20, 4),
            'n_detected': n_det,
        }
    else:
        latency_stats = {
            'mean_steps': None, 'std_steps': None, 'CI_95': None,
            'P_lt_20_queries': 0.0, 'n_detected': 0,
        }

    aggregate = {
        'detection_rate':   round(detection_rate, 4),
        'false_alarm_rate': round(false_alarm_rate, 4),
        'n_trials':         n_completed,
        'n_detected':       detections,
        'n_false_alarms':   false_alarms,
        **latency_stats,
    }

    output = {
        'n_trials':   n_completed,
        'seeds':      seeds[:n_trials],
        'per_trial':  per_trial,
        'aggregate':  aggregate,
        'metadata': {
            'n_clean': n_clean,
            'n_adv':   n_adv,
            'eps':     round(eps, 6),
            'eps_255': round(eps * 255, 2),
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    print(f"\nMulti-trial results saved → {output_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"Detection rate:    {detection_rate:.2%}  ({detections}/{n_completed} trials)")
    print(f"False alarm rate:  {false_alarm_rate:.2%}  ({false_alarms}/{n_completed} trials)")
    if latency_stats.get('mean_steps') is not None:
        print(f"Detection latency: {latency_stats['mean_steps']:.1f}±{latency_stats['std_steps']:.1f} steps  "
              f"95%CI={latency_stats['CI_95']}")
        print(f"P(detect < 20q):   {latency_stats['P_lt_20_queries']:.2%}")
    print(f"{'='*65}")

    return output


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="PRISM campaign detection experiment")
    parser.add_argument('--n-clean',     type=int, default=50)
    parser.add_argument('--n-adv',       type=int, default=100)
    parser.add_argument('--eps',         type=float, default=EPS_LINF_STANDARD)
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--multi-trial', action='store_true',
                        help='Run over 10 seeds and aggregate (paper mode)')
    parser.add_argument('--n-trials',    type=int, default=10)
    parser.add_argument('--output',      default='experiments/campaign/results.json')
    args = parser.parse_args()

    if args.multi_trial:
        multi_out = args.output.replace('.json', '_multitrial.json')
        run_campaign_multitrial(
            n_trials=args.n_trials,
            seeds=list(range(args.n_trials)),
            n_clean=args.n_clean,
            n_adv=args.n_adv,
            eps=args.eps,
            output_path=multi_out,
        )
    else:
        run_campaign_experiment(
            n_clean=args.n_clean,
            n_adv=args.n_adv,
            eps=args.eps,
            seed=args.seed,
        )
