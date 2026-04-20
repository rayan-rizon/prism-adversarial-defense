"""
Real BOCPD Campaign Detection — Multi-Attack, Multi-Trial

Replaces synthetic campaign evidence with REAL PRISM inference score streams.
Each trial feeds real clean images → real adversarial images through PRISM
and monitors the anomaly score stream via BOCPD (L0 campaign detector).

Attacks tested: FGSM, PGD, Square — each in independent trial sets.
Each scenario: 50 clean images → 100 adversarial images.

Multi-trial (≥10 seeds per attack) for publishable statistics:
  - Detection rate (fraction of trials where L0 fired during adversarial phase)
  - False alarm rate (fraction where L0 fired during clean phase)
  - Detection latency with 95% CI (steps after campaign onset)

USAGE
-----
  cd prism/
  python experiments/campaign/run_campaign_real.py --n-trials 10
  python experiments/campaign/run_campaign_real.py --n-trials 10 --attacks FGSM PGD Square

OUTPUT: experiments/campaign/results_real_multitrial.json
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import json, argparse, time, ssl, certifi
from tqdm import tqdm

os.environ.setdefault('SSL_CERT_FILE', certifi.where())
os.environ.setdefault('REQUESTS_CA_BUNDLE', certifi.where())
ssl._create_default_https_context = ssl.create_default_context

from torchvision.models import ResNet18_Weights

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
    print("ERROR: ART not installed.")
    sys.exit(1)

from src.prism import PRISM
from src.sacd.monitor import CampaignMonitor
from src.config import (
    LAYER_NAMES, LAYER_WEIGHTS, DIM_WEIGHTS,
    IMAGENET_MEAN, IMAGENET_STD, EPS_LINF_STANDARD,
)

_MEAN = IMAGENET_MEAN
_STD  = IMAGENET_STD
_PIXEL = T.Compose([T.Resize(224), T.ToTensor()])
_NORM  = T.Normalize(_MEAN, _STD)


class _NormalizedResNet(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.register_buffer('mean', torch.tensor(_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor(_STD).view(1, 3, 1, 1))
    def forward(self, x):
        return self.backbone((x - self.mean) / self.std)


def run_single_trial(
    model, wrapped, ds, attack, n_clean, n_adv, seed, device,
    bocpd_params,
):
    """
    Run one campaign trial: n_clean clean queries → n_adv adversarial queries.
    Returns dict with detection results and full score stream.
    """
    rng = np.random.RandomState(seed)

    # Fresh PRISM with active campaign monitor
    prism = PRISM.from_saved(
        model=model,
        layer_names=LAYER_NAMES,
        layer_weights=LAYER_WEIGHTS,
        dim_weights=DIM_WEIGHTS,
        calibrator_path='models/calibrator.pkl',
        profile_path='models/reference_profiles.pkl',
        ensemble_path='models/ensemble_scorer.pkl',
        campaign_monitor=CampaignMonitor(**bocpd_params),
    )

    indices = rng.choice(len(ds), n_clean + n_adv, replace=False)
    clean_idx = indices[:n_clean]
    adv_idx = indices[n_clean:]

    scores = []
    levels = []
    l0_states = []
    false_alarm_step = None
    detect_step = None

    # Phase 1: Clean
    for t, idx in enumerate(clean_idx):
        pixel_img, _ = ds[int(idx)]
        norm_img = _NORM(pixel_img).unsqueeze(0).to(device)
        _, level, meta = prism.defend(norm_img)
        s = meta.get('anomaly_score', 0.0)
        scores.append(float(s))
        levels.append(level)
        l0 = meta.get('l0_state', {}).get('l0_active', False)
        l0_states.append(l0)
        if l0 and false_alarm_step is None:
            false_alarm_step = t

    # Phase 2: Adversarial
    # Pre-generate adversarials for this trial
    adv_imgs_pixel = []
    for idx in adv_idx:
        pixel_img, _ = ds[int(idx)]
        adv_imgs_pixel.append(pixel_img)

    X_adv_batch = torch.stack(adv_imgs_pixel).numpy()
    try:
        X_adv_np = attack.generate(X_adv_batch)
    except Exception:
        # Per-sample fallback
        X_adv_np = np.zeros_like(X_adv_batch)
        for i, x_np in enumerate(X_adv_batch):
            try:
                X_adv_np[i] = attack.generate(x_np[np.newaxis])[0]
            except Exception:
                X_adv_np[i] = x_np

    for t in range(n_adv):
        x_adv_norm = _NORM(torch.tensor(X_adv_np[t])).unsqueeze(0).to(device)
        _, level, meta = prism.defend(x_adv_norm)
        s = meta.get('anomaly_score', 0.0)
        scores.append(float(s))
        levels.append(level)
        l0 = meta.get('l0_state', {}).get('l0_active', False)
        l0_states.append(l0)
        if l0 and detect_step is None:
            detect_step = t

    clean_scores = np.array(scores[:n_clean])
    adv_scores = np.array(scores[n_clean:])

    return {
        'false_alarm': false_alarm_step is not None,
        'false_alarm_step': false_alarm_step,
        'detected': detect_step is not None,
        'detect_step': int(detect_step) if detect_step is not None else None,
        'detect_global_step': int(n_clean + detect_step) if detect_step is not None else None,
        'clean_score_mean': round(float(clean_scores.mean()), 4),
        'clean_score_std': round(float(clean_scores.std()), 4),
        'adv_score_mean': round(float(adv_scores.mean()), 4),
        'adv_score_std': round(float(adv_scores.std()), 4),
        'seed': seed,
        'scores': scores,  # full stream for offline analysis
    }


def run_campaign_real(
    n_trials=10,
    attacks_to_run=None,
    n_clean=50,
    n_adv=100,
    output_path='experiments/campaign/results_real_multitrial.json',
    device_str=None,
    data_root='./data',
):
    from scipy import stats as _stats

    attacks_to_run = attacks_to_run or ['FGSM', 'PGD', 'Square']
    eps = EPS_LINF_STANDARD

    device = torch.device(device_str) if device_str else \
             torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"{'='*65}")
    print(f"Real BOCPD Campaign Detection — Multi-Attack, Multi-Trial")
    print(f"{'='*65}")
    print(f"  Device: {device}")
    print(f"  Attacks: {attacks_to_run}")
    print(f"  n_trials={n_trials}, n_clean={n_clean}, n_adv={n_adv}")
    print(f"  eps={eps:.4f} ({eps*255:.1f}/255)")
    print()

    # ── Model ──
    backbone = torchvision.models.resnet18(
        weights=ResNet18_Weights.IMAGENET1K_V1
    ).to(device).eval()
    wrapped = _NormalizedResNet(backbone).to(device).eval()

    # ── ART Classifier ──
    device_type = 'gpu' if device.type == 'cuda' else 'cpu'
    classifier = PyTorchClassifier(
        model=wrapped,
        loss=torch.nn.CrossEntropyLoss(),
        input_shape=(3, 224, 224),
        nb_classes=1000,
        clip_values=(0.0, 1.0),
        device_type=device_type,
    )

    attack_factories = {
        'FGSM': lambda: FastGradientMethod(classifier, eps=eps),
        'PGD': lambda: ProjectedGradientDescent(
            classifier, eps=eps, eps_step=eps / 4, max_iter=40, num_random_init=1),
        'Square': lambda: SquareAttack(
            classifier, eps=eps, max_iter=5000, nb_restarts=1),
    }

    ds = torchvision.datasets.CIFAR10(
        data_root, train=False, download=True, transform=_PIXEL
    )

    # BOCPD parameters calibrated for CIFAR-10 clean score distribution
    bocpd_params = dict(
        mu0=7.0, kappa0=5.0, alpha0=3.0, beta0=15.0,
        hazard_rate=1/30, alert_run_length=10, alert_run_prob=0.60,
        warmup_steps=35, l0_factor=0.8, cooldown_steps=30,
    )

    seeds = list(range(n_trials))
    t_global_start = time.time()

    all_results = {}

    for attack_name in attacks_to_run:
        if attack_name not in attack_factories:
            print(f"Unknown attack: {attack_name}. Skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"Campaign attack: {attack_name}")
        print(f"{'='*60}")

        attack = attack_factories[attack_name]()

        per_trial = {}
        detected_latencies = []
        false_alarms = 0
        detections = 0

        for trial_idx, seed in enumerate(seeds):
            print(f"  Trial {trial_idx+1}/{n_trials}  (seed={seed})", end="")

            result = run_single_trial(
                model=backbone, wrapped=wrapped, ds=ds, attack=attack,
                n_clean=n_clean, n_adv=n_adv, seed=seed, device=device,
                bocpd_params=bocpd_params,
            )

            # Don't store full score stream in per_trial (too large)
            trial_summary = {k: v for k, v in result.items() if k != 'scores'}
            # Store truncated scores for verification
            trial_summary['scores_last_20_clean'] = result['scores'][max(0, n_clean-20):n_clean]
            trial_summary['scores_first_20_adv'] = result['scores'][n_clean:n_clean+20]
            per_trial[str(trial_idx)] = trial_summary

            if result['false_alarm']:
                false_alarms += 1
                print(f"  FA@{result['false_alarm_step']}", end="")

            if result['detected']:
                detections += 1
                detected_latencies.append(result['detect_step'])
                print(f"  ✓ L0@step{result['detect_step']}", end="")
            else:
                print(f"  ✗ no detection", end="")
            print()

        n_completed = len(seeds)
        detection_rate = detections / max(n_completed, 1)
        false_alarm_rate = false_alarms / max(n_completed, 1)

        # Latency statistics
        latency_stats = {}
        if detected_latencies:
            arr = np.array(detected_latencies)
            lat_mean = float(arr.mean())
            lat_std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
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
                'std_steps': round(lat_std, 2),
                'CI_95': [round(ci_lo, 2), round(ci_hi, 2)],
                'median': round(float(np.median(arr)), 2),
                'min': int(arr.min()),
                'max': int(arr.max()),
                'P_lt_20_queries': round(p_lt_20, 4),
                'n_detected': n_det,
            }
        else:
            latency_stats = {
                'mean_steps': None, 'std_steps': None, 'CI_95': None,
                'P_lt_20_queries': 0.0, 'n_detected': 0,
            }

        all_results[attack_name] = {
            'detection_rate': round(detection_rate, 4),
            'false_alarm_rate': round(false_alarm_rate, 4),
            'n_trials': n_completed,
            'n_detected': detections,
            'n_false_alarms': false_alarms,
            'latency': latency_stats,
            'per_trial': per_trial,
        }

        print(f"\n  Detection rate:   {detection_rate:.0%} ({detections}/{n_completed})")
        print(f"  False alarm rate: {false_alarm_rate:.0%} ({false_alarms}/{n_completed})")
        if latency_stats.get('mean_steps') is not None:
            print(f"  Latency: {latency_stats['mean_steps']:.1f}±{latency_stats['std_steps']:.1f} "
                  f"steps  CI={latency_stats['CI_95']}")
            print(f"  P(detect < 20q): {latency_stats['P_lt_20_queries']:.0%}")

    elapsed = time.time() - t_global_start

    # ── Global summary ──
    print(f"\n{'='*70}")
    print(f"{'Attack':>10} {'Detect%':>10} {'FA%':>8} {'Latency':>12} {'P<20q':>8}")
    print(f"{'-'*70}")
    for atk in attacks_to_run:
        if atk in all_results:
            r = all_results[atk]
            lat = r['latency']
            lat_str = f"{lat['mean_steps']:.1f}±{lat['std_steps']:.1f}" if lat['mean_steps'] is not None else "N/A"
            p20 = f"{lat['P_lt_20_queries']:.0%}" if lat['P_lt_20_queries'] is not None else "N/A"
            print(f"{atk:>10} {r['detection_rate']:>10.0%} {r['false_alarm_rate']:>8.0%} "
                  f"{lat_str:>12} {p20:>8}")

    output = {
        'per_attack': all_results,
        'metadata': {
            'n_trials': n_trials,
            'n_clean': n_clean,
            'n_adv': n_adv,
            'eps': round(eps, 6),
            'eps_255': round(eps * 255, 2),
            'attacks': attacks_to_run,
            'bocpd_params': bocpd_params,
            'device': str(device),
            'elapsed_s': round(elapsed, 1),
            'evidence_type': 'real PRISM inference (not synthetic scores)',
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved → {output_path}")
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Real BOCPD campaign detection (multi-attack, multi-trial)")
    parser.add_argument('--n-trials', type=int, default=10)
    parser.add_argument('--n-clean', type=int, default=50)
    parser.add_argument('--n-adv', type=int, default=100)
    parser.add_argument('--attacks', nargs='+', default=['FGSM', 'PGD', 'Square'])
    parser.add_argument('--output',
                        default='experiments/campaign/results_real_multitrial.json')
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    run_campaign_real(
        n_trials=args.n_trials,
        attacks_to_run=args.attacks,
        n_clean=args.n_clean,
        n_adv=args.n_adv,
        output_path=args.output,
        device_str=args.device,
    )
