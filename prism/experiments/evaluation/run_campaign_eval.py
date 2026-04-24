"""
SACD Campaign-Stream Evaluation (P0.4)

The IID per-query evaluation in run_evaluation_full.py gives SACD (L0 BOCPD
campaign monitor) nothing to aggregate — every query is independent, so the
temporal component trivially contributes zero. This script tests SACD on
sequential streams where the monitor has something to detect.

Claim to support:
    "Under sustained attack campaigns, SACD detects the campaign in ≤30
     queries and tightens thresholds, reducing effective ASR vs. per-query
     detection, without false-alarming on clean streams."

Scenarios:
  - clean_only:    500 clean queries. Measures false-alarm rate of L0.
  - sustained:     k=20 clean warmup, then 200 queries at mixing ratio ρ
                   (frac adversarial). ρ ∈ {0.5, 0.8, 1.0}.
  - burst:         10 clean, 10 adv, 10 clean, 10 adv, ... Probes reset.
  - low_rate:      ρ=0.1 mixed stream (weakly adversarial). Tests BOCPD
                   prior against naive thresholding.

Metrics per scenario:
  - Time-to-detection (queries from first adversarial to first L0 trigger)
  - Post-detection effective ASR (fraction of adv queries that PASS after L0)
  - False-alarm rate on clean-only stream (L0 must stay inactive)
  - L0-on vs L0-off ASR delta on sustained streams (gate: ≥ 10pp gap)

Go/no-go (5-seed pool):
  - sustained ρ=1.0: ASR(L0-off) − ASR(L0-on) ≥ 0.10  (10pp gap)
  - clean_only: fraction of steps with l0_active ≤ 0.01  (≤ 1% false-alarm)
If either gate misses, SACD (C3) is demoted to appendix per the plan.

USAGE
-----
  cd prism/
  python experiments/evaluation/run_campaign_eval.py \
      --scenarios clean_only sustained burst low_rate \
      --seeds 42 123 456 789 999
"""
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models import ResNet18_Weights
import numpy as np
import json, os, sys, ssl, certifi, time, argparse
from copy import deepcopy
from tqdm import tqdm

os.environ.setdefault('SSL_CERT_FILE', certifi.where())
os.environ.setdefault('REQUESTS_CA_BUNDLE', certifi.where())
ssl._create_default_https_context = ssl.create_default_context

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Route --config CLI flag to PRISM_CONFIG env var BEFORE importing src.config.
from src import bootstrap  # noqa: F401

try:
    from art.attacks.evasion import ProjectedGradientDescent
    from art.estimators.classification import PyTorchClassifier
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False

from src.prism import PRISM
from src.sacd.monitor import CampaignMonitor, NoOpCampaignMonitor
from src.config import (
    LAYER_NAMES, LAYER_WEIGHTS, DIM_WEIGHTS,
    IMAGENET_MEAN, IMAGENET_STD, EPS_LINF_STANDARD,
    EVAL_IDX, DATASET, PATHS,
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
# Scenario stream builders: return list of (kind, img_idx) pairs
#   kind ∈ {'clean', 'adv'}, img_idx = index into sample pool
# ═════════════════════════════════════════════════════════════════════════════

def build_stream(scenario: str, n_clean_pool: int, n_adv_pool: int, rng: np.random.RandomState):
    if scenario == 'clean_only':
        return [('clean', int(rng.randint(n_clean_pool))) for _ in range(500)]

    if scenario == 'sustained':
        # Sub-scenarios ρ ∈ {0.5, 0.8, 1.0} handled by caller via scenario name
        raise ValueError("Use 'sustained_rho050', 'sustained_rho080', 'sustained_rho100'")

    if scenario.startswith('sustained_rho'):
        rho = int(scenario.split('rho')[-1]) / 100.0
        warmup = [('clean', int(rng.randint(n_clean_pool))) for _ in range(20)]
        active = []
        for _ in range(200):
            if rng.rand() < rho:
                active.append(('adv', int(rng.randint(n_adv_pool))))
            else:
                active.append(('clean', int(rng.randint(n_clean_pool))))
        return warmup + active

    if scenario == 'burst':
        # 10 clean, 10 adv, 10 clean, 10 adv, ... for 16 blocks (160 queries)
        stream = []
        for block in range(16):
            kind = 'clean' if block % 2 == 0 else 'adv'
            pool = n_clean_pool if kind == 'clean' else n_adv_pool
            for _ in range(10):
                stream.append((kind, int(rng.randint(pool))))
        return stream

    if scenario == 'low_rate':
        rho = 0.1
        return [
            ('adv' if rng.rand() < rho else 'clean',
             int(rng.randint(n_adv_pool if rng.rand() < rho else n_clean_pool)))
            for _ in range(500)
        ]

    raise ValueError(f"Unknown scenario: {scenario}")


def run_one_scenario(scenario, prism_factory, clean_imgs, adv_imgs, device, seed,
                     thresholds_path=None):
    """
    Run scenario twice: once with CampaignMonitor (L0 on) and once with
    NoOpCampaignMonitor (L0 off, per-query thresholding only). Returns a dict
    with per-step traces and aggregate metrics for each mode.

    Args:
        thresholds_path: Optional pkl with calibrated L0 thresholds (see
            scripts/calibrate_l0_thresholds.py). Overlaid onto CampaignMonitor
            defaults when present.
    """
    rng = np.random.RandomState(seed)
    stream = build_stream(scenario, len(clean_imgs), len(adv_imgs), rng)

    n_clean_pool = len(clean_imgs)
    n_adv_pool   = len(adv_imgs)

    results_per_mode = {}
    for mode in ('l0_on', 'l0_off'):
        monitor = (
            CampaignMonitor(thresholds_path=thresholds_path)
            if mode == 'l0_on' else NoOpCampaignMonitor()
        )
        prism = prism_factory(monitor)

        steps = []
        first_adv_step = None
        first_l0_trigger_step = None
        n_clean_seen = 0; n_clean_alerted = 0  # FPR denominator for this stream
        n_adv_seen = 0;   n_adv_passed = 0     # ASR numerator

        for step_i, (kind, idx) in enumerate(stream):
            img = clean_imgs[idx] if kind == 'clean' else adv_imgs[idx]
            x = _NORMALIZE(img).unsqueeze(0).to(device)
            _, level, meta = prism.defend(x)

            l0_active = bool(meta.get('l0_state', {}).get('l0_active', False))
            if l0_active and first_l0_trigger_step is None:
                first_l0_trigger_step = step_i
            if kind == 'adv' and first_adv_step is None:
                first_adv_step = step_i

            if kind == 'clean':
                n_clean_seen += 1
                if level != 'PASS':
                    n_clean_alerted += 1
            else:
                n_adv_seen += 1
                if level == 'PASS':
                    n_adv_passed += 1

            steps.append({
                'i': step_i, 'kind': kind, 'level': level,
                'score': float(meta.get('anomaly_score', 0.0)),
                'l0_active': l0_active,
            })

        asr = n_adv_passed / max(n_adv_seen, 1)
        fpr = n_clean_alerted / max(n_clean_seen, 1)
        l0_active_frac = sum(1 for s in steps if s['l0_active']) / max(len(steps), 1)

        time_to_detect = None
        if first_adv_step is not None and first_l0_trigger_step is not None \
                and first_l0_trigger_step >= first_adv_step:
            time_to_detect = first_l0_trigger_step - first_adv_step

        results_per_mode[mode] = {
            'ASR': round(asr, 4),
            'FPR_clean_steps': round(fpr, 4),
            'l0_active_fraction': round(l0_active_frac, 4),
            'first_adv_step': first_adv_step,
            'first_l0_trigger_step': first_l0_trigger_step,
            'time_to_detect_queries': time_to_detect,
            'n_clean': n_clean_seen,
            'n_adv': n_adv_seen,
            'n_adv_passed': n_adv_passed,
            'n_clean_alerted': n_clean_alerted,
            'stream_length': len(stream),
        }

    # Gap metric (L0 contribution): ASR(l0_off) - ASR(l0_on)
    asr_gap = results_per_mode['l0_off']['ASR'] - results_per_mode['l0_on']['ASR']
    results_per_mode['asr_gap_pp'] = round(100.0 * asr_gap, 2)
    return results_per_mode


def run_campaign_eval(
    n_test=1000, scenarios=None, seed=42, device_str=None,
    data_root='./data',
    output_path='experiments/evaluation/results_campaign.json',
    n_clean_pool=200, n_adv_pool=200,
    thresholds_path=None,
):
    if not ART_AVAILABLE:
        print("ERROR: ART not installed."); sys.exit(1)

    scenarios = scenarios or [
        'clean_only',
        'sustained_rho050', 'sustained_rho080', 'sustained_rho100',
        'burst', 'low_rate',
    ]
    device = torch.device(device_str) if device_str else \
             torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Scenarios: {scenarios}; seed={seed}")

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    # ── Model ──
    model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model = model.to(device).eval()

    # ── Dataset (dispatches on DATASET: cifar10 / cifar100) ──
    ds = load_test_dataset(root=data_root, download=True, transform=_PIXEL_TRANSFORM)
    eval_indices = list(range(*EVAL_IDX))
    pool_idx = rng.choice(eval_indices, min(n_clean_pool, len(eval_indices)), replace=False)

    # Load clean pool
    clean_imgs = [ds[int(i)][0] for i in pool_idx]

    # Generate adversarial pool (PGD-40) on a disjoint subset
    remaining = list(set(eval_indices) - set(int(i) for i in pool_idx))
    adv_src_idx = rng.choice(remaining, min(n_adv_pool, len(remaining)), replace=False)
    adv_src_imgs = [ds[int(i)][0] for i in adv_src_idx]

    print(f"\nGenerating adversarial pool (PGD-40) on {len(adv_src_imgs)} images...")
    norm_model = _NormalizedResNet(model).to(device).eval()
    device_type = 'gpu' if device.type == 'cuda' else 'cpu'
    classifier = PyTorchClassifier(
        model=norm_model, loss=torch.nn.CrossEntropyLoss(),
        input_shape=(3, 224, 224), nb_classes=1000,
        clip_values=(0.0, 1.0), device_type=device_type,
    )
    eps = EPS_LINF_STANDARD
    pgd = ProjectedGradientDescent(
        classifier, eps=eps, eps_step=eps / 4, max_iter=40, num_random_init=1)
    X_src = torch.stack(adv_src_imgs).numpy()
    X_adv_np = pgd.generate(X_src)
    adv_imgs = [torch.tensor(X_adv_np[i]) for i in range(len(adv_src_imgs))]

    # ── PRISM factory ──
    def prism_factory(monitor):
        return PRISM.from_saved(
            model=model,
            layer_names=LAYER_NAMES,
            calibrator_path=PATHS['calibrator'],
            profile_path=PATHS['reference_profiles'],
            ensemble_path=PATHS['ensemble_scorer'],
            layer_weights=LAYER_WEIGHTS,
            dim_weights=DIM_WEIGHTS,
            campaign_monitor=monitor,
        )

    # ── Resolve L0 thresholds pkl (auto-discover if caller didn't pass one) ──
    default_l0_pkl = os.path.join(
        os.path.dirname(PATHS['calibrator']), 'l0_thresholds.pkl'
    )
    if thresholds_path is None and os.path.exists(default_l0_pkl):
        thresholds_path = default_l0_pkl
        print(f"Using calibrated L0 thresholds from {thresholds_path}")
    elif thresholds_path is None:
        print(f"No calibrated L0 thresholds found at {default_l0_pkl}; "
              f"using CampaignMonitor defaults.")

    # ── Run scenarios ──
    results = {}
    t_start = time.time()
    for scen in scenarios:
        print(f"\n{'='*60}\nScenario: {scen}\n{'='*60}")
        r = run_one_scenario(
            scen, prism_factory, clean_imgs, adv_imgs, device, seed,
            thresholds_path=thresholds_path,
        )
        results[scen] = r
        print(f"  L0-off  ASR={r['l0_off']['ASR']:.4f} FPR={r['l0_off']['FPR_clean_steps']:.4f}")
        print(f"  L0-on   ASR={r['l0_on']['ASR']:.4f} FPR={r['l0_on']['FPR_clean_steps']:.4f} "
              f"(l0_active_frac={r['l0_on']['l0_active_fraction']:.3f})")
        print(f"  ASR gap (L0-off − L0-on): {r['asr_gap_pp']:+.2f} pp")
        if r['l0_on']['time_to_detect_queries'] is not None:
            print(f"  Time-to-detect: {r['l0_on']['time_to_detect_queries']} queries")

    elapsed = time.time() - t_start

    # ── Gate summary ──
    gates = {}
    if 'sustained_rho100' in results:
        gates['sustained_rho100_asr_gap_ge_10pp'] = (
            results['sustained_rho100']['asr_gap_pp'] >= 10.0
        )
    if 'clean_only' in results:
        gates['clean_only_false_alarm_le_1pct'] = (
            results['clean_only']['l0_on']['l0_active_fraction'] <= 0.01
        )
    results['_gates'] = gates
    results['_meta'] = {
        'dataset': DATASET,
        'n_clean_pool': len(clean_imgs),
        'n_adv_pool': len(adv_imgs),
        'seed': seed,
        'device': str(device),
        'scenarios': scenarios,
        'eps': round(eps, 6),
        'pgd_steps': 40,
        'elapsed_s': round(elapsed, 1),
        'l0_thresholds_path': thresholds_path,
        'go_no_go': (
            'sustained_rho100 asr_gap ≥ 10pp AND clean_only l0_active_fraction ≤ 1%; '
            'if either gate fails, SACD (C3) is demoted to appendix per plan P0.4.'
        ),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {output_path}")
    print(f"Gates: {gates}")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SACD campaign-stream evaluation (P0.4)")
    parser.add_argument('--config', default=None,
                        help='YAML config path (routes via PRISM_CONFIG env var).')
    parser.add_argument('--scenarios', nargs='+', default=None,
                        help='Scenarios to run. Default: all six.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-clean-pool', type=int, default=200)
    parser.add_argument('--n-adv-pool', type=int, default=200)
    _default_out = os.path.join(
        os.path.dirname(PATHS['clean_scores']).replace('calibration', 'evaluation')
            or 'experiments/evaluation',
        (os.path.basename(PATHS['clean_scores']).replace('clean_scores.npy', '')
         + 'results_campaign.json')
    )
    parser.add_argument('--output', default=_default_out)
    parser.add_argument('--device', default=None)
    parser.add_argument('--thresholds-path', default=None,
                        help='Path to l0_thresholds.pkl produced by '
                             'scripts/calibrate_l0_thresholds.py. If omitted, '
                             'auto-discovers models/l0_thresholds.pkl and falls '
                             'back to CampaignMonitor defaults.')
    args = parser.parse_args()

    run_campaign_eval(
        scenarios=args.scenarios,
        seed=args.seed,
        device_str=args.device,
        output_path=args.output,
        n_clean_pool=args.n_clean_pool,
        n_adv_pool=args.n_adv_pool,
        thresholds_path=args.thresholds_path,
    )
