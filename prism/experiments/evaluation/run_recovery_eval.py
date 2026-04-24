"""
TAMSH L3-Recovery Evaluation (P0.5)

When PRISM triggers L3 (high-confidence adversarial), the default action is
to reject — which destroys availability. TAMSH's claim is that topology-aware
routing to a matched expert can recover the *correct* prediction on many
rejected inputs. The standard TPR-based ablation cannot surface this because
TPR does not measure availability.

This script compares three post-L3 strategies on the L3-triggered adversarial
subset, using the model's predicted class (top-1) vs. the CIFAR-10 ground
truth label as the correctness signal. Note: the backbone is ImageNet-1000,
so we use a fixed CIFAR-10 → ImageNet-1000 class mapping via the *clean*
backbone prediction on the original clean image (the "oracle" clean label).
Recovery accuracy = P(adversarial routed through strategy → same argmax as
clean-image argmax).

Strategies:
  - reject:      Availability = 0 (baseline).
  - passthrough: Just run the base model on the adversarial. Availability = 1,
                 accuracy typically low under successful attack.
  - tamsh:       Route adversarial activation through the topology-matched
                 expert, return expert logits. Availability = 1, accuracy
                 depends on how well differentiated the experts are.

Go/no-go (5-seed pool): recovery_accuracy(tamsh) − recovery_accuracy(passthrough) ≥ 15pp.
If miss, TAMSH (C4) is demoted to appendix per plan P0.5.

USAGE
-----
  cd prism/
  python experiments/evaluation/run_recovery_eval.py \
      --strategies reject passthrough tamsh \
      --seeds 42 123 456 789 999
"""
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models import ResNet18_Weights
import numpy as np
import json, os, sys, ssl, certifi, time, argparse
from tqdm import tqdm

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
    )
    from art.estimators.classification import PyTorchClassifier
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False

from src.prism import PRISM
from src.sacd.monitor import NoOpCampaignMonitor
from src.tamsh.experts import TopologyAwareMoE, ExpertSubNetwork
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


def _load_moe(experts_path=None):
    """Rebuild TopologyAwareMoE from experts.pkl (same pattern as run_ablation.py)."""
    if experts_path is None:
        experts_path = PATHS['experts']
    if not os.path.exists(experts_path):
        return None
    data = PRISM._load_pickle(experts_path)
    if isinstance(data, TopologyAwareMoE):
        return data
    if isinstance(data, dict) and 'experts' in data:
        rebuilt = []
        for sd in data['experts']:
            net = ExpertSubNetwork(
                input_dim=data['input_dim'],
                output_dim=data['output_dim'],
                hidden_dim=data.get('hidden_dim', 256),
            )
            net.load_state_dict(sd)
            net.eval()
            rebuilt.append(net)
        return TopologyAwareMoE(
            experts=rebuilt,
            expert_ref_diagrams=data['medoid_diagrams'],
        )
    return None


def _oracle_clean_argmax(model, x_pixel, device):
    """Top-1 argmax of the clean image through the backbone — our 'correct' label."""
    mean_t = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std_t  = torch.tensor(_STD,  device=device).view(1, 3, 1, 1)
    with torch.no_grad():
        logits = model((x_pixel - mean_t) / std_t)
    return int(logits.argmax(1).item())


def run_recovery_eval(
    n_test=1000, attack_name='PGD', strategies=None,
    seed=42, device_str=None, data_root='./data',
    output_path='experiments/evaluation/results_recovery.json',
):
    if not ART_AVAILABLE:
        print("ERROR: ART not installed."); sys.exit(1)

    strategies = strategies or ['reject', 'passthrough', 'tamsh']
    device = torch.device(device_str) if device_str else \
             torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Recovery eval: attack={attack_name}, strategies={strategies}, seed={seed}")

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    # ── Model ──
    model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model = model.to(device).eval()

    # ── MoE (TAMSH) ──
    moe = _load_moe()
    if moe is None and 'tamsh' in strategies:
        print("WARNING: models/experts.pkl not found — dropping 'tamsh' strategy. "
              "Train experts with scripts/train_experts.py first.")
        strategies = [s for s in strategies if s != 'tamsh']

    # ── Dataset (dispatches on DATASET: cifar10 / cifar100) ──
    ds = load_test_dataset(root=data_root, download=True, transform=_PIXEL_TRANSFORM)
    eval_indices = list(range(*EVAL_IDX))
    sample_idx = rng.choice(eval_indices, min(n_test, len(eval_indices)), replace=False)
    imgs_pixel = [ds[int(i)][0] for i in sample_idx]
    X_pixel_np = torch.stack(imgs_pixel).numpy()

    # ── Generate adversarials ──
    eps = EPS_LINF_STANDARD
    norm_model = _NormalizedResNet(model).to(device).eval()
    device_type = 'gpu' if device.type == 'cuda' else 'cpu'
    classifier = PyTorchClassifier(
        model=norm_model, loss=torch.nn.CrossEntropyLoss(),
        input_shape=(3, 224, 224), nb_classes=1000,
        clip_values=(0.0, 1.0), device_type=device_type,
    )
    attacks = {
        'FGSM': lambda: FastGradientMethod(classifier, eps=eps),
        'PGD':  lambda: ProjectedGradientDescent(
            classifier, eps=eps, eps_step=eps / 4, max_iter=40, num_random_init=1),
    }
    attack = attacks[attack_name]()
    print(f"Generating {len(sample_idx)} adversarials ({attack_name})...")
    X_adv_np = attack.generate(X_pixel_np)

    # ── Build PRISM (no monitor interference) ──
    prism = PRISM.from_saved(
        model=model,
        layer_names=LAYER_NAMES,
        calibrator_path=PATHS['calibrator'],
        profile_path=PATHS['reference_profiles'],
        ensemble_path=PATHS['ensemble_scorer'],
        layer_weights=LAYER_WEIGHTS,
        dim_weights=DIM_WEIGHTS,
        campaign_monitor=NoOpCampaignMonitor(),
        moe=moe,
    )

    # ── Phase 1: identify L3-triggered adversarials ──
    print("Identifying L3-triggered adversarials...")
    l3_triggered = []  # list of (j, x_adv_pixel, x_clean_pixel, clean_argmax, adv_diagrams, adv_acts)
    for j, img_pixel in enumerate(tqdm(imgs_pixel, desc="  triage")):
        x_clean_pixel = img_pixel.unsqueeze(0).to(device)
        x_adv_pixel   = torch.tensor(X_adv_np[j]).unsqueeze(0).to(device)
        x_adv_norm    = _NORMALIZE(x_adv_pixel.squeeze(0).cpu()).unsqueeze(0).to(device)

        _, level, meta = prism.defend(x_adv_norm)
        if level in ('L3', 'L3_REJECT'):
            clean_argmax = _oracle_clean_argmax(model, x_clean_pixel, device)
            # Re-extract diagrams + activations for this adv (PRISM stateful internals
            # are consumed by defend()); keep things simple with a fresh pass.
            acts = prism.extractor.extract(x_adv_norm)
            diagrams = {
                L: prism.profiler.compute_diagram(acts[L].squeeze(0).cpu().numpy())
                for L in LAYER_NAMES
            }
            l3_triggered.append({
                'j': j,
                'x_clean_pixel': x_clean_pixel,
                'x_adv_pixel': x_adv_pixel,
                'x_adv_norm': x_adv_norm,
                'clean_argmax': clean_argmax,
                'acts': acts,
                'diagrams': diagrams,
            })

    n_l3 = len(l3_triggered)
    n_total = len(imgs_pixel)
    trigger_rate = n_l3 / max(n_total, 1)
    print(f"L3-triggered: {n_l3}/{n_total} = {trigger_rate:.3f}")

    # Sanity guard for the P0.5 gate: if <10% of adversarials trigger L3, the
    # recovery sample is too small to be representative; if >80%, the detector
    # is pathologically aggressive and the gate measures noise. Either way,
    # flag the run so downstream gate checks can treat the gap with suspicion.
    trigger_rate_ok = 0.10 <= trigger_rate <= 0.80
    if not trigger_rate_ok:
        print(f"  WARN P0.5: L3-trigger rate {trigger_rate:.3f} outside [0.10, 0.80]; "
              "recovery gap may be unreliable.")

    if n_l3 == 0:
        print("No L3 triggers — nothing to recover. Aborting.")
        return {'_meta': {
            'n_l3': 0, 'n_total': n_total,
            'trigger_rate': trigger_rate,
            'trigger_rate_ok': trigger_rate_ok,
        }}

    # ── Phase 2: apply each strategy to the L3 subset ──
    results = {}
    expert_selection_counts = {}
    for strat in strategies:
        correct = 0
        available = 0
        expert_uses = {}
        for item in tqdm(l3_triggered, desc=f"  {strat}"):
            clean_argmax = item['clean_argmax']
            if strat == 'reject':
                # Availability 0; correctness 0 by construction.
                continue
            elif strat == 'passthrough':
                with torch.no_grad():
                    logits = norm_model(item['x_adv_pixel'])
                pred = int(logits.argmax(1).item())
                available += 1
                if pred == clean_argmax:
                    correct += 1
            elif strat == 'tamsh':
                assert moe is not None
                # Use the deepest monitored layer's flat activation as expert input.
                last_layer = LAYER_NAMES[-1]
                a = item['acts'][last_layer]
                if a.dim() > 2:
                    a_flat = a.view(a.size(0), -1)
                else:
                    a_flat = a
                expert_out, expert_idx = moe.forward_through_expert(
                    item['diagrams'][last_layer], a_flat.to(device))
                pred = int(expert_out.argmax(1).item())
                expert_uses[expert_idx] = expert_uses.get(expert_idx, 0) + 1
                available += 1
                if pred == clean_argmax:
                    correct += 1
            else:
                raise ValueError(f"Unknown strategy: {strat}")

        recovery_acc = correct / max(n_l3, 1)
        availability = available / max(n_l3, 1)
        ci = wilson_ci(correct, n_l3)

        results[strat] = {
            'recovery_accuracy': round(recovery_acc, 4),
            'recovery_accuracy_CI_95': [round(ci[0], 4), round(ci[1], 4)],
            'availability': round(availability, 4),
            'n_correct': correct,
            'n_l3': n_l3,
        }
        if strat == 'tamsh':
            expert_selection_counts = {int(k): int(v) for k, v in expert_uses.items()}
            results[strat]['expert_selection_counts'] = expert_selection_counts

        print(f"  {strat:>12}: recovery_acc={recovery_acc:.4f} "
              f"CI[{ci[0]:.4f},{ci[1]:.4f}] availability={availability:.4f}")

    # ── Gates ──
    gap_pp = None
    if 'tamsh' in results and 'passthrough' in results:
        gap = results['tamsh']['recovery_accuracy'] - results['passthrough']['recovery_accuracy']
        gap_pp = round(100.0 * gap, 2)
    gates = {
        'tamsh_minus_passthrough_ge_15pp': (gap_pp is not None and gap_pp >= 15.0),
        'trigger_rate_in_band': trigger_rate_ok,
    }

    results['_gates'] = gates
    results['_meta'] = {
        'dataset': DATASET,
        'attack': attack_name,
        'n_test': n_total,
        'n_l3_triggered': n_l3,
        'l3_trigger_rate': round(trigger_rate, 4),
        'trigger_rate_ok': trigger_rate_ok,
        'strategies': strategies,
        'seed': seed,
        'eps': round(eps, 6),
        'device': str(device),
        'go_no_go': (
            'tamsh recovery_accuracy − passthrough recovery_accuracy ≥ 15pp; '
            'if miss, TAMSH (C4) demoted to appendix per plan P0.5.'
        ),
        'oracle_source': (
            "top-1 argmax of the CLEAN backbone on the original (un-attacked) image — "
            "the backbone is ImageNet-1000 so we use its own clean prediction as the "
            "'correct' label, not the CIFAR-10 ground truth."
        ),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {output_path}")
    print(f"Gates: {gates}")
    if gap_pp is not None:
        print(f"TAMSH − passthrough gap: {gap_pp:+.2f} pp")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TAMSH L3-recovery evaluation (P0.5)")
    parser.add_argument('--config', default=None,
                        help='YAML config path (routes via PRISM_CONFIG env var).')
    parser.add_argument('--n-test', type=int, default=1000)
    parser.add_argument('--attack', default='PGD', choices=['FGSM', 'PGD'])
    parser.add_argument('--strategies', nargs='+',
                        default=['reject', 'passthrough', 'tamsh'])
    parser.add_argument('--seed', type=int, default=42)
    _default_out = os.path.join(
        os.path.dirname(PATHS['clean_scores']).replace('calibration', 'evaluation')
            or 'experiments/evaluation',
        (os.path.basename(PATHS['clean_scores']).replace('clean_scores.npy', '')
         + 'results_recovery.json')
    )
    parser.add_argument('--output', default=_default_out)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    run_recovery_eval(
        n_test=args.n_test,
        attack_name=args.attack,
        strategies=args.strategies,
        seed=args.seed,
        output_path=args.output,
        device_str=args.device,
    )
