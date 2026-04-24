"""
Adaptive PGD Evaluation — BPDA-style attack targeting PRISM's detection

Implements an adaptive adversary (Athalye et al. 2018, BPDA) that modifies
PGD's loss to jointly optimise:
  L_total = -CE(f(x'), y) + λ * ||a(x') - a_ref||_2 / D

where:
  - CE is standard cross-entropy (misclassification objective)
  - a(x') are intermediate activations at PRISM-monitored layers
  - a_ref are activations from the clean input x
  - D normalises by activation dimension so λ is scale-invariant
  - λ ∈ {0.0, 0.5, 1.0, 2.0, 5.0, 10.0} sweeps from standard PGD to full evasion

The activation-matching term forces adversarial activations to stay close to
clean activations, directly attacking the topological profiling in TAMM.

P1.4 additions:
  - λ sweep extended to {0, 0.5, 1, 2, 5, 10} (was 0,0.5,1,2,5)
  - --pgd-restarts: random restarts per image, keep worst adversarial
  - --eot-samples: EOT (Athalye 2018) gradient averaging. PRISM's detector
      uses a deterministic hash-based subsample so EOT *should* be a no-op;
      --eot-samples>1 VERIFIES this rather than assumes it.
  - --through-scorer: add DCT-energy term targeting the scorer's 37th feature
      (a lightweight stand-in for a full APGD-CE-through-scorer attack).

USAGE
-----
  cd prism/
  python experiments/evaluation/run_adaptive_pgd.py --n-test 500 --pgd-steps 40
  python experiments/evaluation/run_adaptive_pgd.py --n-test 500 --lambdas 0.0 1.0 5.0
  # Strong variant: 100 steps × 10 restarts, λ sweep incl. 10, through-scorer loss
  python experiments/evaluation/run_adaptive_pgd.py --n-test 1000 --pgd-steps 100 \
      --pgd-restarts 10 --through-scorer \
      --lambdas 0 0.5 1 2 5 10

EVAL SPLIT: CIFAR-10 test indices 8000-9999 (same as run_evaluation_full.py)
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
from src.prism import PRISM
from src.sacd.monitor import NoOpCampaignMonitor
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


def per_tier_fpr(clean_levels, n_clean):
    fp_l1 = n_clean - clean_levels.get('PASS', 0)
    fp_l2 = clean_levels.get('L2', 0) + clean_levels.get('L3', 0) + clean_levels.get('L3_REJECT', 0)
    fp_l3 = clean_levels.get('L3', 0) + clean_levels.get('L3_REJECT', 0)
    return {
        'FPR_L1_plus': round(fp_l1 / max(n_clean, 1), 4),
        'FPR_L2_plus': round(fp_l2 / max(n_clean, 1), 4),
        'FPR_L3_plus': round(fp_l3 / max(n_clean, 1), 4),
        'target_L1': 0.10, 'target_L2': 0.03, 'target_L3': 0.005,
        'pass_L1': fp_l1 / max(n_clean, 1) <= 0.10,
        'pass_L2': fp_l2 / max(n_clean, 1) <= 0.03,
        'pass_L3': fp_l3 / max(n_clean, 1) <= 0.005,
    }


def _hf_energy_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Differentiable proxy for the DCT high-frequency energy feature.

    Computes the log sum of squared high-frequency FFT magnitudes per channel,
    masking the low-frequency quadrant (top-left H/4 × W/4 region).
    This approximates compute_dct_energy() from persistence_stats.py but is
    fully differentiable via torch.fft.rfft2.

    Args:
        x: (1, C, H, W) tensor in [0, 1].
    Returns:
        Scalar tensor (differentiable).
    """
    H, W = x.shape[-2], x.shape[-1]
    fft = torch.fft.rfft2(x)          # (1, C, H, W//2+1) complex
    mag = fft.abs() ** 2              # magnitude squared
    # zero out low-frequency quadrant
    mag = mag.clone()
    mag[:, :, : H // 4, : W // 4] = 0.0
    return torch.log(mag.sum() + 1e-8)


def adaptive_pgd_attack(
    model, x_pixel, eps, steps, step_size, lam, layer_names, device,
    through_scorer: bool = False,
    eot_samples: int = 1,
):
    """
    Generate one adaptive PGD adversarial example.

    Loss = -CE(f(x'), y_pred)
           + λ * Σ_layer ||a_layer(x') - a_layer(x)||₂ / D_layer   [activation match]
           + 0.5 * |HF_energy(x') - HF_energy(x)|                  [DCT match, through_scorer only]

    The activation-matching term directly targets PRISM's TAMM module.
    The DCT-energy matching term targets the 37th feature used by the ensemble
    scorer and is active only when `through_scorer=True`.  Both terms force the
    adversary to erase the signals PRISM monitors, providing a stronger adaptive
    attack than activation-only evasion.

    Args:
        model: Backbone model (un-normalised input, ImageNet output).
        x_pixel: (1, 3, H, W) pixel-space tensor in [0, 1].
        eps: L∞ perturbation budget.
        steps: Number of PGD iterations.
        step_size: Per-step L∞ step size.
        lam: Weight of activation-matching loss (0 = standard PGD).
        layer_names: Layers to match activations on.
        device: torch device.
        through_scorer: If True, also minimise the DCT high-frequency energy
                        difference (proxy for the ensemble scorer's 37th feature).
    Returns:
        x_adv: (1, 3, H, W) adversarial example in [0, 1].
    """
    mean_t = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std_t  = torch.tensor(_STD,  device=device).view(1, 3, 1, 1)
    ce_loss = torch.nn.CrossEntropyLoss()

    x = x_pixel.clone().to(device)

    # Pre-compute clean DCT energy reference (constant, no grad needed)
    with torch.no_grad():
        clean_hf_energy = _hf_energy_torch(x).detach() if through_scorer else None

    # Get clean activations and predicted label (no grad needed)
    hooks = {}
    clean_acts = {}
    handles = []

    def make_hook(name, target_dict):
        def hook_fn(module, inp, out):
            target_dict[name] = out
        return hook_fn

    module_dict = dict(model.named_modules())
    for name in layer_names:
        h = module_dict[name].register_forward_hook(make_hook(name, clean_acts))
        handles.append(h)

    with torch.no_grad():
        x_norm = (x - mean_t) / std_t
        logits_clean = model(x_norm)
        y_pred = logits_clean.argmax(dim=1)

    # Detach clean activations
    clean_acts_detached = {k: v.detach().clone() for k, v in clean_acts.items()}
    for h in handles:
        h.remove()

    # Initialise adversarial with uniform random perturbation
    delta = torch.zeros_like(x, requires_grad=True)
    delta.data.uniform_(-eps, eps)
    delta.data = torch.clamp(x + delta.data, 0.0, 1.0) - x

    for step_i in range(steps):
        # EOT (Athalye 2018): average gradient over eot_samples stochastic
        # forward passes. Our detector is deterministic so eot_samples>1 is
        # a verification that EOT is a no-op; still implement it correctly.
        grad_accum = torch.zeros_like(delta)
        for _ in range(max(eot_samples, 1)):
            adv_acts = {}
            handles2 = []
            for name in layer_names:
                h = module_dict[name].register_forward_hook(make_hook(name, adv_acts))
                handles2.append(h)

            x_adv = x + delta
            x_adv_norm = (x_adv - mean_t) / std_t
            logits = model(x_adv_norm)

            # Misclassification loss (maximise = negate CE)
            loss_ce = -ce_loss(logits, y_pred)

            # Activation matching loss
            loss_act = torch.tensor(0.0, device=device)
            if lam > 0:
                for name in layer_names:
                    a_adv = adv_acts[name]
                    a_clean = clean_acts_detached[name]
                    D = float(a_adv.numel())
                    loss_act = loss_act + torch.norm(a_adv - a_clean, p=2) / max(D, 1.0)

            # DCT high-frequency energy matching (targets scorer's 37th feature)
            loss_dct = torch.tensor(0.0, device=device)
            if through_scorer and clean_hf_energy is not None:
                adv_hf_energy = _hf_energy_torch(x_adv)
                loss_dct = (adv_hf_energy - clean_hf_energy).abs()

            loss_total = loss_ce + lam * loss_act + 0.5 * loss_dct
            loss_total.backward()

            for h in handles2:
                h.remove()

            grad_accum = grad_accum + delta.grad.detach()
            delta.grad = None

        grad = grad_accum / max(eot_samples, 1)

        # PGD step (L∞)
        delta.data = delta.data - step_size * grad.sign()
        delta.data = torch.clamp(delta.data, -eps, eps)
        delta.data = torch.clamp(x + delta.data, 0.0, 1.0) - x

    return (x + delta.detach()).clamp(0.0, 1.0)


def adaptive_pgd_attack_with_restarts(
    model, x_pixel, eps, steps, step_size, lam, layer_names, device,
    prism, mean_t, std_t,
    through_scorer: bool = False,
    eot_samples: int = 1,
    num_restarts: int = 1,
):
    """
    Run adaptive PGD with `num_restarts` random initialisations and keep the
    most-evasive adversarial — i.e. the one that (a) evades PRISM if any does,
    or (b) has the lowest PRISM score if all are detected.

    This matches Athalye/Carlini's best-practice for detector evaluation: an
    adversary gets multiple attempts per image, and the defender must survive
    the worst of them.
    """
    best_x_adv = None
    best_evaded = False
    best_score = float('inf')  # lower PRISM score = more evasive

    for _restart in range(max(num_restarts, 1)):
        x_adv_pixel = adaptive_pgd_attack(
            model, x_pixel, eps, steps, step_size, lam, layer_names, device,
            through_scorer=through_scorer,
            eot_samples=eot_samples,
        )
        x_adv_norm = ((x_adv_pixel - mean_t) / std_t)
        _, lv, info = prism.defend(x_adv_norm)
        evaded = (lv == 'PASS')
        score = float(info.get('score', info.get('prism_score', 0.0))) if isinstance(info, dict) else 0.0

        if evaded and not best_evaded:
            best_x_adv, best_evaded, best_score = x_adv_pixel, True, score
        elif evaded and best_evaded and score < best_score:
            best_x_adv, best_score = x_adv_pixel, score
        elif not evaded and not best_evaded and score < best_score:
            best_x_adv, best_score = x_adv_pixel, score

    return best_x_adv if best_x_adv is not None else x_adv_pixel


def run_adaptive_pgd(
    n_test=500, lambdas=None, pgd_steps=40, seed=42,
    output_path='experiments/evaluation/results_adaptive_pgd.json',
    device_str=None, data_root='./data',
    through_scorer=False,
    pgd_restarts=1,
    eot_samples=1,
):
    eps = EPS_LINF_STANDARD
    step_size = eps / 4  # 2/255

    if lambdas is None:
        # P1.4: include λ=10 to probe saturation of the activation-matching loss
        lambdas = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]

    device = torch.device(device_str) if device_str else \
             torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Adaptive PGD: n={n_test}, steps={pgd_steps}, restarts={pgd_restarts}, "
          f"eot_samples={eot_samples}, eps={eps:.4f}, "
          f"lambdas={lambdas}, through_scorer={through_scorer}")
    print(f"Eval split: CIFAR-10 test[{EVAL_IDX[0]}-{EVAL_IDX[1]-1}]\n")

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    # ── Model ──
    model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model = model.to(device).eval()

    # ── Dataset — dispatch on DATASET (cifar10 / cifar100) ──
    ds = load_test_dataset(root=data_root, download=True, transform=_PIXEL_TRANSFORM)
    eval_indices = list(range(*EVAL_IDX))
    sample_idx = rng.choice(eval_indices, min(n_test, len(eval_indices)), replace=False)

    # Pre-load pixel images
    print(f"Pre-loading {len(sample_idx)} images...")
    imgs_pixel = []
    for i in sample_idx:
        img, _ = ds[int(i)]
        imgs_pixel.append(img)
    print(f"Pre-loaded {len(imgs_pixel)} images\n")

    results = {}
    t_start = time.time()

    for lam in lambdas:
        print(f"\n{'='*60}")
        print(f"Adaptive PGD  λ={lam}")
        print(f"{'='*60}")

        # Fresh PRISM per lambda — routed through PATHS for per-dataset artifacts
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

        tp, fp, fn, tn = 0, 0, 0, 0
        level_clean, level_adv = {}, {}

        mean_t = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
        std_t  = torch.tensor(_STD,  device=device).view(1, 3, 1, 1)

        for j, img_pixel in enumerate(tqdm(imgs_pixel, desc=f"  λ={lam}")):
            x_pixel = img_pixel.unsqueeze(0).to(device)
            x_norm  = _NORMALIZE(img_pixel).unsqueeze(0).to(device)

            # Clean evaluation
            _, lv_c, _ = prism.defend(x_norm)
            level_clean[lv_c] = level_clean.get(lv_c, 0) + 1
            if lv_c == 'PASS':
                tn += 1
            else:
                fp += 1

            # Adaptive adversarial (with optional restarts + EOT)
            if pgd_restarts > 1:
                x_adv_pixel = adaptive_pgd_attack_with_restarts(
                    model, x_pixel, eps, pgd_steps, step_size, lam,
                    LAYER_NAMES, device, prism, mean_t, std_t,
                    through_scorer=through_scorer,
                    eot_samples=eot_samples,
                    num_restarts=pgd_restarts,
                )
            else:
                x_adv_pixel = adaptive_pgd_attack(
                    model, x_pixel, eps, pgd_steps, step_size, lam,
                    LAYER_NAMES, device,
                    through_scorer=through_scorer,
                    eot_samples=eot_samples,
                )
            x_adv_norm = _NORMALIZE(x_adv_pixel.squeeze(0).cpu()).unsqueeze(0).to(device)
            _, lv_a, _ = prism.defend(x_adv_norm)
            level_adv[lv_a] = level_adv.get(lv_a, 0) + 1
            if lv_a != 'PASS':
                tp += 1
            else:
                fn += 1

            if (j + 1) % 100 == 0:
                _tpr = tp / max(tp + fn, 1)
                print(f"  [{j+1}/{len(imgs_pixel)}] TPR={_tpr:.4f}")

        n_adv = tp + fn
        n_clean = fp + tn
        tpr = tp / max(n_adv, 1)
        fpr = fp / max(n_clean, 1)
        prec = tp / max(tp + fp, 1)
        f1 = 2 * prec * tpr / max(prec + tpr, 1e-8)
        tpr_ci = wilson_ci(tp, n_adv)
        fpr_ci = wilson_ci(fp, n_clean)
        tier_fpr = per_tier_fpr(level_clean, n_clean)

        key = f'AdaptivePGD_lambda_{lam}'
        results[key] = {
            'TPR': round(tpr, 4),
            'TPR_CI_95': [round(tpr_ci[0], 4), round(tpr_ci[1], 4)],
            'FPR': round(fpr, 4),
            'FPR_CI_95': [round(fpr_ci[0], 4), round(fpr_ci[1], 4)],
            'Precision': round(prec, 4),
            'F1': round(f1, 4),
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'n_adv': n_adv, 'n_clean': n_clean,
            'per_tier_fpr': tier_fpr,
            'clean_level_distribution': level_clean,
            'adversarial_level_distribution': level_adv,
            'lambda': lam,
            'pgd_steps': pgd_steps,
            'pgd_restarts': pgd_restarts,
            'eot_samples': eot_samples,
            'eps': round(eps, 6),
        }

        status = '✅' if tpr >= 0.85 else ('⚠' if tpr >= 0.70 else '❌')
        print(f"\n  TPR={tpr:.4f} CI[{tpr_ci[0]:.4f}, {tpr_ci[1]:.4f}] {status}")
        print(f"  FPR={fpr:.4f} CI[{fpr_ci[0]:.4f}, {fpr_ci[1]:.4f}]")

    elapsed = time.time() - t_start

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"{'λ':>6} {'TPR':>8} {'FPR':>8} {'F1':>8} {'Status':>9}")
    print(f"{'-'*70}")
    for lam in lambdas:
        key = f'AdaptivePGD_lambda_{lam}'
        r = results[key]
        s = '✅' if r['TPR'] >= 0.85 else ('⚠' if r['TPR'] >= 0.70 else '❌')
        print(f"{lam:>6.1f} {r['TPR']:>8.4f} {r['FPR']:>8.4f} {r['F1']:>8.4f} {s:>9}")

    results['_meta'] = {
        'n_test': n_test,
        'n_actual': int(len(sample_idx)),
        'eval_split': f'CIFAR-10 test idx {EVAL_IDX[0]}-{EVAL_IDX[1]-1}',
        'seed': seed,
        'eps': round(eps, 6),
        'eps_note': '8/255 standard',
        'pgd_steps': pgd_steps,
        'pgd_restarts': pgd_restarts,
        'eot_samples': eot_samples,
        'step_size': round(step_size, 6),
        'lambdas': lambdas,
        'through_scorer': through_scorer,
        'device': str(device),
        'elapsed_s': round(elapsed, 1),
        'attack_design': (
            'BPDA adaptive PGD: combined loss = -CE + λ * '
            'Σ_layer ||a_layer(x_adv) - a_layer(x_clean)||₂ / D_layer'
            + (' + 0.5 * |HF_energy(x_adv) - HF_energy(x_clean)| [through_scorer]'
               if through_scorer else '')
            + '. λ=0 is standard PGD. '
            'Reference: Athalye et al. 2018 (Obfuscated Gradients Give a '
            'False Sense of Security, ICML).'
        ),
        'layer_names': LAYER_NAMES,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {output_path}")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Adaptive PGD evaluation for PRISM")
    parser.add_argument('--config', default=None,
                        help='YAML config path (routes via PRISM_CONFIG env var).')
    parser.add_argument('--n-test', type=int, default=500)
    parser.add_argument('--pgd-steps', type=int, default=40)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lambdas', nargs='+', type=float,
                        default=[0.0, 0.5, 1.0, 2.0, 5.0, 10.0],
                        help='λ sweep (default: 0 0.5 1 2 5 10 — P1.4)')
    parser.add_argument('--pgd-restarts', type=int, default=1,
                        help='Random restarts per image; keep most-evasive adversarial '
                             '(P1.4: 10 for strong evaluation)')
    parser.add_argument('--eot-samples', type=int, default=1,
                        help='EOT gradient-averaging samples (Athalye 2018). '
                             'PRISM is deterministic so >1 is a verification, not a defeat.')
    parser.add_argument('--output', default='experiments/evaluation/results_adaptive_pgd.json')
    parser.add_argument('--device', default=None)
    parser.add_argument('--through-scorer', action='store_true',
                        help='Add a DCT high-frequency energy matching term to the '
                             'loss (coefficient 0.5), targeting the ensemble scorer\'s '
                             '37th feature. Produces a stronger adaptive attack.')
    args = parser.parse_args()

    run_adaptive_pgd(
        n_test=args.n_test,
        lambdas=args.lambdas,
        pgd_steps=args.pgd_steps,
        seed=args.seed,
        output_path=args.output,
        device_str=args.device,
        through_scorer=args.through_scorer,
        pgd_restarts=args.pgd_restarts,
        eot_samples=args.eot_samples,
    )
