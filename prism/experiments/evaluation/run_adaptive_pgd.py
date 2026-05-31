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

EVAL SPLIT: active dataset test indices from src.config.EVAL_IDX.
"""
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
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
    BACKBONE_MEAN, BACKBONE_STD, BACKBONE_INPUT_SIZE, BACKBONE_NUM_CLASSES,
    EPS_LINF_STANDARD,
    EVAL_IDX, DATASET, PATHS,
)
from src.data_loader import load_test_dataset

_MEAN = BACKBONE_MEAN
_STD  = BACKBONE_STD
if BACKBONE_INPUT_SIZE == 32:
    _PIXEL_TRANSFORM = T.Compose([T.ToTensor()])
else:
    _PIXEL_TRANSFORM = T.Compose([T.Resize(BACKBONE_INPUT_SIZE), T.ToTensor()])
_NORMALIZE       = T.Normalize(mean=_MEAN, std=_STD)


# Backward-compat alias — _NormalizedBackbone in src.models is the same wrapper.
from src.models import load_backbone, _NormalizedBackbone as _NormalizedResNet


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


def _append_jsonl(path, row):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(row, sort_keys=True) + '\n')


def _load_completed_lambdas(path):
    completed = {}
    if not path or not os.path.exists(path):
        return completed
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get('event') == 'lambda_done' and 'key' in row and 'result' in row:
                completed[row['key']] = row['result']
    return completed


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


def _softmax_entropy_torch(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    return -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)


def _logit_profile_torch(logits: torch.Tensor) -> torch.Tensor:
    """
    Differentiable mirror of compute_logit_profile_features().

    Returned order:
      [top_confidence, top2_probability_gap, top3_probability_mass,
       logit_margin, logit_range, centered_logit_l2, logsumexp_energy,
       probability_l2]
    """
    probs = torch.softmax(logits, dim=1)
    n_cls = logits.shape[1]
    p_sorted, _ = torch.sort(probs, dim=1, descending=True)
    z_sorted, _ = torch.sort(logits, dim=1, descending=True)
    top_conf = p_sorted[:, 0]
    top2_gap = p_sorted[:, 0] - (p_sorted[:, 1] if n_cls > 1 else 0.0)
    top3_mass = p_sorted[:, : min(3, n_cls)].sum(dim=1)
    margin = z_sorted[:, 0] - (z_sorted[:, 1] if n_cls > 1 else 0.0)
    logit_range = z_sorted[:, 0] - z_sorted[:, -1]
    centered_l2 = torch.norm(logits - logits.mean(dim=1, keepdim=True), p=2, dim=1)
    energy = torch.logsumexp(logits, dim=1)
    prob_l2 = torch.norm(probs, p=2, dim=1)
    return torch.stack([
        top_conf, top2_gap, top3_mass, margin,
        logit_range, centered_l2, energy, prob_l2,
    ], dim=1)


def _shift_reflect_torch(pix: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    _, _, height, width = pix.shape
    padded = F.pad(pix, (1, 1, 1, 1), mode='reflect')
    y0 = 1 - int(dy)
    x0 = 1 - int(dx)
    return padded[:, :, y0:y0 + height, x0:x0 + width]


def _stability_transforms_torch(pix: torch.Tensor):
    _, _, height, width = pix.shape
    avg3 = F.avg_pool2d(
        F.pad(pix, (1, 1, 1, 1), mode='reflect'),
        kernel_size=3,
        stride=1,
    )
    transforms = [
        avg3,
        _shift_reflect_torch(pix, dy=1, dx=0),
        _shift_reflect_torch(pix, dy=0, dx=1),
    ]
    if height >= 2 and width >= 2:
        low = F.avg_pool2d(pix, kernel_size=2, stride=2)
        transforms.append(
            F.interpolate(low, size=(height, width), mode='bilinear', align_corners=False)
        )
    return transforms


def _stability_summary_torch(
    model: torch.nn.Module,
    x_pixel: torch.Tensor,
    logits: torch.Tensor,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
) -> torch.Tensor:
    """
    Differentiable surrogate for the deployed stability-v2 block.

    The deployed block contains a hard top-1-changed indicator.  The surrogate
    uses probability disagreement for that slot so gradients remain useful.
    """
    pa = torch.softmax(logits, dim=1)
    rows = []
    for transformed in _stability_transforms_torch(x_pixel.clamp(0.0, 1.0)):
        transformed_norm = (transformed.clamp(0.0, 1.0) - mean_t) / std_t
        logits_b = model(transformed_norm)
        pb = torch.softmax(logits_b, dim=1)
        m = 0.5 * (pa + pb)
        js = 0.5 * (pa * (torch.log(pa.clamp_min(1e-12)) - torch.log(m.clamp_min(1e-12)))).sum(dim=1)
        js = js + 0.5 * (pb * (torch.log(pb.clamp_min(1e-12)) - torch.log(m.clamp_min(1e-12)))).sum(dim=1)
        top1_proxy = 1.0 - (pa * pb).sum(dim=1)
        conf_delta = (pa.max(dim=1).values - pb.max(dim=1).values).abs()
        za = torch.sort(logits, dim=1, descending=True).values
        zb = torch.sort(logits_b, dim=1, descending=True).values
        margin_a = za[:, 0] - (za[:, 1] if za.shape[1] > 1 else 0.0)
        margin_b = zb[:, 0] - (zb[:, 1] if zb.shape[1] > 1 else 0.0)
        margin_delta = (margin_a - margin_b).abs()
        rows.append(torch.stack([js, top1_proxy, conf_delta, margin_delta], dim=1))
    arr = torch.stack(rows, dim=1)
    return torch.stack([
        arr[:, :, 0].max(dim=1).values,
        arr[:, :, 0].mean(dim=1),
        arr[:, :, 1].max(dim=1).values,
        arr[:, :, 1].mean(dim=1),
        arr[:, :, 2].max(dim=1).values,
        arr[:, :, 2].mean(dim=1),
        arr[:, :, 3].max(dim=1).values,
        arr[:, :, 3].mean(dim=1),
    ], dim=1)


def _pred_logit_grad_norm_torch(
    logits: torch.Tensor,
    x_norm: torch.Tensor,
    create_graph: bool,
) -> torch.Tensor:
    pred = logits.argmax(dim=1)
    selected = logits.gather(1, pred.view(-1, 1)).sum()
    (grad_x,) = torch.autograd.grad(
        selected,
        x_norm,
        create_graph=create_graph,
        retain_graph=True,
    )
    return grad_x.flatten(1).norm(p=2, dim=1)


def _ensemble_feature_flags(
    scorer,
    through_scorer: bool,
    ensemble_complete: bool,
    include_gradnorm: bool,
):
    if not ensemble_complete:
        return {
            'dct': bool(through_scorer),
            'entropy': False,
            'logit_profile': False,
            'stability': False,
            'grad_norm': False,
        }
    if scorer is None:
        return {
            'dct': True,
            'entropy': True,
            'logit_profile': True,
            'stability': True,
            'grad_norm': bool(include_gradnorm),
        }
    return {
        'dct': bool(getattr(scorer, 'use_dct', False) or through_scorer),
        'entropy': bool(getattr(scorer, 'use_softmax_entropy', False)),
        'logit_profile': bool(getattr(scorer, 'use_logit_profile_features', False)),
        'stability': bool(getattr(scorer, 'use_stability_features', False)),
        'grad_norm': bool(include_gradnorm and getattr(scorer, 'use_grad_norm', False)),
    }


def _side_feature_vector_torch(
    model: torch.nn.Module,
    x_pixel: torch.Tensor,
    x_norm: torch.Tensor,
    logits: torch.Tensor,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    flags,
    create_graph: bool,
) -> torch.Tensor:
    parts = []
    if flags.get('dct', False):
        parts.append(_hf_energy_torch(x_pixel).view(1, 1))
    if flags.get('entropy', False):
        parts.append(_softmax_entropy_torch(logits).view(1, 1))
    if flags.get('logit_profile', False):
        parts.append(_logit_profile_torch(logits))
    if flags.get('stability', False):
        parts.append(_stability_summary_torch(model, x_pixel, logits, mean_t, std_t))
    if flags.get('grad_norm', False):
        parts.append(_pred_logit_grad_norm_torch(logits, x_norm, create_graph=create_graph).view(1, 1))
    if not parts:
        return torch.zeros((1, 0), device=x_pixel.device, dtype=x_pixel.dtype)
    return torch.cat(parts, dim=1)


def _relative_feature_mse(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    denom = ref.detach().abs() + 1.0
    return torch.mean(((x - ref.detach()) / denom) ** 2)


def _side_logistic_surrogate(side_features: torch.Tensor, scorer):
    """
    Actual deployed side-channel logistic logit, with unavailable TDA columns
    neutralized at their training mean.  If side-quadratic is enabled, this
    also includes the deployed quadratic side interactions.
    """
    if scorer is None or not getattr(scorer, '_logistic_fitted', False):
        return None
    if getattr(scorer, 'feature_mean', None) is None or getattr(scorer, 'feature_std', None) is None:
        return None
    if getattr(scorer, 'logistic_weights', None) is None:
        return None

    raw_dim = int(getattr(scorer, 'n_features', 0))
    weights_np = np.asarray(scorer.logistic_weights, dtype=np.float32).reshape(-1)
    mean_np = np.asarray(scorer.feature_mean, dtype=np.float32).reshape(-1)
    std_np = np.asarray(scorer.feature_std, dtype=np.float32).reshape(-1)
    if raw_dim <= 0 or len(mean_np) != len(std_np) or len(weights_np) != len(mean_np):
        return None

    side_start = int(getattr(scorer, 'quadratic_feature_start', 36))
    side_start = min(max(side_start, 0), raw_dim)
    expected_side = raw_dim - side_start
    if expected_side != side_features.shape[1]:
        return None

    device = side_features.device
    dtype = side_features.dtype
    raw_mean = torch.as_tensor(mean_np[:raw_dim], device=device, dtype=dtype).view(1, -1)
    raw = raw_mean.clone()
    raw[:, side_start:] = side_features

    model_features = raw
    if bool(getattr(scorer, 'use_side_quadratic_features', False)):
        side = raw[:, side_start:]
        tri = torch.triu_indices(side.shape[1], side.shape[1], device=device)
        quad = side[:, tri[0]] * side[:, tri[1]]
        model_features = torch.cat([raw, quad], dim=1)

    if model_features.shape[1] != len(weights_np):
        return None
    mean = torch.as_tensor(mean_np, device=device, dtype=dtype).view(1, -1)
    std = torch.as_tensor(std_np, device=device, dtype=dtype).view(1, -1)
    weights = torch.as_tensor(weights_np, device=device, dtype=dtype).view(1, -1)
    feat_norm = (model_features - mean) / (std + 1e-8)
    bias = torch.as_tensor(float(scorer.logistic_bias), device=device, dtype=dtype)
    return (feat_norm * weights).sum(dim=1) + bias


def adaptive_pgd_attack(
    model, x_pixel, eps, steps, step_size, lam, layer_names, device,
    through_scorer: bool = False,
    eot_samples: int = 1,
    ensemble_complete: bool = False,
    ec_match_coeff: float = 0.5,
    ec_score_coeff: float = 0.25,
    ec_include_gradnorm: bool = True,
    scorer=None,
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

    With `ensemble_complete=True`, the legacy DCT-only scorer term is replaced
    by a differentiable side-channel surrogate that matches DCT, entropy,
    logit-profile, stability-v2, and grad-norm features to the clean input and
    also minimizes the fitted side-quadratic logistic score when available.

    Args:
        model: CIFAR backbone model receiving normalised input.
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
    feature_flags = _ensemble_feature_flags(
        scorer=scorer,
        through_scorer=through_scorer,
        ensemble_complete=ensemble_complete,
        include_gradnorm=ec_include_gradnorm,
    )

    # Pre-compute clean DCT energy reference (constant, no grad needed)
    with torch.no_grad():
        clean_hf_energy = _hf_energy_torch(x).detach() if through_scorer and not ensemble_complete else None

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

    clean_side_features = None
    clean_side_logit = None
    if ensemble_complete:
        x_norm_ref = ((x.detach() - mean_t) / std_t).requires_grad_(feature_flags.get('grad_norm', False))
        with torch.enable_grad():
            logits_ref = model(x_norm_ref)
            clean_side_features = _side_feature_vector_torch(
                model=model,
                x_pixel=x.detach(),
                x_norm=x_norm_ref,
                logits=logits_ref,
                mean_t=mean_t,
                std_t=std_t,
                flags=feature_flags,
                create_graph=False,
            ).detach()
            clean_side_logit_tmp = _side_logistic_surrogate(clean_side_features, scorer)
            if clean_side_logit_tmp is not None:
                clean_side_logit = clean_side_logit_tmp.detach()

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

            for h in handles2:
                h.remove()

            loss_scorer = torch.tensor(0.0, device=device)
            if (
                ensemble_complete
                and clean_side_features is not None
                and clean_side_features.numel() > 0
            ):
                adv_side_features = _side_feature_vector_torch(
                    model=model,
                    x_pixel=x_adv,
                    x_norm=x_adv_norm,
                    logits=logits,
                    mean_t=mean_t,
                    std_t=std_t,
                    flags=feature_flags,
                    create_graph=True,
                )
                loss_side_match = _relative_feature_mse(adv_side_features, clean_side_features)
                loss_side_score = torch.tensor(0.0, device=device)
                adv_side_logit = _side_logistic_surrogate(adv_side_features, scorer)
                if adv_side_logit is not None and clean_side_logit is not None:
                    loss_side_score = F.softplus(adv_side_logit - clean_side_logit).mean()
                loss_scorer = ec_match_coeff * loss_side_match + ec_score_coeff * loss_side_score
            elif through_scorer and clean_hf_energy is not None:
                # DCT high-frequency energy matching (legacy through-scorer mode)
                adv_hf_energy = _hf_energy_torch(x_adv)
                loss_scorer = 0.5 * (adv_hf_energy - clean_hf_energy).abs()

            loss_total = loss_ce + lam * loss_act + loss_scorer
            model.zero_grad(set_to_none=True)
            loss_total.backward()

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
    ensemble_complete: bool = False,
    ec_match_coeff: float = 0.5,
    ec_score_coeff: float = 0.25,
    ec_include_gradnorm: bool = True,
):
    """
    Run adaptive PGD with `num_restarts` random initialisations and keep the
    strongest valid candidate. Selection prefers classifier-successful
    adversarials first, then PRISM evasion, then the lowest PRISM score.

    This matches Athalye/Carlini's best-practice for detector evaluation: an
    adversary gets multiple attempts per image, and the defender must survive
    the worst of them.
    """
    best_x_adv = None
    best_rank = None

    with torch.no_grad():
        clean_pred = int(model(((x_pixel.to(device) - mean_t) / std_t)).argmax(dim=1).item())

    for _restart in range(max(num_restarts, 1)):
        x_adv_pixel = adaptive_pgd_attack(
            model, x_pixel, eps, steps, step_size, lam, layer_names, device,
            through_scorer=through_scorer,
            eot_samples=eot_samples,
            ensemble_complete=ensemble_complete,
            ec_match_coeff=ec_match_coeff,
            ec_score_coeff=ec_score_coeff,
            ec_include_gradnorm=ec_include_gradnorm,
            scorer=getattr(prism, 'scorer', None),
        )
        x_adv_norm = ((x_adv_pixel - mean_t) / std_t)
        with torch.no_grad():
            adv_pred = int(model(x_adv_norm).argmax(dim=1).item())
        _, lv, info = prism.defend(x_adv_norm, pixel_image=x_adv_pixel)
        evaded = (lv == 'PASS')
        successful = (adv_pred != clean_pred)
        score = float(
            info.get('anomaly_score', info.get('score', info.get('prism_score', 0.0)))
        ) if isinstance(info, dict) else 0.0

        # Prefer valid adversarials first, then PRISM evasion, then lower score.
        rank = (1 if successful else 0, 1 if evaded else 0, -score)
        if best_rank is None or rank > best_rank:
            best_x_adv, best_rank = x_adv_pixel, rank

    return best_x_adv if best_x_adv is not None else x_adv_pixel


def run_adaptive_pgd(
    n_test=500, lambdas=None, pgd_steps=40, seed=42,
    output_path='experiments/evaluation/results_adaptive_pgd.json',
    device_str=None, data_root='./data',
    through_scorer=False,
    pgd_restarts=1,
    eot_samples=1,
    eot_verify_samples=20,
    checkpoint_jsonl=None,
    resume=False,
    ensemble_complete=False,
    ec_match_coeff=0.5,
    ec_score_coeff=0.25,
    ec_include_gradnorm=True,
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
          f"lambdas={lambdas}, through_scorer={through_scorer}, "
          f"ensemble_complete={ensemble_complete}")
    if ensemble_complete:
        print(
            "Ensemble-complete surrogate: DCT/entropy/logit-profile/"
            "stability-v2/grad-norm feature matching plus fitted side-quadratic "
            f"logistic score, match_coeff={ec_match_coeff}, score_coeff={ec_score_coeff}, "
            f"gradnorm={ec_include_gradnorm}"
        )
    print(f"Eval split: {DATASET.upper()} test[{EVAL_IDX[0]}-{EVAL_IDX[1]-1}]\n")
    if checkpoint_jsonl is None:
        root, ext = os.path.splitext(output_path)
        checkpoint_jsonl = f"{root}.jsonl" if ext else f"{output_path}.jsonl"
    print(f"Checkpoint JSONL: {checkpoint_jsonl}  resume={resume}")
    if eot_verify_samples > 1 and eot_samples == 1:
        print(
            f"EOT verification: detector/model path is deterministic; recording "
            f"n={eot_verify_samples} verification samples without repeating "
            "identical gradients during optimization."
        )

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    # ── Model ──
    # Active CIFAR-trained backbone from the current config.
    model = load_backbone(device)

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
    completed = _load_completed_lambdas(checkpoint_jsonl) if resume else {}
    if completed:
        results.update(completed)
        print(f"Resume: loaded {len(completed)} completed lambda result(s).")
    t_start = time.time()

    for lam in lambdas:
        key = f'AdaptivePGD_lambda_{lam}'
        if key in completed:
            print(f"\nSkipping λ={lam}; completed result found in checkpoint.")
            continue
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
        attack_success = 0
        detected_success = 0
        evaded_success = 0

        mean_t = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
        std_t  = torch.tensor(_STD,  device=device).view(1, 3, 1, 1)

        for j, img_pixel in enumerate(tqdm(imgs_pixel, desc=f"  λ={lam}")):
            x_pixel = img_pixel.unsqueeze(0).to(device)
            x_norm  = _NORMALIZE(img_pixel).unsqueeze(0).to(device)
            with torch.no_grad():
                clean_pred = int(model(x_norm).argmax(dim=1).item())

            # Clean evaluation
            _, lv_c, _ = prism.defend(x_norm, pixel_image=img_pixel)
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
                    ensemble_complete=ensemble_complete,
                    ec_match_coeff=ec_match_coeff,
                    ec_score_coeff=ec_score_coeff,
                    ec_include_gradnorm=ec_include_gradnorm,
                )
            else:
                x_adv_pixel = adaptive_pgd_attack(
                    model, x_pixel, eps, pgd_steps, step_size, lam,
                    LAYER_NAMES, device,
                    through_scorer=through_scorer,
                    eot_samples=eot_samples,
                    ensemble_complete=ensemble_complete,
                    ec_match_coeff=ec_match_coeff,
                    ec_score_coeff=ec_score_coeff,
                    ec_include_gradnorm=ec_include_gradnorm,
                    scorer=getattr(prism, 'scorer', None),
                )
            x_adv_norm = _NORMALIZE(x_adv_pixel.squeeze(0).cpu()).unsqueeze(0).to(device)
            with torch.no_grad():
                adv_pred = int(model(x_adv_norm).argmax(dim=1).item())
            is_success = (adv_pred != clean_pred)
            if is_success:
                attack_success += 1
            _, lv_a, _ = prism.defend(x_adv_norm, pixel_image=x_adv_pixel)
            level_adv[lv_a] = level_adv.get(lv_a, 0) + 1
            if lv_a != 'PASS':
                tp += 1
                if is_success:
                    detected_success += 1
            else:
                fn += 1
                if is_success:
                    evaded_success += 1

            if (j + 1) % 100 == 0:
                _tpr = tp / max(tp + fn, 1)
                _asr = attack_success / max(j + 1, 1)
                print(f"  [{j+1}/{len(imgs_pixel)}] TPR={_tpr:.4f}")
                _append_jsonl(checkpoint_jsonl, {
                    'event': 'progress',
                    'key': key,
                    'lambda': lam,
                    'processed': int(j + 1),
                    'n_total': int(len(imgs_pixel)),
                    'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn),
                    'TPR': round(float(_tpr), 6),
                    'model_ASR': round(float(_asr), 6),
                    'n_successful_adv': int(attack_success),
                    'timestamp': time.time(),
                })

        n_adv = tp + fn
        n_clean = fp + tn
        tpr = tp / max(n_adv, 1)
        fpr = fp / max(n_clean, 1)
        asr = attack_success / max(n_adv, 1)
        tpr_success = detected_success / max(attack_success, 1)
        evasion_success = evaded_success / max(attack_success, 1)
        undetected_success = evaded_success / max(n_adv, 1)
        prec = tp / max(tp + fp, 1)
        f1 = 2 * prec * tpr / max(prec + tpr, 1e-8)
        tpr_ci = wilson_ci(tp, n_adv)
        fpr_ci = wilson_ci(fp, n_clean)
        tier_fpr = per_tier_fpr(level_clean, n_clean)

        results[key] = {
            'TPR': round(tpr, 4),
            'TPR_CI_95': [round(tpr_ci[0], 4), round(tpr_ci[1], 4)],
            'FPR': round(fpr, 4),
            'FPR_CI_95': [round(fpr_ci[0], 4), round(fpr_ci[1], 4)],
            'Precision': round(prec, 4),
            'F1': round(f1, 4),
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'n_adv': n_adv, 'n_clean': n_clean,
            'n_successful_adv': int(attack_success),
            'model_ASR': round(asr, 4),
            'TPR_on_successful_attacks': round(tpr_success, 4),
            'evasion_rate_on_successful_attacks': round(evasion_success, 4),
            'undetected_success_rate': round(undetected_success, 4),
            'detected_successful_adv': int(detected_success),
            'evaded_successful_adv': int(evaded_success),
            'per_tier_fpr': tier_fpr,
            'clean_level_distribution': level_clean,
            'adversarial_level_distribution': level_adv,
            'lambda': lam,
            'pgd_steps': pgd_steps,
            'pgd_restarts': pgd_restarts,
            'eot_samples': eot_samples,
            'ensemble_complete': bool(ensemble_complete),
            'ec_match_coeff': float(ec_match_coeff),
            'ec_score_coeff': float(ec_score_coeff),
            'ec_include_gradnorm': bool(ec_include_gradnorm),
            'eps': round(eps, 6),
        }

        status = '✅' if tpr >= 0.85 else ('⚠' if tpr >= 0.70 else '❌')
        print(f"\n  TPR={tpr:.4f} CI[{tpr_ci[0]:.4f}, {tpr_ci[1]:.4f}] {status}")
        print(f"  Model ASR={asr:.4f}; TPR on successful attacks={tpr_success:.4f}")
        print(f"  FPR={fpr:.4f} CI[{fpr_ci[0]:.4f}, {fpr_ci[1]:.4f}]")
        _append_jsonl(checkpoint_jsonl, {
            'event': 'lambda_done',
            'key': key,
            'lambda': lam,
            'result': results[key],
            'timestamp': time.time(),
        })

    elapsed = time.time() - t_start

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"{'λ':>6} {'TPR':>8} {'ASR':>8} {'FPR':>8} {'F1':>8} {'Status':>9}")
    print(f"{'-'*70}")
    for lam in lambdas:
        key = f'AdaptivePGD_lambda_{lam}'
        r = results[key]
        s = '✅' if r['TPR'] >= 0.85 else ('⚠' if r['TPR'] >= 0.70 else '❌')
        print(f"{lam:>6.1f} {r['TPR']:>8.4f} {r['model_ASR']:>8.4f} "
              f"{r['FPR']:>8.4f} {r['F1']:>8.4f} {s:>9}")

    results['_meta'] = {
        'n_test': n_test,
        'n_actual': int(len(sample_idx)),
        'dataset': DATASET,
        'eval_split': f'{DATASET.upper()} test idx {EVAL_IDX[0]}-{EVAL_IDX[1]-1}',
        'seed': seed,
        'eps': round(eps, 6),
        'eps_note': '8/255 standard',
        'pgd_steps': pgd_steps,
        'pgd_restarts': pgd_restarts,
        'eot_samples': eot_samples,
        'eot_verify_samples': eot_verify_samples,
        'eot_verification': (
            'PRISM uses deterministic forward scoring and deterministic '
            'hash-based TDA subsampling; eot_verify_samples records the '
            'Appendix-B n=20 verification setting. Set --eot-samples 20 to '
            'force repeated identical gradient averaging.'
        ),
        'step_size': round(step_size, 6),
        'lambdas': lambdas,
        'through_scorer': through_scorer,
        'ensemble_complete': bool(ensemble_complete),
        'ec_match_coeff': float(ec_match_coeff),
        'ec_score_coeff': float(ec_score_coeff),
        'ec_include_gradnorm': bool(ec_include_gradnorm),
        'ensemble_complete_terms': [
            'activation_matching_for_TDA',
            'dct_high_frequency_energy',
            'softmax_entropy',
            'logit_profile_8',
            'stability_v2_8',
            'input_grad_norm',
            'side_quadratic_logistic_surrogate',
        ] if ensemble_complete else [],
        'lambda_zero_note': (
            'CE plus ensemble side-channel evasion when ensemble_complete=True; '
            'standard PGD otherwise.'
        ),
        'device': str(device),
        'elapsed_s': round(elapsed, 1),
        'checkpoint_jsonl': checkpoint_jsonl,
        'resume': bool(resume),
        'attack_design': (
            'BPDA adaptive PGD: combined loss = -CE + lambda * '
            'sum_layer ||a_layer(x_adv) - a_layer(x_clean)||_2 / D_layer'
            + (
                ' + ensemble-complete side-channel surrogate '
                '(DCT, entropy, logit profile, stability-v2, grad-norm, '
                'side-quadratic logistic score)'
                if ensemble_complete else
                (' + 0.5 * |HF_energy(x_adv) - HF_energy(x_clean)| [through_scorer]'
                 if through_scorer else '')
            )
            + (
                '. lambda=0 keeps the ensemble side-channel surrogate active. '
                if ensemble_complete else
                '. lambda=0 is standard PGD. '
            )
            + 'Reference: Athalye et al. 2018 (Obfuscated Gradients Give a '
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
    parser.add_argument('--eot-verify-samples', type=int, default=20,
                        help='Appendix-B EOT verification count recorded in metadata. '
                             'The detector is deterministic; use --eot-samples to '
                             'force repeated gradient averaging if desired.')
    parser.add_argument('--output', default='experiments/evaluation/results_adaptive_pgd.json')
    parser.add_argument('--device', default=None)
    parser.add_argument('--through-scorer', action='store_true',
                        help='Add a DCT high-frequency energy matching term to the '
                             'loss (coefficient 0.5), targeting the ensemble scorer\'s '
                             '37th feature. Produces a stronger adaptive attack.')
    parser.add_argument('--ensemble-complete', action='store_true',
                        help='Target the deployed ensemble side channels too: DCT, '
                             'softmax entropy, logit profile, stability-v2, grad-norm, '
                             'and side-quadratic logistic score when present.')
    parser.add_argument('--ec-match-coeff', type=float, default=0.5,
                        help='Coefficient for ensemble-complete side-feature matching.')
    parser.add_argument('--ec-score-coeff', type=float, default=0.25,
                        help='Coefficient for the fitted side-quadratic logistic score surrogate.')
    parser.add_argument('--skip-gradnorm-surrogate', action='store_true',
                        help='Disable the second-order grad-norm surrogate if VRAM/time is too high.')
    parser.add_argument('--checkpoint-jsonl', default=None,
                        help='Append progress and completed lambda results to this JSONL file.')
    parser.add_argument('--resume', action='store_true',
                        help='Skip lambda values already completed in --checkpoint-jsonl.')
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
        eot_verify_samples=args.eot_verify_samples,
        checkpoint_jsonl=args.checkpoint_jsonl,
        resume=args.resume,
        ensemble_complete=args.ensemble_complete,
        ec_match_coeff=args.ec_match_coeff,
        ec_score_coeff=args.ec_score_coeff,
        ec_include_gradnorm=not args.skip_gradnorm_surrogate,
    )
