"""
Native PyTorch Carlini-Wagner L2 attack.

Extracted from experiments/evaluation/run_evaluation_full.py to be shared
across training, evaluation, and ablation scripts.  Keeps the entire
optimisation on GPU — no NumPy bounce per step — giving ~24× speedup
over ART's CarliniL2Method on an RTX 5090.

Mathematical equivalence: identical tanh-space parameterisation, binary
search over c, confidence=0 default, and per-sample best-adversarial
tracking as ART's implementation.  Only the execution backend differs.

Usage
-----
    from src.attacks.cw_torch import cw_l2_attack_torch, TorchCWGenerator

    # Low-level (returns torch tensors + stats dict):
    adv, stats = cw_l2_attack_torch(norm_model, x_pixel, device)

    # High-level wrapper with ART-compatible .generate(x_np) API:
    gen = TorchCWGenerator(norm_model, device, max_iter=40, bss=5)
    x_adv_np = gen.generate(x_np)  # np.ndarray in, np.ndarray out
"""
import torch
import numpy as np


def _atanh_stable(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable arctanh: clamp to avoid ±inf at boundaries."""
    x = torch.clamp(x, -0.999999, 0.999999)
    return 0.5 * torch.log((1.0 + x) / (1.0 - x))


def cw_l2_attack_torch(
    norm_model: torch.nn.Module,
    x_pixel: torch.Tensor,
    device: torch.device,
    max_iter: int = 40,
    binary_search_steps: int = 5,
    learning_rate: float = 0.01,
    confidence: float = 0.0,
    initial_const: float = 0.01,
) -> tuple:
    """
    Native PyTorch untargeted C&W-L2 attack.

    This is the standard tanh-space Carlini-Wagner objective:
        min ||x' - x||_2^2 + c * max(logit_y - max_{i!=y} logit_i + kappa, 0)

    ART's NumPy-driven implementation is correct but very slow because every
    optimisation step bounces tensors through the ART estimator interface.
    This path keeps the entire optimisation on GPU while preserving the same
    labels, confidence=0 default, binary search over c, and per-sample
    best-adversarial tracking.

    Args:
        norm_model: A model that accepts pixel-space [0,1] input and returns
                    logits.  Normalisation must be baked in (see
                    _NormalizedResNet in the calling scripts).
        x_pixel:    Input tensor of shape (B, C, H, W) in [0, 1].
        device:     torch.device for computation.
        max_iter:   Optimiser steps per binary-search round.
        binary_search_steps: Number of binary-search rounds over c.
        learning_rate: Adam learning rate.
        confidence: Margin kappa (0 = untargeted misclassification).
        initial_const: Starting value of c for the binary search.

    Returns:
        (best_adv, stats) where best_adv is a (B, C, H, W) tensor clamped
        to [0, 1] and stats is a dict with attack_success count, L2 norms,
        total optimizer steps, and per-sample success mask.
    """
    norm_model.eval()
    x = x_pixel.detach().to(device).clamp(0.0, 1.0)
    batch = x.size(0)

    with torch.no_grad():
        y = norm_model(x).argmax(dim=1)

    w_orig = _atanh_stable(x * 2.0 - 1.0)
    lower = torch.zeros(batch, device=device)
    upper = torch.full((batch,), float('inf'), device=device)
    const = torch.full((batch,), float(initial_const), device=device)

    best_l2 = torch.full((batch,), float('inf'), device=device)
    best_adv = x.clone()
    best_success = torch.zeros(batch, dtype=torch.bool, device=device)

    total_steps = 0
    for _ in range(binary_search_steps):
        w = w_orig.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([w], lr=learning_rate)
        step_success = torch.zeros(batch, dtype=torch.bool, device=device)

        for _iter in range(max_iter):
            adv = 0.5 * (torch.tanh(w) + 1.0)
            logits = norm_model(adv)

            real = logits.gather(1, y.view(-1, 1)).squeeze(1)
            mask = torch.nn.functional.one_hot(y, num_classes=logits.size(1)).bool()
            other = logits.masked_fill(mask, -1e9).max(dim=1).values
            margin = torch.clamp(real - other + confidence, min=0.0)
            l2 = (adv - x).flatten(1).pow(2).sum(dim=1)
            loss = (l2 + const * margin).sum()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_steps += 1

            with torch.no_grad():
                adv_updated = 0.5 * (torch.tanh(w) + 1.0)
                logits_updated = norm_model(adv_updated)
                pred = logits_updated.argmax(dim=1)
                success = pred.ne(y)
                l2_updated = (adv_updated - x).flatten(1).pow(2).sum(dim=1)
                improved = success & (l2_updated < best_l2)
                if improved.any():
                    best_l2[improved] = l2_updated[improved]
                    best_adv[improved] = adv_updated.detach()[improved]
                    best_success[improved] = True
                step_success |= success

        with torch.no_grad():
            upper = torch.where(step_success, torch.minimum(upper, const), upper)
            lower = torch.where(step_success, lower, torch.maximum(lower, const))
            has_upper = torch.isfinite(upper)
            bisected = (lower + upper) / 2.0
            const = torch.where(
                has_upper,
                bisected,
                torch.where(step_success, const, const * 10.0),
            )

    l2_out = torch.where(best_success, best_l2, torch.zeros_like(best_l2))
    stats = {
        'attack_success': int(best_success.sum().item()),
        'batch_size': int(batch),
        'total_optimizer_steps': int(total_steps),
        'mean_l2_success': (
            float(best_l2[best_success].sqrt().mean().item())
            if best_success.any() else None
        ),
        'max_l2_success': (
            float(best_l2[best_success].sqrt().max().item())
            if best_success.any() else None
        ),
        'mean_l2_all': float(l2_out.sqrt().mean().item()),
        'success_mask': best_success.detach().cpu().numpy().astype(bool),
        'success_l2': (
            best_l2[best_success].sqrt().detach().cpu().numpy()
            if best_success.any() else np.array([], dtype=np.float32)
        ),
    }
    return best_adv.detach().clamp(0.0, 1.0), stats


class TorchCWGenerator:
    """Wraps native torch CW to expose ART's .generate(x_np) interface.

    This allows _batch_generate_adversarials() and batch_generate_adversarials()
    to call gen.generate(x_np_chunk) identically to any ART attack object,
    while the actual computation runs on GPU via cw_l2_attack_torch().

    Usage:
        gen = TorchCWGenerator(norm_model, device, max_iter=40, bss=5)
        x_adv_np = gen.generate(x_np)  # (N, C, H, W) np.ndarray
    """

    def __init__(self, norm_model, device, max_iter=40, bss=5,
                 lr=0.01, confidence=0.0):
        self._model = norm_model
        self._device = device
        self._max_iter = max_iter
        self._bss = bss
        self._lr = lr
        self._confidence = confidence

    def generate(self, x_np):
        """ART-compatible interface: np.ndarray in → np.ndarray out."""
        x = torch.tensor(x_np, dtype=torch.float32, device=self._device)
        adv, _ = cw_l2_attack_torch(
            self._model, x, self._device,
            max_iter=self._max_iter,
            binary_search_steps=self._bss,
            learning_rate=self._lr,
            confidence=self._confidence,
        )
        return adv.detach().cpu().numpy()
