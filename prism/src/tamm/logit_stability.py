"""
Deterministic logit-stability feature extraction.

This module owns the pixel transforms used by training, calibration, and
runtime inference so the feature-space contract cannot drift between phases.
"""
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .persistence_stats import (
    compute_logit_stability_features,
    compute_logit_stability_summary,
)


DEFAULT_STABILITY_FEATURE_COUNT = 8


def _normalise_pixel_tensor(
    pix: torch.Tensor,
    mean: List[float],
    std: List[float],
) -> torch.Tensor:
    mean_t = torch.tensor(mean, device=pix.device, dtype=pix.dtype).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=pix.device, dtype=pix.dtype).view(1, -1, 1, 1)
    return (pix - mean_t) / std_t


def _pixel_tensor(
    x_norm: torch.Tensor,
    img_pixel: Optional[torch.Tensor],
    mean: List[float],
    std: List[float],
) -> torch.Tensor:
    if img_pixel is not None:
        pix = img_pixel.detach().to(device=x_norm.device, dtype=x_norm.dtype)
        if pix.dim() == 3:
            pix = pix.unsqueeze(0)
        return pix.clamp(0.0, 1.0)

    mean_t = torch.tensor(mean, device=x_norm.device, dtype=x_norm.dtype).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=x_norm.device, dtype=x_norm.dtype).view(1, -1, 1, 1)
    return (x_norm.detach() * std_t + mean_t).clamp(0.0, 1.0)


def _shift_reflect(pix: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    _, _, height, width = pix.shape
    padded = F.pad(pix, (1, 1, 1, 1), mode='reflect')
    y0 = 1 - int(dy)
    x0 = 1 - int(dx)
    return padded[:, :, y0:y0 + height, x0:x0 + width]


def _stability_transforms(pix: torch.Tensor) -> List[torch.Tensor]:
    _, _, height, width = pix.shape
    avg3 = F.avg_pool2d(
        F.pad(pix, (1, 1, 1, 1), mode='reflect'),
        kernel_size=3,
        stride=1,
    )
    transforms = [
        avg3,
        _shift_reflect(pix, dy=1, dx=0),
        _shift_reflect(pix, dy=0, dx=1),
    ]
    if height >= 2 and width >= 2:
        low = F.avg_pool2d(pix, kernel_size=2, stride=2)
        transforms.append(
            F.interpolate(low, size=(height, width), mode='bilinear', align_corners=False)
        )
    return transforms


def compute_input_stability_features(
    model: torch.nn.Module,
    x_norm: torch.Tensor,
    img_pixel: Optional[torch.Tensor],
    mean: List[float],
    std: List[float],
    logits_np: Optional[np.ndarray] = None,
    feature_count: int = DEFAULT_STABILITY_FEATURE_COUNT,
) -> np.ndarray:
    """
    Compute deterministic transform-consistency features for one input.

    feature_count <= 4 preserves the legacy one-transform block used by older
    42-feature artifacts. feature_count >= 8 activates the v2 multi-transform
    summary used by the current research candidate.
    """
    with torch.no_grad():
        if logits_np is None:
            logits_np = model(x_norm).squeeze(0).detach().cpu().numpy()

        pix = _pixel_tensor(x_norm, img_pixel, mean, std)
        transforms = _stability_transforms(pix)
        transformed_logits = []
        for transformed in transforms:
            transformed_norm = _normalise_pixel_tensor(
                transformed.clamp(0.0, 1.0),
                mean,
                std,
            )
            transformed_logits.append(
                model(transformed_norm).squeeze(0).detach().cpu().numpy()
            )

    if feature_count <= 4:
        return compute_logit_stability_features(logits_np, transformed_logits[0])

    features = compute_logit_stability_summary(logits_np, transformed_logits)
    if feature_count == features.size:
        return features
    if feature_count < features.size:
        return features[:feature_count].astype(np.float32)
    out = np.zeros(feature_count, dtype=np.float32)
    out[:features.size] = features
    return out
