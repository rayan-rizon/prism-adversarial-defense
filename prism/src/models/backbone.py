"""
Centralised backbone loader for the PRISM pipeline.

Single source of truth: every pipeline script (build_profile_testset,
train_ensemble_scorer, calibrate_ensemble, compute_ensemble_val_fpr,
train_experts, run_evaluation_full) calls `load_backbone(device)` instead
of constructing its own ResNet. Guarantees architecture / normalisation /
weights parity across profiling, training, calibration, and evaluation.

The loader follows three rules:
  1. The backbone is CIFAR-10-trained — uses `cifar_resnet18` with the
     checkpoint at `BACKBONE_CHECKPOINT_PATH` (default
     `models/cifar_resnet18.pt`). If the checkpoint is missing it raises
     a hard error pointing at `scripts/pretrain_cifar_backbone.py`.
  2. The returned module is wrapped in `_NormalizedBackbone` which applies
     `(x - mean) / std` internally. Callers pass pixel-space tensors in
     [0, 1] and get logits out. The wrapper is the same pattern used in
     the prior pipeline so PRISM.from_saved keeps working.
  3. The wrapper exposes the standard `layer2 / layer3 / layer4` hook
     points (forwarded from the wrapped CIFARResNet18) so activation
     extraction is unchanged.
"""
from typing import List, Optional

import torch
import torch.nn as nn

from .cifar_resnet import cifar_resnet18, CIFARResNet18
from ..config import (
    BACKBONE_MEAN, BACKBONE_STD, BACKBONE_CHECKPOINT_PATH,
    BACKBONE_NUM_CLASSES,
)


class _NormalizedBackbone(nn.Module):
    """
    Wraps a CIFAR-trained backbone so that attack code can operate in pixel
    space [0, 1] while the network internally sees the appropriately
    normalised input. Used as the attack target by ART, AutoAttack, and the
    native PyTorch CW: perturbations are bounded in pixel space, the
    wrapper applies `(x - mean) / std` inside the forward pass.

    PRISM itself receives the *unwrapped* backbone (see `load_backbone`)
    and is called with already-normalised inputs — that convention pre-
    dates this refactor and is preserved because the activation extractor
    looks up `model.layer2 / layer3 / layer4` by name via
    `named_modules()`, which only works when the resnet is the top-level
    module. Mixing the two conventions in one wrapper would require
    invasive changes to the extractor; the two-object setup is simpler.
    """

    def __init__(self, model: nn.Module,
                 mean: List[float] = BACKBONE_MEAN,
                 std:  List[float] = BACKBONE_STD):
        super().__init__()
        self._model = model
        self.register_buffer('_mean', torch.tensor(mean).view(3, 1, 1))
        self.register_buffer('_std',  torch.tensor(std).view(3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model((x - self._mean) / self._std)


def load_backbone(device: torch.device,
                  checkpoint_path: Optional[str] = None,
                  num_classes: Optional[int] = None,
                  eval_mode: bool = True,
                  wrap: bool = False) -> nn.Module:
    """
    Load the CIFAR-trained backbone and move it to *device*.

    Returned object depends on `wrap`:
      - `wrap=False` (default): returns the raw `CIFARResNet18`. This is the
        object PRISM consumes — the activation extractor walks
        `model.layer2/layer3/layer4` via `named_modules()`, and PRISM's
        forward pass receives already-normalised inputs from the data
        pipeline.
      - `wrap=True`: returns the same backbone wrapped in
        `_NormalizedBackbone`. This is the object attack code (ART, AutoAttack,
        cw_torch) consumes — perturbations are computed in pixel space,
        normalisation happens inside the model.

    Args:
        device: Target torch device.
        checkpoint_path: Path to the pretrained CIFAR ResNet-18 checkpoint.
            Defaults to `BACKBONE_CHECKPOINT_PATH` (config-driven, normally
            `models/cifar_resnet18.pt`).
        num_classes: Number of output classes. Defaults to
            `BACKBONE_NUM_CLASSES` from the active config (10 for CIFAR-10,
            100 for CIFAR-100). The checkpoint must have been trained with
            the matching head dimension.
        eval_mode: Whether to call .eval() before returning. Default True.
            Set False only for the pretraining script itself.
        wrap: See above.

    Returns:
        A `CIFARResNet18` (default) or a `_NormalizedBackbone` (`wrap=True`),
        on `device`, in eval() mode.

    Raises:
        FileNotFoundError: When the checkpoint does not exist. The error
            message tells the operator to run `pretrain_cifar_backbone.py`.
    """
    from pathlib import Path

    if num_classes is None:
        num_classes = BACKBONE_NUM_CLASSES

    ckpt = checkpoint_path or BACKBONE_CHECKPOINT_PATH
    if not Path(ckpt).exists():
        raise FileNotFoundError(
            f"CIFAR backbone checkpoint not found at '{ckpt}'.\n"
            f"  Run:  python scripts/pretrain_cifar_backbone.py\n"
            f"  This produces a CIFAR-trained ResNet-18 in ~1 hour on an "
            f"RTX 5090 (200 epochs)."
        )

    backbone = cifar_resnet18(
        num_classes=num_classes,
        checkpoint_path=ckpt,
        map_location=str(device),
    )
    backbone = backbone.to(device)
    if eval_mode:
        backbone.eval()
    if wrap:
        wrapped = _NormalizedBackbone(backbone).to(device)
        if eval_mode:
            wrapped.eval()
        return wrapped
    return backbone
