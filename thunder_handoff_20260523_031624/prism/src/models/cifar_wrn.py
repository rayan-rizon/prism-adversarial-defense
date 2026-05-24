"""
WideResNet-28-10 for CIFAR-32x32 inputs (Zagoruyko & Komodakis 2016).

Used by PRISM Stretch A as a second backbone architecture to validate that
the TDA/conformal detection pipeline generalises beyond ResNet-18.

Layer naming is designed so PRISM's ActivationExtractor can hook into the
same depth positions as ResNet-18 via configs/wrn_cifar10.yaml:
  layer1 → widths 16→160, 32×32 spatial  (≈ ResNet layer2 role)
  layer2 → widths 160→320, 16×16 spatial (≈ ResNet layer3 role)
  layer3 → widths 320→640, 8×8 spatial   (≈ ResNet layer4 role)

Expected clean test accuracy on CIFAR-10: 95.5–96.1% (standard result for
WRN-28-10 with cosine-annealing, no dropout, no cutout).

Usage (factory function mirrors cifar_resnet18 signature):
    from src.models.cifar_wrn import cifar_wrn28_10
    model = cifar_wrn28_10(num_classes=10)                       # random init
    model = cifar_wrn28_10(checkpoint_path='models/cifar_wrn28_10.pt')
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ────────────────────────────────────────────────────────────

class _WRNBasicBlock(nn.Module):
    """Pre-activation basic block (Zagoruyko & Komodakis 2016, WRN paper)."""

    def __init__(self, in_planes: int, out_planes: int, stride: int = 1,
                 dropout_rate: float = 0.0):
        super().__init__()
        self.bn1   = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.dropout = (nn.Dropout(p=dropout_rate) if dropout_rate > 0.0
                        else nn.Identity())
        # Projection shortcut when dimensions change.
        if stride != 1 or in_planes != out_planes:
            self.shortcut: nn.Module = nn.Conv2d(
                in_planes, out_planes, kernel_size=1, stride=stride, bias=False
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-activation: BN→ReLU before the first conv.
        out = F.relu(self.bn1(x), inplace=True)
        # Projection shortcut reads from the pre-activated tensor (standard WRN).
        sc  = self.shortcut(out) if not isinstance(self.shortcut, nn.Identity) else x
        out = self.conv1(out)
        out = self.dropout(F.relu(self.bn2(out), inplace=True))
        out = self.conv2(out)
        return out + sc


# ── Network ────────────────────────────────────────────────────────────────────

class CIFARWideResNet(nn.Module):
    """
    WRN-depth-widen for CIFAR 32×32 inputs.

    Default: depth=28, widen_factor=10 → WRN-28-10 with 36.5M parameters.
    Layer naming (`layer1 / layer2 / layer3`) is compatible with PRISM's
    ActivationExtractor when used with `configs/wrn_cifar10.yaml`.

    Input: (B, 3, 32, 32) — already normalised by BACKBONE_MEAN/STD.
    """

    def __init__(self, depth: int = 28, widen_factor: int = 10,
                 num_classes: int = 10, dropout_rate: float = 0.0):
        super().__init__()
        assert (depth - 4) % 6 == 0, 'WRN depth must satisfy (depth-4) % 6 == 0'
        n = (depth - 4) // 6
        k = widen_factor
        widths = [16, 16 * k, 32 * k, 64 * k]   # [16, 160, 320, 640] for k=10

        # Stem — single 3×3 conv, preserves 32×32 spatial resolution.
        self.conv1 = nn.Conv2d(3, widths[0], kernel_size=3, stride=1,
                               padding=1, bias=False)

        # Three residual stages with widening.  Names match the config hook points.
        self.layer1 = self._make_layer(widths[0], widths[1], n, stride=1,
                                       dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(widths[1], widths[2], n, stride=2,
                                       dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(widths[2], widths[3], n, stride=2,
                                       dropout_rate=dropout_rate)

        # Final BN + pooling + classifier.
        self.bn   = nn.BatchNorm2d(widths[3])
        self.fc   = nn.Linear(widths[3], num_classes)

        # Kaiming init (same convention as CIFARResNet18).
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_planes: int, out_planes: int, num_blocks: int,
                    stride: int, dropout_rate: float) -> nn.Sequential:
        layers: List[nn.Module] = []
        for i in range(num_blocks):
            layers.append(_WRNBasicBlock(
                in_planes  if i == 0 else out_planes,
                out_planes,
                stride     if i == 0 else 1,
                dropout_rate=dropout_rate,
            ))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn(out), inplace=True)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


# ── Factory ────────────────────────────────────────────────────────────────────

def cifar_wrn28_10(num_classes: int = 10,
                   checkpoint_path: Optional[str] = None,
                   map_location: str = 'cpu') -> CIFARWideResNet:
    """
    Construct a CIFAR WRN-28-10, optionally loading a pretrained checkpoint.

    Signature mirrors `cifar_resnet18` so `backbone.py` can dispatch to either
    with identical call syntax.

    Args:
        num_classes:     Output classes (10 for CIFAR-10, 100 for CIFAR-100).
        checkpoint_path: Optional path to a state_dict produced by
                         `scripts/pretrain_wrn_backbone.py`.
        map_location:    Device hint for torch.load ('cpu' or 'cuda:0').

    Returns:
        A CIFARWideResNet in eval() mode when a checkpoint is loaded,
        otherwise in train() mode with Kaiming-initialised weights.
    """
    model = CIFARWideResNet(depth=28, widen_factor=10,
                            num_classes=num_classes)
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location=map_location,
                           weights_only=True)
        # Support both raw state_dict and dict-wrapped checkpoints.
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        # Strip optional "module." prefix from DataParallel-saved checkpoints.
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
        model.load_state_dict(state)
        model.eval()
    return model
