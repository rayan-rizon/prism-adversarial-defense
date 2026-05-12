"""
CIFAR-10-adapted ResNet-18.

Standard CIFAR ResNet-18, as used by Madry et al. 2018 ("Towards Deep
Learning Models Resistant to Adversarial Attacks"), TRADES, MART, and the
RobustBench benchmark. Differences from the torchvision ImageNet ResNet-18:

  1. First conv: 3x3 stride=1 (was 7x7 stride=2)
  2. No max-pool after first conv (was 3x3 stride=2 maxpool)
  3. Final FC: 512 -> 10 (was 512 -> 1000)
  4. Input expects 32x32 RGB (was 224x224)

The intermediate-block names match torchvision exactly so all downstream
TDA / activation-extraction code keeps working unchanged:
  conv1 -> bn1 -> relu -> [layer1] -> layer2 -> layer3 -> layer4 -> avgpool -> fc

Training reference (matches the published research norm):
  - Optimiser: SGD, lr=0.1, momentum=0.9, weight_decay=5e-4
  - Schedule: 200 epochs, cosine annealing to lr=0
  - Augmentation: RandomCrop(32, padding=4) + RandomHorizontalFlip
  - Expected clean test accuracy: 94-95%

Citation: He et al. 2016, "Deep Residual Learning for Image Recognition",
adapted for CIFAR per the original paper's §4.2 (CIFAR-10 experiments).
"""
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    """Standard ResNet basic block (2 convs + identity skip), as in He et al. 2016."""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = _conv3x3(in_planes, planes, stride)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2   = nn.BatchNorm2d(planes)

        self.shortcut: nn.Module
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class CIFARResNet18(nn.Module):
    """
    CIFAR-10 ResNet-18.  Layer naming matches torchvision so existing
    `layer2 / layer3 / layer4` hook points work unchanged.

    Input shape: (B, 3, 32, 32), pixel values in [0, 1] (un-normalised).
    Normalisation is applied externally by `_NormalizedBackbone`.

    Args:
        num_classes: Output dimension. 10 for CIFAR-10 (default), 100 for CIFAR-100.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.in_planes = 64

        # ── Stem ─────────────────────────────────────────────────────────────
        # 3x3 stride=1 instead of ImageNet 7x7 stride=2; no maxpool. Preserves
        # 32x32 spatial resolution into layer1, so the network has enough depth
        # before the first spatial downsample.
        self.conv1 = _conv3x3(3, 64, stride=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.maxpool = nn.Identity()  # placeholder: keeps the public API parity with torchvision

        # ── Residual stages ─────────────────────────────────────────────────
        # ResNet-18 layout: 2 BasicBlocks per stage, channels 64/128/256/512.
        self.layer1 = self._make_layer( 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(512, num_blocks=2, stride=2)

        # ── Head ─────────────────────────────────────────────────────────────
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(512 * BasicBlock.expansion, num_classes)

        # Kaiming init (standard for ResNet)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers: List[nn.Module] = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.maxpool(out)         # no-op; preserved for hook parity
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def cifar_resnet18(num_classes: int = 10,
                   checkpoint_path: Optional[str] = None,
                   map_location: str = 'cpu') -> CIFARResNet18:
    """
    Construct a CIFAR ResNet-18, optionally loading a pretrained checkpoint.

    Args:
        num_classes: Output classes (10 for CIFAR-10, 100 for CIFAR-100).
        checkpoint_path: Optional path to a state_dict produced by
            `scripts/pretrain_cifar_backbone.py`. When None, the model is
            returned with Kaiming-initialised random weights (training mode).
        map_location: Device hint for torch.load. Use 'cpu' or 'cuda:0'.

    Returns:
        A CIFARResNet18 module in eval() mode when a checkpoint is loaded,
        otherwise in train() mode (uninitialised weights).
    """
    model = CIFARResNet18(num_classes=num_classes)
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location=map_location)
        # Support both raw state_dict and dict-wrapped checkpoints
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        # Strip optional "module." prefix from DataParallel-saved checkpoints
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
        model.load_state_dict(state)
        model.eval()
    return model
