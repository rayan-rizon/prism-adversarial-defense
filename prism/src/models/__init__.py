"""
PRISM models module.

Contains the CIFAR-10-adapted ResNet-18 backbone and a centralised loader.
All pipeline scripts (build_profile, train_ensemble_scorer, calibrate, eval)
import the backbone from here — never from torchvision directly — so the
model used at training, profiling, calibration, and evaluation time is
guaranteed identical.

This eliminates the prior ImageNet-pretrained ResNet-18 / CIFAR-10 data
mismatch that produced 41% top-1 clean confidence and a ~7% C&W TPR ceiling.
With a CIFAR-10-trained backbone, clean-data confidence is ~95% and the
detection features (softmax entropy, DCT energy, TDA persistence statistics)
operate in their designed signal regime.
"""

from .cifar_resnet import CIFARResNet18, cifar_resnet18
from .backbone import load_backbone, _NormalizedBackbone

__all__ = [
    'CIFARResNet18',
    'cifar_resnet18',
    'load_backbone',
    '_NormalizedBackbone',
]
