"""
PRISM dataset dispatcher — routes torchvision loaders based on DATASET
from the active config (default.yaml → cifar10, cifar100.yaml → cifar100).

Scripts should replace direct calls like `datasets.CIFAR10(...)` with
`load_test_dataset(...)` so a single --config flip switches datasets.

The default transforms match the CIFAR-10-trained backbone (32x32 native,
CIFAR-10 channel statistics). The legacy 224x224 ImageNet path is kept
behind the `BACKBONE_INPUT_SIZE` config knob for the rare case where the
ImageNet evaluation track is rerun.
"""
from typing import Optional

import torchvision
import torchvision.transforms as transforms

from src.config import (
    DATASET,
    BACKBONE_MEAN, BACKBONE_STD,
    BACKBONE_INPUT_SIZE,
    # Backward-compat re-exports for scripts still importing the old names.
    IMAGENET_MEAN, IMAGENET_STD,  # noqa: F401  -- aliased in src.config
)


_NORMALIZE = transforms.Normalize(mean=BACKBONE_MEAN, std=BACKBONE_STD)


def _build_test_transforms():
    """
    Build the canonical (normalised, pixel-space) test-time transform pair.

    Most callers want `_DEFAULT_TEST_TRANSFORM` (returns a tensor that has
    already had `(x - mean) / std` applied — directly model-ready). Attack
    code calls `_DEFAULT_PIXEL_TRANSFORM` instead to get a [0, 1] pixel-space
    tensor, then relies on `_NormalizedBackbone` to apply the normalisation
    inside the model forward pass. Externalising the normalisation is what
    makes the C&W L2 budget honest (perturbations are bounded in pixel
    space, not in pre-normalised space).
    """
    if BACKBONE_INPUT_SIZE != 32:
        # ImageNet-style track — resize first.
        return (
            transforms.Compose([
                transforms.Resize(BACKBONE_INPUT_SIZE),
                transforms.ToTensor(),
                _NORMALIZE,
            ]),
            transforms.Compose([
                transforms.Resize(BACKBONE_INPUT_SIZE),
                transforms.ToTensor(),
            ]),
        )
    # CIFAR-10 native — no resize.
    return (
        transforms.Compose([
            transforms.ToTensor(),
            _NORMALIZE,
        ]),
        transforms.Compose([
            transforms.ToTensor(),
        ]),
    )


_DEFAULT_TEST_TRANSFORM, _DEFAULT_PIXEL_TRANSFORM = _build_test_transforms()


def _resolve_class(dataset: str):
    key = (dataset or 'cifar10').lower()
    if key == 'cifar10':
        return torchvision.datasets.CIFAR10
    if key == 'cifar100':
        return torchvision.datasets.CIFAR100
    raise ValueError(f"Unsupported dataset '{dataset}'. Expected 'cifar10' or 'cifar100'.")


def load_test_dataset(root: str = './data',
                       download: bool = True,
                       transform: Optional[transforms.Compose] = None,
                       dataset: Optional[str] = None):
    """Return the torchvision test split for the active config's dataset.

    Args:
        root: Dataset root directory. Default './data' (matches Vast.ai).
        download: Auto-download if missing.
        transform: Image transform pipeline. Defaults to native-resolution
            CIFAR transform with backbone-correct normalization.
        dataset: Override the DATASET constant (rarely needed).
    """
    ds_cls = _resolve_class(dataset or DATASET)
    return ds_cls(root=root, train=False, download=download,
                  transform=transform or _DEFAULT_TEST_TRANSFORM)


def load_train_dataset(root: str = './data',
                        download: bool = True,
                        transform: Optional[transforms.Compose] = None,
                        dataset: Optional[str] = None):
    """Training split counterpart to load_test_dataset()."""
    ds_cls = _resolve_class(dataset or DATASET)
    return ds_cls(root=root, train=True, download=download,
                  transform=transform or _DEFAULT_TEST_TRANSFORM)
