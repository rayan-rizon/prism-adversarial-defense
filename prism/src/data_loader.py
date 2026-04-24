"""
PRISM dataset dispatcher — routes torchvision loaders based on DATASET
from the active config (default.yaml → cifar10, cifar100.yaml → cifar100).

Scripts should replace direct calls like `datasets.CIFAR10(...)` with
`load_test_dataset(...)` so a single --config flip switches datasets.
"""
from typing import Optional

import torchvision
import torchvision.transforms as transforms

from src.config import DATASET, IMAGENET_MEAN, IMAGENET_STD


_NORMALIZE = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
_DEFAULT_TEST_TRANSFORM = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    _NORMALIZE,
])


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
        transform: Image transform pipeline. Defaults to 224-resize + ImageNet normalize.
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
