"""
PRISM dataset dispatcher — routes torchvision loaders based on DATASET
from the active config (default.yaml → cifar10, cifar100.yaml → cifar100).

Scripts should replace direct calls like `datasets.CIFAR10(...)` with
`load_test_dataset(...)` so a single --config flip switches datasets.

The default transforms match the active CIFAR-trained backbone (32x32 native,
dataset-specific channel statistics). Non-32x32 configs can still request
resizing through the `BACKBONE_INPUT_SIZE` config knob.
"""
from typing import Optional

import torch
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
    if DATASET == 'imagenet':
        # ImageNet-grade inputs: standard Resize(256) -> CenterCrop(224) so the
        # tensor is square (a bare Resize(224) would leave the long side
        # uncropped). Normalised view for the model, pixel [0,1] view for
        # attack code (perturbation budget stays honest in pixel space).
        resize = max(BACKBONE_INPUT_SIZE + 32, 256)
        return (
            transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(BACKBONE_INPUT_SIZE),
                transforms.ToTensor(),
                _NORMALIZE,
            ]),
            transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(BACKBONE_INPUT_SIZE),
                transforms.ToTensor(),
            ]),
        )
    if BACKBONE_INPUT_SIZE != 32:
        # Non-native-size config — resize first.
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


# ── ImageNet-100 pooled-ImageFolder dispatch (Exp 2, 224x224 scaling run) ─────
#
# ImageNet does not ship as one flat indexable test file, so the CIFAR
# split-by-flat-index convention (profile [0,5000), cal [5000,7000),
# val [7000,8000), eval [8000,10000)) has nothing to index into. We bridge
# the gap with ONE deterministic pool: take an ImageFolder over the staged
# ImageNet-100 directory, apply a fixed-seed permutation, and expose the
# permuted view to every script. Flat indices are then globally disjoint
# clean images that each span all 100 classes, so:
#   * profile / cal / val / eval ranges stay disjoint by construction;
#   * eval is a held-out slice of the same pool -- identical in spirit to the
#     disclosed CIFAR protocol (all splits carved from one pool, disjoint
#     indices), not weaker.
# The pool directory and permutation seed are read from the active config's
# data.imagenet_dir / data.imagenet_pool_seed keys. The 100-class subset and
# its 0..99 labels come straight from the ImageFolder's sorted class dirs, so
# they line up with the fine-tuned 100-way head from
# scripts/pretrain_imagenet100_backbone.py -- no wnid remapping needed.

_IMAGENET_POOL_CACHE = {}


class _PermutedImageFolder(torch.utils.data.Dataset):
    """ImageFolder viewed through a fixed-seed permutation, transform applied."""

    def __init__(self, root: str, transform, seed: int):
        self._base = torchvision.datasets.ImageFolder(root)
        g = torch.Generator().manual_seed(int(seed))
        self._perm = torch.randperm(len(self._base), generator=g).tolist()
        self._transform = transform
        self.classes = self._base.classes

    def __len__(self):
        return len(self._perm)

    def __getitem__(self, idx):
        img, label = self._base[self._perm[idx]]
        if self._transform is not None:
            img = self._transform(img)
        return img, label


def _imagenet_pool(transform, train: bool):
    from src.config import _CFG  # active config dict
    data_cfg = _CFG.get('data', {}) if isinstance(_CFG, dict) else {}
    root = data_cfg.get('imagenet_dir')
    if not root:
        raise ValueError(
            "data.imagenet_dir is not set in the active config. Point it at the "
            "staged ImageNet-100 directory (ImageFolder layout: one subdir per "
            "class). See configs/imagenet.yaml and REVISION_EXP2_IMAGENET_RUNBOOK.md."
        )
    seed = int(data_cfg.get('imagenet_pool_seed', 1234))
    key = (root, seed, id(transform))
    if key not in _IMAGENET_POOL_CACHE:
        _IMAGENET_POOL_CACHE[key] = _PermutedImageFolder(root, transform, seed)
    return _IMAGENET_POOL_CACHE[key]


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
    key = (dataset or DATASET).lower()
    if key == 'imagenet':
        return _imagenet_pool(transform or _DEFAULT_TEST_TRANSFORM, train=False)
    ds_cls = _resolve_class(key)
    return ds_cls(root=root, train=False, download=download,
                  transform=transform or _DEFAULT_TEST_TRANSFORM)


def load_train_dataset(root: str = './data',
                        download: bool = True,
                        transform: Optional[transforms.Compose] = None,
                        dataset: Optional[str] = None):
    """Training split counterpart to load_test_dataset().

    For ImageNet the 'train' and 'test' loaders return the SAME deterministic
    permuted pool (see `_imagenet_pool`): the flat split indices already carve
    disjoint profile/cal/val/eval slices out of it, so a separate train file is
    neither needed nor available.
    """
    key = (dataset or DATASET).lower()
    if key == 'imagenet':
        return _imagenet_pool(transform or _DEFAULT_TEST_TRANSFORM, train=True)
    ds_cls = _resolve_class(key)
    return ds_cls(root=root, train=True, download=download,
                  transform=transform or _DEFAULT_TEST_TRANSFORM)
