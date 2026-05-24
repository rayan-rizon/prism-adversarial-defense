"""
Pretrain the CIFAR ResNet-18 backbone for PRISM.

Standard, fully-reproducible training schedule that matches the research
norm used by Madry et al. 2018, TRADES, MART, and the RobustBench
benchmark for clean (non-adversarially-trained) baselines.

Supports both CIFAR-10 (10 classes, ~94-95% acc) and CIFAR-100 (100
classes, ~76-78% acc) via --dataset / --num-classes flags.

Schedule
========
  - Optimiser:  SGD, lr = 0.1, momentum = 0.9, nesterov = True,
                weight_decay = 5e-4
  - Schedule:   200 epochs, cosine annealing to lr = 0
  - Batch:      256
  - Augment:    RandomCrop(32, padding=4) + RandomHorizontalFlip
  - Precision:  AMP (autocast + GradScaler)
  - Loss:       CrossEntropyLoss
  - Seed:       42 (deterministic data shuffling)

Expected outcome
================
  CIFAR-10:  94.5-95.5 % test acc, ~30-45 min on RTX 5090
  CIFAR-100: 76-78 % test acc, ~30-45 min on RTX 5090

Usage
=====
    python scripts/pretrain_cifar_backbone.py
    python scripts/pretrain_cifar_backbone.py --dataset cifar100 --num-classes 100 \\
        --output models/cifar100/cifar_resnet18_c100.pt --min-test-acc 0.73

Provenance
==========
On a successful save, a sidecar JSON is written next to the checkpoint
(e.g. ``models/cifar_resnet18.acc.json``) recording the final test
accuracy, training schedule, seed, dataset, and a sha256 prefix of the
saved file. Every downstream stage (``scripts/verify_backbone_acc.py``,
``run_smoke_test.sh`` Step 0, ``run_vastai_full.sh`` Step 0a) verifies
this sidecar before doing any work. A missing or mismatched sidecar is
treated as a stale-cache event and forces a fresh training run.

The previous ``--fast`` shortcut (5 epochs, gate disabled) was removed:
it produced a 51%-acc backbone that silently passed the prior shape-only
check and poisoned every downstream artifact, collapsing detector TPR to
the FPR baseline (see logs/smoke/step0_backbone.log circa 2026-05-13).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

# ── SSL fix ─────────────────────────────────────────────────────────────
import ssl, certifi
os.environ.setdefault('SSL_CERT_FILE', certifi.where())
os.environ.setdefault('REQUESTS_CA_BUNDLE', certifi.where())
ssl._create_default_https_context = ssl.create_default_context

# Ensure src/ is importable when run from the prism/ working directory
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from src.models.cifar_resnet import cifar_resnet18
from src.config import BACKBONE_MEAN, BACKBONE_STD, BACKBONE_CHECKPOINT_PATH
from src.perf import setup_perf_flags


# ── Reproducibility ─────────────────────────────────────────────────────
def _set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Dataset channel statistics ─────────────────────────────────────────
_DATASET_STATS = {
    'cifar10':  {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2470, 0.2435, 0.2616]},
    'cifar100': {'mean': [0.5071, 0.4867, 0.4408], 'std': [0.2675, 0.2565, 0.2761]},
}


# ── Data ────────────────────────────────────────────────────────────────
def _build_dataloaders(data_root: str, batch_size: int, num_workers: int,
                       dataset: str = 'cifar10',
                       train_subset: int | None = None,
                       test_subset: int | None = None,
                       ) -> Tuple[DataLoader, DataLoader]:
    stats = _DATASET_STATS[dataset]
    normalize = T.Normalize(mean=stats['mean'], std=stats['std'])

    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        normalize,
    ])

    ds_cls = (torchvision.datasets.CIFAR10 if dataset == 'cifar10'
              else torchvision.datasets.CIFAR100)
    train_ds = ds_cls(
        root=data_root, train=True,  download=True, transform=train_tf,
    )
    test_ds  = ds_cls(
        root=data_root, train=False, download=True, transform=test_tf,
    )
    if train_subset is not None:
        train_ds = Subset(train_ds, list(range(min(train_subset, len(train_ds)))))
    if test_subset is not None:
        test_ds = Subset(test_ds, list(range(min(test_subset, len(test_ds)))))

    # persistent_workers keeps the worker processes alive between epochs,
    # avoiding the ~1-2 s fork cost per epoch. prefetch_factor lets each
    # worker stage 4 batches ahead, hiding host→device transfer behind
    # GPU compute. Both are safe and standard.
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )
    test_dl = DataLoader(
        test_ds, batch_size=512, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )
    return train_dl, test_dl


# ── Train / Eval loops ──────────────────────────────────────────────────
def _train_one_epoch(model: nn.Module, loader: DataLoader,
                     optim: torch.optim.Optimizer, device: torch.device,
                     scaler: 'torch.cuda.amp.GradScaler',
                     use_amp: bool
                     ) -> Tuple[float, float]:
    """One training epoch with optional AMP. When `use_amp` is True the
    forward+loss runs under autocast(float16) and gradients are unscaled
    by `scaler` before the optimiser step — standard mixed-precision
    pattern from torch.cuda.amp."""
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optim.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optim.step()
        loss_sum += loss.item() * x.size(0)
        correct  += (logits.argmax(1) == y).sum().item()
        total    += x.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
              use_amp: bool = False
              ) -> Tuple[float, float]:
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    ctx = (torch.amp.autocast('cuda', dtype=torch.float16)
           if use_amp else torch.amp.autocast('cuda', enabled=False))
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with ctx:
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        loss_sum += loss.item() * x.size(0)
        correct  += (logits.argmax(1) == y).sum().item()
        total    += x.size(0)
    return loss_sum / total, correct / total


# ── Main ────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description='Pretrain CIFAR ResNet-18 backbone for PRISM.'
    )
    parser.add_argument('--dataset',      default='cifar10',
                        choices=['cifar10', 'cifar100'],
                        help='Dataset to train on (default: cifar10).')
    parser.add_argument('--num-classes',  type=int,   default=None,
                        help='Number of output classes. Auto-detected from '
                             '--dataset if not specified (10 or 100).')
    parser.add_argument('--epochs',       type=int,   default=200)
    parser.add_argument('--batch-size',   type=int,   default=256)
    parser.add_argument('--lr',           type=float, default=0.1)
    parser.add_argument('--momentum',     type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--num-workers',  type=int,   default=8)
    parser.add_argument('--no-amp',       action='store_true',
                        help='Disable mixed-precision training.')
    parser.add_argument('--seed',         type=int,   default=42)
    parser.add_argument('--data-root',    default='./data')
    parser.add_argument('--output',       default=None,
                        help='Output state_dict path. Default: config-driven.')
    parser.add_argument('--min-test-acc', type=float, default=None,
                        help='Hard floor on final test accuracy. Refuses to '
                             'save below this. Default: 0.93 for CIFAR-10, '
                             '0.73 for CIFAR-100. Cannot be lowered below '
                             'these floors — the downstream detector pipeline '
                             'is statistically vacuous below them.')
    parser.add_argument('--train-subset', type=int, default=None,
                        help='Optional number of training images to use for a '
                             'bounded local smoke run.')
    parser.add_argument('--test-subset', type=int, default=None,
                        help='Optional number of test images to use for a '
                             'bounded local smoke run.')
    parser.add_argument('--allow-undertrained-smoke', action='store_true',
                        help='Smoke-only escape hatch: allow saving a checkpoint '
                             'below the publishable accuracy floor.')
    args = parser.parse_args()

    if args.num_classes is None:
        args.num_classes = 100 if args.dataset == 'cifar100' else 10
    if args.output is None:
        args.output = BACKBONE_CHECKPOINT_PATH
    _MIN_FLOOR = 0.73 if args.dataset == 'cifar100' else 0.93
    if args.min_test_acc is None:
        args.min_test_acc = _MIN_FLOOR
    if args.min_test_acc < _MIN_FLOOR and not args.allow_undertrained_smoke:
        # We deliberately do not allow loosening below the publishable floor.
        # An undertrained backbone makes attacks ill-defined and collapses
        # detector TPR to the FPR baseline — see plan §"Why this collapses
        # TPR — and not FPR".
        print(
            f'ERROR: --min-test-acc {args.min_test_acc:.4f} is below the '
            f'publishable floor {_MIN_FLOOR:.4f} for {args.dataset}. '
            f'Refusing to weaken the gate.', flush=True,
        )
        return 2

    stats = _DATASET_STATS[args.dataset]
    _set_seed(args.seed)
    setup_perf_flags(verbose=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = (not args.no_amp) and (device.type == 'cuda')
    print(f'Device: {device}')
    print(f'Dataset:   {args.dataset} ({args.num_classes} classes)')
    print(f'Backbone:  cifar_resnet18 (CIFAR-adapted: 3x3 stem, no maxpool)')
    print(f'Train:     {args.epochs} epochs, batch={args.batch_size}, '
          f'lr={args.lr}→0 cosine, momentum={args.momentum}, wd={args.weight_decay}')
    print(f'Augment:   RandomCrop(32, pad=4) + RandomHorizontalFlip')
    print(f'Normalize: mean={stats["mean"]}, std={stats["std"]}')
    print(f'Precision: {"AMP (FP16)" if use_amp else "FP32"},  num_workers={args.num_workers}')

    train_dl, test_dl = _build_dataloaders(
        args.data_root, args.batch_size, args.num_workers,
        dataset=args.dataset,
        train_subset=args.train_subset,
        test_subset=args.test_subset,
    )
    print(f'Loaded {args.dataset.upper()}: train={len(train_dl.dataset)}, '
          f'test={len(test_dl.dataset)}')

    model = cifar_resnet18(num_classes=args.num_classes).to(device)
    optim = torch.optim.SGD(
        model.parameters(),
        lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay, nesterov=True,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs, eta_min=0.0,
    )
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_acc = 0.0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.time()
        train_loss, train_acc = _train_one_epoch(model, train_dl, optim, device,
                                                  scaler=scaler, use_amp=use_amp)
        test_loss,  test_acc  = _evaluate(model, test_dl, device, use_amp=use_amp)
        sched.step()
        ep_dt = time.time() - ep_t0
        print(
            f'[{epoch:3d}/{args.epochs}]  '
            f'lr={sched.get_last_lr()[0]:.4f}  '
            f'train_loss={train_loss:.4f} acc={train_acc:.4f}  '
            f'test_loss={test_loss:.4f} acc={test_acc:.4f}  '
            f'({ep_dt:.1f}s)',
            flush=True,
        )
        best_acc = max(best_acc, test_acc)

    total_dt = time.time() - t0
    print(f'\nFinal test accuracy: {test_acc:.4f}  (best across epochs: {best_acc:.4f})')
    print(f'Total wall-clock: {total_dt/60:.1f} min')

    if test_acc < args.min_test_acc and not args.allow_undertrained_smoke:
        print(
            f'\nREFUSING TO SAVE: final test accuracy {test_acc:.4f} < '
            f'gate {args.min_test_acc:.4f}. Inspect the training curve and '
            f'increase --epochs or --lr-tuning if needed.',
            flush=True,
        )
        return 1
    if test_acc < args.min_test_acc:
        print(
            f'\n[WARN] Saving undertrained smoke backbone: acc {test_acc:.4f} < '
            f'publishable gate {args.min_test_acc:.4f}. '
            f'This checkpoint is for local integration smoke only.',
            flush=True,
        )

    # Save raw state_dict (most portable form for downstream load_backbone()).
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f'\n[OK] Saved backbone -> {output_path}  ({size_mb:.1f} MB)')
    print(f'     final test accuracy: {test_acc:.4f}')

    # Provenance sidecar — required by every downstream gate. Hashing the
    # written file (rather than the state_dict bytes) means a hand-edited or
    # truncated checkpoint produces a SHA mismatch and the gate fails closed.
    h = hashlib.sha256()
    with open(output_path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    sidecar_path = output_path.with_suffix('.acc.json')
    sidecar = {
        'test_acc':         round(float(test_acc), 6),
        'best_test_acc':    round(float(best_acc), 6),
        'epochs':           int(args.epochs),
        'batch_size':       int(args.batch_size),
        'lr':               float(args.lr),
        'momentum':         float(args.momentum),
        'weight_decay':     float(args.weight_decay),
        'seed':             int(args.seed),
        'dataset':          args.dataset,
        'num_classes':      int(args.num_classes),
        'min_test_acc_gate': float(args.min_test_acc),
        'sha256_first16':   h.hexdigest()[:16],
        'checkpoint':       str(output_path),
        'recipe_version':   'madry2018-cifar-resnet18-v1',
        'train_subset':     args.train_subset,
        'test_subset':      args.test_subset,
        'allow_undertrained_smoke': bool(args.allow_undertrained_smoke),
    }
    with open(sidecar_path, 'w') as f:
        json.dump(sidecar, f, indent=2, sort_keys=True)
    print(f'[OK] Wrote provenance sidecar -> {sidecar_path}')
    print(f'     sha256_first16={sidecar["sha256_first16"]}  '
          f'recipe={sidecar["recipe_version"]}')
    print(f'\nNext step: python scripts/build_profile_testset.py')
    return 0


if __name__ == '__main__':
    sys.exit(main())
