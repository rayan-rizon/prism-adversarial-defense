"""
Pretrain WRN-28-10 backbone on CIFAR-10 for PRISM Stretch A.

Mirrors pretrain_cifar_backbone.py exactly (same schedule, same gates,
same provenance sidecar format) but uses CIFARWideResNet instead of
CIFARResNet18 and writes to models/cifar_wrn28_10.pt by default.

Schedule (same as ResNet-18 to enable fair comparison):
  - Optimiser:  SGD, lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4
  - Schedule:   200 epochs, cosine annealing to lr=0
  - Batch:      128 (WRN-28-10 is 3× heavier; 128 fits safely in 48 GB)
  - Augment:    RandomCrop(32, padding=4) + RandomHorizontalFlip
  - Precision:  AMP (autocast + GradScaler)
  - Seed:       42

Expected outcome:
  CIFAR-10:  95.5–96.1 % test acc, ~40–55 min on A6000/A100

Usage:
    python scripts/pretrain_wrn_backbone.py
    python scripts/pretrain_wrn_backbone.py --output models/cifar_wrn28_10.pt

Output: models/cifar_wrn28_10.pt  +  models/cifar_wrn28_10.acc.json
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

# SSL fix (same as pretrain_cifar_backbone.py)
import ssl, certifi
os.environ.setdefault('SSL_CERT_FILE', certifi.where())
os.environ.setdefault('REQUESTS_CA_BUNDLE', certifi.where())
ssl._create_default_https_context = ssl.create_default_context

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from src.models.cifar_wrn import cifar_wrn28_10
from src.perf import setup_perf_flags

_CIFAR10_STATS = {'mean': [0.4914, 0.4822, 0.4465],
                  'std':  [0.2470, 0.2435, 0.2616]}
_DEFAULT_OUTPUT = 'models/cifar_wrn28_10.pt'
_MIN_FLOOR      = 0.93   # same floor as ResNet-18; WRN-28-10 should hit 95.5%+


def _set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_dataloaders(data_root: str, batch_size: int,
                       num_workers: int) -> Tuple[DataLoader, DataLoader]:
    stats = _CIFAR10_STATS
    normalize = T.Normalize(mean=stats['mean'], std=stats['std'])
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])
    test_tf = T.Compose([T.ToTensor(), normalize])

    train_ds = torchvision.datasets.CIFAR10(
        root=data_root, train=True,  download=True, transform=train_tf)
    test_ds  = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=test_tf)

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )
    test_dl = DataLoader(
        test_ds, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )
    return train_dl, test_dl


def _train_one_epoch(model, loader, optim, device, scaler,
                     use_amp) -> Tuple[float, float]:
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optim.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = model(x)
                loss   = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            logits = model(x)
            loss   = F.cross_entropy(logits, y)
            loss.backward()
            optim.step()
        loss_sum += loss.item() * x.size(0)
        correct  += (logits.argmax(1) == y).sum().item()
        total    += x.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def _evaluate(model, loader, device, use_amp=False) -> Tuple[float, float]:
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    ctx = (torch.amp.autocast('cuda', dtype=torch.float16)
           if use_amp else torch.amp.autocast('cuda', enabled=False))
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with ctx:
            logits = model(x)
            loss   = F.cross_entropy(logits, y)
        loss_sum += loss.item() * x.size(0)
        correct  += (logits.argmax(1) == y).sum().item()
        total    += x.size(0)
    return loss_sum / total, correct / total


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Pretrain WRN-28-10 backbone for PRISM Stretch A.')
    parser.add_argument('--epochs',       type=int,   default=200)
    parser.add_argument('--batch-size',   type=int,   default=128,
                        help='Default 128 (WRN is 3× heavier than ResNet-18).')
    parser.add_argument('--lr',           type=float, default=0.1)
    parser.add_argument('--momentum',     type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--num-workers',  type=int,   default=6)
    parser.add_argument('--no-amp',       action='store_true')
    parser.add_argument('--seed',         type=int,   default=42)
    parser.add_argument('--data-root',    default='./data')
    parser.add_argument('--output',       default=_DEFAULT_OUTPUT)
    parser.add_argument('--min-test-acc', type=float, default=None,
                        help=f'Hard floor. Default: {_MIN_FLOOR}.')
    args = parser.parse_args()

    if args.min_test_acc is None:
        args.min_test_acc = _MIN_FLOOR
    if args.min_test_acc < _MIN_FLOOR:
        print(f'ERROR: --min-test-acc {args.min_test_acc} is below the '
              f'publishable floor {_MIN_FLOOR}.', flush=True)
        return 2

    _set_seed(args.seed)
    setup_perf_flags(verbose=True)
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = (not args.no_amp) and (device.type == 'cuda')
    stats   = _CIFAR10_STATS

    print(f'Device:    {device}')
    print(f'Backbone:  WRN-28-10  (~36.5 M params)')
    print(f'Dataset:   CIFAR-10 (10 classes)')
    print(f'Train:     {args.epochs} epochs, batch={args.batch_size}, '
          f'lr={args.lr}→0 cosine, momentum={args.momentum}, wd={args.weight_decay}')
    print(f'Augment:   RandomCrop(32, pad=4) + RandomHorizontalFlip')
    print(f'Normalize: mean={stats["mean"]}, std={stats["std"]}')
    print(f'Precision: {"AMP (FP16)" if use_amp else "FP32"},  '
          f'num_workers={args.num_workers}', flush=True)

    train_dl, test_dl = _build_dataloaders(
        args.data_root, args.batch_size, args.num_workers)
    print(f'Loaded CIFAR-10: train={len(train_dl.dataset)}, '
          f'test={len(test_dl.dataset)}', flush=True)

    model  = cifar_wrn28_10(num_classes=10).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'WRN-28-10 parameters: {n_params/1e6:.2f} M', flush=True)

    optim  = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay, nesterov=True)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs, eta_min=0.0)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_acc = 0.0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.time()
        train_loss, train_acc = _train_one_epoch(
            model, train_dl, optim, device, scaler, use_amp)
        test_loss,  test_acc  = _evaluate(model, test_dl, device, use_amp)
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
    print(f'\nFinal test accuracy: {test_acc:.4f}  '
          f'(best across epochs: {best_acc:.4f})')
    print(f'Total wall-clock: {total_dt/60:.1f} min')

    if test_acc < args.min_test_acc:
        print(
            f'\nREFUSING TO SAVE: final test accuracy {test_acc:.4f} < '
            f'gate {args.min_test_acc:.4f}.',
            flush=True,
        )
        return 1

    # Save state_dict + provenance sidecar.
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f'\n[OK] Saved WRN-28-10 -> {output_path}  ({size_mb:.1f} MB)')
    print(f'     final test accuracy: {test_acc:.4f}')

    h = hashlib.sha256()
    with open(output_path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    sidecar_path = output_path.with_suffix('.acc.json')
    sidecar = {
        'test_acc':          round(float(test_acc), 6),
        'best_test_acc':     round(float(best_acc), 6),
        'epochs':            int(args.epochs),
        'batch_size':        int(args.batch_size),
        'lr':                float(args.lr),
        'momentum':          float(args.momentum),
        'weight_decay':      float(args.weight_decay),
        'seed':              int(args.seed),
        'dataset':           'cifar10',
        'num_classes':       10,
        'min_test_acc_gate': float(args.min_test_acc),
        'sha256_first16':    h.hexdigest()[:16],
        'checkpoint':        str(output_path),
        'recipe_version':    'madry2018-cifar-wrn28x10-v1',
        'arch':              'wrn28_10',
    }
    with open(sidecar_path, 'w') as f:
        json.dump(sidecar, f, indent=2, sort_keys=True)
    print(f'[OK] Wrote provenance sidecar -> {sidecar_path}')
    print(f'     sha256_first16={sidecar["sha256_first16"]}  '
          f'recipe={sidecar["recipe_version"]}')
    print(f'\nNext step: PRISM_CONFIG=configs/wrn_cifar10.yaml '
          f'python scripts/build_profile_testset.py')
    return 0


if __name__ == '__main__':
    sys.exit(main())
