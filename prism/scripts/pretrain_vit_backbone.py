"""
Fine-tune a non-CNN ViT-B/16 backbone for PRISM.

This is the architecture-agnostic extension path: torchvision ViT-B/16 is
fine-tuned on CIFAR and saved as a plain state_dict that src.models.load_backbone
can reload when model.arch=vit_b_16.

Recommended Vast.ai run:
    python scripts/pretrain_vit_backbone.py --dataset cifar10 --epochs 20 \
        --batch-size 96 --lr 3e-5 --output models/vit_cifar10/vit_b16_cifar10.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import ssl
import sys
import time
from pathlib import Path
from typing import Tuple

import certifi
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T
from torchvision.models import ViT_B_16_Weights, vit_b_16

os.environ.setdefault('SSL_CERT_FILE', certifi.where())
os.environ.setdefault('REQUESTS_CA_BUNDLE', certifi.where())
ssl._create_default_https_context = ssl.create_default_context

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from src.perf import setup_perf_flags


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_model(num_classes: int, weights_name: str) -> nn.Module:
    weights = None
    if weights_name == 'imagenet':
        weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model


def _build_dataloaders(
    data_root: str,
    dataset: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    train_subset: int | None,
    test_subset: int | None,
) -> Tuple[DataLoader, DataLoader]:
    normalize = T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
    train_tf = T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.70, 1.00)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])
    test_tf = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        normalize,
    ])
    ds_cls = torchvision.datasets.CIFAR100 if dataset == 'cifar100' else torchvision.datasets.CIFAR10
    train_ds = ds_cls(root=data_root, train=True, download=True, transform=train_tf)
    test_ds = ds_cls(root=data_root, train=False, download=True, transform=test_tf)
    if train_subset is not None:
        train_ds = Subset(train_ds, list(range(min(train_subset, len(train_ds)))))
    if test_subset is not None:
        test_ds = Subset(test_ds, list(range(min(test_subset, len(test_ds)))))

    common = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **common)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)
    return train_dl, test_dl


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
) -> Tuple[float, float]:
    model.train()
    total = correct = 0
    loss_sum = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
        loss_sum += float(loss.item()) * x.size(0)
        correct += int((logits.argmax(1) == y).sum().item())
        total += int(y.numel())
    return loss_sum / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool) -> Tuple[float, float]:
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    ctx = torch.amp.autocast('cuda', dtype=torch.float16) if use_amp else torch.amp.autocast('cuda', enabled=False)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with ctx:
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        loss_sum += float(loss.item()) * x.size(0)
        correct += int((logits.argmax(1) == y).sum().item())
        total += int(y.numel())
    return loss_sum / max(total, 1), correct / max(total, 1)


def _sha256_first16(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:16]


def main() -> int:
    parser = argparse.ArgumentParser(description='Fine-tune torchvision ViT-B/16 for PRISM.')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--num-classes', type=int, default=None)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=96)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--weight-decay', type=float, default=5e-2)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data-root', default='./data')
    parser.add_argument('--output', default='models/vit_cifar10/vit_b16_cifar10.pt')
    parser.add_argument('--weights', default='imagenet', choices=['imagenet', 'none'])
    parser.add_argument('--freeze-encoder', action='store_true',
                        help='Train only the classification head. Useful for a quick smoke run, not paper results.')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--min-test-acc', type=float, default=None)
    parser.add_argument('--train-subset', type=int, default=None)
    parser.add_argument('--test-subset', type=int, default=None)
    parser.add_argument('--allow-undertrained-smoke', action='store_true')
    args = parser.parse_args()

    if args.num_classes is None:
        args.num_classes = 100 if args.dataset == 'cifar100' else 10
    floor = 0.90 if args.dataset == 'cifar10' else 0.70
    if args.min_test_acc is None:
        args.min_test_acc = floor
    if args.min_test_acc < floor and not args.allow_undertrained_smoke:
        print(f'ERROR: --min-test-acc {args.min_test_acc:.4f} is below publishable floor {floor:.4f}.')
        return 2

    _set_seed(args.seed)
    setup_perf_flags(verbose=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = (not args.no_amp) and device.type == 'cuda'
    print(f'Device: {device}')
    print(f'Backbone: ViT-B/16 ({args.weights} weights), image_size={args.image_size}')
    print(f'Dataset: {args.dataset}, classes={args.num_classes}')
    print(f'Train: epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}, wd={args.weight_decay}')
    print(f'Precision: {"AMP" if use_amp else "FP32"}, workers={args.num_workers}')

    train_dl, test_dl = _build_dataloaders(
        args.data_root, args.dataset, args.image_size, args.batch_size,
        args.num_workers, args.train_subset, args.test_subset,
    )
    model = _build_model(args.num_classes, args.weights).to(device)
    if args.freeze_encoder:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith('heads.')

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_acc = 0.0
    best_state = None
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.time()
        train_loss, train_acc = _train_one_epoch(model, train_dl, optimizer, device, scaler, use_amp)
        test_loss, test_acc = _evaluate(model, test_dl, device, use_amp)
        scheduler.step()
        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        print(
            f'[{epoch:3d}/{args.epochs}] lr={scheduler.get_last_lr()[0]:.6f} '
            f'train_loss={train_loss:.4f} acc={train_acc:.4f} '
            f'test_loss={test_loss:.4f} acc={test_acc:.4f} ({time.time() - ep_t0:.1f}s)',
            flush=True,
        )

    final_acc = test_acc
    print(f'\nFinal test accuracy: {final_acc:.4f}; best: {best_acc:.4f}; wall={((time.time() - t0) / 60):.1f} min')
    if best_acc < args.min_test_acc and not args.allow_undertrained_smoke:
        print(f'REFUSING TO SAVE: best accuracy {best_acc:.4f} < gate {args.min_test_acc:.4f}.')
        return 1

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state if best_state is not None else model.state_dict(), out)
    sidecar = {
        'arch': 'vit_b_16',
        'dataset': args.dataset,
        'num_classes': int(args.num_classes),
        'image_size': int(args.image_size),
        'weights': args.weights,
        'epochs': int(args.epochs),
        'batch_size': int(args.batch_size),
        'lr': float(args.lr),
        'weight_decay': float(args.weight_decay),
        'seed': int(args.seed),
        'test_acc': round(float(final_acc), 6),
        'best_test_acc': round(float(best_acc), 6),
        'min_test_acc_gate': float(args.min_test_acc),
        'allow_undertrained_smoke': bool(args.allow_undertrained_smoke),
        'train_subset': args.train_subset,
        'test_subset': args.test_subset,
        'checkpoint': str(out),
        'sha256_first16': _sha256_first16(out),
        'recipe_version': 'torchvision-vit-b16-cifar-finetune-v1',
    }
    with open(out.with_suffix('.acc.json'), 'w') as f:
        json.dump(sidecar, f, indent=2, sort_keys=True)
    print(f'[OK] Saved ViT checkpoint -> {out}')
    print(f'[OK] Wrote sidecar -> {out.with_suffix(".acc.json")}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
