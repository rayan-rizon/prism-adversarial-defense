"""
Fine-tune a ResNet-50 backbone on ImageNet-100 for the PRISM Exp 2 scaling run.

Stock torchvision ResNet-50 (ImageNet-1k init) with the classifier replaced by
a 100-way head, fine-tuned on the staged ImageNet-100 ImageFolder, saved as a
plain state_dict that src.models.load_backbone reloads when model.arch=resnet50.
The 100 classes / labels 0..99 come from the ImageFolder's sorted class dirs,
so they line up with src.data_loader._imagenet_pool (same ImageFolder, same
sort) — no wnid remapping anywhere in the pipeline.

Recommended Vast.ai run:
    python scripts/pretrain_imagenet100_backbone.py \
        --data-dir data/imagenet100 --epochs 15 --batch-size 128 --lr 1e-3 \
        --output models/imagenet/resnet50_imagenet100.pt

Notes
-----
* The fine-tuning split here is internal to this script (a held-out 10% of the
  ImageFolder for accuracy reporting). It does NOT have to align with the PRISM
  detector splits in configs/imagenet.yaml — those are carved later, by index,
  over the same pool. This script only needs an accurate classifier.
* Backbone-freeze: by default only layer4 + fc are unfrozen (fast, enough for a
  100-class subset). Pass --full-finetune to train all layers.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import ssl
import sys
import time
from pathlib import Path

import certifi
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T
from torchvision.models import ResNet50_Weights, resnet50

os.environ.setdefault('SSL_CERT_FILE', certifi.where())
os.environ.setdefault('REQUESTS_CA_BUNDLE', certifi.where())
ssl._create_default_https_context = ssl.create_default_context

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from src.perf import setup_perf_flags  # noqa: E402

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_model(num_classes: int, full_finetune: bool) -> nn.Module:
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    if not full_finetune:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.layer4.parameters():
            p.requires_grad = True
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)  # always trainable
    return model


def _build_loaders(data_dir: str, image_size: int, batch_size: int,
                   num_workers: int, val_frac: float, seed: int):
    normalize = T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
    train_tf = T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.60, 1.00)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])
    eval_tf = T.Compose([
        T.Resize(max(image_size + 32, 256)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        normalize,
    ])
    base_train = torchvision.datasets.ImageFolder(data_dir, transform=train_tf)
    base_eval = torchvision.datasets.ImageFolder(data_dir, transform=eval_tf)
    n = len(base_train)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = int(n * val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_loader = DataLoader(Subset(base_train, train_idx), batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(Subset(base_eval, val_idx), batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, base_train.classes


@torch.no_grad()
def _evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', required=True,
                    help='ImageNet-100 ImageFolder dir (one subdir per class).')
    ap.add_argument('--output', default='models/imagenet/resnet50_imagenet100.pt')
    ap.add_argument('--num-classes', type=int, default=100)
    ap.add_argument('--image-size', type=int, default=224)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--val-frac', type=float, default=0.10)
    ap.add_argument('--full-finetune', action='store_true',
                    help='Unfreeze all layers (slower; default trains layer4+fc).')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    _set_seed(args.seed)
    setup_perf_flags()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not Path(args.data_dir).is_dir():
        raise FileNotFoundError(
            f"--data-dir '{args.data_dir}' not found. Stage the ImageNet-100 "
            f"ImageFolder there first (see configs/imagenet.yaml header)."
        )

    train_loader, val_loader, classes = _build_loaders(
        args.data_dir, args.image_size, args.batch_size,
        args.num_workers, args.val_frac, args.seed)
    if len(classes) != args.num_classes:
        raise ValueError(
            f"ImageFolder has {len(classes)} classes but --num-classes="
            f"{args.num_classes}. Set them equal (or restage the subset)."
        )

    model = _build_model(args.num_classes, args.full_finetune).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9,
                          weight_decay=args.weight_decay, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')

    best_acc = 0.0
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()
        sched.step()
        acc = _evaluate(model, val_loader, device)
        print(f"epoch {epoch:2d}/{args.epochs}  loss={running/len(train_loader):.4f}  "
              f"val_acc={acc:.4f}  ({time.time()-t0:.0f}s)")
        if acc >= best_acc:
            best_acc = acc
            torch.save(model.state_dict(), out)

    meta = {
        'arch': 'resnet50', 'num_classes': args.num_classes,
        'image_size': args.image_size, 'best_val_acc': best_acc,
        'full_finetune': args.full_finetune, 'classes': classes,
        'data_dir': args.data_dir, 'seed': args.seed,
    }
    out.with_suffix('.meta.json').write_text(json.dumps(meta, indent=2))
    print(f"\nSaved best checkpoint (val_acc={best_acc:.4f}) -> {out}")
    print(f"Metadata -> {out.with_suffix('.meta.json')}")


if __name__ == '__main__':
    main()
