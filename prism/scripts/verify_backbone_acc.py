"""
Backbone-accuracy gate — single source of truth.

Every stage of the PRISM pipeline (smoke and Vast.ai) calls this utility
before doing any work that depends on backbone competence. It exists to
prevent the silent failure mode in which an undertrained checkpoint
(e.g. the 51%-acc 3-epoch artifact from logs/smoke/step0_backbone.log)
passes the prior shape-only verification and goes on to poison every
downstream artifact — reference profiles, ensemble scorer, calibrator.

When the gate trips, no TDA features, no logistic regression, and no
conformal thresholds are computed. The pipeline halts loudly with a
message that points at `scripts/pretrain_cifar_backbone.py`.

CLI
---
    # As a gate (exits non-zero on failure):
    python scripts/verify_backbone_acc.py \\
        --checkpoint models/cifar_resnet18.pt \\
        --min-acc 0.93 --n 1000 \\
        [--sidecar models/cifar_resnet18.acc.json]

    # As a library:
    from scripts.verify_backbone_acc import verify_backbone_acc
    acc, n = verify_backbone_acc('models/cifar_resnet18.pt', n=200)

The optional --sidecar flag adds a second check: the sha256 of the
checkpoint file must match the value recorded by pretrain_cifar_backbone.py
in the sidecar JSON. A mismatch is treated as a stale-cache event and the
gate fails, forcing a fresh pretrain run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader

# Ensure src/ is importable when run from the prism/ project root.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from src.models import cifar_resnet18
from src.config import (
    BACKBONE_CHECKPOINT_PATH,
    BACKBONE_NUM_CLASSES,
    BACKBONE_MEAN,
    BACKBONE_STD,
    DATASET,
)
from src.data_loader import load_test_dataset


def _sha256_first16(path: str | Path) -> str:
    """Hash the checkpoint bytes — same recipe as the sidecar writer."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:16]


def verify_backbone_acc(
    checkpoint_path: str | Path,
    n: int = 200,
    device: str | None = None,
    data_root: str = './data',
) -> Tuple[float, int]:
    """Load the checkpoint and measure clean test accuracy on N images.

    Returns:
        (acc, n_eval) where acc is in [0, 1] and n_eval ≤ n.

    Raises:
        FileNotFoundError if the checkpoint is missing.
    """
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Backbone checkpoint not found: {ckpt}. "
            f"Run scripts/pretrain_cifar_backbone.py first (~50-70 min on RTX 5090)."
        )

    dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = cifar_resnet18(
        num_classes=BACKBONE_NUM_CLASSES,
        checkpoint_path=str(ckpt),
        map_location=str(dev),
    ).to(dev).eval()

    # Use the canonical test transform (already normalised) so this matches
    # exactly what every downstream stage feeds the model.
    test_ds = load_test_dataset(root=data_root, download=True)
    n_eval = min(n, len(test_ds))
    loader = DataLoader(
        torch.utils.data.Subset(test_ds, list(range(n_eval))),
        batch_size=128, shuffle=False, num_workers=0,
    )

    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(dev)
            y = y.to(dev)
            pred = model(x).argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    return correct / max(total, 1), total


def _load_sidecar(sidecar_path: Path) -> dict:
    if not sidecar_path.exists():
        raise FileNotFoundError(
            f"Sidecar JSON not found: {sidecar_path}. The checkpoint at "
            f"{BACKBONE_CHECKPOINT_PATH} predates the accuracy-gate fix or "
            f"was hand-copied without provenance. Re-run pretrain to regenerate."
        )
    with open(sidecar_path) as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Verify a CIFAR backbone checkpoint meets the accuracy gate.'
    )
    parser.add_argument('--checkpoint', default=BACKBONE_CHECKPOINT_PATH,
                        help=f'Path to state_dict (default: {BACKBONE_CHECKPOINT_PATH}).')
    parser.add_argument('--min-acc', type=float, required=True,
                        help='Hard floor on clean test accuracy (e.g. 0.93).')
    parser.add_argument('--n', type=int, default=1000,
                        help='Number of test images to evaluate (default: 1000).')
    parser.add_argument('--data-root', default='./data')
    parser.add_argument('--sidecar', default=None,
                        help='Optional sidecar JSON. If given, sha256_first16 '
                             'must match the checkpoint file.')
    parser.add_argument('--device', default=None,
                        help='Override device (default: cuda if available).')
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    print(f'Verifying backbone: {ckpt}', flush=True)
    print(f'  Gate:     min_acc={args.min_acc:.4f}  n={args.n}  dataset={DATASET}',
          flush=True)

    # ── Sidecar SHA check ──────────────────────────────────────────────────
    if args.sidecar is not None:
        try:
            meta = _load_sidecar(Path(args.sidecar))
        except FileNotFoundError as e:
            print(f'[FAIL] {e}', flush=True)
            return 2
        recorded = meta.get('sha256_first16', '')
        actual = _sha256_first16(ckpt)
        if recorded != actual:
            print(
                f'[FAIL] Sidecar SHA mismatch: sidecar={recorded} '
                f'file={actual}. Checkpoint has been replaced or corrupted '
                f'since training. Delete {ckpt} and re-run pretrain.',
                flush=True,
            )
            return 3
        print(
            f'  Sidecar:  sha256_first16 OK ({actual})  '
            f'trained {meta.get("epochs", "?")} epochs  '
            f'recorded test_acc={meta.get("test_acc", "?"):.4f}',
            flush=True,
        )

    # ── Empirical accuracy check ───────────────────────────────────────────
    try:
        acc, n_eval = verify_backbone_acc(
            ckpt, n=args.n, device=args.device, data_root=args.data_root,
        )
    except FileNotFoundError as e:
        print(f'[FAIL] {e}', flush=True)
        return 2

    print(f'  Measured: acc={acc:.4f}  n_eval={n_eval}', flush=True)

    if acc < args.min_acc:
        print(
            f'[FAIL] Backbone test accuracy {acc:.4f} < gate {args.min_acc:.4f}. '
            f'A {acc:.0%}-accurate backbone produces noisy decision boundaries; '
            f'FGSM/PGD/Square gradients computed against it do not yield '
            f'meaningfully adversarial inputs, so TDA features cannot separate '
            f'clean from adversarial. Smoke gate will collapse to TPR ≈ FPR.',
            flush=True,
        )
        print(
            f'       Fix: rm {ckpt} && python scripts/pretrain_cifar_backbone.py',
            flush=True,
        )
        return 1

    print(f'[OK] Backbone passes accuracy gate ({acc:.4f} ≥ {args.min_acc:.4f}).',
          flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
