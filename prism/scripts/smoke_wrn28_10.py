"""
Smoke test: WRN-28-10 throughput + VRAM probe on the local GPU.

Goal: decide whether Stretch A (CIFAR-10 WRN-28-10 backbone for the
PRISM "second architecture" experiment) can run locally, or must go to
Vast.ai.

What it does:
  1. Build WRN-28-10 (CIFAR-10) via robustbench's reference implementation.
  2. Probe maximum batch size that fits in available VRAM (binary search).
  3. Time forward+backward at the largest feasible batch for ~50 iters.
  4. Extrapolate to 200-epoch CIFAR-10 training time (50,000 images/epoch).
  5. Report a recommendation: LOCAL_OK / LOCAL_SLOW / USE_VAST.

Reads no project state, writes nothing. Safe to delete after.
"""
from __future__ import annotations

import sys
import time
import gc
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class _WRNBasicBlock(nn.Module):
    """Pre-activation basic block (Zagoruyko & Komodakis 2016, WRN paper)."""
    def __init__(self, in_planes: int, out_planes: int, stride: int = 1,
                 dropout_rate: float = 0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                      stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(x), inplace=True)
        # Apply shortcut from pre-activated x for the projection case,
        # following the standard WRN convention.
        shortcut = self.shortcut(out) if not isinstance(self.shortcut, nn.Identity) else x
        out = self.conv1(out)
        out = F.relu(self.bn2(out), inplace=True)
        out = self.dropout(out)
        out = self.conv2(out)
        return out + shortcut


class _WideResNet(nn.Module):
    """WRN-depth-widen, standard Zagoruyko & Komodakis WRN for CIFAR (32x32)."""
    def __init__(self, depth: int = 28, widen_factor: int = 10,
                 num_classes: int = 10, dropout_rate: float = 0.0):
        super().__init__()
        assert (depth - 4) % 6 == 0, 'WRN depth must satisfy (depth-4) % 6 == 0'
        n = (depth - 4) // 6
        k = widen_factor
        widths = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = nn.Conv2d(3, widths[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.layer1 = self._make_layer(widths[0], widths[1], n, stride=1,
                                       dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(widths[1], widths[2], n, stride=2,
                                       dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(widths[2], widths[3], n, stride=2,
                                       dropout_rate=dropout_rate)
        self.bn = nn.BatchNorm2d(widths[3])
        self.linear = nn.Linear(widths[3], num_classes)

    def _make_layer(self, in_planes: int, out_planes: int, num_blocks: int,
                    stride: int, dropout_rate: float) -> nn.Sequential:
        layers = []
        for i in range(num_blocks):
            s = stride if i == 0 else 1
            in_p = in_planes if i == 0 else out_planes
            layers.append(_WRNBasicBlock(in_p, out_planes, stride=s,
                                         dropout_rate=dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn(out), inplace=True)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.linear(out)


def _build_wrn28_10(num_classes: int = 10) -> nn.Module:
    """Inline WRN-28-10, no external import (robustbench has a stale
    `pkg_resources` dep)."""
    return _WideResNet(depth=28, widen_factor=10, num_classes=num_classes)


def _probe_max_batch(device: torch.device,
                     candidates=(256, 192, 128, 96, 64, 48, 32),
                     use_amp: bool = True) -> int:
    """Binary-search-ish probe: largest batch that runs one fwd+bwd without OOM."""
    for bs in candidates:
        try:
            torch.cuda.empty_cache()
            model = _build_wrn28_10().to(device)
            x = torch.randn(bs, 3, 32, 32, device=device)
            y = torch.randint(0, 10, (bs,), device=device)
            opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    logits = model(x)
                    loss = F.cross_entropy(logits, y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                opt.step()
            torch.cuda.synchronize()
            peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f'  batch={bs:>4d}  OK   peak_alloc={peak_mb:7.1f} MiB')
            del model, x, y, opt, scaler, logits, loss
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gc.collect()
            return bs
        except torch.cuda.OutOfMemoryError:
            print(f'  batch={bs:>4d}  OOM')
            try:
                del model
            except Exception:
                pass
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gc.collect()
            continue
    raise RuntimeError('No batch size fits in VRAM -- GPU is too small for WRN-28-10.')


def _time_iters(device: torch.device, batch_size: int, n_iters: int = 50,
                use_amp: bool = True) -> tuple[float, float]:
    """Returns (mean_iter_seconds, images_per_second)."""
    torch.cuda.empty_cache()
    model = _build_wrn28_10().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # Warmup (3 iters) -- first iter includes CUDA kernel JIT.
    x = torch.randn(batch_size, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                loss = F.cross_entropy(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(n_iters):
        opt.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                loss = F.cross_entropy(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
    torch.cuda.synchronize()
    dt = time.time() - t0

    iter_s = dt / n_iters
    imgs_s = batch_size / iter_s
    return iter_s, imgs_s


def main() -> int:
    if not torch.cuda.is_available():
        print('CUDA not available -- Stretch A is infeasible locally.')
        return 1

    device = torch.device('cuda')
    name = torch.cuda.get_device_name(0)
    total_vram_mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
    print(f'GPU: {name}   total VRAM: {total_vram_mb:.0f} MiB')
    print(f'torch: {torch.__version__}   cuda: {torch.version.cuda}')
    print()

    # WRN-28-10 model size
    m = _build_wrn28_10()
    n_params = sum(p.numel() for p in m.parameters())
    print(f'WRN-28-10 parameter count: {n_params/1e6:.2f} M')
    del m
    gc.collect()

    # Pascal (GTX 1060) gets limited speedup from AMP -- autocast falls back
    # to FP32 for BN and some convs. Probe FP32 only to avoid wasted cycles.
    # Cap probe at batch=256 to stay under the 6 GB VRAM ceiling.
    print()
    print('== Probing max batch size in FP32 ==')
    bs_fp32 = _probe_max_batch(device, use_amp=False)
    bs_amp = bs_fp32  # parity for verdict logic
    use_amp = False
    batch_size = bs_fp32
    print()
    print(f'== Timing 10 iters at batch={batch_size}, AMP={use_amp} ==')
    iter_s, imgs_s = _time_iters(device, batch_size, n_iters=10, use_amp=use_amp)
    print(f'  mean iter time: {iter_s*1000:.1f} ms')
    print(f'  throughput:     {imgs_s:.0f} images/sec')

    # Extrapolate to 200-epoch CIFAR-10 training
    images_per_epoch = 50_000
    epochs = 200
    iters_per_epoch = -(-images_per_epoch // batch_size)  # ceil
    seconds_per_epoch = iters_per_epoch * iter_s
    total_seconds = seconds_per_epoch * epochs
    total_hours = total_seconds / 3600

    # Add ~15% overhead for data loading + eval (conservative on slow disks)
    overhead_factor = 1.15
    realistic_hours = total_hours * overhead_factor

    print()
    print('== 200-epoch CIFAR-10 WRN-28-10 extrapolation ==')
    print(f'  iters per epoch: {iters_per_epoch} @ batch {batch_size}')
    print(f'  epoch time:      {seconds_per_epoch:.1f} s ({seconds_per_epoch/60:.1f} min)')
    print(f'  pure training:   {total_hours:.1f} h')
    print(f'  with overhead:   {realistic_hours:.1f} h  (~{realistic_hours/24:.1f} days)')

    # Recommendation gate
    print()
    print('== Verdict ==')
    if realistic_hours < 6:
        verdict = 'LOCAL_OK -- train locally, fits inside a single sitting.'
    elif realistic_hours < 16:
        verdict = ('LOCAL_OK_OVERNIGHT -- train locally overnight (one shot, '
                   'no babysitting needed).')
    elif realistic_hours < 36:
        verdict = ('LOCAL_SLOW -- feasible but painful: ~1.5 days of dedicated '
                   'GPU. Consider Vast.ai if you need to iterate.')
    else:
        verdict = ('USE_VAST -- local training would take 1.5+ days; spend the '
                   '$1-2 of Vast.ai time instead.')

    print(f'  Estimated wall-clock: {realistic_hours:.1f} h')
    print(f'  Max batch (AMP):      {bs_amp}')
    print(f'  Max batch (FP32):     {bs_fp32}')
    print(f'  -> {verdict}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

