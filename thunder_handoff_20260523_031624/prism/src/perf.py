"""
GPU performance flags — single source of truth.

Entry-point scripts call `setup_perf_flags()` once at startup to enable
hardware-level optimisations that are safe for this pipeline:

  - `torch.backends.cudnn.benchmark = True`
        Lets cuDNN benchmark and cache the fastest kernel for each
        (input_shape, conv_params) pair. The pipeline uses fixed shapes
        (B, 3, 32, 32) almost everywhere, so the lookup hits cache
        after the first batch. Gain: 5-25 % on convolutions.

  - `torch.backends.cuda.matmul.allow_tf32 = True`
    `torch.backends.cudnn.allow_tf32 = True`
        Enables TensorFloat-32 on Ampere/Hopper/Blackwell GPUs (3000-series
        and newer; RTX 5090 included). TF32 uses 10-bit mantissa matmul
        accumulators while keeping FP32 storage — numerically identical
        for our training/eval purposes; gain: 15-30 % on matmul/conv.

  - `torch.set_float32_matmul_precision('high')`
        Same effect at the PyTorch high-level API; some kernels honour
        only this flag.

This module is intentionally minimal: it does NOT enable AMP/autocast,
because those require per-call instrumentation and we use them
explicitly in the pretraining script. It does NOT pin to a specific
device — callers do that themselves.

Usage
-----
    from src.perf import setup_perf_flags
    setup_perf_flags()   # call once at the top of main()
"""
import torch


def setup_perf_flags(verbose: bool = False) -> None:
    """Enable safe, deterministic-equivalent GPU perf flags."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision('high')
        except AttributeError:
            # Older PyTorch (<1.12) doesn't have these flags
            pass
        if verbose:
            cap = torch.cuda.get_device_capability()
            name = torch.cuda.get_device_name()
            print(f"[perf] {name} (compute {cap[0]}.{cap[1]}); "
                  f"cudnn.benchmark=ON, TF32=ON")
    elif verbose:
        print("[perf] CUDA not available — flags no-op")
