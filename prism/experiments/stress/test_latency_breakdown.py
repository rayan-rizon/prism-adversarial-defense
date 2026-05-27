"""
P2: Per-component latency breakdown of the deployed PRISM forward pass.

The paper's latency table reports end-to-end mean/p50/p95 only. Reviewers
ask: "where does time go?" This script times each component on n=100
clean VAL-split inputs and reports a per-stage breakdown:
  1. backbone forward (logits)
  2. TAMM activation extraction
  3. TDA persistence diagram computation per layer
  4. ensemble feature extraction (DCT, grad-norm, softmax-entropy,
     logit-profile, stability) — separately
  5. ensemble logistic + Wasserstein fusion (score computation)

Output: per-stage mean / p50 / p95 / contribution-percent on RTX 4090.
"""
import os, sys, time, json, pickle
import numpy as np
import torch
import torchvision.transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
PRISM_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, PRISM_ROOT)

from src import bootstrap  # noqa: F401
from src.tamm.extractor import ActivationExtractor
from src.tamm.tda import TopologicalProfiler
from src.tamm.scorer import TopologicalScorer
from src.tamm.logit_stability import compute_input_stability_features
from src.tamm.persistence_stats import compute_logit_profile_features
from src.cadg.ensemble_scorer import PersistenceEnsembleScorer
from src.config import (
    LAYER_NAMES, LAYER_WEIGHTS, DIM_WEIGHTS,
    BACKBONE_MEAN, BACKBONE_STD, PATHS, N_SUBSAMPLE, MAX_DIM, VAL_IDX,
)
from src.data_loader import load_test_dataset
from src.models import load_backbone


_PIXEL_TRANSFORM = T.Compose([T.ToTensor()])
_NORMALIZE = T.Normalize(mean=BACKBONE_MEAN, std=BACKBONE_STD)


def main():
    n = 100
    seed = 42

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')
    if device == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    backbone = load_backbone(device=device)
    extractor = ActivationExtractor(backbone, LAYER_NAMES)
    profiler = TopologicalProfiler(n_subsample=N_SUBSAMPLE, max_dim=MAX_DIM)
    with open(PATHS['reference_profiles'], 'rb') as f:
        ref = pickle.load(f)
    base = TopologicalScorer(ref_profiles=ref, layer_names=LAYER_NAMES,
                             layer_weights=LAYER_WEIGHTS, dim_weights=DIM_WEIGHTS)
    ens = PersistenceEnsembleScorer.load(PATHS['ensemble_scorer'], base, LAYER_NAMES)

    ds = load_test_dataset(root='./data', transform=_PIXEL_TRANSFORM)
    rng = np.random.RandomState(seed)
    pick = sorted(rng.choice(list(range(*VAL_IDX)), n, replace=False).tolist())

    stab_count = int(getattr(ens, 'stability_feature_count', 8) or 8)

    times = {k: [] for k in [
        'backbone_fwd', 'tamm_extract', 'tda_diagrams',
        'grad_norm', 'logits_fwd', 'logit_profile', 'stability', 'dct_prep',
        'ensemble_fusion', 'total',
    ]}

    # Warmup (CUDA kernel compile, etc.)
    for idx in pick[:5]:
        img_pixel, _ = ds[int(idx)]
        x = _NORMALIZE(img_pixel).unsqueeze(0).to(device)
        with torch.no_grad():
            _ = backbone(x)
        _ = extractor.extract(x)

    def _sync():
        if device == 'cuda':
            torch.cuda.synchronize()

    for idx in pick:
        img_pixel, _ = ds[int(idx)]
        x = _NORMALIZE(img_pixel).unsqueeze(0).to(device)

        _sync(); t0 = time.perf_counter()

        # Stage 1: backbone forward (logits)
        with torch.no_grad():
            logits_t = backbone(x)
        _sync(); t1 = time.perf_counter()
        times['backbone_fwd'].append((t1 - t0) * 1000)

        logits_np = logits_t.squeeze(0).cpu().numpy()

        # Stage 2: TAMM activation extraction
        acts = extractor.extract(x)
        _sync(); t2 = time.perf_counter()
        times['tamm_extract'].append((t2 - t1) * 1000)

        # Stage 3: TDA persistence diagrams (CPU-bound, dominant in literature)
        dgms = {}
        for L in LAYER_NAMES:
            dgms[L] = profiler.compute_diagram(acts[L].squeeze(0).cpu().numpy())
        t3 = time.perf_counter()
        times['tda_diagrams'].append((t3 - t2) * 1000)

        # Stage 4a: gradient-norm
        x_g = x.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            lg = backbone(x_g)
            pred = int(lg.argmax(1).item())
            (gx,) = torch.autograd.grad(lg[0, pred], x_g)
        gn = float(gx.norm().item())
        _sync(); t4 = time.perf_counter()
        times['grad_norm'].append((t4 - t3) * 1000)

        # Stage 4b: logit profile features (cheap, on CPU)
        lp = compute_logit_profile_features(logits_np)
        t5 = time.perf_counter()
        times['logit_profile'].append((t5 - t4) * 1000)

        # Stage 4c: stability features (involves M=5 jittered forward passes)
        stab = compute_input_stability_features(
            model=backbone, x_norm=x, img_pixel=img_pixel,
            mean=BACKBONE_MEAN, std=BACKBONE_STD,
            logits_np=logits_np, feature_count=stab_count,
        )
        _sync(); t6 = time.perf_counter()
        times['stability'].append((t6 - t5) * 1000)

        # Stage 4d: DCT image prep (cpu->np)
        img_np = img_pixel.detach().cpu().numpy()
        t7 = time.perf_counter()
        times['dct_prep'].append((t7 - t6) * 1000)

        # Stage 5: ensemble logistic + Wasserstein fusion
        score = ens.score(
            dgms, image=img_np, grad_norm=gn, logits=logits_np,
            logit_profile_features=lp, stability_features=stab,
        )
        _sync(); t8 = time.perf_counter()
        times['ensemble_fusion'].append((t8 - t7) * 1000)
        times['total'].append((t8 - t0) * 1000)
        # store logits_fwd as a derived re-forward measurement (no-op)
        times['logits_fwd'].append(0.0)

    # Aggregate
    print()
    print(f'Per-stage latency over n={n} clean VAL images (RTX/GTX, ms):')
    print(f'{"stage":<22s}  {"mean":>8s}  {"p50":>8s}  {"p95":>8s}  {"%":>6s}')
    print('-' * 60)
    total_mean = float(np.mean(times['total']))
    rows = []
    for k in ['backbone_fwd', 'tamm_extract', 'tda_diagrams',
              'grad_norm', 'logit_profile', 'stability', 'dct_prep',
              'ensemble_fusion', 'total']:
        a = np.array(times[k])
        m, p50, p95 = float(a.mean()), float(np.percentile(a, 50)), float(np.percentile(a, 95))
        pct = (m / total_mean * 100) if k != 'total' else 100.0
        rows.append({'stage': k, 'mean_ms': round(m, 3),
                     'p50_ms': round(p50, 3), 'p95_ms': round(p95, 3),
                     'pct_of_total': round(pct, 1)})
        print(f'  {k:<20s}  {m:>8.2f}  {p50:>8.2f}  {p95:>8.2f}  {pct:>5.1f}%')

    out_path = os.path.join(HERE, 'results_latency_breakdown.json')
    with open(out_path, 'w') as f:
        json.dump({
            'device': device,
            'gpu_name': torch.cuda.get_device_name(0) if device == 'cuda' else None,
            'n_samples': n, 'seed': seed, 'split': 'VAL',
            'eval_idx_range': VAL_IDX, 'stages': rows,
        }, f, indent=2)
    print(f'\nwrote {out_path}')


if __name__ == '__main__':
    main()
