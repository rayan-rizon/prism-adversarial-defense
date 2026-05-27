"""
Generate REAL clean ensemble scores by running the deployed PRISM pipeline
on N clean CIFAR-10 test images. Replaces the synthetic Gaussian /
bootstrap stream in test_l0_long_clean_stream.py with actual scores
from the deployed scorer.

Outputs: experiments/stress/real_clean_scores_n{N}.npy
Mirrors the score-computation path used by
scripts/calibrate_l0_thresholds.py::_compute_score_stream so the
emitted scores are exactly what the L0 monitor sees in production.
"""
import os
import sys
import time
import argparse
import numpy as np
import torch
import torchvision.transforms as T
import pickle
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
PRISM_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, PRISM_ROOT)

# Route --config CLI flag to PRISM_CONFIG env var BEFORE importing src.config.
from src import bootstrap  # noqa: F401

from src.tamm.extractor import ActivationExtractor
from src.tamm.tda import TopologicalProfiler
from src.tamm.scorer import TopologicalScorer
from src.cadg.ensemble_scorer import PersistenceEnsembleScorer
from src.tamm.logit_stability import compute_input_stability_features
from src.tamm.persistence_stats import compute_logit_profile_features
from src.config import (
    LAYER_NAMES, LAYER_WEIGHTS, DIM_WEIGHTS,
    BACKBONE_MEAN, BACKBONE_STD, BACKBONE_INPUT_SIZE,
    EVAL_IDX, VAL_IDX, PATHS, N_SUBSAMPLE, MAX_DIM,
)
from src.data_loader import load_test_dataset
from src.models import load_backbone, _NormalizedBackbone


_NORMALIZE = T.Normalize(mean=BACKBONE_MEAN, std=BACKBONE_STD)
_PIXEL_TRANSFORM = T.Compose([T.ToTensor()])  # 32x32 native


def _stability_features(model, x_norm, img_pixel, logits_np, feature_count):
    return compute_input_stability_features(
        model=model,
        x_norm=x_norm,
        img_pixel=img_pixel,
        mean=BACKBONE_MEAN,
        std=BACKBONE_STD,
        logits_np=logits_np,
        feature_count=feature_count,
    )


def compute_score(model_normed, model_raw, profiler, extractor, ensemble,
                  img_pixel, device):
    # Use the raw backbone (with hooks) for activation extraction, but compute
    # logits / grad / stability on the normalised forward path matching
    # PRISM.defend()'s production code.
    x_norm = _NORMALIZE(img_pixel).unsqueeze(0).to(device)
    acts = extractor.extract(x_norm)
    dgms = {
        L: profiler.compute_diagram(acts[L].squeeze(0).cpu().numpy())
        for L in LAYER_NAMES
    }
    use_dct = getattr(ensemble, 'use_dct', False)
    use_grad_norm = getattr(ensemble, 'use_grad_norm', False)
    use_softmax_entropy = getattr(ensemble, 'use_softmax_entropy', False)
    use_logit_profile_features = getattr(ensemble, 'use_logit_profile_features', False)
    use_stability_features = getattr(ensemble, 'use_stability_features', False)
    # IMPORTANT: read the count off the loaded scorer (it is 8 in the
    # deployed pkl, not the default 4) so the feature vector matches
    # the trained logistic's expected dimension.
    stab_count = int(getattr(ensemble, 'stability_feature_count', 4) or 4)

    img_np = img_pixel.detach().cpu().numpy() if use_dct else None

    grad_norm = None
    if use_grad_norm:
        x_g = x_norm.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            logits_g = model_normed(x_g)
            pred_idx = int(logits_g.argmax(1).item())
            (grad_x,) = torch.autograd.grad(logits_g[0, pred_idx], x_g)
        grad_norm = float(grad_x.norm().item())

    logits_np = None
    if use_softmax_entropy or use_logit_profile_features or use_stability_features:
        with torch.no_grad():
            logits = model_normed(x_norm)
        logits_np = logits.squeeze(0).detach().cpu().numpy()
    logit_profile_features = None
    if use_logit_profile_features:
        logit_profile_features = compute_logit_profile_features(logits_np)
    stability_features = None
    if use_stability_features:
        stability_features = _stability_features(
            model_normed, x_norm, img_pixel, logits_np, stab_count
        )

    return ensemble.score(
        dgms,
        image=img_np,
        grad_norm=grad_norm,
        logits=logits_np,
        logit_profile_features=logit_profile_features,
        stability_features=stability_features,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n-clean', type=int, default=2000)
    p.add_argument('--output', default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--split', choices=('val', 'eval'), default='val',
                   help='val=[7000,8000) (same as calibrate_l0_thresholds.py); '
                        'eval=[8000,10000) (campaign experiments)')
    args = p.parse_args()

    out = args.output or os.path.join(
        HERE, f'real_clean_scores_{args.split}_n{args.n_clean}.npy'
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')
    if args.split == 'val':
        lo, hi = VAL_IDX[0], VAL_IDX[1]
    else:
        lo, hi = EVAL_IDX[0], EVAL_IDX[1]
    print(f'loading dataset (CIFAR-10 test, {args.split.upper()}_IDX[{lo}:{hi}))...')

    testset = load_test_dataset(root='./data', transform=_PIXEL_TRANSFORM)
    eligible = list(range(lo, hi))
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(eligible, size=min(args.n_clean, len(eligible)), replace=False)
    pick = sorted(pick.tolist())
    print(f'sampling {len(pick)} clean images from index range [{lo}, {hi})')

    print('loading models...')
    # Match calibrate_l0_thresholds.py: hooks on the raw backbone; the raw
    # backbone is fed normalised tensors (data pipeline normalises).
    backbone = load_backbone(device=device)
    model_normed = backbone  # alias — already in eval, accepts normalised input
    extractor = ActivationExtractor(backbone, LAYER_NAMES)

    with open(PATHS['reference_profiles'], 'rb') as f:
        ref_profiles = pickle.load(f)
    profiler = TopologicalProfiler(n_subsample=N_SUBSAMPLE, max_dim=MAX_DIM)

    base_scorer = TopologicalScorer(
        ref_profiles=ref_profiles,
        layer_names=LAYER_NAMES,
        layer_weights=LAYER_WEIGHTS,
        dim_weights=DIM_WEIGHTS,
    )
    ensemble = PersistenceEnsembleScorer.load(
        PATHS['ensemble_scorer'], base_scorer=base_scorer, layer_names=LAYER_NAMES,
    )
    print(f'ensemble_scorer flags: use_dct={getattr(ensemble,"use_dct",False)}, '
          f'use_grad_norm={getattr(ensemble,"use_grad_norm",False)}, '
          f'use_softmax_entropy={getattr(ensemble,"use_softmax_entropy",False)}, '
          f'use_logit_profile_features={getattr(ensemble,"use_logit_profile_features",False)}, '
          f'use_stability_features={getattr(ensemble,"use_stability_features",False)}')

    scores = np.empty(len(pick), dtype=np.float32)
    t0 = time.time()
    for i, idx in enumerate(tqdm(pick, desc='score(clean)')):
        img_pixel, _label = testset[idx]  # already tensor via _PIXEL_TRANSFORM
        scores[i] = compute_score(
            model_normed, backbone, profiler, extractor, ensemble,
            img_pixel, device,
        )
    dt = time.time() - t0

    np.save(out, scores)
    print(f'\nwrote {out}')
    print(f'n={len(scores)}, mean={scores.mean():.4f}, std={scores.std():.4f}, '
          f'min={scores.min():.4f}, max={scores.max():.4f}')
    print(f'p50={np.percentile(scores,50):.4f}, p95={np.percentile(scores,95):.4f}, '
          f'p99={np.percentile(scores,99):.4f}')
    print(f'time: {dt:.1f}s ({dt/len(scores)*1000:.1f} ms/sample)')


if __name__ == '__main__':
    main()
