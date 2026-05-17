"""
Audit PRISM score distributions by attack family.

This script is a diagnostic gate, not a publishable evaluation. It uses a
held-out split to report clean/attack score quantiles, scorer components,
feature statistics, response levels, and base attack success so detector
failures can be attributed before expensive multi-seed Vast.ai runs.
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import torchvision.transforms as T
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import bootstrap  # noqa: F401
from src.perf import setup_perf_flags
setup_perf_flags()

_ART_IMPORT_ERROR = None
try:
    from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, SquareAttack
    from art.estimators.classification import PyTorchClassifier
    ART_AVAILABLE = True
except Exception as exc:
    _ART_IMPORT_ERROR = exc
    ART_AVAILABLE = False

_AA_IMPORT_ERROR = None
try:
    from autoattack import AutoAttack as _AutoAttack
    AA_AVAILABLE = True
except Exception as exc:
    _AA_IMPORT_ERROR = exc
    AA_AVAILABLE = False

from src.attacks.cw_torch import cw_l2_attack_torch
from src.cadg.ensemble_scorer import PersistenceEnsembleScorer
from src.config import (
    BACKBONE_INPUT_SIZE, BACKBONE_MEAN, BACKBONE_NUM_CLASSES, BACKBONE_STD,
    CAL_IDX, DATASET, DIM_WEIGHTS, EPS_LINF_STANDARD, EVAL_IDX, LAYER_NAMES,
    LAYER_WEIGHTS, MAX_DIM, N_SUBSAMPLE, PATHS, VAL_IDX,
)
from src.data_loader import load_test_dataset
from src.models import load_backbone
from src.tamm.extractor import ActivationExtractor
from src.tamm.logit_stability import compute_input_stability_features
from src.tamm.persistence_stats import compute_logit_profile_features
from src.tamm.scorer import TopologicalScorer
from src.tamm.tda import TopologicalProfiler


if BACKBONE_INPUT_SIZE == 32:
    _PIXEL_TRANSFORM = T.Compose([T.ToTensor()])
else:
    _PIXEL_TRANSFORM = T.Compose([T.Resize(BACKBONE_INPUT_SIZE), T.ToTensor()])
_NORMALIZE = T.Normalize(mean=BACKBONE_MEAN, std=BACKBONE_STD)


def _split_range(name):
    return {'cal': CAL_IDX, 'val': VAL_IDX, 'eval': EVAL_IDX}[name]


def _quantiles(values):
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return {}
    return {
        'mean': round(float(arr.mean()), 6),
        'std': round(float(arr.std()), 6),
        'min': round(float(arr.min()), 6),
        'p01': round(float(np.percentile(arr, 1)), 6),
        'p05': round(float(np.percentile(arr, 5)), 6),
        'p25': round(float(np.percentile(arr, 25)), 6),
        'p50': round(float(np.percentile(arr, 50)), 6),
        'p75': round(float(np.percentile(arr, 75)), 6),
        'p95': round(float(np.percentile(arr, 95)), 6),
        'p99': round(float(np.percentile(arr, 99)), 6),
        'max': round(float(arr.max()), 6),
    }


def _level_counts(levels):
    out = {}
    for lv in levels:
        out[lv] = out.get(lv, 0) + 1
    return out


def _input_grad_norm(model, x_norm, device):
    x_g = x_norm.detach().clone().to(device).requires_grad_(True)
    with torch.enable_grad():
        logits = model(x_g)
        pred_idx = int(logits.argmax(1).item())
        (grad_x,) = torch.autograd.grad(logits[0, pred_idx], x_g)
    return float(grad_x.norm().item())


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


def _score_pixels(imgs_pixel, model, extractor, profiler, scorer, calibrator, device, label):
    scores, w_scores, raw_logits, centered_logits, probs, levels = [], [], [], [], [], []
    features = []
    use_dct = getattr(scorer, 'use_dct', False)
    use_grad_norm = getattr(scorer, 'use_grad_norm', False)
    use_entropy = getattr(scorer, 'use_softmax_entropy', False)
    use_logit_profile = getattr(scorer, 'use_logit_profile_features', False)
    use_stability = getattr(scorer, 'use_stability_features', False)
    stability_feature_count = int(getattr(scorer, 'stability_feature_count', 4))

    for img_pixel in tqdm(imgs_pixel, desc=f"score[{label}]"):
        x_norm = _NORMALIZE(img_pixel).unsqueeze(0).to(device)
        acts = extractor.extract(x_norm)
        dgms = {
            layer: profiler.compute_diagram(acts[layer].squeeze(0).cpu().numpy())
            for layer in LAYER_NAMES
        }
        grad_norm = _input_grad_norm(model, x_norm, device) if use_grad_norm else None
        logits_np = None
        if use_entropy or use_logit_profile or use_stability:
            with torch.no_grad():
                logits_np = model(x_norm).squeeze(0).detach().cpu().numpy()
        logit_profile_features = None
        if use_logit_profile:
            logit_profile_features = compute_logit_profile_features(logits_np)
        stability_features = None
        if use_stability:
            stability_features = _stability_features(
                model, x_norm, img_pixel, logits_np, stability_feature_count,
            )
        comps = scorer.score_components(
            dgms,
            image=img_pixel.detach().cpu().numpy() if use_dct else None,
            grad_norm=grad_norm,
            logits=logits_np,
            logit_profile_features=logit_profile_features,
            stability_features=stability_features,
        )
        score = float(comps['score'])
        scores.append(score)
        w_scores.append(float(comps.get('w_score', 0.0)))
        raw_logits.append(float(comps.get('raw_logit', 0.0)))
        centered_logits.append(float(comps.get('logit_centered', 0.0)))
        probs.append(float(comps.get('logit_prob', 0.0)))
        levels.append(calibrator.classify(score, l0_active=False))
        if 'features' in comps:
            features.append(np.asarray(comps['features'], dtype=np.float32))

    feat_summary = {}
    if features:
        mat = np.stack(features, axis=0)
        feat_summary = {
            'n_features': int(mat.shape[1]),
            'mean': [round(float(v), 6) for v in mat.mean(axis=0).tolist()],
            'std': [round(float(v), 6) for v in mat.std(axis=0).tolist()],
        }

    return {
        'n': len(scores),
        'score_quantiles': _quantiles(scores),
        'w_score_quantiles': _quantiles(w_scores),
        'raw_logit_quantiles': _quantiles(raw_logits),
        'centered_logit_quantiles': _quantiles(centered_logits),
        'prob_quantiles': _quantiles(probs),
        'level_distribution': _level_counts(levels),
        'feature_summary': feat_summary,
        '_scores': np.asarray(scores, dtype=np.float32),
    }


def _base_success(norm_model, clean_np, adv_np, device, batch_size=256):
    clean_pred, adv_pred = [], []
    with torch.no_grad():
        for s in range(0, len(clean_np), batch_size):
            e = min(s + batch_size, len(clean_np))
            clean = torch.tensor(clean_np[s:e], device=device, dtype=torch.float32)
            adv = torch.tensor(adv_np[s:e], device=device, dtype=torch.float32)
            clean_pred.append(norm_model(clean).argmax(dim=1).cpu().numpy())
            adv_pred.append(norm_model(adv).argmax(dim=1).cpu().numpy())
    mask = np.concatenate(adv_pred) != np.concatenate(clean_pred)
    return {
        'base_attack_success': int(mask.sum()),
        'base_attack_success_rate': round(float(mask.mean()), 6),
    }


def _generate_attack(name, clean_np, norm_model, classifier, device, args):
    if name == 'FGSM':
        return FastGradientMethod(classifier, eps=EPS_LINF_STANDARD).generate(clean_np)
    if name == 'PGD':
        atk = ProjectedGradientDescent(
            classifier, eps=EPS_LINF_STANDARD, eps_step=EPS_LINF_STANDARD / 4,
            max_iter=args.pgd_steps, num_random_init=1,
        )
        return atk.generate(clean_np)
    if name == 'Square':
        atk = SquareAttack(
            classifier, eps=EPS_LINF_STANDARD, max_iter=args.square_max_iter,
            nb_restarts=1, verbose=False,
        )
        return atk.generate(clean_np)
    if name == 'CW':
        out = np.zeros_like(clean_np)
        for s in range(0, len(clean_np), args.cw_chunk):
            e = min(s + args.cw_chunk, len(clean_np))
            chunk = torch.tensor(clean_np[s:e], device=device, dtype=torch.float32)
            adv, _ = cw_l2_attack_torch(
                norm_model, chunk, device, max_iter=args.cw_max_iter,
                binary_search_steps=args.cw_bss, learning_rate=0.01,
                confidence=0.0,
            )
            out[s:e] = adv.detach().cpu().numpy()
        return out
    if name == 'AutoAttack':
        if not AA_AVAILABLE:
            raise RuntimeError('AutoAttack not installed')
        x = torch.tensor(clean_np, device=device, dtype=torch.float32)
        with torch.no_grad():
            y = norm_model(x).argmax(dim=1)
        adversary = _AutoAttack(
            norm_model, norm='Linf', eps=EPS_LINF_STANDARD,
            version=args.aa_version, device=device,
        )
        out = torch.empty_like(x)
        for s in range(0, len(clean_np), args.aa_chunk):
            e = min(s + args.aa_chunk, len(clean_np))
            out[s:e] = adversary.run_standard_evaluation(x[s:e], y[s:e], bs=args.aa_chunk)
        return out.detach().cpu().numpy()
    raise ValueError(f"Unknown attack: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, help='YAML config routed by bootstrap.')
    parser.add_argument('--split', choices=['cal', 'val', 'eval'], default='val')
    parser.add_argument('--n', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--attacks', nargs='+', default=['FGSM', 'PGD', 'Square', 'AutoAttack'])
    parser.add_argument('--data-root', default='./data')
    parser.add_argument('--output', default='experiments/calibration/score_audit.json')
    parser.add_argument('--pgd-steps', type=int, default=40)
    parser.add_argument('--square-max-iter', type=int, default=1000)
    parser.add_argument('--cw-max-iter', type=int, default=40)
    parser.add_argument('--cw-bss', type=int, default=5)
    parser.add_argument('--cw-chunk', type=int, default=128)
    parser.add_argument('--aa-version', default='standard')
    parser.add_argument('--aa-chunk', type=int, default=64)
    args = parser.parse_args()

    if not ART_AVAILABLE and any(a in args.attacks for a in ('FGSM', 'PGD', 'Square')):
        raise RuntimeError(
            'ART unavailable; install/fix adversarial-robustness-toolbox. '
            f'Import error: {_ART_IMPORT_ERROR}'
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.RandomState(args.seed)

    with open(PATHS['reference_profiles'], 'rb') as f:
        ref_profiles = pickle.load(f)
    with open(PATHS['calibrator'], 'rb') as f:
        calibrator = pickle.load(f)
    base_scorer = TopologicalScorer(
        ref_profiles=ref_profiles, layer_names=LAYER_NAMES,
        layer_weights=LAYER_WEIGHTS, dim_weights=DIM_WEIGHTS,
    )
    scorer = PersistenceEnsembleScorer.load(PATHS['ensemble_scorer'], base_scorer, LAYER_NAMES)

    model = load_backbone(device)
    norm_model = load_backbone(device, wrap=True)
    extractor = ActivationExtractor(model, LAYER_NAMES)
    profiler = TopologicalProfiler(n_subsample=N_SUBSAMPLE, max_dim=MAX_DIM)

    dataset = load_test_dataset(root=args.data_root, download=True, transform=_PIXEL_TRANSFORM)
    split = _split_range(args.split)
    pool = list(range(*split))
    sample_idx = rng.choice(pool, min(args.n, len(pool)), replace=False)
    imgs_pixel = [dataset[int(i)][0] for i in sample_idx]
    clean_np = torch.stack(imgs_pixel).numpy()

    classifier = PyTorchClassifier(
        model=norm_model, loss=torch.nn.CrossEntropyLoss(),
        input_shape=(3, BACKBONE_INPUT_SIZE, BACKBONE_INPUT_SIZE),
        nb_classes=BACKBONE_NUM_CLASSES, clip_values=(0.0, 1.0),
        device_type='gpu' if device.type == 'cuda' else 'cpu',
    )

    t0 = time.time()
    clean_summary = _score_pixels(
        imgs_pixel, model, extractor, profiler, scorer, calibrator, device, 'clean'
    )
    clean_scores = clean_summary.pop('_scores')

    results = {
        '_meta': {
            'dataset': DATASET,
            'split': args.split,
            'split_indices': [int(split[0]), int(split[1])],
            'n': int(len(sample_idx)),
            'seed': args.seed,
            'eps_linf': round(float(EPS_LINF_STANDARD), 6),
            'feature_space_version': getattr(scorer, 'feature_space_version', None),
            'stability_feature_count': getattr(scorer, 'stability_feature_count', None),
            'selection_objective': getattr(scorer, 'selection_objective', None),
            'training_attacks': getattr(scorer, 'training_attacks', None),
            'training_attack_counts': getattr(scorer, 'training_attack_counts', None),
        },
        'clean': clean_summary,
        'attacks': {},
    }

    l1 = float(calibrator.thresholds['L1'])
    for attack_name in args.attacks:
        print(f"\n=== Audit attack: {attack_name} ===", flush=True)
        adv_np = _generate_attack(attack_name, clean_np, norm_model, classifier, device, args)
        adv_pixels = [torch.tensor(x) for x in adv_np]
        adv_summary = _score_pixels(
            adv_pixels, model, extractor, profiler, scorer, calibrator, device, attack_name
        )
        adv_scores = adv_summary.pop('_scores')
        attack_result = {
            **adv_summary,
            **_base_success(norm_model, clean_np, adv_np, device),
            'tpr_at_L1': round(float(np.mean(adv_scores > l1)), 6),
            'miss_rate_at_L1': round(float(np.mean(adv_scores <= l1)), 6),
            'clean_overlap_fraction': round(float(np.mean(adv_scores <= np.percentile(clean_scores, 90))), 6),
        }
        results['attacks'][attack_name] = attack_result

    extractor.cleanup()
    results['_meta']['elapsed_s'] = round(float(time.time() - t0), 2)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote score audit -> {args.output}")


if __name__ == '__main__':
    main()
