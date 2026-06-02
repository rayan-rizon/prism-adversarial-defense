"""
Vast.ai paper-canonical CW: bss=9, max_iter=100, kappa in {0, 10, 20},
n=1000 per seed, 5 seeds (42, 123, 456, 789, 999), EVAL split [8000, 9999).

This matches the canonical CW configuration in main_attacks.tex
(`results_cw_n1000_ms5_seed*`: max_iter=100, binary_search_steps=9)
and re-runs the kappa sweep at that headline attack strength so the
appendix tab:strong_cw drop can be directly compared to the Table
tab:main 0.938 headline.

Differs from vastai_stronger_cw.py only in CW hyperparameters (bss=9,
max_iter=100 vs bss=5, max_iter=40). Everything else identical.
"""
import os, sys, time, json, pickle, math
import numpy as np
import torch
import torchvision.transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
PRISM_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, PRISM_ROOT)

from src import bootstrap  # noqa
from src.tamm.extractor import ActivationExtractor
from src.tamm.tda import TopologicalProfiler
from src.tamm.scorer import TopologicalScorer
from src.tamm.logit_stability import compute_input_stability_features
from src.tamm.persistence_stats import compute_logit_profile_features
from src.cadg.ensemble_scorer import PersistenceEnsembleScorer
from src.attacks.cw_torch import cw_l2_attack_torch
from src.config import (
    LAYER_NAMES, LAYER_WEIGHTS, DIM_WEIGHTS,
    BACKBONE_MEAN, BACKBONE_STD,
    PATHS, N_SUBSAMPLE, MAX_DIM, EVAL_IDX,
)
from src.data_loader import load_test_dataset
from src.models import load_backbone

_PIXEL_TRANSFORM = T.Compose([T.ToTensor()])
_NORMALIZE = T.Normalize(mean=BACKBONE_MEAN, std=BACKBONE_STD)

N_PER_SEED = 1000
SEEDS = [42, 123, 456, 789, 999]
KAPPAS = [0.0, 10.0, 20.0]
BATCH_SIZE = 500  # RTX 4090 can handle larger batches
CW_MAX_ITER = 100        # paper canonical
CW_BSS = 9               # paper canonical


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    halfw = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - halfw), min(1.0, centre + halfw))


def compute_score(backbone, profiler, extractor, ens, img_pixel, device):
    x_norm = _NORMALIZE(img_pixel).unsqueeze(0).to(device)
    acts = extractor.extract(x_norm)
    dgms = {L: profiler.compute_diagram(acts[L].squeeze(0).cpu().numpy())
            for L in LAYER_NAMES}
    use_grad = getattr(ens, 'use_grad_norm', False)
    use_sm = getattr(ens, 'use_softmax_entropy', False)
    use_lp = getattr(ens, 'use_logit_profile_features', False)
    use_st = getattr(ens, 'use_stability_features', False)
    use_dct = getattr(ens, 'use_dct', False)
    stab_count = int(getattr(ens, 'stability_feature_count', 8) or 8)
    img_np = img_pixel.detach().cpu().numpy() if use_dct else None
    gn = None
    if use_grad:
        x_g = x_norm.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            lg = backbone(x_g)
            pred = int(lg.argmax(1).item())
            (gx,) = torch.autograd.grad(lg[0, pred], x_g)
        gn = float(gx.norm().item())
    logits_np = None
    if use_sm or use_lp or use_st:
        with torch.no_grad():
            logits_np = backbone(x_norm).squeeze(0).cpu().numpy()
    lp = compute_logit_profile_features(logits_np) if use_lp else None
    stab = compute_input_stability_features(
        model=backbone, x_norm=x_norm, img_pixel=img_pixel,
        mean=BACKBONE_MEAN, std=BACKBONE_STD,
        logits_np=logits_np, feature_count=stab_count,
    ) if use_st else None
    return float(ens.score(dgms, image=img_np, grad_norm=gn, logits=logits_np,
                           logit_profile_features=lp, stability_features=stab))


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device} (CW canonical bss={CW_BSS}, max_iter={CW_MAX_ITER}, '
          f'n_per_seed={N_PER_SEED}, seeds={SEEDS}, kappas={KAPPAS})')

    backbone = load_backbone(device=device)
    norm_backbone = load_backbone(device=device, wrap=True)
    extractor = ActivationExtractor(backbone, LAYER_NAMES)
    profiler = TopologicalProfiler(n_subsample=N_SUBSAMPLE, max_dim=MAX_DIM)
    with open(PATHS['reference_profiles'], 'rb') as f:
        ref = pickle.load(f)
    base = TopologicalScorer(ref_profiles=ref, layer_names=LAYER_NAMES,
                             layer_weights=LAYER_WEIGHTS, dim_weights=DIM_WEIGHTS)
    ens = PersistenceEnsembleScorer.load(PATHS['ensemble_scorer'], base, LAYER_NAMES)
    with open(PATHS['calibrator'], 'rb') as f:
        calib = pickle.load(f)
    thr = calib.thresholds
    print(f'thresholds: {thr}')

    ds = load_test_dataset(root='./data', transform=_PIXEL_TRANSFORM)
    eval_pool = list(range(*EVAL_IDX))

    results = {
        'n_per_seed': N_PER_SEED, 'seeds': SEEDS, 'kappas': KAPPAS,
        'cw_max_iter': CW_MAX_ITER, 'cw_bss': CW_BSS,
        'split': 'EVAL', 'eval_idx_range': list(EVAL_IDX),
        'thresholds': {k: float(v) for k, v in thr.items()},
        'per_seed': {}, 'aggregate': {},
    }
    clean_pool_all = []

    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        pick = sorted(rng.choice(eval_pool, N_PER_SEED, replace=False).tolist())
        print(f'\n[seed {seed}] sampling {N_PER_SEED} from EVAL...')
        clean_stack = torch.stack([ds[int(i)][0] for i in pick]).to(device)

        # Clean scoring
        print(f'  scoring clean (n={N_PER_SEED})...')
        t0 = time.time()
        clean_scores = np.array([
            compute_score(backbone, profiler, extractor, ens,
                          ds[int(i)][0], device)
            for i in pick
        ])
        clean_pool_all.append(clean_scores)
        fpr_l1 = int(np.sum(clean_scores > thr['L1']))
        fpr_l2 = int(np.sum(clean_scores > thr['L2']))
        fpr_l3 = int(np.sum(clean_scores > thr['L3']))
        print(f'    clean mean={clean_scores.mean():.3f} '
              f'FPR L1={fpr_l1/N_PER_SEED:.4f} L2={fpr_l2/N_PER_SEED:.4f} '
              f'L3={fpr_l3/N_PER_SEED:.4f}  ({time.time()-t0:.1f}s)')

        results['per_seed'][str(seed)] = {
            'clean': {'mean': float(clean_scores.mean()),
                      'std': float(clean_scores.std()),
                      'FPR_L1_count': fpr_l1, 'FPR_L2_count': fpr_l2,
                      'FPR_L3_count': fpr_l3, 'n': N_PER_SEED},
            'cw_by_kappa': {},
        }

        for kappa in KAPPAS:
            t0 = time.time()
            print(f'  CW canonical kappa={kappa} '
                  f'(bss={CW_BSS}, max_iter={CW_MAX_ITER}, batch={BATCH_SIZE})...')
            adv_chunks = []
            for s in range(0, N_PER_SEED, BATCH_SIZE):
                e = min(s + BATCH_SIZE, N_PER_SEED)
                adv_b, _ = cw_l2_attack_torch(
                    norm_backbone, clean_stack[s:e], device,
                    max_iter=CW_MAX_ITER, binary_search_steps=CW_BSS,
                    learning_rate=0.01, confidence=kappa, initial_const=0.01,
                )
                adv_chunks.append(adv_b.detach())
            adv_stack = torch.cat(adv_chunks, dim=0)
            with torch.no_grad():
                clean_pred = norm_backbone(clean_stack).argmax(1)
                adv_pred = norm_backbone(adv_stack).argmax(1)
            attack_success = float((adv_pred != clean_pred).float().mean().item())
            l2 = (adv_stack - clean_stack).flatten(1).norm(dim=1).mean().item()
            dt_attack = time.time() - t0
            print(f'    success={attack_success:.3f} L2={l2:.4f} '
                  f'time={dt_attack:.1f}s  scoring adv...')

            t1 = time.time()
            adv_scores = np.array([
                compute_score(backbone, profiler, extractor, ens,
                              adv_stack[i].cpu(), device)
                for i in range(N_PER_SEED)
            ])
            dt_score = time.time() - t1
            tpr_l1 = int(np.sum(adv_scores > thr['L1']))
            tpr_l2 = int(np.sum(adv_scores > thr['L2']))
            tpr_l3 = int(np.sum(adv_scores > thr['L3']))
            print(f'    adv mean={adv_scores.mean():.3f} '
                  f'TPR L1={tpr_l1/N_PER_SEED:.4f} L2={tpr_l2/N_PER_SEED:.4f} '
                  f'L3={tpr_l3/N_PER_SEED:.4f} score_time={dt_score:.1f}s')

            results['per_seed'][str(seed)]['cw_by_kappa'][str(kappa)] = {
                'attack_success': attack_success,
                'mean_L2': float(l2),
                'attack_time_s': round(dt_attack, 1),
                'adv_mean_score': float(adv_scores.mean()),
                'adv_std_score': float(adv_scores.std()),
                'TPR_L1_count': tpr_l1, 'TPR_L2_count': tpr_l2,
                'TPR_L3_count': tpr_l3, 'n': N_PER_SEED,
            }

    n_total = N_PER_SEED * len(SEEDS)
    pooled_clean = np.concatenate(clean_pool_all)
    pooled_fpr = {
        'L1': float((pooled_clean > thr['L1']).mean()),
        'L2': float((pooled_clean > thr['L2']).mean()),
        'L3': float((pooled_clean > thr['L3']).mean()),
    }
    results['aggregate']['clean'] = {
        'mean': float(pooled_clean.mean()),
        'std': float(pooled_clean.std()),
        'FPR': pooled_fpr, 'n_total': n_total,
    }
    print(f'\npooled clean FPR: {pooled_fpr}')

    print(f'\n  kappa | success | L2    | TPR(L1) [CI95]              | TPR(L2)  | TPR(L3)')
    for kappa in KAPPAS:
        succ = np.mean([results['per_seed'][str(s)]['cw_by_kappa'][str(kappa)]['attack_success'] for s in SEEDS])
        l2 = np.mean([results['per_seed'][str(s)]['cw_by_kappa'][str(kappa)]['mean_L2'] for s in SEEDS])
        tpr_l1_k = sum(results['per_seed'][str(s)]['cw_by_kappa'][str(kappa)]['TPR_L1_count'] for s in SEEDS)
        tpr_l2_k = sum(results['per_seed'][str(s)]['cw_by_kappa'][str(kappa)]['TPR_L2_count'] for s in SEEDS)
        tpr_l3_k = sum(results['per_seed'][str(s)]['cw_by_kappa'][str(kappa)]['TPR_L3_count'] for s in SEEDS)
        tpr_l1 = tpr_l1_k / n_total
        tpr_l2 = tpr_l2_k / n_total
        tpr_l3 = tpr_l3_k / n_total
        ci1 = wilson(tpr_l1_k, n_total)
        ci2 = wilson(tpr_l2_k, n_total)
        ci3 = wilson(tpr_l3_k, n_total)
        results['aggregate'][f'kappa_{kappa}'] = {
            'attack_success_mean': float(succ), 'L2_mean': float(l2),
            'TPR_L1': tpr_l1, 'TPR_L1_CI95': list(ci1),
            'TPR_L2': tpr_l2, 'TPR_L2_CI95': list(ci2),
            'TPR_L3': tpr_l3, 'TPR_L3_CI95': list(ci3),
            'n_total': n_total,
        }
        print(f'    {kappa:>4.1f} |  {succ:.3f}  | {l2:.3f} | '
              f'{tpr_l1:.4f} [{ci1[0]:.3f},{ci1[1]:.3f}] | '
              f'{tpr_l2:.4f} | {tpr_l3:.4f}')

    _suffix = os.environ.get('PRISM_OUT_SUFFIX', '')
    _suffix = f'_{_suffix}' if _suffix else ''
    out_path = os.path.join(HERE, f'vastai_stronger_cw_canonical{_suffix}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nwrote {out_path}')


if __name__ == '__main__':
    main()
