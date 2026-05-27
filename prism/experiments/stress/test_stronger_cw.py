"""
P2: Stronger CW evaluation (confidence margin kappa sweep).

The paper's canonical CW configuration uses confidence=0 (Table~main).
Reviewers may ask: "what about high-confidence CW with margin > 0?"
This script reruns CW with kappa in {0, 10, 20} on the same VAL-split
clean inputs and reports TPR at the L1/L2/L3 conformal thresholds.

Output: experiments/stress/results_stronger_cw.json
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
from src.attacks.cw_torch import cw_l2_attack_torch
from src.config import (
    LAYER_NAMES, LAYER_WEIGHTS, DIM_WEIGHTS,
    BACKBONE_MEAN, BACKBONE_STD, PATHS, N_SUBSAMPLE, MAX_DIM, VAL_IDX,
)
from src.data_loader import load_test_dataset
from src.models import load_backbone


_PIXEL_TRANSFORM = T.Compose([T.ToTensor()])
_NORMALIZE = T.Normalize(mean=BACKBONE_MEAN, std=BACKBONE_STD)


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
    return ens.score(dgms, image=img_np, grad_norm=gn, logits=logits_np,
                     logit_profile_features=lp, stability_features=stab)


def main():
    n = 100
    kappas = [0.0, 10.0, 20.0]
    seed = 42

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

    # Load components
    backbone = load_backbone(device=device)
    norm_backbone = load_backbone(device=device, wrap=True)  # for CW (pixel input)
    extractor = ActivationExtractor(backbone, LAYER_NAMES)
    profiler = TopologicalProfiler(n_subsample=N_SUBSAMPLE, max_dim=MAX_DIM)
    with open(PATHS['reference_profiles'], 'rb') as f:
        ref = pickle.load(f)
    base = TopologicalScorer(ref_profiles=ref, layer_names=LAYER_NAMES,
                             layer_weights=LAYER_WEIGHTS, dim_weights=DIM_WEIGHTS)
    ens = PersistenceEnsembleScorer.load(PATHS['ensemble_scorer'], base, LAYER_NAMES)
    with open(PATHS['calibrator'], 'rb') as f:
        calib = pickle.load(f)
    thresholds = calib.thresholds
    print(f'thresholds: {thresholds}')

    ds = load_test_dataset(root='./data', transform=_PIXEL_TRANSFORM)
    rng = np.random.RandomState(seed)
    pick = sorted(rng.choice(list(range(*VAL_IDX)), n, replace=False).tolist())

    # Stack clean
    clean_stack = torch.stack([ds[int(i)][0] for i in pick]).to(device)
    print(f'clean batch: {clean_stack.shape}')

    # Clean scores for FPR
    print(f'\nscoring clean (n={n})...')
    clean_scores = np.array([
        compute_score(backbone, profiler, extractor, ens, ds[int(i)][0], device)
        for i in pick
    ])
    fpr_l1 = float(np.mean(clean_scores > thresholds['L1']))
    fpr_l2 = float(np.mean(clean_scores > thresholds['L2']))
    fpr_l3 = float(np.mean(clean_scores > thresholds['L3']))
    print(f'clean mean={clean_scores.mean():.3f}  '
          f'FPR L1={fpr_l1:.3f}  L2={fpr_l2:.3f}  L3={fpr_l3:.3f}')

    results = {
        'n': n, 'seed': seed, 'split': 'VAL', 'eval_idx_range': VAL_IDX,
        'thresholds': {k: float(v) for k, v in thresholds.items()},
        'clean': {
            'mean': float(clean_scores.mean()), 'std': float(clean_scores.std()),
            'FPR_L1': fpr_l1, 'FPR_L2': fpr_l2, 'FPR_L3': fpr_l3,
        },
        'cw_runs': [],
    }

    for kappa in kappas:
        print(f'\n--- CW kappa={kappa} ---')
        t0 = time.time()
        adv_pixel, stats = cw_l2_attack_torch(
            norm_backbone, clean_stack, device,
            max_iter=40, binary_search_steps=5,
            learning_rate=0.01, confidence=kappa, initial_const=0.01,
        )
        attack_dt = time.time() - t0
        # CW attack success: ARG max(adv_logits) != arg max(clean_logits)
        with torch.no_grad():
            clean_logits = norm_backbone(clean_stack)
            adv_logits = norm_backbone(adv_pixel)
        clean_pred = clean_logits.argmax(1)
        adv_pred = adv_logits.argmax(1)
        attack_success = float((adv_pred != clean_pred).float().mean().item())
        # L2 distortion
        l2 = (adv_pixel - clean_stack).flatten(1).norm(dim=1).mean().item()
        print(f'  attack_success={attack_success:.3f}  mean_L2={l2:.4f}  time={attack_dt:.1f}s')

        # Score adversarials
        adv_scores = np.array([
            compute_score(backbone, profiler, extractor, ens,
                          adv_pixel[i].cpu(), device)
            for i in range(n)
        ])
        tpr_l1 = float(np.mean(adv_scores > thresholds['L1']))
        tpr_l2 = float(np.mean(adv_scores > thresholds['L2']))
        tpr_l3 = float(np.mean(adv_scores > thresholds['L3']))
        print(f'  adv mean={adv_scores.mean():.3f}  '
              f'TPR L1={tpr_l1:.3f}  L2={tpr_l2:.3f}  L3={tpr_l3:.3f}')

        results['cw_runs'].append({
            'kappa': kappa,
            'attack_success': attack_success,
            'mean_L2': float(l2),
            'time_s': round(attack_dt, 1),
            'adv_mean_score': float(adv_scores.mean()),
            'adv_std_score': float(adv_scores.std()),
            'TPR_L1': tpr_l1, 'TPR_L2': tpr_l2, 'TPR_L3': tpr_l3,
        })

    out_path = os.path.join(HERE, 'results_stronger_cw.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nwrote {out_path}')

    print('\n  kappa | attack_success | mean_L2 | TPR(L1) | TPR(L2) | TPR(L3)')
    print('  ' + '-' * 70)
    for r in results['cw_runs']:
        print(f'    {r["kappa"]:>4.1f} |    {r["attack_success"]:.3f}      | '
              f' {r["mean_L2"]:.3f}  |  {r["TPR_L1"]:.3f}  |  {r["TPR_L2"]:.3f}  |  {r["TPR_L3"]:.3f}')


if __name__ == '__main__':
    main()
