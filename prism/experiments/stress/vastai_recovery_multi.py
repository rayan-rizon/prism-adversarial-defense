"""
Vast.ai paper-grade multi-attack recovery: FGSM, Square, CW
on n=1000 per seed, 5 seeds (42, 123, 456, 789, 999), EVAL split.

For each attack:
  1. Generate adversarials (batched on GPU)
  2. Score with ensemble; collect L3-rejected subset
  3. Apply 4 recovery policies (reject, passthrough, uniform, topology, force_pgd)
  4. Report mean recovery accuracy with Wilson 95% CIs over pooled per-seed L3 counts

Output: vastai_recovery_multi.json
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
from src.tamsh.experts import TopologyAwareMoE, ExpertSubNetwork
from src.attacks.cw_torch import cw_l2_attack_torch
from src.config import (
    LAYER_NAMES, LAYER_WEIGHTS, DIM_WEIGHTS,
    BACKBONE_MEAN, BACKBONE_STD, BACKBONE_NUM_CLASSES, EPS_LINF_STANDARD,
    PATHS, N_SUBSAMPLE, MAX_DIM, EVAL_IDX, BACKBONE_INPUT_SIZE,
)
from src.data_loader import load_test_dataset
from src.models import load_backbone

from art.attacks.evasion import FastGradientMethod, SquareAttack
from art.estimators.classification import PyTorchClassifier

_PIXEL_TRANSFORM = T.Compose([T.ToTensor()])
_NORMALIZE = T.Normalize(mean=BACKBONE_MEAN, std=BACKBONE_STD)

N_PER_SEED = 1000
SEEDS = [42, 123, 456, 789, 999]
SQUARE_MAX_ITER = 5000  # paper-matching


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    halfw = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - halfw), min(1.0, centre + halfw))


def load_moe(device):
    data = pickle.load(open(PATHS['experts'], 'rb'))
    experts = []
    for sd in data['experts']:
        e = ExpertSubNetwork(data['input_dim'], data['output_dim'], data['hidden_dim'])
        e.load_state_dict(sd); e.eval().to(device)
        experts.append(e)
    moe = TopologyAwareMoE(
        experts=experts, expert_ref_diagrams=data['medoid_diagrams'],
        comparison_dim=1, comparison_mode='combined',
        h0_weight=1.0, h1_weight=1.0,
    )
    return moe, data['expert_names']


def compute_score_and_acts(backbone, profiler, extractor, ens, img_pixel, device):
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
    score = float(ens.score(dgms, image=img_np, grad_norm=gn, logits=logits_np,
                            logit_profile_features=lp, stability_features=stab))
    a4 = acts[LAYER_NAMES[-1]]
    pooled = torch.nn.functional.adaptive_avg_pool2d(a4, 1).flatten(1)
    return score, dgms, pooled


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

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
    L3 = calib.thresholds['L3']
    print(f'L3 threshold: {L3:.4f}')
    moe, expert_names = load_moe(device)
    print(f'experts: {expert_names}')

    ds = load_test_dataset(root='./data', transform=_PIXEL_TRANSFORM)
    eval_pool = list(range(*EVAL_IDX))

    classifier = PyTorchClassifier(
        model=norm_backbone, loss=torch.nn.CrossEntropyLoss(),
        input_shape=(3, BACKBONE_INPUT_SIZE, BACKBONE_INPUT_SIZE),
        nb_classes=BACKBONE_NUM_CLASSES, clip_values=(0.0, 1.0),
        device_type='gpu' if device == 'cuda' else 'cpu',
    )

    results = {
        'n_per_seed': N_PER_SEED, 'seeds': SEEDS,
        'split': 'EVAL', 'eval_idx_range': list(EVAL_IDX),
        'L3_threshold': float(L3),
        'per_seed': {}, 'aggregate': {},
    }

    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        pick = sorted(rng.choice(eval_pool, N_PER_SEED, replace=False).tolist())
        print(f'\n=== [seed {seed}] generating clean batch ===')
        clean_stack = torch.stack([ds[int(i)][0] for i in pick]).to(device)
        labels = torch.tensor([ds[int(i)][1] for i in pick]).to(device)

        # Generate adversarials
        attacks_adv = {}
        print(f'  FGSM...'); t0 = time.time()
        fgsm = FastGradientMethod(estimator=classifier, eps=EPS_LINF_STANDARD)
        attacks_adv['FGSM'] = torch.tensor(fgsm.generate(clean_stack.cpu().numpy())).to(device)
        print(f'    done in {time.time()-t0:.1f}s')

        print(f'  Square (max_iter={SQUARE_MAX_ITER})...'); t0 = time.time()
        sq = SquareAttack(estimator=classifier, eps=EPS_LINF_STANDARD,
                          max_iter=SQUARE_MAX_ITER, nb_restarts=1,
                          batch_size=min(250, N_PER_SEED))
        attacks_adv['Square'] = torch.tensor(sq.generate(clean_stack.cpu().numpy())).to(device)
        print(f'    done in {time.time()-t0:.1f}s')

        print(f'  CW (kappa=0)...'); t0 = time.time()
        adv_chunks = []
        for s in range(0, N_PER_SEED, 250):
            e = min(s + 250, N_PER_SEED)
            adv_b, _ = cw_l2_attack_torch(
                norm_backbone, clean_stack[s:e], device,
                max_iter=40, binary_search_steps=5,
                learning_rate=0.01, confidence=0.0, initial_const=0.01,
            )
            adv_chunks.append(adv_b.detach())
        attacks_adv['CW'] = torch.cat(adv_chunks, dim=0)
        print(f'    done in {time.time()-t0:.1f}s')

        results['per_seed'][str(seed)] = {'attacks': {}}

        for attack_name, adv_stack in attacks_adv.items():
            print(f'  --- {attack_name} (scoring + L3 routing) ---'); t0 = time.time()
            n_l3 = 0
            pass_corr = 0
            uni_corr = 0
            topo_corr = 0
            force_corr = 0
            oracle_corr = 0
            for i in range(N_PER_SEED):
                img = adv_stack[i].cpu()
                score, dgms, pooled = compute_score_and_acts(
                    backbone, profiler, extractor, ens, img, device,
                )
                if score <= L3:
                    continue
                n_l3 += 1
                true_lab = int(labels[i].item())
                # passthrough: base backbone prediction on adv
                with torch.no_grad():
                    base_pred = int(norm_backbone(adv_stack[i:i+1]).argmax(1).item())
                pass_corr += int(base_pred == true_lab)
                # uniform
                with torch.no_grad():
                    out_uni, _ = moe.forward_uniform(pooled)
                uni_corr += int(int(out_uni.argmax(1).item()) == true_lab)
                # topology (combined H0+H1)
                with torch.no_grad():
                    out_topo, _ = moe.forward_through_expert(
                        dgms[LAYER_NAMES[-1]], pooled,
                    )
                topo_corr += int(int(out_topo.argmax(1).item()) == true_lab)
                # force PGD specialist (expert index 2)
                with torch.no_grad():
                    out_force = moe.experts[2](pooled)
                force_corr += int(int(out_force.argmax(1).item()) == true_lab)
                # oracle: best of 4
                any_ok = 0
                for k in range(len(moe.experts)):
                    with torch.no_grad():
                        pk = int(moe.experts[k](pooled).argmax(1).item())
                    if pk == true_lab:
                        any_ok = 1; break
                oracle_corr += any_ok

            dt = time.time() - t0
            trigger_rate = n_l3 / N_PER_SEED
            results['per_seed'][str(seed)]['attacks'][attack_name] = {
                'L3_trigger_rate': trigger_rate, 'n_L3': n_l3,
                'pass_correct': pass_corr,
                'uniform_correct': uni_corr,
                'topology_correct': topo_corr,
                'force_pgd_correct': force_corr,
                'oracle_correct': oracle_corr,
                'score_route_time_s': round(dt, 1),
            }
            if n_l3 > 0:
                print(f'    trigger={trigger_rate:.3f} n_L3={n_l3} '
                      f'pass={pass_corr/n_l3:.3f} uni={uni_corr/n_l3:.3f} '
                      f'topo={topo_corr/n_l3:.3f} force={force_corr/n_l3:.3f} '
                      f'oracle={oracle_corr/n_l3:.3f}  ({dt:.1f}s)')
            else:
                print(f'    trigger={trigger_rate:.3f} n_L3=0  ({dt:.1f}s)')

    # Aggregate
    print('\n=== POOLED AGGREGATE (5 seeds) ===')
    for attack in ['FGSM', 'Square', 'CW']:
        total_l3 = sum(results['per_seed'][str(s)]['attacks'][attack]['n_L3'] for s in SEEDS)
        if total_l3 == 0:
            results['aggregate'][attack] = {'n_L3_total': 0}
            print(f'{attack}: 0 L3 across all seeds')
            continue
        pass_t = sum(results['per_seed'][str(s)]['attacks'][attack]['pass_correct'] for s in SEEDS)
        uni_t = sum(results['per_seed'][str(s)]['attacks'][attack]['uniform_correct'] for s in SEEDS)
        topo_t = sum(results['per_seed'][str(s)]['attacks'][attack]['topology_correct'] for s in SEEDS)
        force_t = sum(results['per_seed'][str(s)]['attacks'][attack]['force_pgd_correct'] for s in SEEDS)
        oracle_t = sum(results['per_seed'][str(s)]['attacks'][attack]['oracle_correct'] for s in SEEDS)
        trigger_mean = np.mean([results['per_seed'][str(s)]['attacks'][attack]['L3_trigger_rate'] for s in SEEDS])
        agg = {
            'n_L3_total': total_l3,
            'L3_trigger_rate_mean': float(trigger_mean),
            'pass_acc': pass_t / total_l3, 'pass_CI95': list(wilson(pass_t, total_l3)),
            'uniform_acc': uni_t / total_l3, 'uniform_CI95': list(wilson(uni_t, total_l3)),
            'topology_acc': topo_t / total_l3, 'topology_CI95': list(wilson(topo_t, total_l3)),
            'force_pgd_acc': force_t / total_l3, 'force_pgd_CI95': list(wilson(force_t, total_l3)),
            'oracle_acc': oracle_t / total_l3, 'oracle_CI95': list(wilson(oracle_t, total_l3)),
            'gap_topo_vs_pass_pp': (topo_t / total_l3 - pass_t / total_l3) * 100,
            'gap_uniform_vs_pass_pp': (uni_t / total_l3 - pass_t / total_l3) * 100,
            'gap_force_vs_pass_pp': (force_t / total_l3 - pass_t / total_l3) * 100,
            'gap_oracle_vs_pass_pp': (oracle_t / total_l3 - pass_t / total_l3) * 100,
        }
        results['aggregate'][attack] = agg
        print(f'{attack:>6s}: n_L3={total_l3}/{N_PER_SEED*len(SEEDS)}  trig={trigger_mean:.3f}  '
              f'pass={agg["pass_acc"]:.3f}  uni={agg["uniform_acc"]:.3f}  '
              f'topo={agg["topology_acc"]:.3f} ({agg["gap_topo_vs_pass_pp"]:+.1f}pp)  '
              f'force={agg["force_pgd_acc"]:.3f}  oracle={agg["oracle_acc"]:.3f}')

    _suffix = os.environ.get('PRISM_OUT_SUFFIX', '')
    _suffix = f'_{_suffix}' if _suffix else ''
    out_path = os.path.join(HERE, f'vastai_recovery_multi{_suffix}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nwrote {out_path}')


if __name__ == '__main__':
    main()
