"""
P1: Extend TAMSH recovery beyond PGD-only.

The paper's recovery table (Table~tab:recovery) reports recovery accuracy
on the PGD-L3-rejected subset only. This test runs the same protocol on
FGSM, Square, and CW-L2 (kappa=0) adversarials, on n=100 VAL-split
inputs. Reports passthrough, uniform-router, and topology-router
(combined H0+H1) recovery accuracies per attack.

Outputs: experiments/stress/results_recovery_multi_attack.json
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
from src.tamsh.experts import TopologyAwareMoE, ExpertSubNetwork
from src.attacks.cw_torch import cw_l2_attack_torch
from src.config import (
    LAYER_NAMES, LAYER_WEIGHTS, DIM_WEIGHTS,
    BACKBONE_MEAN, BACKBONE_STD, BACKBONE_NUM_CLASSES, EPS_LINF_STANDARD,
    PATHS, N_SUBSAMPLE, MAX_DIM, VAL_IDX, BACKBONE_INPUT_SIZE,
)
from src.data_loader import load_test_dataset
from src.models import load_backbone

try:
    from art.attacks.evasion import FastGradientMethod, SquareAttack
    from art.estimators.classification import PyTorchClassifier
except ImportError:
    print('ERROR: ART not installed. Recovery test needs FGSM/Square attacks.')
    sys.exit(1)


_PIXEL_TRANSFORM = T.Compose([T.ToTensor()])
_NORMALIZE = T.Normalize(mean=BACKBONE_MEAN, std=BACKBONE_STD)


def load_moe(device):
    data = pickle.load(open(PATHS['experts'], 'rb'))
    experts = []
    for sd in data['experts']:
        e = ExpertSubNetwork(data['input_dim'], data['output_dim'], data['hidden_dim'])
        e.load_state_dict(sd)
        e.eval().to(device)
        experts.append(e)
    moe = TopologyAwareMoE(
        experts=experts,
        expert_ref_diagrams=data['medoid_diagrams'],
        comparison_dim=1,
        comparison_mode='combined',
        h0_weight=1.0,
        h1_weight=1.0,
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
    score = ens.score(dgms, image=img_np, grad_norm=gn, logits=logits_np,
                      logit_profile_features=lp, stability_features=stab)
    # layer4 pooled activation for expert input (matches training)
    a4 = acts[LAYER_NAMES[-1]]  # (1, C, H, W)
    pooled = torch.nn.functional.adaptive_avg_pool2d(a4, 1).flatten(1)  # (1, C)
    return float(score), dgms, pooled


def recovery_accuracy(moe, dgms, activation, true_label, mode):
    """mode: 'pass', 'uniform', 'topology', 'force_pgd'"""
    if mode == 'pass':
        # No recovery — using base backbone prediction post-attack
        return None  # handled separately
    if mode == 'uniform':
        out, _ = moe.forward_uniform(activation)
    elif mode == 'topology':
        out, _ = moe.forward_through_expert(dgms[LAYER_NAMES[-1]], activation)
    elif mode == 'force_pgd':
        # Index 2 = 'expert2_pgd'
        with torch.no_grad():
            out = moe.experts[2](activation)
    pred = int(out.argmax(1).item())
    return int(pred == true_label)


def main():
    n = 80
    seed = 42

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
    thresholds = calib.thresholds
    L3 = thresholds['L3']
    print(f'L3 threshold: {L3:.4f}')

    moe, expert_names = load_moe(device)
    print(f'experts: {expert_names}')

    ds = load_test_dataset(root='./data', transform=_PIXEL_TRANSFORM)
    rng = np.random.RandomState(seed)
    pick = sorted(rng.choice(list(range(*VAL_IDX)), n, replace=False).tolist())
    clean_stack = torch.stack([ds[int(i)][0] for i in pick]).to(device)
    labels = torch.tensor([ds[int(i)][1] for i in pick]).to(device)
    print(f'clean batch: {clean_stack.shape}, labels[:10]={labels[:10].tolist()}')

    # ART classifier (for FGSM, Square)
    classifier = PyTorchClassifier(
        model=norm_backbone, loss=torch.nn.CrossEntropyLoss(),
        input_shape=(3, BACKBONE_INPUT_SIZE, BACKBONE_INPUT_SIZE),
        nb_classes=BACKBONE_NUM_CLASSES,
        clip_values=(0.0, 1.0),
        device_type='gpu' if device == 'cuda' else 'cpu',
    )

    attacks = {}
    # FGSM
    print('\ngenerating FGSM...')
    t0 = time.time()
    fgsm = FastGradientMethod(estimator=classifier, eps=EPS_LINF_STANDARD)
    attacks['FGSM'] = torch.tensor(fgsm.generate(clean_stack.cpu().numpy())).to(device)
    print(f'  done in {time.time()-t0:.1f}s')

    # Square (reduced max_iter for local speed)
    print('generating Square (max_iter=500)...')
    t0 = time.time()
    sq = SquareAttack(estimator=classifier, eps=EPS_LINF_STANDARD,
                      max_iter=500, nb_restarts=1, batch_size=n)
    attacks['Square'] = torch.tensor(sq.generate(clean_stack.cpu().numpy())).to(device)
    print(f'  done in {time.time()-t0:.1f}s')

    # CW
    print('generating CW (kappa=0, max_iter=40, bss=5)...')
    t0 = time.time()
    adv_cw, _ = cw_l2_attack_torch(
        norm_backbone, clean_stack, device,
        max_iter=40, binary_search_steps=5,
        learning_rate=0.01, confidence=0.0, initial_const=0.01,
    )
    attacks['CW'] = adv_cw.detach()
    print(f'  done in {time.time()-t0:.1f}s')

    # Run protocol per attack
    results = {
        'n': n, 'seed': seed, 'split': 'VAL',
        'L3_threshold': float(L3),
        'attacks': {},
    }
    for attack_name, adv_stack in attacks.items():
        print(f'\n=== {attack_name} ===')
        # Score all adversarials; collect L3-triggered subset
        l3_indices = []
        adv_acts_l3 = []
        adv_dgms_l3 = []
        true_labels_l3 = []
        # Also track base classifier prediction (passthrough recovery)
        base_correct_l3 = []
        for i in range(n):
            img = adv_stack[i].cpu()
            score, dgms, pooled = compute_score_and_acts(
                backbone, profiler, extractor, ens, img, device,
            )
            if score > L3:
                l3_indices.append(i)
                adv_acts_l3.append(pooled)
                adv_dgms_l3.append(dgms)
                # Base classifier prediction on adversarial
                with torch.no_grad():
                    pred = int(norm_backbone(adv_stack[i:i+1]).argmax(1).item())
                base_correct_l3.append(int(pred == labels[i].item()))
                true_labels_l3.append(int(labels[i].item()))
        n_l3 = len(l3_indices)
        trigger_rate = n_l3 / n
        print(f'  L3 trigger rate: {trigger_rate:.3f} ({n_l3}/{n})')
        if n_l3 == 0:
            print('  (no L3-rejected samples, recovery undefined)')
            results['attacks'][attack_name] = {
                'L3_trigger_rate': trigger_rate, 'n_L3': 0,
            }
            continue

        # Compute recovery under each policy
        pass_acc = float(np.mean(base_correct_l3))
        uniform_correct = [
            recovery_accuracy(moe, dgms, act, lab, 'uniform')
            for dgms, act, lab in zip(adv_dgms_l3, adv_acts_l3, true_labels_l3)
        ]
        topo_correct = [
            recovery_accuracy(moe, dgms, act, lab, 'topology')
            for dgms, act, lab in zip(adv_dgms_l3, adv_acts_l3, true_labels_l3)
        ]
        force_correct = [
            recovery_accuracy(moe, dgms, act, lab, 'force_pgd')
            for dgms, act, lab in zip(adv_dgms_l3, adv_acts_l3, true_labels_l3)
        ]
        # Oracle: per-input best-of-K
        oracle_correct = []
        for dgms, act, lab in zip(adv_dgms_l3, adv_acts_l3, true_labels_l3):
            any_correct = 0
            for k in range(len(moe.experts)):
                with torch.no_grad():
                    pred = int(moe.experts[k](act).argmax(1).item())
                if pred == lab:
                    any_correct = 1
                    break
            oracle_correct.append(any_correct)

        uni_acc = float(np.mean(uniform_correct))
        topo_acc = float(np.mean(topo_correct))
        force_acc = float(np.mean(force_correct))
        oracle_acc = float(np.mean(oracle_correct))
        print(f'  passthrough recovery: {pass_acc:.3f}')
        print(f'  uniform router:       {uni_acc:.3f}  '
              f'(gap vs pass: {(uni_acc-pass_acc)*100:+.1f}pp)')
        print(f'  topology router:      {topo_acc:.3f}  '
              f'(gap vs pass: {(topo_acc-pass_acc)*100:+.1f}pp)')
        print(f'  force-PGD-expert:     {force_acc:.3f}  '
              f'(gap vs pass: {(force_acc-pass_acc)*100:+.1f}pp)')
        print(f'  oracle (best of 4):   {oracle_acc:.3f}  '
              f'(gap vs pass: {(oracle_acc-pass_acc)*100:+.1f}pp)')

        results['attacks'][attack_name] = {
            'L3_trigger_rate': trigger_rate, 'n_L3': n_l3,
            'passthrough': pass_acc, 'uniform': uni_acc,
            'topology': topo_acc, 'force_pgd': force_acc,
            'oracle': oracle_acc,
            'gap_topo_vs_pass_pp': (topo_acc - pass_acc) * 100,
            'gap_force_vs_pass_pp': (force_acc - pass_acc) * 100,
            'gap_oracle_vs_pass_pp': (oracle_acc - pass_acc) * 100,
        }

    out_path = os.path.join(HERE, 'results_recovery_multi_attack.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nwrote {out_path}')

    # Summary table
    print('\n  attack | trig rate | passthrough | topology | force_PGD | oracle')
    print('  ' + '-' * 70)
    for k, r in results['attacks'].items():
        if r.get('n_L3', 0) == 0:
            print(f'  {k:>6s} | n/a (no L3)')
            continue
        print(f'  {k:>6s} |   {r["L3_trigger_rate"]:.3f}   '
              f'|    {r["passthrough"]:.3f}    |  {r["topology"]:.3f}  '
              f'|   {r["force_pgd"]:.3f}   | {r["oracle"]:.3f}')


if __name__ == '__main__':
    main()
