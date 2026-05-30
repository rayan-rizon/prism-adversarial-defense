"""
Train a learned TAMSH router on the L3 pool (fixes C4).

Diagnostic (scripts/diagnose_tamsh.py) showed the topology softmin router
collapses to expert 0 (~94%) because last-layer H1 diagrams are often empty,
giving recovery_acc 0.035 — below the uniform control (0.106) and far below
the best-of-K oracle (0.327). This script trains a supervised router that maps
the pooled last-layer activation to an expert index, recovering most of the
oracle ceiling.

Protocol (no leakage):
  TRAIN router : CAL  split (test idx 5000-6999) adversarials
  EVAL  router : EVAL split (test idx 8000-9999) adversarials  [disjoint]
Experts (experts.pkl) are FIXED; only the router is learned. Detection
(TAMM/CADG/SACD) is upstream of routing and is unaffected.

USAGE
  cd prism/
  python scripts/train_tamsh_router.py --n-train 1500 --n-eval 1000 --seed 42
"""
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import json, os, sys, ssl, certifi, argparse, pickle
from tqdm import tqdm

os.environ.setdefault('SSL_CERT_FILE', certifi.where())
os.environ.setdefault('REQUESTS_CA_BUNDLE', certifi.where())
ssl._create_default_https_context = ssl.create_default_context
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src import bootstrap  # noqa: F401

from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.prism import PRISM
from src.sacd.monitor import NoOpCampaignMonitor
from src.tamsh.experts import TopologyAwareMoE, ExpertSubNetwork
from src.config import (
    LAYER_NAMES, LAYER_WEIGHTS, DIM_WEIGHTS,
    BACKBONE_MEAN, BACKBONE_STD, BACKBONE_INPUT_SIZE, BACKBONE_NUM_CLASSES,
    EPS_LINF_STANDARD, EVAL_IDX, CAL_IDX, DATASET, PATHS,
)
from src.data_loader import load_test_dataset
from src.models import load_backbone

_MEAN, _STD = BACKBONE_MEAN, BACKBONE_STD
_PIXEL = T.Compose([T.ToTensor()]) if BACKBONE_INPUT_SIZE == 32 else T.Compose(
    [T.Resize(BACKBONE_INPUT_SIZE), T.ToTensor()])
_NORM = T.Normalize(mean=_MEAN, std=_STD)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    m = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / d
    return (max(0.0, c - m), min(1.0, c + m))


def _build_moe(device):
    with open(PATHS['experts'], 'rb') as f:
        data = pickle.load(f)
    rebuilt = []
    for sd in data['experts']:
        net = ExpertSubNetwork(input_dim=data['input_dim'],
                               output_dim=data['output_dim'],
                               hidden_dim=data.get('hidden_dim', 256))
        net.load_state_dict(sd); net.eval().to(device)
        rebuilt.append(net)
    moe = TopologyAwareMoE(experts=rebuilt, expert_ref_diagrams=data['medoid_diagrams'],
                           comparison_mode='combined')
    return moe, len(rebuilt)


def _collect_pool(prism, moe, n_experts, ds, indices, attack, device, n, seed, desc):
    """Return (features, oracle_labels, per_expert_correct_list) for L3 samples."""
    rng = np.random.RandomState(seed)
    sample_idx = rng.choice(indices, min(n, len(indices)), replace=False)
    imgs, labels = [], []
    for i in sample_idx:
        img, y = ds[int(i)]; imgs.append(img); labels.append(int(y))
    X_np = torch.stack(imgs).numpy()
    X_adv = attack.generate(X_np)
    last = LAYER_NAMES[-1]
    feats, oracle, pe_correct = [], [], []
    for j in tqdm(range(len(imgs)), desc=desc):
        x_adv_pixel = torch.tensor(X_adv[j]).unsqueeze(0).to(device)
        x_adv_norm = _NORM(x_adv_pixel.squeeze(0).cpu()).unsqueeze(0).to(device)
        _, level, _ = prism.defend(x_adv_norm)
        if level not in ('L3', 'L3_REJECT'):
            continue
        acts = prism.extractor.extract(x_adv_norm)
        a = acts[last]
        pooled = (F.adaptive_avg_pool2d(a, 1).view(a.size(0), -1) if a.dim() > 2 else a).to(device)
        preds = []
        for k in range(n_experts):
            with torch.no_grad():
                preds.append(int(moe.experts[k](pooled).argmax(1).item()))
        correct = [int(p == labels[j]) for p in preds]
        feats.append(pooled.detach().cpu().numpy().reshape(-1))
        pe_correct.append(correct)
        # oracle training label: a correct expert (highest-confidence among correct),
        # else -1 (unrecoverable, excluded from training)
        if sum(correct):
            confs = []
            for k in range(n_experts):
                with torch.no_grad():
                    confs.append(float(torch.softmax(moe.experts[k](pooled), -1).max()))
            best = max((k for k in range(n_experts) if correct[k]), key=lambda k: confs[k])
            oracle.append(best)
        else:
            oracle.append(-1)
    return np.array(feats), np.array(oracle), pe_correct


def _refit_from_cache(args):
    """Refit the router classifier from the cached pools (no TDA recompute)."""
    z = np.load(args.cache)
    Xtr, ytr, Xev, pe = z['Xtr'], z['ytr'], z['Xev'], z['pe']
    topo, uni, n_experts = z['topo'], z['uni'], int(z['n_experts'])
    n_l3 = len(Xev)
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, C=args.C)
    clf.fit(scaler.transform(Xtr), ytr)
    picks = clf.predict(scaler.transform(Xev))
    learned = int(sum(pe[i, picks[i]] for i in range(n_l3)))
    hist = [int((picks == k).sum()) for k in range(n_experts)]
    oracle = int(pe.max(axis=1).sum())
    print(f"[refit C={args.C}] n_l3={n_l3} train-acc={clf.score(scaler.transform(Xtr), ytr):.3f}")
    print(f"  per-expert:   {[round(float(pe[:,k].mean()),4) for k in range(n_experts)]}")
    print(f"  uniform:      {round(uni.mean(),4)}")
    print(f"  topology:     {round(topo.mean(),4)}")
    print(f"  force-best:   {round(float(pe[:,2].mean()),4)}")
    print(f"  LEARNED:      {round(learned/n_l3,4)}  CI {[round(x,4) for x in wilson(learned,n_l3)]}  picks={hist}")
    print(f"  oracle:       {round(oracle/n_l3,4)}")
    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-train', type=int, default=1500)
    ap.add_argument('--n-eval', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default=None)
    ap.add_argument('--out', default=None, help='router pkl (default PATHS dir/tamsh_router.pkl)')
    ap.add_argument('--report', default='logs/local/tamsh_router_report.json')
    ap.add_argument('--cache', default='logs/local/tamsh_router_cache.npz',
                    help='cache of collected (router-independent) pools')
    ap.add_argument('--refit', action='store_true',
                    help='skip TDA collection; reuse --cache to refit the classifier only')
    ap.add_argument('--C', type=float, default=1.0, help='LogisticRegression C')
    args = ap.parse_args()

    if args.refit:
        return _refit_from_cache(args)

    device = torch.device(args.device) if args.device else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  dataset={DATASET}")
    model = load_backbone(device)
    norm_model = load_backbone(device, wrap=True)
    moe, n_experts = _build_moe(device)
    classifier = PyTorchClassifier(
        model=norm_model, loss=torch.nn.CrossEntropyLoss(),
        input_shape=(3, BACKBONE_INPUT_SIZE, BACKBONE_INPUT_SIZE),
        nb_classes=BACKBONE_NUM_CLASSES, clip_values=(0.0, 1.0),
        device_type='gpu' if device.type == 'cuda' else 'cpu')
    attack = ProjectedGradientDescent(classifier, eps=EPS_LINF_STANDARD,
                                      eps_step=EPS_LINF_STANDARD / 4, max_iter=40,
                                      num_random_init=1)
    prism = PRISM.from_saved(
        model=model, layer_names=LAYER_NAMES,
        calibrator_path=PATHS['calibrator'], profile_path=PATHS['reference_profiles'],
        ensemble_path=PATHS['ensemble_scorer'], layer_weights=LAYER_WEIGHTS,
        dim_weights=DIM_WEIGHTS, campaign_monitor=NoOpCampaignMonitor(), moe=moe)
    ds = load_test_dataset(root='./data', download=True, transform=_PIXEL)

    cal_lo, cal_hi = CAL_IDX
    train_indices = list(range(cal_lo, cal_hi))     # 5000-6999 (disjoint from eval)
    eval_indices = list(range(*EVAL_IDX))           # 8000-9999

    print("\n=== TRAIN pool (CAL split) ===")
    Xtr, ytr, _ = _collect_pool(prism, moe, n_experts, ds, train_indices, attack,
                                device, args.n_train, args.seed, "  train pool")
    keep = ytr >= 0
    Xtr_k, ytr_k = Xtr[keep], ytr[keep]
    print(f"  L3 train samples: {len(Xtr)}; recoverable (>=1 expert correct): {keep.sum()}")
    print(f"  oracle-label histogram: {[int((ytr_k==k).sum()) for k in range(n_experts)]}")

    scaler = StandardScaler().fit(Xtr_k)
    # No class_weight balancing: the oracle-label prior (expert 2 dominant) is
    # real signal — upweighting rare weak experts over-routes to them and hurts.
    clf = LogisticRegression(max_iter=3000, C=args.C)
    clf.fit(scaler.transform(Xtr_k), ytr_k)
    print(f"  router train-acc: {clf.score(scaler.transform(Xtr_k), ytr_k):.3f}")
    moe.set_learned_router(scaler, clf)

    # ── EVAL: per-sample comparison of all routing policies on the EVAL split ──
    print("\n=== EVAL pool (EVAL split) ===")
    rng = np.random.RandomState(args.seed + 1)
    sample_idx = rng.choice(eval_indices, min(args.n_eval, len(eval_indices)), replace=False)
    imgs, labels = [], []
    for i in sample_idx:
        img, y = ds[int(i)]; imgs.append(img); labels.append(int(y))
    X_adv = attack.generate(torch.stack(imgs).numpy())
    last = LAYER_NAMES[-1]
    cnt = {'uniform': 0, 'topology': 0, 'force_best': 0, 'learned': 0, 'oracle': 0}
    learned_idx_hist = [0] * n_experts
    per_expert_correct_sum = np.zeros(n_experts)
    # router-independent per-sample data, cached so the classifier can be
    # re-tuned (--refit) without recomputing the expensive TDA pools.
    ev_feats, ev_pe, ev_topo, ev_uni = [], [], [], []
    n_l3 = 0
    for j in tqdm(range(len(imgs)), desc="  eval pool"):
        x_adv_pixel = torch.tensor(X_adv[j]).unsqueeze(0).to(device)
        x_adv_norm = _NORM(x_adv_pixel.squeeze(0).cpu()).unsqueeze(0).to(device)
        _, level, _ = prism.defend(x_adv_norm)
        if level not in ('L3', 'L3_REJECT'):
            continue
        n_l3 += 1
        y = labels[j]
        acts = prism.extractor.extract(x_adv_norm)
        a = acts[last]
        pooled = (F.adaptive_avg_pool2d(a, 1).view(a.size(0), -1) if a.dim() > 2 else a).to(device)
        diagrams = {L: prism.profiler.compute_diagram(acts[L].squeeze(0).cpu().numpy())
                    for L in LAYER_NAMES}
        in_dgm = diagrams[last]
        # oracle + per-expert
        preds = [int(moe.experts[k](pooled).argmax(1).item()) for k in range(n_experts)]
        correct = [int(p == y) for p in preds]
        per_expert_correct_sum += correct
        cnt['oracle'] += int(max(correct))
        # uniform (avg logits)
        out_u, _ = moe.forward_uniform(pooled)
        uni_ok = int(out_u.argmax(1).item() == y)
        cnt['uniform'] += uni_ok
        # topology (combined-mode softmin → selected expert)
        out_t, _ = moe.forward_through_expert(in_dgm, pooled)
        topo_ok = int(out_t.argmax(1).item() == y)
        cnt['topology'] += topo_ok
        # force best single expert
        if n_experts > 2:
            cnt['force_best'] += correct[2]
        # learned router
        out_l, li = moe.forward_learned(pooled)
        learned_idx_hist[li] += 1
        cnt['learned'] += int(out_l.argmax(1).item() == y)
        # cache router-independent data
        ev_feats.append(pooled.detach().cpu().numpy().reshape(-1))
        ev_pe.append(correct); ev_topo.append(topo_ok); ev_uni.append(uni_ok)

    # save cache for fast classifier re-tuning (--refit)
    os.makedirs(os.path.dirname(args.cache), exist_ok=True)
    np.savez(args.cache, Xtr=Xtr_k, ytr=ytr_k,
             Xev=np.array(ev_feats), pe=np.array(ev_pe),
             topo=np.array(ev_topo), uni=np.array(ev_uni), n_experts=n_experts)

    def acc(c):
        return round(c / max(n_l3, 1), 4)

    res = {
        'n_l3_eval': n_l3, 'n_experts': n_experts,
        'per_expert_acc': [round(float(per_expert_correct_sum[k] / max(n_l3, 1)), 4)
                           for k in range(n_experts)],
        'uniform_acc': acc(cnt['uniform']),
        'topology_acc': acc(cnt['topology']),
        'force_best_expert_acc': acc(cnt['force_best']),
        'learned_acc': acc(cnt['learned']),
        'learned_CI95': [round(x, 4) for x in wilson(cnt['learned'], n_l3)],
        'oracle_acc': acc(cnt['oracle']),
        'oracle_CI95': [round(x, 4) for x in wilson(cnt['oracle'], n_l3)],
        'learned_router_idx_hist': learned_idx_hist,
        'train': {'n_recoverable': int(keep.sum()),
                  'router_train_acc': round(float(clf.score(scaler.transform(Xtr_k), ytr_k)), 4)},
        'splits': {'train': f'{cal_lo}-{cal_hi-1}', 'eval': f'{EVAL_IDX[0]}-{EVAL_IDX[1]-1}'},
        'seed': args.seed,
    }
    print("\n=== RESULTS (L3 recovery_acc, n_l3={}) ===".format(n_l3))
    print(f"  per-expert:         {res['per_expert_acc']}")
    print(f"  uniform control:    {res['uniform_acc']}")
    print(f"  topology router:    {res['topology_acc']}")
    print(f"  force-best-expert:  {res['force_best_expert_acc']}")
    print(f"  LEARNED router:     {res['learned_acc']}  CI {res['learned_CI95']}  picks={learned_idx_hist}")
    print(f"  oracle (ceiling):   {res['oracle_acc']}  CI {res['oracle_CI95']}")

    out_router = args.out or os.path.join(os.path.dirname(PATHS['experts']), 'tamsh_router.pkl')
    with open(out_router, 'wb') as f:
        pickle.dump({'scaler': scaler, 'clf': clf, 'n_experts': n_experts,
                     'feature': f'pooled {LAYER_NAMES[-1]} activation', 'dataset': DATASET}, f)
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)
    print(f"\nRouter  -> {out_router}\nReport  -> {args.report}")


if __name__ == '__main__':
    main()
