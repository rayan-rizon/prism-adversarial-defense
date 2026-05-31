"""
Audit small tests (local), reusing the EXACT deployed detection pipeline
(build_prism + prism.defend non-PASS == detection) from run_ablation_paper.py.

TEST A -- PGD-50x10 sensitivity.
  The deployed main-table PGD number (0.987) was generated at PGD-40x1
  (run_ablation_paper.py). We re-run at the RobustBench-standard PGD-50x10 on
  the SAME per-seed image samples (deterministic rng) so the comparison is
  paired, and check whether PRISM's PGD detection TPR holds.

TEST B -- Disjoint eval windows (honest CIs).
  The main numbers subsample n=1000 from the shared [8000,10000] window per
  seed, so the 5 pools overlap (~1929 unique). Here each seed gets a DISJOINT
  400-image window; pooling the five gives n=2000 independent draws, so the
  pooled Wilson CI is honest. Point estimates should match the main table;
  the CI is the deliverable.

Both modes write per-unit checkpoints and a final aggregate JSON.

USAGE
  cd prism/
  python experiments/audit/run_audit_tests.py --test pgd50      --n 1000
  python experiments/audit/run_audit_tests.py --test disjoint   --window 400
  python experiments/audit/run_audit_tests.py --test both
"""
import os, sys, json, argparse, ssl, certifi
import numpy as np
import torch
import torchvision.transforms as T

os.environ.setdefault('SSL_CERT_FILE', certifi.where())
os.environ.setdefault('REQUESTS_CA_BUNDLE', certifi.where())
ssl._create_default_https_context = ssl.create_default_context
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src import bootstrap  # noqa: F401

from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, SquareAttack
from art.estimators.classification import PyTorchClassifier
from src.config import (EVAL_IDX, BACKBONE_INPUT_SIZE, BACKBONE_NUM_CLASSES,
                        EPS_LINF_STANDARD, DATASET)
from src.data_loader import load_test_dataset
from src.models import load_backbone
# Reuse the faithful, deployed detection helpers.
from experiments.ablation.run_ablation_paper import (
    build_prism, batch_generate_adversarials, _NORMALIZE, wilson_ci, EPS,
)

_PIXEL = T.Compose([T.ToTensor()]) if BACKBONE_INPUT_SIZE == 32 else T.Compose(
    [T.Resize(BACKBONE_INPUT_SIZE), T.ToTensor()])
FULL_CFG = {'use_ensemble': True, 'use_moe': True, 'tda_only': False}
SEEDS = [42, 123, 456, 789, 999]


def _classifier(device):
    norm = load_backbone(device, wrap=True)
    return PyTorchClassifier(
        model=norm, loss=torch.nn.CrossEntropyLoss(),
        input_shape=(3, BACKBONE_INPUT_SIZE, BACKBONE_INPUT_SIZE),
        nb_classes=BACKBONE_NUM_CLASSES, clip_values=(0.0, 1.0),
        device_type='gpu' if device.type == 'cuda' else 'cpu')


def _build_attacks(clf, names):
    reg = {
        'FGSM':  lambda: FastGradientMethod(clf, eps=EPS),
        'PGD40': lambda: ProjectedGradientDescent(clf, eps=EPS, eps_step=EPS/4,
                                                  max_iter=40, num_random_init=1),
        'PGD50': lambda: ProjectedGradientDescent(clf, eps=EPS, eps_step=EPS/4,
                                                  max_iter=50, num_random_init=10),
        'Square': lambda: SquareAttack(clf, eps=EPS, max_iter=5000, nb_restarts=1),
    }
    return {n: reg[n]() for n in names}


def _clean_counts(prism, imgs):
    fp = tn = 0
    for img in imgs:
        _, lv, _ = prism.defend(_NORMALIZE(img).unsqueeze(0).to(_DEV))
        if lv == 'PASS': tn += 1
        else: fp += 1
    return fp, tn


def _adv_counts(prism, X_adv_np):
    tp = fn = 0
    for j in range(len(X_adv_np)):
        _, lv, _ = prism.defend(_NORMALIZE(torch.tensor(X_adv_np[j])).unsqueeze(0).to(_DEV))
        if lv != 'PASS': tp += 1
        else: fn += 1
    return tp, fn


def _eval_unit(prism, clf, imgs, attack_names):
    """Clean detection once + adv detection per attack on one image set."""
    fp, tn = _clean_counts(prism, imgs)
    adv = batch_generate_adversarials(_build_attacks(clf, attack_names), imgs)
    out = {'_clean': {'FP': fp, 'TN': tn, 'FPR': fp / max(fp + tn, 1)}}
    for name in attack_names:
        tp, fn = _adv_counts(prism, adv[name])
        out[name] = {'TP': tp, 'FN': fn, 'TPR': tp / max(tp + fn, 1)}
    return out


def test_pgd50(n, outdir):
    """Per-seed random sample (paired w/ deployed), PGD40 vs PGD50."""
    os.makedirs(outdir, exist_ok=True)
    ds = load_test_dataset(root='./data', download=True, transform=_PIXEL)
    model = load_backbone(_DEV)
    clf = _classifier(_DEV)
    prism = build_prism(FULL_CFG, model, _DEV)
    eval_pool = list(range(*EVAL_IDX))
    per_seed = {}
    for s in SEEDS:
        ck = os.path.join(outdir, f'pgd50_seed{s}.json')
        if os.path.exists(ck):
            per_seed[s] = json.load(open(ck)); print(f"[pgd50 seed {s}] cached"); continue
        rng = np.random.RandomState(s)
        idx = rng.choice(eval_pool, min(n, len(eval_pool)), replace=False)
        imgs = [ds[int(i)][0] for i in idx]
        print(f"[pgd50 seed {s}] n={len(imgs)} -- generating + detecting...")
        r = _eval_unit(prism, clf, imgs, ['PGD40', 'PGD50'])
        json.dump(r, open(ck, 'w'), indent=2); per_seed[s] = r
        print(f"  seed {s}: PGD40 TPR={r['PGD40']['TPR']:.4f}  PGD50 TPR={r['PGD50']['TPR']:.4f}  FPR={r['_clean']['FPR']:.4f}")
    # aggregate
    agg = {}
    for atk in ['PGD40', 'PGD50']:
        tprs = [per_seed[s][atk]['TPR'] for s in SEEDS]
        tp = sum(per_seed[s][atk]['TP'] for s in SEEDS); fn = sum(per_seed[s][atk]['FN'] for s in SEEDS)
        agg[atk] = {'TPR_mean': round(float(np.mean(tprs)), 4),
                    'TPR_std': round(float(np.std(tprs, ddof=1)), 4),
                    'pooled_TPR': round(tp / max(tp + fn, 1), 4),
                    'pooled_CI95': [round(v, 4) for v in wilson_ci(tp, tp + fn)],
                    'per_seed': [round(x, 4) for x in tprs]}
    fp = sum(per_seed[s]['_clean']['FP'] for s in SEEDS); tn = sum(per_seed[s]['_clean']['TN'] for s in SEEDS)
    agg['_clean_FPR_pooled'] = round(fp / max(fp + tn, 1), 4)
    agg['_note'] = 'PGD40 should reproduce deployed 0.987; PGD50 = RobustBench standard'
    json.dump(agg, open(os.path.join(outdir, 'pgd50_aggregate.json'), 'w'), indent=2)
    print("\n=== TEST A (PGD-50x10) ===")
    print(f"  PGD40 (deployed config): TPR {agg['PGD40']['TPR_mean']} (pooled {agg['PGD40']['pooled_TPR']})")
    print(f"  PGD50 (field standard):  TPR {agg['PGD50']['TPR_mean']} (pooled {agg['PGD50']['pooled_TPR']}) CI {agg['PGD50']['pooled_CI95']}")
    print(f"  clean FPR pooled: {agg['_clean_FPR_pooled']}")
    return agg


def test_disjoint(window, outdir):
    """5 disjoint windows of `window` imgs -> independent pooled CI."""
    os.makedirs(outdir, exist_ok=True)
    ds = load_test_dataset(root='./data', download=True, transform=_PIXEL)
    model = load_backbone(_DEV)
    clf = _classifier(_DEV)
    prism = build_prism(FULL_CFG, model, _DEV)
    lo, hi = EVAL_IDX
    attacks = ['FGSM', 'PGD40', 'Square']
    per_win = {}
    for wi, s in enumerate(SEEDS):
        ck = os.path.join(outdir, f'disjoint_win{wi}.json')
        if os.path.exists(ck):
            per_win[wi] = json.load(open(ck)); print(f"[disjoint win {wi}] cached"); continue
        w_lo = lo + wi * window
        idx = list(range(w_lo, min(w_lo + window, hi)))
        imgs = [ds[int(i)][0] for i in idx]
        print(f"[disjoint win {wi}] idx[{w_lo}-{w_lo+len(imgs)-1}] n={len(imgs)} ...")
        r = _eval_unit(prism, clf, imgs, attacks)
        r['_idx_range'] = [w_lo, w_lo + len(imgs)]
        json.dump(r, open(ck, 'w'), indent=2); per_win[wi] = r
        print(f"  win {wi}: " + " ".join(f"{a}={r[a]['TPR']:.3f}" for a in attacks) + f"  FPR={r['_clean']['FPR']:.3f}")
    agg = {}
    for atk in attacks:
        tp = sum(per_win[w][atk]['TP'] for w in per_win); fn = sum(per_win[w][atk]['FN'] for w in per_win)
        tprs = [per_win[w][atk]['TPR'] for w in per_win]
        agg[atk] = {'pooled_TPR': round(tp / max(tp + fn, 1), 4),
                    'pooled_CI95_independent': [round(v, 4) for v in wilson_ci(tp, tp + fn)],
                    'per_window_TPR': [round(x, 4) for x in tprs],
                    'window_std': round(float(np.std(tprs, ddof=1)), 4), 'n_unique': tp + fn}
    fp = sum(per_win[w]['_clean']['FP'] for w in per_win); tn = sum(per_win[w]['_clean']['TN'] for w in per_win)
    agg['_clean_FPR_pooled'] = round(fp / max(fp + tn, 1), 4)
    agg['_note'] = 'Disjoint windows -> independent pooled draws; CI is honest (cf. overlapping main table).'
    json.dump(agg, open(os.path.join(outdir, 'disjoint_aggregate.json'), 'w'), indent=2)
    print("\n=== TEST B (disjoint windows) ===")
    for atk in attacks:
        print(f"  {atk:7} pooled TPR {agg[atk]['pooled_TPR']}  honest CI {agg[atk]['pooled_CI95_independent']}  (n_unique={agg[atk]['n_unique']})")
    print(f"  clean FPR pooled: {agg['_clean_FPR_pooled']}")
    return agg


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', choices=['pgd50', 'disjoint', 'both'], default='both')
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--window', type=int, default=400)
    ap.add_argument('--outdir', default='experiments/audit/results')
    ap.add_argument('--device', default=None)
    args = ap.parse_args()
    _DEV = torch.device(args.device) if args.device else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    globals()['_DEV'] = _DEV
    print(f"Device: {_DEV}  dataset={DATASET}  eps={EPS:.5f}")
    if args.test in ('pgd50', 'both'):
        test_pgd50(args.n, args.outdir)
    if args.test in ('disjoint', 'both'):
        test_disjoint(args.window, args.outdir)
    print("\nALL AUDIT TESTS DONE")
