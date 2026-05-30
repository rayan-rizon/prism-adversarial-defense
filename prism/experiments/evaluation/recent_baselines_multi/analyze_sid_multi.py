"""Aggregate SID across settings (WRN, CIFAR-100) and compare to PRISM + the
existing matched-FPR baselines (LID/Maha/ODIN/Energy) from baselines_multi.

Handles partial data (prints whatever seeds are present so far).
"""
import json, glob, os
import numpy as np

here = os.path.dirname(__file__)
ATTACKS = ['FGSM', 'PGD', 'Square']

# Reference rows already in the paper (tables/baselines_multi.tex) for context.
PRISM = {
    'wrn_cifar10':  {'FGSM': 0.985, 'PGD': 0.995, 'Square': 0.944, 'mean': 0.975, 'FPR': 0.073},
    'cifar100':     {'FGSM': 0.835, 'PGD': 0.984, 'Square': 0.672, 'mean': 0.830, 'FPR': 0.077},
    'cifar10_rn18': {'FGSM': 0.882, 'PGD': 0.987, 'Square': 0.886, 'mean': 0.918, 'FPR': 0.074},
}
BEST_BASELINE = {  # strongest non-PRISM baseline mean TPR per setting, for context
    'wrn_cifar10':  ('Mahalanobis', 0.810),
    'cifar100':     ('ODIN', 0.404),
    'cifar10_rn18': ('Mahalanobis', 0.582),
}

def agg(cfg):
    fs = sorted(glob.glob(os.path.join(here, f'results_sid_{cfg}_seed*.json')))
    rows = {a: [] for a in ATTACKS}
    fprs = []
    for f in fs:
        d = json.load(open(f, encoding='utf-8'))
        s = d.get('SID', {})
        for a in ATTACKS:
            if a in s:
                rows[a].append(s[a]['TPR']);
        # one FPR (SID FPR is attack-agnostic; take FGSM cell)
        if 'FGSM' in s:
            fprs.append(s['FGSM']['FPR'])
    n = len(fs)
    if n == 0:
        return None
    means = {a: float(np.mean(rows[a])) if rows[a] else None for a in ATTACKS}
    stds = {a: (float(np.std(rows[a], ddof=1)) if len(rows[a]) > 1 else 0.0) for a in ATTACKS}
    valid = [means[a] for a in ATTACKS if means[a] is not None]
    return {'n_seeds': n, 'means': means, 'stds': stds,
            'mean_tpr': float(np.mean(valid)) if valid else None,
            'fpr': float(np.mean(fprs)) if fprs else None}

# CIFAR-10/RN18 SID from the earlier recent_baselines run (already have it).
def cifar10_rn18_sid():
    p = os.path.join(here, '..', 'recent_baselines', 'results_baselines_recent_aggregate.json')
    if not os.path.exists(p):
        return None
    d = json.load(open(p, encoding='utf-8'))['aggregate'].get('SID', {})
    means = {a: d[a]['TPR_mean'] for a in ATTACKS if a in d}
    fpr = d.get('FGSM', {}).get('FPR_mean')
    valid = list(means.values())
    return {'n_seeds': 5, 'means': means, 'stds': {a: d[a]['TPR_std'] for a in ATTACKS if a in d},
            'mean_tpr': float(np.mean(valid)) if valid else None, 'fpr': fpr}

SETTINGS = [('CIFAR-10/ResNet-18', 'cifar10_rn18', cifar10_rn18_sid()),
            ('CIFAR-10/WRN-28-10', 'wrn_cifar10', agg('wrn_cifar10')),
            ('CIFAR-100/ResNet-18', 'cifar100', agg('cifar100'))]

for name, key, sid in SETTINGS:
    print(f"\n=== {name} ===")
    if sid is None:
        print("  SID: (no results yet)")
        continue
    m, sd = sid['means'], sid['stds']
    cells = "  ".join(f"{a}={m[a]:.3f}" if m[a] is not None else f"{a}=--" for a in ATTACKS)
    print(f"  SID   ({sid['n_seeds']} seed): {cells}  mean={sid['mean_tpr']:.3f}  FPR={sid['fpr']:.3f}")
    p = PRISM[key]
    print(f"  PRISM          : FGSM={p['FGSM']:.3f}  PGD={p['PGD']:.3f}  Square={p['Square']:.3f}  mean={p['mean']:.3f}  FPR={p['FPR']:.3f}")
    bb = BEST_BASELINE[key]
    if sid['mean_tpr'] is not None:
        print(f"  -> PRISM beats SID by +{p['mean']-sid['mean_tpr']:.3f} mean TPR; "
              f"strongest classic baseline={bb[0]} ({bb[1]:.3f})")
