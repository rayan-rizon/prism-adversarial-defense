"""Average the SpectralDefense cross-attack TPR matrix across seeds and
report the in-distribution (diagonal) vs cross-attack (off-diagonal) gap.

Diagonal  = detector trained AND evaluated on the same attack (in-dist).
Off-diag  = trained on attack A, evaluated on attack B (transfer).
"""
import json, glob, os
import numpy as np

here = os.path.dirname(__file__)
files = sorted(glob.glob(os.path.join(here, 'results_baselines_recent_seed*.json')))
attacks = ['FGSM', 'PGD', 'Square', 'CW']

# mat[train][eval] -> list of per-seed L1 TPR
acc = {t: {e: [] for e in attacks} for t in attacks}
for f in files:
    d = json.load(open(f, encoding='utf-8'))
    cm = d.get('_spectral_cross_attack', {})
    for t in attacks:
        for e in attacks:
            v = cm.get(t, {}).get(e, {}).get('L1', {}).get('TPR')
            if v is not None:
                acc[t][e].append(v)

print(f"SpectralDefense cross-attack L1 TPR (mean over {len(files)} seeds)")
print("rows=train, cols=eval\n")
print("train\\eval " + "".join(f"{e:>9}" for e in attacks))
for t in attacks:
    row = f"{t:>9} "
    for e in attacks:
        row += f"{np.mean(acc[t][e]):>9.3f}"
    print(row)

diag = [np.mean(acc[a][a]) for a in attacks]
offs = [np.mean(acc[t][e]) for t in attacks for e in attacks if t != e]
print(f"\nin-dist (diagonal) mean TPR     = {np.mean(diag):.3f}")
print(f"cross-attack (off-diag) mean TPR = {np.mean(offs):.3f}")
print(f"in-dist - cross gap              = {np.mean(diag)-np.mean(offs):+.3f}")

# the cleanest collapse: gradient-L_inf-trained -> non-gradient eval
grad_train = ['FGSM', 'PGD']
hard_eval = ['Square', 'CW']
collapse = [np.mean(acc[t][e]) for t in grad_train for e in hard_eval]
ggrad = [np.mean(acc[t][e]) for t in grad_train for e in grad_train]
print(f"\nFGSM/PGD-trained -> FGSM/PGD-eval  mean = {np.mean(ggrad):.3f}")
print(f"FGSM/PGD-trained -> Square/CW-eval mean = {np.mean(collapse):.3f}  (collapse)")
