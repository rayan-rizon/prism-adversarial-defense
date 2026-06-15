"""Pool per-corruption cert-restoration JSONs into a frozen-vs-restored table."""
import json, glob, os, sys
import numpy as np

d = sys.argv[1] if len(sys.argv) > 1 else 'experiments/stress/cert_restore'
ALPHAS = {'L1': 0.10, 'L2': 0.03, 'L3': 0.005}
cells = {}
for f in sorted(glob.glob(os.path.join(d, 'restore_*.json'))):
    j = json.load(open(f))
    cells.update(j['per_cell'])
print(f"cells pooled: {len(cells)}\n")
print(f"{'tier':5} {'alpha':7} {'mean frozen FPR':16} {'mean restored FPR':18} {'cells restored<=a'}")
print('-' * 70)
for tier, a in ALPHAS.items():
    fz = [v['frozen_FPR'][tier] for v in cells.values()]
    rs = [v['restored_FPR'][tier] for v in cells.values()]
    within = sum(r <= a for r in rs)
    print(f"{tier:5} {a:<7} {np.mean(fz):<16.4f} {np.mean(rs):<18.4f} {within}/{len(rs)}")
print("\nWorst frozen cells (L1):")
worst = sorted(cells.items(), key=lambda kv: -kv[1]['frozen_FPR']['L1'])[:4]
for k, v in worst:
    print(f"  {k:28} frozen L1 {v['frozen_FPR']['L1']:.3f} -> restored {v['restored_FPR']['L1']:.3f}")
