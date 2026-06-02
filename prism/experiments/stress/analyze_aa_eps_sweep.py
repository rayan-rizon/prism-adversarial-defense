"""
Parse vastai_autoattack_eps_sweep.json and print final pooled TPR numbers
to fill into appendix.tex tab:aa_eps_sweep.
"""
import json, sys, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(HERE, 'vastai_autoattack_eps_sweep.json')

if len(sys.argv) > 1:
    JSON_PATH = sys.argv[1]

with open(JSON_PATH) as f:
    d = json.load(f)

SEEDS = d['seeds']
EPS_LIST = d['eps_list']
ATK_ORDER = d['sub_attacks']
N_PER_SEED = d['n_per_seed']
N_TOTAL = N_PER_SEED * len(SEEDS)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    halfw = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - halfw), min(1.0, centre + halfw))


print(f"\n=== Pooled results ({len(SEEDS)} seeds x {N_PER_SEED} = {N_TOTAL} adversarials per cell) ===\n")
print(f"{'Attack':12s} {'eps':6s}  L1      L2      L3     CI95(L1)")
print("-" * 70)

# Pooled clean FPR
if 'aggregate' in d and 'clean' in d['aggregate']:
    ag_c = d['aggregate']['clean']
    print(f"clean FPR: L1={ag_c['FPR']['L1']:.4f}  L2={ag_c['FPR']['L2']:.4f}  L3={ag_c['FPR']['L3']:.4f}")
    print()

results = {}
for eps in EPS_LIST:
    for atk in ATK_ORDER:
        l1 = l2 = l3 = 0
        seeds_with_data = 0
        for seed in SEEDS:
            s = str(seed)
            try:
                r = d['per_seed'][s]['by_eps'][str(eps)]['sub_attack'][atk]
            except KeyError:
                continue
            l1 += r['TPR_L1_count']
            l2 += r['TPR_L2_count']
            l3 += r['TPR_L3_count']
            seeds_with_data += 1
        if seeds_with_data < len(SEEDS):
            print(f"  [WARNING] only {seeds_with_data}/{len(SEEDS)} seeds available for {atk} eps={eps}")
        n = seeds_with_data * N_PER_SEED
        ci = wilson(l1, n) if n > 0 else (0.0, 0.0)
        tpr1 = l1 / n if n > 0 else 0.0
        tpr2 = l2 / n if n > 0 else 0.0
        tpr3 = l3 / n if n > 0 else 0.0
        results[(atk, eps)] = (tpr1, tpr2, tpr3, ci)
        eps_frac = f"{round(eps*255)}/255"
        print(f"{atk:12s} {eps_frac:6s}  {tpr1:.4f}  {tpr2:.4f}  {tpr3:.4f}  [{ci[0]:.3f},{ci[1]:.3f}]")

# LaTeX table rows
print("\n=== LaTeX rows (for tab:aa_eps_sweep) ===")
atk_names = {'apgd-ce': 'APGD-CE', 'apgd-dlr': 'APGD-DLR', 'fab-t': 'FAB-T', 'square': 'Square'}
# From vastai_autoattack.json (8/255) -- hardcoded reference
ref_8 = {
    'apgd-ce':  (0.9708, 0.9608, 0.9500),
    'apgd-dlr': (0.9112, 0.7782, 0.3156),
    'fab-t':    (0.9314, 0.8338, 0.4494),
    'square':   (0.7062, 0.5080, 0.1590),
}
eps12 = EPS_LIST[0]
eps16 = EPS_LIST[1]
for atk in ATK_ORDER:
    r8 = ref_8[atk]
    r12 = results[(atk, eps12)]
    r16 = results[(atk, eps16)]
    name = atk_names.get(atk, atk)
    row = (f"{name:8s} & {r8[0]:.4f} & {r12[0]:.4f} & {r16[0]:.4f} "
           f"& {r8[1]:.4f} & {r12[1]:.4f} & {r16[1]:.4f} "
           f"& {r8[2]:.4f} & {r12[2]:.4f} & {r16[2]:.4f} \\\\")
    print(row)
