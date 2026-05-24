import json, statistics, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "vastai_full_download_2026-05-20_0830UTC" / "post_fix_local_2026-05-21" / "experiments"
SEEDS = [42, 123, 456, 789, 999]


def load(d, name):
    f = BASE / d / f"results_recovery_{name}_seed{42}.json"
    return f


def wilson_ci(p, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def collect(directory, name_prefix):
    pool = {}
    triggers = []
    for s in SEEDS:
        f = BASE / directory / f"results_recovery_{name_prefix}_seed{s}.json"
        d = json.load(open(f))
        triggers.append(d["_meta"]["l3_trigger_rate"])
        for k in ("reject", "passthrough", "tamsh", "tamsh_ensemble",
                  "tamsh_force", "tamsh_uniform"):
            if k in d and isinstance(d[k], dict):
                pool.setdefault(k, []).append(d[k]["recovery_accuracy"])
    return pool, triggers


print("=" * 70)
print("RECOVERY DIR (results_recovery_post_fix_seed*.json)")
print("=" * 70)
pool, trigs = collect("recovery", "post_fix")
print(f"trigger rates: {trigs}  mean={statistics.mean(trigs):.3f}")
for k, vals in pool.items():
    m = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0
    lo, hi = wilson_ci(m, 802 * 5)
    print(f"  {k:<18} n={len(vals)} mean={m:.4f} std={sd:.4f}  WilsonCI(pooled n=4010)=[{lo:.4f},{hi:.4f}]  per-seed={[f'{v:.4f}' for v in vals]}")

print()
print("=" * 70)
print("RECOVERY_UNIFORM DIR (results_recovery_uniform_seed*.json)")
print("=" * 70)
pool, trigs = collect("recovery_uniform", "uniform")
print(f"trigger rates: {trigs}  mean={statistics.mean(trigs):.3f}")
for k, vals in pool.items():
    m = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0
    print(f"  {k:<18} n={len(vals)} mean={m:.4f} std={sd:.4f}  per-seed={[f'{v:.4f}' for v in vals]}")

# Gap calc — per-seed deltas (pairing recovery_post_fix and recovery_uniform might differ)
print()
print("=== PAIRED GAPS (uniform pool: passthrough as baseline) ===")
pool_u, _ = collect("recovery_uniform", "uniform")
if "passthrough" in pool_u:
    base = pool_u["passthrough"]
    for k in ("tamsh", "tamsh_uniform", "tamsh_force"):
        if k in pool_u:
            per_seed_gap = [(pool_u[k][i] - base[i]) * 100 for i in range(len(base))]
            m = statistics.mean(per_seed_gap)
            sd = statistics.stdev(per_seed_gap) if len(per_seed_gap) > 1 else 0
            print(f"  {k:<18}: mean gap = {m:+.2f}pp, std = {sd:.2f}  per-seed={[f'{v:+.2f}' for v in per_seed_gap]}")

print()
print("=== PAIRED GAPS (recovery_post_fix pool: passthrough as baseline) ===")
pool_r, _ = collect("recovery", "post_fix")
if "passthrough" in pool_r:
    base = pool_r["passthrough"]
    for k in ("tamsh", "tamsh_force"):
        if k in pool_r:
            per_seed_gap = [(pool_r[k][i] - base[i]) * 100 for i in range(len(base))]
            m = statistics.mean(per_seed_gap)
            sd = statistics.stdev(per_seed_gap) if len(per_seed_gap) > 1 else 0
            print(f"  {k:<18}: mean gap = {m:+.2f}pp, std = {sd:.2f}  per-seed={[f'{v:+.2f}' for v in per_seed_gap]}")
