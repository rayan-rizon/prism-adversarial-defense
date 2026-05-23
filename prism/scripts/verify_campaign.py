import json, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "vastai_full_download_2026-05-20_0830UTC" / "post_fix_local_2026-05-21" / "experiments" / "evaluation"
SEEDS = [42, 123, 456, 789, 999]

scens = {}
for s in SEEDS:
    d = json.load(open(BASE / f"results_campaign_local_seed{s}.json"))
    for name, scen in d.items():
        if not isinstance(scen, dict) or "l0_on" not in scen:
            continue
        l0 = scen["l0_on"]
        scens.setdefault(name, {"ttd": [], "fpr": [], "l0_frac": []})
        ttd = l0.get("time_to_detect_queries")
        if ttd is not None:
            scens[name]["ttd"].append(ttd)
        scens[name]["fpr"].append(l0.get("FPR_clean_steps"))
        scens[name]["l0_frac"].append(l0.get("l0_active_fraction"))

print(f"{'scenario':<22}  {'TTD':>16}  {'range':>10}  {'FPR_clean':>14}  {'L0_active':>10}")
for n, v in scens.items():
    ttd = v["ttd"]
    fpr = [x for x in v["fpr"] if x is not None]
    l0f = [x for x in v["l0_frac"] if x is not None]
    if ttd:
        tm = statistics.mean(ttd); ts = statistics.stdev(ttd) if len(ttd) > 1 else 0.0
        ttd_str = f"{tm:5.2f} ± {ts:.2f}"
        rng = f"[{min(ttd)}, {max(ttd)}]"
    else:
        ttd_str = "---"
        rng = "---"
    fm = statistics.mean(fpr) if fpr else 0
    fs = statistics.stdev(fpr) if len(fpr) > 1 else 0
    lm = statistics.mean(l0f) if l0f else 0
    print(f"{n:<22}  {ttd_str:>16}  {rng:>10}  {fm:.3f} ± {fs:.3f}  {lm:>10.3f}  per-seed-ttd={ttd}")
