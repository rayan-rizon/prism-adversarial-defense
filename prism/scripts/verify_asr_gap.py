import json, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "vastai_full_download_2026-05-20_0830UTC" / "post_fix_local_2026-05-21" / "experiments" / "evaluation"
SEEDS = [42, 123, 456, 789, 999]

agg = {}
for s in SEEDS:
    d = json.load(open(BASE / f"results_campaign_local_seed{s}.json"))
    for name, scen in d.items():
        if not isinstance(scen, dict) or "l0_on" not in scen:
            continue
        agg.setdefault(name, {"asr_on": [], "asr_off": [], "gap_pp": []})
        agg[name]["asr_on"].append(scen["l0_on"]["ASR"])
        agg[name]["asr_off"].append(scen["l0_off"]["ASR"])
        # Compute paired gap_pp = (off - on) * 100
        agg[name]["gap_pp"].append((scen["l0_off"]["ASR"] - scen["l0_on"]["ASR"]) * 100)

print(f"{'scenario':<22} {'ASR_off':>12} {'ASR_on':>12} {'gap_pp':>14}")
for n, v in agg.items():
    on = v["asr_on"]
    off = v["asr_off"]
    gap = v["gap_pp"]
    print(f"{n:<22} {statistics.mean(off):.4f}±{(statistics.stdev(off) if len(off)>1 else 0):.4f} "
          f"{statistics.mean(on):.4f}±{(statistics.stdev(on) if len(on)>1 else 0):.4f} "
          f"{statistics.mean(gap):+.3f}±{(statistics.stdev(gap) if len(gap)>1 else 0):.3f}")
