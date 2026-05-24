import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "vastai_full_download_2026-05-20_0830UTC" / "project"

files = [
    PROJECT / "experiments" / "evaluation" / "results_cw_n1000_ms5_seed42.json",
    PROJECT / "experiments" / "evaluation" / "results_fast_n1000_ms5_seed42.json",
]

for f in files:
    if not f.exists():
        continue
    d = json.load(open(f))
    print(f"=== {f.name} ===")
    for k, v in d.items():
        if not isinstance(v, dict):
            continue
        sq = v.get("score_quantiles")
        if sq:
            print(f"  {k}:")
            for sub, sub_v in sq.items():
                print(f"    {sub}: {sub_v}")
        else:
            print(f"  {k} keys: {list(v.keys())[:12]}")
