"""Verify SACD campaign numbers used in the paper."""
from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEEDS = [42, 123, 456, 789, 999]


def resolve_experiments_root(arg: str | None) -> Path:
    candidates = []
    if arg:
        candidates.append(Path(arg))
    if os.environ.get("PRISM_EXPERIMENTS_ROOT"):
        candidates.append(Path(os.environ["PRISM_EXPERIMENTS_ROOT"]))
    candidates.extend(
        [
            ROOT / "Cifar 10" / "post_fix_local_2026-05-21" / "experiments",
            ROOT
            / "vastai_full_download_2026-05-20_0830UTC"
            / "post_fix_local_2026-05-21"
            / "experiments",
        ]
    )

    required = [
        Path("evaluation") / f"results_campaign_local_seed{s}.json"
        for s in SEEDS
    ]
    for candidate in candidates:
        base = candidate.resolve()
        if all((base / rel).exists() for rel in required):
            return base
    checked = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Could not locate campaign artifacts. Checked:\n  " + checked
    )


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-root")
    args = parser.parse_args()

    base = resolve_experiments_root(args.experiments_root)
    scens: dict[str, dict[str, list[float]]] = {}
    for seed in SEEDS:
        data = load(base / "evaluation" / f"results_campaign_local_seed{seed}.json")
        for name, scen in data.items():
            if not isinstance(scen, dict) or "l0_on" not in scen:
                continue
            l0 = scen["l0_on"]
            scens.setdefault(name, {"ttd": [], "fpr": [], "l0_frac": []})
            ttd = l0.get("time_to_detect_queries")
            if ttd is not None:
                scens[name]["ttd"].append(ttd)
            if l0.get("FPR_clean_steps") is not None:
                scens[name]["fpr"].append(l0["FPR_clean_steps"])
            if l0.get("l0_active_fraction") is not None:
                scens[name]["l0_frac"].append(l0["l0_active_fraction"])

    print(f"source: {base}")
    print(
        f"{'scenario':<22}  {'TTD':>16}  {'range':>10}  "
        f"{'FPR_clean':>14}  {'L0_active':>10}"
    )
    for name, values in scens.items():
        ttd = values["ttd"]
        fpr = values["fpr"]
        l0f = values["l0_frac"]
        if ttd:
            ttd_mean = statistics.mean(ttd)
            ttd_std = statistics.stdev(ttd) if len(ttd) > 1 else 0.0
            ttd_str = f"{ttd_mean:5.2f} +/- {ttd_std:.2f}"
            rng = f"[{min(ttd)}, {max(ttd)}]"
        else:
            ttd_str = "---"
            rng = "---"
        fpr_mean = statistics.mean(fpr) if fpr else 0.0
        fpr_std = statistics.stdev(fpr) if len(fpr) > 1 else 0.0
        l0_mean = statistics.mean(l0f) if l0f else 0.0
        l0_std = statistics.stdev(l0f) if len(l0f) > 1 else 0.0
        print(
            f"{name:<22}  {ttd_str:>16}  {rng:>10}  "
            f"{fpr_mean:.3f} +/- {fpr_std:.3f}  "
            f"{l0_mean:.3f} +/- {l0_std:.3f}  per-seed-ttd={ttd}"
        )


if __name__ == "__main__":
    main()
