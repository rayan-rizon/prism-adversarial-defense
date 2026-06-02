"""Verify L0-on/off ASR gap diagnostics for SACD campaign streams."""
from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEEDS = [42, 123, 456, 789, 999]


def resolve_evaluation_root(arg: str | None) -> Path:
    candidates = []
    if arg:
        candidates.append(Path(arg))
    if os.environ.get("PRISM_EVALUATION_ROOT"):
        candidates.append(Path(os.environ["PRISM_EVALUATION_ROOT"]))
    candidates.extend(
        [
            ROOT
            / "Cifar 10"
            / "post_fix_local_2026-05-21"
            / "experiments"
            / "evaluation",
            ROOT
            / "vastai_full_download_2026-05-20_0830UTC"
            / "post_fix_local_2026-05-21"
            / "experiments"
            / "evaluation",
        ]
    )
    required = [Path(f"results_campaign_local_seed{s}.json") for s in SEEDS]
    for candidate in candidates:
        base = candidate.resolve()
        if all((base / rel).exists() for rel in required):
            return base
    checked = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Could not locate campaign evaluation artifacts. Checked:\n  " + checked
    )


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root")
    args = parser.parse_args()

    base = resolve_evaluation_root(args.evaluation_root)
    agg: dict[str, dict[str, list[float]]] = {}
    for seed in SEEDS:
        data = load(base / f"results_campaign_local_seed{seed}.json")
        for name, scen in data.items():
            if not isinstance(scen, dict) or "l0_on" not in scen:
                continue
            agg.setdefault(name, {"asr_on": [], "asr_off": [], "gap_pp": []})
            agg[name]["asr_on"].append(scen["l0_on"]["ASR"])
            agg[name]["asr_off"].append(scen["l0_off"]["ASR"])
            agg[name]["gap_pp"].append(
                (scen["l0_off"]["ASR"] - scen["l0_on"]["ASR"]) * 100
            )

    print(f"source: {base}")
    print(f"{'scenario':<22} {'ASR_off':>12} {'ASR_on':>12} {'gap_pp':>14}")
    for name, values in agg.items():
        on = values["asr_on"]
        off = values["asr_off"]
        gap = values["gap_pp"]
        print(
            f"{name:<22} "
            f"{statistics.mean(off):.4f}+/-{(statistics.stdev(off) if len(off) > 1 else 0):.4f} "
            f"{statistics.mean(on):.4f}+/-{(statistics.stdev(on) if len(on) > 1 else 0):.4f} "
            f"{statistics.mean(gap):+.3f}+/-{(statistics.stdev(gap) if len(gap) > 1 else 0):.3f}"
        )


if __name__ == "__main__":
    main()
