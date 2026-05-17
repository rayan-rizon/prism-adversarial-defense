#!/usr/bin/env python3
"""Check whether a Vast.ai full PRISM run meets the research gates.

This script is intentionally read-only. It validates the JSON files produced by
run_vastai_full.sh and exits non-zero on missing files or target misses.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


ATTACK_TARGETS = {
    "FGSM": 0.85,
    "PGD": 0.90,
    "Square": 0.85,
    "CW": 0.85,
    "AutoAttack": 0.90,
}
FPR_TARGETS = {"L1": 0.10, "L2": 0.03, "L3": 0.005}
LATENCY_TARGET_MS = 100.0


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _latest(pattern: str) -> Path | None:
    matches = [Path(p) for p in glob.glob(pattern)]
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def _attack_tpr(result: dict[str, Any], attack: str) -> float | None:
    aggregate = result.get("aggregate") or {}
    row = aggregate.get(attack) or {}
    if "TPR_mean" in row:
        return float(row["TPR_mean"])
    if "TPR" in row:
        return float(row["TPR"])
    gate = result.get("target_metric_gate") or (result.get("_meta") or {}).get("target_metric_gate") or {}
    gate_attack = (gate.get("attacks") or {}).get(attack)
    if gate_attack and "TPR" in gate_attack:
        return float(gate_attack["TPR"])
    return None


def _gate_fpr(result: dict[str, Any], tier: str) -> float | None:
    gate = result.get("target_metric_gate") or (result.get("_meta") or {}).get("target_metric_gate") or {}
    gate_fpr = gate.get("fpr", {})
    row = gate_fpr.get(tier)
    if row and "FPR" in row:
        return float(row["FPR"])
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", default="experiments/evaluation")
    parser.add_argument("--fast-result", default=None)
    parser.add_argument("--cw-result", default=None)
    parser.add_argument(
        "--attacks",
        nargs="+",
        default=["FGSM", "PGD", "Square", "CW", "AutoAttack"],
        choices=sorted(ATTACK_TARGETS),
    )
    parser.add_argument(
        "--calibration-report",
        default="experiments/calibration/ensemble_fpr_report.json",
    )
    parser.add_argument("--latency-file", default="experiments/evaluation/results_latency_standalone.json")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    failures: list[str] = []

    fast_path = Path(args.fast_result) if args.fast_result else _latest(str(eval_dir / "results_fast_n*_ms5.json"))
    cw_path = Path(args.cw_result) if args.cw_result else _latest(str(eval_dir / "results_cw_n*_ms5.json"))
    if fast_path is None:
        failures.append("missing fast attack result: results_fast_n*_ms5.json")
        fast = {}
    else:
        fast = _load(fast_path)
    if cw_path is None:
        failures.append("missing CW result: results_cw_n*_ms5.json")
        cw = {}
    else:
        cw = _load(cw_path)

    print("Vast.ai full-gate check")
    print(f"  fast: {fast_path or 'MISSING'}")
    print(f"  cw:   {cw_path or 'MISSING'}")

    result_by_attack = {
        "FGSM": fast,
        "PGD": fast,
        "Square": fast,
        "AutoAttack": fast,
        "CW": cw,
    }
    for attack in args.attacks:
        target = ATTACK_TARGETS[attack]
        tpr = _attack_tpr(result_by_attack[attack], attack)
        ok = tpr is not None and tpr >= target
        print(f"  {attack:10s} TPR={tpr if tpr is not None else 'NA'} target>={target:.2f} {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"{attack} TPR miss")

    cal_path = Path(args.calibration_report)
    if cal_path.exists():
        cal = _load(cal_path)
        for tier, target in FPR_TARGETS.items():
            row = (cal.get("tiers") or {}).get(tier, {})
            fpr = row.get("FPR")
            ok = fpr is not None and float(fpr) <= target
            print(f"  {tier:10s} val FPR={fpr if fpr is not None else 'NA'} target<={target:.3f} {'PASS' if ok else 'FAIL'}")
            if not ok:
                failures.append(f"{tier} validation FPR miss")
    else:
        failures.append(f"missing validation FPR report: {cal_path}")

    for tier, target in FPR_TARGETS.items():
        observed = [_gate_fpr(r, tier) for r in (fast, cw) if r]
        observed = [v for v in observed if v is not None]
        if observed:
            worst = max(observed)
            ok = worst <= target
            print(f"  {tier:10s} eval FPR={worst:.4f} target<={target:.3f} {'PASS' if ok else 'FAIL'}")
            if not ok:
                failures.append(f"{tier} eval FPR miss")

    latency_path = Path(args.latency_file)
    if latency_path.exists():
        latency = (_load(latency_path).get("_meta") or {}).get("latency", {})
        mean_ms = latency.get("mean_ms")
        ok = mean_ms is not None and float(mean_ms) < LATENCY_TARGET_MS
        print(f"  latency    mean_ms={mean_ms if mean_ms is not None else 'NA'} target<{LATENCY_TARGET_MS:.0f} {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append("latency miss")
    else:
        failures.append(f"missing latency result: {latency_path}")

    if failures:
        print("\nFAIL")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
