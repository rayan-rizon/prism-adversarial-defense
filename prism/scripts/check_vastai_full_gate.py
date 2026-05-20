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
POOLED_WILSON_LOWER_TARGET = 0.80
DEFAULT_SEEDS = [42, 123, 456, 789, 999]
EXPECTED_FEATURE_SPACE = "pixel-stability-v2+logitprofile+sidequad+gradnorm"
EXPECTED_N_FEATURES = 55
EXPECTED_TRAINING_ATTACKS = {"FGSM", "PGD", "Square"}
EXPECTED_REQUESTED_WEIGHTS = {"FGSM": 1.0, "PGD": 1.0, "Square": 1.0}


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _latest(pattern: str) -> Path | None:
    matches = [Path(p) for p in glob.glob(pattern)]
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def _result_paths(eval_dir: Path, kind: str) -> list[Path]:
    """Prefer multi-seed aggregate files; fall back to seed-specific files."""
    aggregate_paths = sorted(eval_dir.glob(f"results_{kind}_n*_ms5.json"))
    if aggregate_paths:
        return aggregate_paths
    seed_paths = sorted(eval_dir.glob(f"results_{kind}_n*_ms5_seed*.json"))
    if seed_paths:
        return seed_paths
    return sorted(eval_dir.glob(f"results_{kind}_n*_ms5*.json"))


def _load_many(paths: list[Path]) -> list[tuple[Path, dict[str, Any]]]:
    loaded: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        try:
            loaded.append((path, _load(path)))
        except Exception as exc:
            print(f"  WARN: skipping unreadable result {path.name}: {exc}")
    return loaded


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


def _gate_pooled_wilson_lower(result: dict[str, Any]) -> float | None:
    gate = result.get("target_metric_gate") or (result.get("_meta") or {}).get("target_metric_gate") or {}
    value = gate.get("pooled_wilson_lower")
    return float(value) if value is not None else None


def _iter_seed_results(path: Path, result: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    per_seed = result.get("per_seed")
    if isinstance(per_seed, dict) and per_seed:
        return [(f"{path.name}:seed={seed}", seed_result) for seed, seed_result in per_seed.items()]
    return [(path.name, result)]


def _metadata(result: dict[str, Any]) -> dict[str, Any]:
    return result.get("_meta") or result.get("metadata") or {}


def _validate_result_contract(
    records: list[tuple[Path, dict[str, Any]]],
    *,
    expected_attacks: set[str],
    expected_seeds: set[int],
    expected_n_test: int,
    kind: str,
) -> list[str]:
    failures: list[str] = []
    for path, result in records:
        seeds = result.get("seeds")
        if seeds is not None:
            seen = {int(seed) for seed in seeds}
            if seen != expected_seeds:
                failures.append(f"{path.name} seeds={sorted(seen)}, expected {sorted(expected_seeds)}")
        aggregate_attacks = set((result.get("aggregate") or {}).keys())
        if aggregate_attacks and not expected_attacks.issubset(aggregate_attacks):
            failures.append(
                f"{path.name} aggregate attacks={sorted(aggregate_attacks)}, "
                f"expected at least {sorted(expected_attacks)}"
            )

        for label, seed_result in _iter_seed_results(path, result):
            meta = _metadata(seed_result)
            seed = meta.get("seed")
            if seed is not None and int(seed) not in expected_seeds:
                failures.append(f"{label} unexpected seed={seed}")
            n_test = meta.get("n_test")
            if n_test is not None and int(n_test) != expected_n_test:
                failures.append(f"{label} n_test={n_test}, expected {expected_n_test}")
            attacks = set(meta.get("attacks") or [])
            if attacks and not expected_attacks.issubset(attacks):
                failures.append(
                    f"{label} eval attacks={sorted(attacks)}, expected at least {sorted(expected_attacks)}"
                )
            for attack in expected_attacks:
                row = seed_result.get(attack)
                if row is None:
                    failures.append(f"{label} missing attack result: {attack}")
                elif "error" in row:
                    failures.append(f"{label} {attack} error: {row['error']}")

            ensemble = meta.get("ensemble") or {}
            if ensemble:
                training_attacks = set(ensemble.get("training_attacks") or [])
                if training_attacks != EXPECTED_TRAINING_ATTACKS:
                    failures.append(
                        f"{label} training_attacks={sorted(training_attacks)}, "
                        f"expected {sorted(EXPECTED_TRAINING_ATTACKS)}"
                    )
                if bool(ensemble.get("use_grad_norm")) is not True:
                    failures.append(f"{label} use_grad_norm={ensemble.get('use_grad_norm')}, expected True")
                if int(ensemble.get("n_features") or 0) != EXPECTED_N_FEATURES:
                    failures.append(
                        f"{label} n_features={ensemble.get('n_features')}, expected {EXPECTED_N_FEATURES}"
                    )
                if ensemble.get("feature_space_version") != EXPECTED_FEATURE_SPACE:
                    failures.append(
                        f"{label} feature_space_version={ensemble.get('feature_space_version')}, "
                        f"expected {EXPECTED_FEATURE_SPACE}"
                    )
                if bool(ensemble.get("balanced_attacks")) is not True:
                    failures.append(f"{label} balanced_attacks={ensemble.get('balanced_attacks')}, expected True")
                requested = ensemble.get("requested_oversample_weights") or {}
                for attack, expected in EXPECTED_REQUESTED_WEIGHTS.items():
                    actual = requested.get(attack)
                    try:
                        actual_f = float(actual)
                    except Exception:
                        actual_f = None
                    if actual_f is None or abs(actual_f - expected) > 1e-6:
                        failures.append(
                            f"{label} requested_oversample_weights[{attack}]={actual}, "
                            f"expected {expected}"
                        )
                if "CW" in requested or "AutoAttack" in requested:
                    failures.append(
                        f"{label} requested_oversample_weights includes unsupported attacks: {requested}"
                    )
            elif kind in {"fast", "cw"}:
                failures.append(f"{label} missing ensemble provenance metadata")
    return failures


def _worst_attack_tpr(records: list[tuple[Path, dict[str, Any]]], attack: str) -> float | None:
    values: list[float] = []
    for _, result in records:
        tpr = _attack_tpr(result, attack)
        if tpr is not None:
            values.append(float(tpr))
    return min(values) if values else None


def _worst_tier_fpr(records: list[tuple[Path, dict[str, Any]]], tier: str) -> float | None:
    values: list[float] = []
    for _, result in records:
        fpr = _gate_fpr(result, tier)
        if fpr is not None:
            values.append(float(fpr))
    return max(values) if values else None


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
    parser.add_argument("--expected-n-test", type=int, default=1000)
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--calibration-report",
        default="experiments/calibration/ensemble_fpr_report.json",
    )
    parser.add_argument("--latency-file", default="experiments/evaluation/results_latency_standalone.json")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    failures: list[str] = []

    fast_paths = [Path(args.fast_result)] if args.fast_result else _result_paths(eval_dir, "fast")
    cw_paths = [Path(args.cw_result)] if args.cw_result else _result_paths(eval_dir, "cw")
    fast_records = _load_many(fast_paths) if fast_paths else []
    cw_records = _load_many(cw_paths) if cw_paths else []
    requested_attacks = set(args.attacks)
    fast_expected_attacks = requested_attacks - {"CW"}
    cw_expected_attacks = requested_attacks & {"CW"}

    if not fast_records and fast_expected_attacks:
        failures.append("missing fast attack result: results_fast_n*_ms5*.json")
    if not cw_records and cw_expected_attacks:
        failures.append("missing CW result: results_cw_n*_ms5*.json")

    print("Vast.ai full-gate check")
    print(
        "  fast: "
        + (", ".join(path.name for path, _ in fast_records) if fast_records else "MISSING")
    )
    print(
        "  cw:   "
        + (", ".join(path.name for path, _ in cw_records) if cw_records else "MISSING")
    )

    expected_seeds = {int(seed) for seed in args.expected_seeds}
    failures.extend(
        _validate_result_contract(
            fast_records,
            expected_attacks=fast_expected_attacks,
            expected_seeds=expected_seeds,
            expected_n_test=args.expected_n_test,
            kind="fast",
        )
    )
    if cw_expected_attacks:
        failures.extend(
            _validate_result_contract(
                cw_records,
                expected_attacks=cw_expected_attacks,
                expected_seeds=expected_seeds,
                expected_n_test=args.expected_n_test,
                kind="cw",
            )
        )

    for attack in args.attacks:
        target = ATTACK_TARGETS[attack]
        records = fast_records if attack != "CW" else cw_records
        tpr = _worst_attack_tpr(records, attack)
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
        observed = [_worst_tier_fpr(records, tier) for records in (fast_records, cw_records)]
        observed = [v for v in observed if v is not None]
        if observed:
            worst = max(observed)
            ok = worst <= target
            print(f"  {tier:10s} eval FPR={worst:.4f} target<={target:.3f} {'PASS' if ok else 'FAIL'}")
            if not ok:
                failures.append(f"{tier} eval FPR miss")

    pooled_lowers = [
        _gate_pooled_wilson_lower(result)
        for records in (fast_records, cw_records)
        for _, result in records
    ]
    pooled_lowers = [v for v in pooled_lowers if v is not None]
    if pooled_lowers:
        worst_lower = min(pooled_lowers)
        ok = worst_lower >= POOLED_WILSON_LOWER_TARGET
        print(
            f"  pooled    Wilson lower={worst_lower:.4f} "
            f"target>={POOLED_WILSON_LOWER_TARGET:.2f} {'PASS' if ok else 'FAIL'}"
        )
        if not ok:
            failures.append("pooled Wilson lower miss")
    else:
        failures.append("missing pooled Wilson lower in target_metric_gate")

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
