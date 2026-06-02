"""Verify TAMSH recovery table numbers from canonical post-fix artifacts."""
from __future__ import annotations

import argparse
import json
import math
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
        Path("recovery_uniform") / f"results_recovery_uniform_seed{s}.json"
        for s in SEEDS
    ]
    for candidate in candidates:
        base = candidate.resolve()
        if all((base / rel).exists() for rel in required):
            return base
    checked = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Could not locate recovery_uniform artifacts. Checked:\n  " + checked
    )


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def wilson_ci(p: float, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def collect(base: Path, directory: str, name_prefix: str):
    pool: dict[str, list[float]] = {}
    triggers = []
    for seed in SEEDS:
        path = base / directory / f"results_recovery_{name_prefix}_seed{seed}.json"
        data = load(path)
        triggers.append(data["_meta"]["l3_trigger_rate"])
        for key in (
            "reject",
            "passthrough",
            "tamsh",
            "tamsh_ensemble",
            "tamsh_force",
            "tamsh_uniform",
        ):
            if key in data and isinstance(data[key], dict):
                pool.setdefault(key, []).append(data[key]["recovery_accuracy"])
    return pool, triggers


def print_pool(title: str, pool: dict[str, list[float]], triggers: list[float]) -> None:
    print("=" * 70)
    print(title)
    print("=" * 70)
    print(f"trigger rates: {triggers}  mean={statistics.mean(triggers):.3f}")
    for key, vals in pool.items():
        avg = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        lo, hi = wilson_ci(avg, 802 * 5)
        print(
            f"  {key:<18} n={len(vals)} mean={avg:.4f} std={sd:.4f} "
            f"WilsonCI(pooled n=4010)=[{lo:.4f},{hi:.4f}] "
            f"per-seed={[f'{v:.4f}' for v in vals]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-root")
    args = parser.parse_args()

    base = resolve_experiments_root(args.experiments_root)
    print(f"source: {base}")

    recovery_uniform, triggers_uniform = collect(base, "recovery_uniform", "uniform")
    print_pool("RECOVERY_UNIFORM DIR", recovery_uniform, triggers_uniform)

    recovery_dir = base / "recovery"
    if recovery_dir.exists():
        recovery, triggers = collect(base, "recovery", "post_fix")
        print()
        print_pool("RECOVERY DIR", recovery, triggers)

    print()
    print("=== PAIRED GAPS (uniform pool: passthrough baseline) ===")
    base_vals = recovery_uniform.get("passthrough", [])
    for key in ("tamsh", "tamsh_uniform", "tamsh_force"):
        if key in recovery_uniform and base_vals:
            gaps = [
                (recovery_uniform[key][i] - base_vals[i]) * 100
                for i in range(len(base_vals))
            ]
            avg = statistics.mean(gaps)
            sd = statistics.stdev(gaps) if len(gaps) > 1 else 0.0
            print(
                f"  {key:<18}: mean gap = {avg:+.2f}pp, std = {sd:.2f} "
                f"per-seed={[f'{v:+.2f}' for v in gaps]}"
            )


if __name__ == "__main__":
    main()
