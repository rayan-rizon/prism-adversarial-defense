#!/usr/bin/env python3
"""Generate local research gate verdict from evaluation artifacts.

Usage:
  python scripts/generate_local_research_gate_report.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


Z95 = 1.959963984540054


@dataclass(frozen=True)
class Targets:
    fgsm_tpr: float = 0.85
    pgd_tpr: float = 0.90
    square_tpr: float = 0.85
    fpr_l1: float = 0.10
    fpr_l2: float = 0.03
    fpr_l3: float = 0.005
    latency_ms: float = 100.0
    pooled_wilson_lower: float = 0.80


def wilson_ci(k: int, n: int, z: float = Z95) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    margin = (
        z
        * (((p * (1.0 - p)) + (z * z) / (4.0 * n)) / n) ** 0.5
        / denom
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    eval_paths = [
        repo_root / "experiments" / "evaluation" / "local_research_fast_n1000_ms5_seed42.json",
        repo_root / "experiments" / "evaluation" / "local_research_fast_n1000_seed123_square5000.json",
        repo_root / "experiments" / "evaluation" / "local_research_fast_n1000_seed456_square5000.json",
    ]
    fpr_path = repo_root / "experiments" / "calibration" / "local_research_fpr_report.json"

    targets = Targets()
    attack_targets = {
        "FGSM": targets.fgsm_tpr,
        "PGD": targets.pgd_tpr,
        "Square": targets.square_tpr,
    }

    eval_data = [load_json(p) for p in eval_paths]
    fpr_data = load_json(fpr_path)

    per_seed = []
    pooled_counts: Dict[str, Dict[str, int]] = {
        "FGSM": {"TP": 0, "n_adv": 0},
        "PGD": {"TP": 0, "n_adv": 0},
        "Square": {"TP": 0, "n_adv": 0},
    }
    pooled_all_tp = 0
    pooled_all_n = 0
    latency_values: List[float] = []

    for data, src in zip(eval_data, eval_paths):
        seed = int(data["_meta"]["seed"])
        attacks = {}
        seed_pass = True
        for attack, target in attack_targets.items():
            tpr = float(data[attack]["TPR"])
            passed = tpr >= target
            attacks[attack] = {"TPR": tpr, "target": target, "passed": passed}
            seed_pass = seed_pass and passed
            pooled_counts[attack]["TP"] += int(data[attack]["TP"])
            pooled_counts[attack]["n_adv"] += int(data[attack]["n_adv"])
            pooled_all_tp += int(data[attack]["TP"])
            pooled_all_n += int(data[attack]["n_adv"])

        l1 = float(data["FGSM"]["per_tier_fpr"]["FPR_L1_plus"])
        l2 = float(data["FGSM"]["per_tier_fpr"]["FPR_L2_plus"])
        l3 = float(data["FGSM"]["per_tier_fpr"]["FPR_L3_plus"])
        fpr_gate = {
            "L1": {"FPR": l1, "target": targets.fpr_l1, "passed": l1 <= targets.fpr_l1},
            "L2": {"FPR": l2, "target": targets.fpr_l2, "passed": l2 <= targets.fpr_l2},
            "L3": {"FPR": l3, "target": targets.fpr_l3, "passed": l3 <= targets.fpr_l3},
        }
        seed_pass = seed_pass and all(t["passed"] for t in fpr_gate.values())

        lat = float(data["_meta"]["latency"]["mean_ms"])
        lat_pass = lat < targets.latency_ms
        latency_values.append(lat)
        seed_pass = seed_pass and lat_pass

        per_seed.append(
            {
                "seed": seed,
                "source_file": str(src.relative_to(repo_root)).replace("\\", "/"),
                "attacks": attacks,
                "fpr": fpr_gate,
                "latency_mean_ms": lat,
                "latency_target_ms": targets.latency_ms,
                "latency_passed": lat_pass,
                "passed": seed_pass,
            }
        )

    pooled_attacks = {}
    pooled_attack_gate_pass = True
    for attack, counts in pooled_counts.items():
        tp = counts["TP"]
        n_adv = counts["n_adv"]
        tpr = tp / n_adv if n_adv else 0.0
        ci_lo, ci_hi = wilson_ci(tp, n_adv)
        passed = tpr >= attack_targets[attack]
        pooled_attacks[attack] = {
            "TP": tp,
            "n_adv": n_adv,
            "TPR": tpr,
            "TPR_CI_95": [ci_lo, ci_hi],
            "target": attack_targets[attack],
            "passed": passed,
        }
        pooled_attack_gate_pass = pooled_attack_gate_pass and passed

    pooled_tpr = pooled_all_tp / pooled_all_n if pooled_all_n else 0.0
    pooled_lo, pooled_hi = wilson_ci(pooled_all_tp, pooled_all_n)
    pooled_wilson_passed = pooled_lo >= targets.pooled_wilson_lower

    val_fpr_gate = {
        "L1": {
            "FPR": float(fpr_data["tiers"]["L1"]["FPR"]),
            "target": targets.fpr_l1,
            "passed": bool(fpr_data["tiers"]["L1"]["passed"]),
        },
        "L2": {
            "FPR": float(fpr_data["tiers"]["L2"]["FPR"]),
            "target": targets.fpr_l2,
            "passed": bool(fpr_data["tiers"]["L2"]["passed"]),
        },
        "L3": {
            "FPR": float(fpr_data["tiers"]["L3"]["FPR"]),
            "target": targets.fpr_l3,
            "passed": bool(fpr_data["tiers"]["L3"]["passed"]),
        },
    }

    mean_latency = sum(latency_values) / len(latency_values) if latency_values else 0.0
    max_latency = max(latency_values) if latency_values else 0.0
    latency_gate_passed = max_latency < targets.latency_ms

    all_seed_passed = all(s["passed"] for s in per_seed)
    overall_passed = (
        all_seed_passed
        and pooled_attack_gate_pass
        and pooled_wilson_passed
        and latency_gate_passed
        and all(v["passed"] for v in val_fpr_gate.values())
    )

    report = {
        "report_name": "local_research_gate_seed42_123_456",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_policy": {
            "attacks": attack_targets,
            "fpr": {
                "L1": targets.fpr_l1,
                "L2": targets.fpr_l2,
                "L3": targets.fpr_l3,
            },
            "latency_mean_ms_lt": targets.latency_ms,
            "pooled_wilson_lower_ge": targets.pooled_wilson_lower,
        },
        "inputs": {
            "evaluation_files": [str(p.relative_to(repo_root)).replace("\\", "/") for p in eval_paths],
            "validation_fpr_file": str(fpr_path.relative_to(repo_root)).replace("\\", "/"),
        },
        "per_seed": per_seed,
        "pooled_attacks": pooled_attacks,
        "pooled_all_attacks": {
            "TP": pooled_all_tp,
            "n_adv": pooled_all_n,
            "TPR": pooled_tpr,
            "TPR_CI_95": [pooled_lo, pooled_hi],
            "wilson_lower": pooled_lo,
            "wilson_lower_target": targets.pooled_wilson_lower,
            "passed": pooled_wilson_passed,
        },
        "validation_fpr_gate": val_fpr_gate,
        "latency_gate": {
            "seed_mean_ms_values": latency_values,
            "mean_of_seed_means_ms": mean_latency,
            "max_seed_mean_ms": max_latency,
            "target_ms": targets.latency_ms,
            "passed": latency_gate_passed,
        },
        "gate_summary": {
            "all_seed_passed": all_seed_passed,
            "pooled_attack_gate_passed": pooled_attack_gate_pass,
            "pooled_wilson_gate_passed": pooled_wilson_passed,
            "validation_fpr_gate_passed": all(v["passed"] for v in val_fpr_gate.values()),
            "latency_gate_passed": latency_gate_passed,
            "overall_passed": overall_passed,
        },
    }

    out_json = repo_root / "experiments" / "evaluation" / "local_research_gate_seed42_123_456.json"
    out_md = repo_root / "experiments" / "evaluation" / "local_research_gate_seed42_123_456.md"

    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Local Research Gate (Seeds 42/123/456)",
        "",
        f"- Generated (UTC): `{report['generated_at_utc']}`",
        f"- Overall gate: `{'PASS' if overall_passed else 'FAIL'}`",
        "",
        "## Per-seed attack gate",
    ]
    for s in per_seed:
        lines.append(
            f"- Seed {s['seed']}: FGSM={s['attacks']['FGSM']['TPR']:.3f} "
            f"(target {targets.fgsm_tpr:.2f}), PGD={s['attacks']['PGD']['TPR']:.3f} "
            f"(target {targets.pgd_tpr:.2f}), Square={s['attacks']['Square']['TPR']:.3f} "
            f"(target {targets.square_tpr:.2f}), pass={s['passed']}"
        )

    lines.extend(
        [
            "",
            "## Pooled attack gate (3,000 examples per attack)",
        ]
    )
    for atk in ("FGSM", "PGD", "Square"):
        m = pooled_attacks[atk]
        lo, hi = m["TPR_CI_95"]
        lines.append(
            f"- {atk}: TPR={m['TPR']:.4f} [95% CI {lo:.4f}, {hi:.4f}] "
            f"target={m['target']:.2f}, pass={m['passed']}"
        )

    lines.extend(
        [
            "",
            "## Calibration + latency",
            f"- Validation FPR: L1={val_fpr_gate['L1']['FPR']:.3f}, "
            f"L2={val_fpr_gate['L2']['FPR']:.3f}, L3={val_fpr_gate['L3']['FPR']:.3f} "
            f"(all pass={all(v['passed'] for v in val_fpr_gate.values())})",
            f"- Pooled Wilson lower (all attacks): {pooled_lo:.4f} "
            f"(target >= {targets.pooled_wilson_lower:.2f}, pass={pooled_wilson_passed})",
            f"- Seed latency means (ms): {', '.join(f'{x:.2f}' for x in latency_values)} "
            f"(max={max_latency:.2f}, target<{targets.latency_ms:.0f}, pass={latency_gate_passed})",
            "",
            "## Decision",
            f"- Gate result: `{'PASS' if overall_passed else 'FAIL'}`",
            "- CW-L2/AutoAttack remain deferred to Vast.ai until fast-attack local gate is stable.",
        ]
    )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
