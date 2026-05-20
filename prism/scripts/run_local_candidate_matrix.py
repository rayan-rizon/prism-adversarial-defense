#!/usr/bin/env python
"""
Run the local PRISM scorer candidate matrix.

The matrix is intentionally limited to fast attacks. CW-L2 and AutoAttack stay
deferred until the local FGSM/PGD/Square gate is healthy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ENSEMBLE = PROJECT_ROOT / "models" / "ensemble_scorer.pkl"
CANONICAL_CALIBRATOR = PROJECT_ROOT / "models" / "calibrator.pkl"
CANONICAL_FPR = PROJECT_ROOT / "experiments" / "calibration" / "ensemble_fpr_report.json"

ATTACK_TARGETS = {"FGSM": 0.85, "PGD": 0.90, "Square": 0.85}
FPR_TARGETS = {"L1": 0.10, "L2": 0.03, "L3": 0.005}


@dataclass(frozen=True)
class Candidate:
    key: str
    label: str
    train_args: List[str]
    grad_norm: bool = False


CANDIDATES = {
    "A": Candidate(
        key="A",
        label="A_46_sidequad_balanced",
        train_args=[
            "--balanced-attacks",
            "--use-stability-features",
            "--use-side-quadratic-features",
        ],
    ),
    "B": Candidate(
        key="B",
        label="B_54_logitprofile_sidequad_balanced",
        train_args=[
            "--balanced-attacks",
            "--use-stability-features",
            "--use-logit-profile-features",
            "--use-side-quadratic-features",
        ],
    ),
    "C": Candidate(
        key="C",
        label="C_54_logitprofile_sidequad_fgsm2p5_pgd2p5",
        train_args=[
            "--fgsm-oversample", "2.5",
            "--pgd-oversample", "2.5",
            "--use-stability-features",
            "--use-logit-profile-features",
            "--use-side-quadratic-features",
        ],
    ),
    "D": Candidate(
        key="D",
        label="D_55_logitprofile_gradnorm_balanced",
        train_args=[
            "--balanced-attacks",
            "--use-stability-features",
            "--use-logit-profile-features",
            "--use-side-quadratic-features",
            "--use-grad-norm",
        ],
        grad_norm=True,
    ),
    "E": Candidate(
        key="E",
        label="E_55_logitprofile_gradnorm_attackheads_fgsm2p5_pgd2p5",
        train_args=[
            "--fgsm-oversample", "2.5",
            "--pgd-oversample", "2.5",
            "--use-stability-features",
            "--use-logit-profile-features",
            "--use-side-quadratic-features",
            "--use-grad-norm",
            "--attack-heads",
            "--score-channel-aggregation", "max",
        ],
        grad_norm=True,
    ),
    "F": Candidate(
        key="F",
        label="F_55_logitprofile_gradnorm_fgsm2p5_pgd2p5_square1p5",
        train_args=[
            "--fgsm-oversample", "2.5",
            "--pgd-oversample", "2.5",
            "--square-oversample", "1.5",
            "--use-stability-features",
            "--use-logit-profile-features",
            "--use-side-quadratic-features",
            "--use-grad-norm",
        ],
        grad_norm=True,
    ),
}


def run(cmd: List[str], *, log_path: Path, env: Dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n$ " + " ".join(cmd), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def snapshot_artifacts(snapshot_dir: Path) -> Dict[str, Path]:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "ensemble": (CANONICAL_ENSEMBLE, snapshot_dir / "ensemble_scorer.pkl"),
        "calibrator": (CANONICAL_CALIBRATOR, snapshot_dir / "calibrator.pkl"),
        "fpr": (CANONICAL_FPR, snapshot_dir / "ensemble_fpr_report.json"),
    }
    existing = {}
    for key, (src, dst) in mapping.items():
        if src.exists():
            shutil.copy2(src, dst)
            existing[key] = dst
    return existing


def restore_artifacts(snapshot: Dict[str, Path]) -> None:
    targets = {
        "ensemble": CANONICAL_ENSEMBLE,
        "calibrator": CANONICAL_CALIBRATOR,
        "fpr": CANONICAL_FPR,
    }
    for key, src in snapshot.items():
        dst = targets[key]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def wilson_lower(successes: int, total: int, z: float = 1.959963984540054) -> float:
    if total <= 0:
        return 0.0
    p = successes / total
    denom = 1 + z * z / total
    centre = p + z * z / (2 * total)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    return max(0.0, (centre - margin) / denom)


def summarize_candidate(
    candidate: Candidate,
    eval_report: Dict[str, Any],
    fpr_report: Dict[str, Any],
    ensemble: Dict[str, Any],
) -> Dict[str, Any]:
    attacks: Dict[str, Any] = {}
    total_tp = 0
    total_adv = 0
    worst_case_tpr = 1.0

    for attack, target in ATTACK_TARGETS.items():
        row = eval_report.get(attack, {})
        tpr = float(row.get("TPR", 0.0))
        fpr = float(row.get("FPR", 1.0))
        n_adv = int(row.get("n_adv", 0))
        tp = int(row.get("TP", round(tpr * n_adv)))
        total_tp += tp
        total_adv += n_adv
        worst_case_tpr = min(worst_case_tpr, tpr)
        attacks[attack] = {
            "TPR": tpr,
            "target": target,
            "passed": tpr >= target,
            "FPR": fpr,
            "n_adv": n_adv,
            "TP": tp,
            "base_attack_success_rate": row.get("base_attack_success_rate"),
            "detector_TPR_on_base_success": row.get("detector_TPR_on_base_success"),
        }

    fpr_tiers = fpr_report.get("tiers", {})
    fpr_summary = {}
    for tier, target in FPR_TARGETS.items():
        fpr = float(fpr_tiers.get(tier, {}).get("FPR", 1.0))
        fpr_summary[tier] = {
            "FPR": fpr,
            "target": target,
            "passed": fpr <= target + 1e-12,
        }
    latency = eval_report.get("_meta", {}).get("latency", {})
    pooled_lower = wilson_lower(total_tp, total_adv)

    return {
        "candidate": candidate.key,
        "label": candidate.label,
        "feature_contract": ensemble.get("feature_space_version"),
        "n_features": ensemble.get("n_features"),
        "use_grad_norm": bool(ensemble.get("use_grad_norm", False)),
        "training_attack_counts": ensemble.get("training_attack_counts", {}),
        "training_source_split": ensemble.get("training_source_split"),
        "training_source_description": ensemble.get("training_source_description"),
        "balanced_attacks": bool(ensemble.get("balanced_attacks", False)),
        "fgsm_oversample": ensemble.get("fgsm_oversample"),
        "pgd_oversample": ensemble.get("pgd_oversample"),
        "selection_objective": ensemble.get("selection_objective"),
        "per_attack_validation_metrics": ensemble.get("per_attack_validation_metrics", {}),
        "attacks": attacks,
        "worst_case_tpr": worst_case_tpr,
        "mean_tpr": sum(row["TPR"] for row in attacks.values()) / max(len(attacks), 1),
        "pooled_tpr": total_tp / total_adv if total_adv else 0.0,
        "pooled_wilson_lower": pooled_lower,
        "fpr": fpr_summary,
        "latency": latency,
        "passes_attack_targets": all(row["passed"] for row in attacks.values()),
        "passes_fpr_targets": all(row["passed"] for row in fpr_summary.values()),
        "passes_latency": bool(latency.get("pass", False)),
        "passes_pooled_wilson": pooled_lower >= 0.80,
    }


def score_for_selection(summary: Dict[str, Any]) -> tuple:
    return (
        int(summary["passes_fpr_targets"]),
        summary["worst_case_tpr"],
        summary["mean_tpr"],
        int(summary["passes_latency"]),
    )


def parse_candidates(values: Iterable[str]) -> List[Candidate]:
    out: List[Candidate] = []
    for value in values:
        key = value.upper()
        if key not in CANDIDATES:
            raise SystemExit(f"Unknown candidate {value!r}; choose from {', '.join(CANDIDATES)}")
        out.append(CANDIDATES[key])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", nargs="+", default=["A", "B", "C", "D", "E", "F"],
                        help="Candidate keys to run: A B C D E F")
    parser.add_argument("--n-train", type=int, default=1500,
                        help="Adversarial training budget per candidate. Use 1500+ for promotion diagnostics when feasible.")
    parser.add_argument("--n-test", type=int, default=100,
                        help="Evaluation samples per attack for the local candidate pass.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--square-max-iter", type=int, default=500,
                        help="Square queries for candidate smoke. Use 5000 for canonical promotion where feasible.")
    parser.add_argument("--source-split", choices=["profile", "test-profile", "train"], default="profile",
                        help="Source split for scorer training. Default profile aligns with the conformal split protocol.")
    parser.add_argument("--gen-chunk", type=int, default=16)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--out-dir", default="experiments/evaluation/candidate_matrix")
    parser.add_argument("--keep-winner", action=argparse.BooleanOptionalAction, default=True,
                        help="Copy the selected winner back to canonical model/calibrator paths.")
    parser.add_argument("--allow-grad-norm-winner", action=argparse.BooleanOptionalAction, default=True,
                        help="Allow the grad-norm winner to become canonical. Disable only for legacy no-grad-norm reproductions.")
    args = parser.parse_args()

    candidates = parse_candidates(args.candidates)
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    initial_snapshot = snapshot_artifacts(out_dir / "_initial_artifacts")

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("OMP_NUM_THREADS", "4")

    summaries: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for candidate in candidates:
        cand_dir = out_dir / candidate.label
        cand_dir.mkdir(parents=True, exist_ok=True)
        eval_out = cand_dir / f"eval_n{args.n_test}_seed{args.seed}.json"

        train_cmd = [
            sys.executable,
            "scripts/train_ensemble_scorer.py",
            "--n-train", str(args.n_train),
            "--pgd-train-steps", "40",
            "--square-train-max-iter", str(args.square_max_iter),
            "--selection-objective", "worst_case_tpr",
            "--source-split", args.source_split,
            "--output", str(CANONICAL_ENSEMBLE),
            *candidate.train_args,
        ]
        calibrate_cmd = [sys.executable, "scripts/calibrate_ensemble.py"]
        fpr_cmd = [sys.executable, "scripts/compute_ensemble_val_fpr.py"]
        eval_cmd = [
            sys.executable,
            "experiments/evaluation/run_evaluation_full.py",
            "--n-test", str(args.n_test),
            "--attacks", "FGSM", "PGD", "Square",
            "--seed", str(args.seed),
            "--square-max-iter", str(args.square_max_iter),
            "--gen-chunk", str(args.gen_chunk),
            "--checkpoint-interval", str(args.checkpoint_interval),
            "--output", str(eval_out),
        ]

        try:
            run(train_cmd, log_path=cand_dir / "01_train.log", env=env)
            run(calibrate_cmd, log_path=cand_dir / "02_calibrate.log", env=env)
            run(fpr_cmd, log_path=cand_dir / "03_fpr.log", env=env)
            run(eval_cmd, log_path=cand_dir / "04_eval.log", env=env)

            with CANONICAL_ENSEMBLE.open("rb") as f:
                ensemble = pickle.load(f)
            if not isinstance(ensemble, dict):
                raise TypeError(f"Expected dict ensemble artifact, got {type(ensemble).__name__}")

            copy_if_exists(CANONICAL_ENSEMBLE, cand_dir / "ensemble_scorer.pkl")
            copy_if_exists(CANONICAL_CALIBRATOR, cand_dir / "calibrator.pkl")
            copy_if_exists(CANONICAL_FPR, cand_dir / "ensemble_fpr_report.json")

            summary = summarize_candidate(
                candidate=candidate,
                eval_report=load_json(eval_out),
                fpr_report=load_json(CANONICAL_FPR),
                ensemble=ensemble,
            )
            summaries.append(summary)
            (cand_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            print(
                f"[SUMMARY] {candidate.label}: worst_case_tpr={summary['worst_case_tpr']:.3f}, "
                f"mean_tpr={summary['mean_tpr']:.3f}, pooled_wilson_lower={summary['pooled_wilson_lower']:.3f}, "
                f"fpr_pass={summary['passes_fpr_targets']}"
            )
        except Exception as exc:
            failures.append({"candidate": candidate.key, "label": candidate.label, "error": str(exc)})
            print(f"[FAIL] {candidate.label}: {exc}", file=sys.stderr)

    if not summaries:
        restore_artifacts(initial_snapshot)
        matrix = {"status": "failed", "failures": failures}
        (out_dir / "matrix_summary.json").write_text(json.dumps(matrix, indent=2), encoding="utf-8")
        return 1

    research_winner = max(summaries, key=score_for_selection)
    keepable = [
        s for s in summaries
        if s["passes_fpr_targets"] and (args.allow_grad_norm_winner or not s["use_grad_norm"])
    ]
    canonical_winner = max(keepable, key=score_for_selection) if keepable else None

    matrix = {
        "status": "complete",
        "n_train": args.n_train,
        "n_test": args.n_test,
        "seed": args.seed,
        "square_max_iter": args.square_max_iter,
        "targets": {"attacks": ATTACK_TARGETS, "fpr": FPR_TARGETS, "pooled_wilson_lower": 0.80},
        "research_winner": research_winner["label"],
        "canonical_winner": canonical_winner["label"] if canonical_winner else None,
        "all_candidates": summaries,
        "failures": failures,
    }

    if args.keep_winner and canonical_winner is not None:
        winner_dir = out_dir / canonical_winner["label"]
        copy_if_exists(winner_dir / "ensemble_scorer.pkl", CANONICAL_ENSEMBLE)
        copy_if_exists(winner_dir / "calibrator.pkl", CANONICAL_CALIBRATOR)
        copy_if_exists(winner_dir / "ensemble_fpr_report.json", CANONICAL_FPR)
        matrix["kept_canonical_winner"] = True
    else:
        restore_artifacts(initial_snapshot)
        matrix["kept_canonical_winner"] = False

    summary_path = out_dir / "matrix_summary.json"
    summary_path.write_text(json.dumps(matrix, indent=2, sort_keys=True), encoding="utf-8")

    print("\nCandidate matrix complete.")
    print(f"Summary: {summary_path}")
    print(f"Research winner: {matrix['research_winner']}")
    print(f"Canonical winner: {matrix['canonical_winner'] or 'none'}")
    if research_winner["use_grad_norm"] and not args.allow_grad_norm_winner:
        print("Note: grad-norm candidate was not promoted to canonical contract.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
