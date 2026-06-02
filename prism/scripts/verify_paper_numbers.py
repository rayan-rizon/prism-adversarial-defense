"""
Recompute headline paper numbers from the canonical post-fix JSONs and
assert that key table-source values still appear in the active .tex files.

Run:
    python prism/scripts/verify_paper_numbers.py
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[2]
# The canonical CIFAR-10 artifacts were reorganised out of the original
# `vastai_full_download_2026-05-20_0830UTC/` snapshot into `Cifar 10/`, which
# preserves the same `project/` (Vast.ai) and `post_fix_local_2026-05-21/`
# (post-fix local) subtrees. Allow an override and fall back to the legacy
# snapshot name if a checkout still uses it.
import os
_DEFAULT_VASTAI = ROOT / "Cifar 10"
_LEGACY_VASTAI = ROOT / "vastai_full_download_2026-05-20_0830UTC"
VASTAI = Path(os.environ.get("PRISM_ARTIFACT_ROOT", "")) if os.environ.get("PRISM_ARTIFACT_ROOT") else (
    _DEFAULT_VASTAI if _DEFAULT_VASTAI.exists() else _LEGACY_VASTAI
)
POST_FIX = VASTAI / "post_fix_local_2026-05-21"
PROJECT = VASTAI / "project"
PAPER = ROOT / "prism" / "paper"

SEEDS = [42, 123, 456, 789, 999]


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def fmt(x, d=3):
    if isinstance(x, float):
        return f"{x:.{d}f}"
    return str(x)


def std(xs):
    return stdev(xs) if len(xs) > 1 else 0.0


def require_tex_contains(rel_path: str, needles: list[str]) -> None:
    path = PAPER / rel_path
    text = path.read_text(encoding="utf-8")
    missing = [needle for needle in needles if needle not in text]
    if missing:
        raise AssertionError(
            f"{path} missing expected generated values: {missing}"
        )


# ──────────────────────────────────────────────────────────────────────────
def check_main_attacks():
    print("\n=== TABLE: main_attacks.tex ===")
    abl = load(POST_FIX / "experiments" / "ablation" / "results_ablation_multiseed.json")
    rows = {}
    for attack in ("FGSM", "PGD", "Square"):
        tprs = [abl["per_seed"][str(s)]["Full PRISM"][attack]["TPR"] for s in SEEDS]
        fprs = [abl["per_seed"][str(s)]["Full PRISM"][attack]["FPR"] for s in SEEDS]
        rows[attack] = (mean(tprs), std(tprs), mean(fprs), tprs, fprs)

    # CW canonical (max_iter=100, bss=9) — Vast.ai gate run, NOT the
    # post-fix weakened CW (max_iter=5, bss=5) which only feeds the
    # detector-fidelity ablation in Appendix app:weak_cw.
    cw_tprs, cw_fprs = [], []
    for s in SEEDS:
        f = PROJECT / "experiments" / "evaluation" / f"results_cw_n1000_ms5_seed{s}.json"
        d = load(f)
        cw_node = d.get("CW") or d.get("CW-L2") or d.get("CW_L2") or d
        cw_tprs.append(cw_node["TPR"])
        cw_fprs.append(cw_node["FPR"])
    rows["CW-L2"] = (mean(cw_tprs), std(cw_tprs), mean(cw_fprs), cw_tprs, cw_fprs)

    # AutoAttack — from fast_n1000_ms5
    aa_tprs, aa_fprs = [], []
    for s in SEEDS:
        f = PROJECT / "experiments" / "evaluation" / f"results_fast_n1000_ms5_seed{s}.json"
        if not f.exists():
            continue
        d = load(f)
        # peek schema once
        if s == 42 and not aa_tprs:
            print(f"  fast_n1000_ms5 seed42 top keys: {list(d.keys())[:15]}")
        aa = d.get("AutoAttack") or d.get("autoattack") or d.get("AA")
        if aa is None and "results" in d:
            aa = d["results"].get("AutoAttack") or d["results"].get("autoattack")
        if isinstance(aa, dict):
            aa_tprs.append(aa.get("TPR"))
            aa_fprs.append(aa.get("FPR"))
    if aa_tprs and aa_tprs[0] is not None:
        rows["AutoAttack"] = (mean(aa_tprs), std(aa_tprs), mean(aa_fprs), aa_tprs, aa_fprs)

    print(f"  {'attack':<11} {'mean':>6} {'std':>6} {'FPR':>6}   per-seed TPRs")
    for k, (m, s, f, tprs, fprs) in rows.items():
        print(f"  {k:<11} {fmt(m):>6} {fmt(s):>6} {fmt(f):>6}   {[fmt(t) for t in tprs]}")
    return rows


# ──────────────────────────────────────────────────────────────────────────
def check_ablation():
    print("\n=== TABLE: ablation.tex ===")
    abl = load(POST_FIX / "experiments" / "ablation" / "results_ablation_multiseed.json")
    seed0 = abl["per_seed"][str(SEEDS[0])]
    arms = list(seed0.keys())
    summary = {}
    print(f"  {'arm':<22} {'mean_TPR':>9} {'std':>6} {'mean_FPR':>9}  "
          f"{'FGSM':>6} {'PGD':>6} {'Sq':>6}")
    for arm in arms:
        per_attack = {}
        for attack in ("FGSM", "PGD", "Square"):
            tprs = [abl["per_seed"][str(s)][arm][attack]["TPR"] for s in SEEDS]
            fprs = [abl["per_seed"][str(s)][arm][attack]["FPR"] for s in SEEDS]
            per_attack[attack] = (mean(tprs), std(tprs), mean(fprs))
        # Compute mean across attacks at PER-SEED level then aggregate
        per_seed_means = []
        per_seed_fpr_means = []
        for s in SEEDS:
            m_tpr = mean([abl["per_seed"][str(s)][arm][a]["TPR"] for a in ("FGSM", "PGD", "Square")])
            m_fpr = mean([abl["per_seed"][str(s)][arm][a]["FPR"] for a in ("FGSM", "PGD", "Square")])
            per_seed_means.append(m_tpr)
            per_seed_fpr_means.append(m_fpr)
        mean_tpr = mean(per_seed_means)
        std_tpr = std(per_seed_means)
        mean_fpr = mean(per_seed_fpr_means)
        summary[arm] = (mean_tpr, std_tpr, mean_fpr, per_attack)
        print(f"  {arm:<22} {fmt(mean_tpr):>9} {fmt(std_tpr):>6} {fmt(mean_fpr):>9}  "
              f"{fmt(per_attack['FGSM'][0]):>6} {fmt(per_attack['PGD'][0]):>6} "
              f"{fmt(per_attack['Square'][0]):>6}")

    # Marginal gap claim (Full - Ensemble-no-TDA)
    if "Full PRISM" in summary and "Ensemble-no-TDA" in summary:
        gap = summary["Full PRISM"][0] - summary["Ensemble-no-TDA"][0]
        print(f"\n  CLAIM CHECK: Full - Ensemble-no-TDA = {gap*100:.2f} pp  (paper: +13.1 pp)")
    return summary


# ──────────────────────────────────────────────────────────────────────────
def check_baselines():
    print("\n=== TABLE: baselines.tex ===")
    agg = load(POST_FIX / "baselines" / "results_baselines_aggregate.json")
    print(f"  {'detector':<14} {'mean_TPR':>9} {'mean_FPR':>9}  "
          f"FGSM         PGD          Square")
    prism_mean = 0.918
    for det in agg["detectors"]:
        d = agg["aggregate"][det]
        m = d["mean_TPR_across_attacks"]
        fprs = [d[a]["FPR_mean"] for a in ("FGSM", "PGD", "Square")]
        mean_fpr = mean(fprs)
        print(f"  {det:<14} {fmt(m):>9} {fmt(mean_fpr):>9}  "
              f"{fmt(d['FGSM']['TPR_mean'])}±{fmt(d['FGSM']['TPR_std'],3)}  "
              f"{fmt(d['PGD']['TPR_mean'])}±{fmt(d['PGD']['TPR_std'],3)}  "
              f"{fmt(d['Square']['TPR_mean'])}±{fmt(d['Square']['TPR_std'],3)}")
    # Best baseline gap
    best = max(agg["detectors"], key=lambda dt: agg["aggregate"][dt]["mean_TPR_across_attacks"])
    best_v = agg["aggregate"][best]["mean_TPR_across_attacks"]
    gap = (prism_mean - best_v) * 100
    print(f"\n  Best baseline = {best} @ {best_v:.4f}  ->  PRISM(0.918) - {best}({best_v:.3f}) = {gap:.2f} pp")
    return agg


# ──────────────────────────────────────────────────────────────────────────
def check_recovery():
    """
    Paper recovery.tex sources every row from the RECOVERY_UNIFORM pool
    (per the table header comment: 'post-fix 5-seed recovery_uniform pool ...').
    The other RECOVERY pool exists but uses a slightly different L3-trigger
    composition, so mixing pools shifts std on the paired gap.
    This function reads ONLY recovery_uniform so the printed numbers
    match the paper table exactly. Gaps are reported as PAIRED per-seed
    deltas, matching how the table was assembled.
    """
    print("\n=== TABLE: recovery.tex (source: recovery_uniform pool) ===")
    per_seed_rows = []
    for s in SEEDS:
        f = POST_FIX / "experiments" / "recovery_uniform" / f"results_recovery_uniform_seed{s}.json"
        d = load(f)
        meta = d.get("_meta", {})
        row = {
            "seed": s,
            "trig_rate": meta.get("l3_trigger_rate"),
            "n_l3": meta.get("n_l3_triggered"),
            "passthrough": d["passthrough"]["recovery_accuracy"],
            "tamsh": d["tamsh"]["recovery_accuracy"],
            "tamsh_uniform": d["tamsh_uniform"]["recovery_accuracy"],
            "tamsh_force": d["tamsh_force"]["recovery_accuracy"],
        }
        per_seed_rows.append(row)

    # Per-seed table
    print(f"  {'seed':>5} {'trig':>7} {'n_l3':>5} "
          f"{'pthru':>7} {'tamsh':>7} {'unif':>7} {'force':>7}")
    for r in per_seed_rows:
        print(f"  {r['seed']:>5} {r['trig_rate']:>7.3f} {r['n_l3']:>5} "
              f"{r['passthrough']:>7.4f} {r['tamsh']:>7.4f} "
              f"{r['tamsh_uniform']:>7.4f} {r['tamsh_force']:>7.4f}")

    # Aggregate
    print()
    for c in ("trig_rate", "passthrough", "tamsh", "tamsh_uniform", "tamsh_force"):
        vals = [r[c] for r in per_seed_rows]
        print(f"  {c:>16}  mean={mean(vals):.4f}  std={std(vals):.4f}")

    # Paired gaps (matches paper recovery.tex)
    print("\n  Paired-per-seed gaps vs passthrough (matches paper):")
    base = [r["passthrough"] for r in per_seed_rows]
    for c in ("tamsh", "tamsh_uniform", "tamsh_force"):
        deltas = [r[c] - r["passthrough"] for r in per_seed_rows]
        m = mean(deltas) * 100
        sd = std(deltas) * 100
        print(f"    {c:>16}  +{m:.2f}pp +/- {sd:.2f}pp")

    # Side-note: the other RECOVERY pool also has a tamsh_ensemble variant
    # that is NOT in the paper. Print as informational diagnostic.
    print("\n  [Note] The separate RECOVERY pool also contains "
          "tamsh_ensemble (soft-mix over experts).")
    print("         Not in paper because it lives in a pool with different "
          "L3-trigger composition.")
    extra = []
    for s in SEEDS:
        f = POST_FIX / "experiments" / "recovery" / f"results_recovery_post_fix_seed{s}.json"
        if not f.exists():
            continue
        d = load(f)
        e = d.get("tamsh_ensemble")
        if e and isinstance(e, dict):
            extra.append(e.get("recovery_accuracy"))
    if extra:
        print(f"         tamsh_ensemble (n={len(extra)}): mean={mean(extra):.4f}  std={std(extra):.4f}")
    return per_seed_rows


# ──────────────────────────────────────────────────────────────────────────
def check_campaign():
    print("\n=== TABLE: campaign.tex ===")
    scenarios = {}
    fa_clean_l0_on = []
    fa_clean_l0_off = []
    for s in SEEDS:
        d = load(POST_FIX / "experiments" / "evaluation" / f"results_campaign_local_seed{s}.json")
        for scen_name, scen_data in d.items():
            if not isinstance(scen_data, dict):
                continue
            if "l0_on" in scen_data:
                ttd = scen_data["l0_on"].get("time_to_detect_queries")
                l0_act_on = scen_data["l0_on"].get("l0_active_fraction")
                l0_act_off = scen_data["l0_off"].get("l0_active_fraction")
                scenarios.setdefault(scen_name, {"ttd": [], "l0_on_frac": [], "l0_off_frac": []})
                if ttd is not None:
                    scenarios[scen_name]["ttd"].append(ttd)
                if l0_act_on is not None:
                    scenarios[scen_name]["l0_on_frac"].append(l0_act_on)
                if l0_act_off is not None:
                    scenarios[scen_name]["l0_off_frac"].append(l0_act_off)
            if "clean" in scen_name.lower():
                if "l0_on" in scen_data:
                    fa_clean_l0_on.append(scen_data["l0_on"].get("l0_active_fraction"))
                    fa_clean_l0_off.append(scen_data["l0_off"].get("l0_active_fraction"))

    for name, data in scenarios.items():
        ttd = data["ttd"]
        on_frac = data["l0_on_frac"]
        off_frac = data["l0_off_frac"]
        if ttd:
            m = mean(ttd)
            sd = std(ttd)
            print(f"  {name:<28} TTD mean={fmt(m,2):>5}±{fmt(sd,2):<5} "
                  f"L0_active(on)={fmt(mean(on_frac))}  L0_active(off)={fmt(mean(off_frac))}  "
                  f"per-seed TTD={ttd}")
        else:
            print(f"  {name:<28} TTD=None  L0_active(on)={fmt(mean(on_frac)) if on_frac else 'NA'}")

    print(f"\n  Clean-only L0 activation (on): {fa_clean_l0_on}")
    print(f"  Clean-only L0 activation (off): {fa_clean_l0_off}")
    return scenarios


# ──────────────────────────────────────────────────────────────────────────
def check_adaptive_pgd():
    print("\n=== TABLE: adaptive_pgd.tex ===")
    summary = ROOT / "prism" / "experiments" / "evaluation" / "ensemble_complete_adaptive_pgd_summary.json"
    if not summary.exists():
        summary = PROJECT / "experiments" / "evaluation" / "ensemble_complete_adaptive_pgd_summary.json"
    if summary.exists():
        d = load(summary)
        print("  ensemble-complete adaptive PGD summary found")
        scan = d.get("lambda_scan_n50_pooled", {})
        for lam, row in sorted(scan.items(), key=lambda kv: float(kv[0])):
            ci = row.get("TPR_CI_95", [None, None])
            print(
                f"  scan lambda={float(lam):<4.1f} "
                f"TPR={fmt(row.get('TPR'))} CI=[{fmt(ci[0])}, {fmt(ci[1])}] "
                f"TPR_succ={fmt(row.get('TPR_on_successful_attacks'))} "
                f"FPR={fmt(row.get('FPR'))} n={row.get('n_adv')}"
            )
        confirm = d.get("worst_lambda_n200_pooled", {})
        if confirm:
            ci = confirm.get("TPR_CI_95", [None, None])
            print(
                f"  confirm lambda=10.0 TPR={fmt(confirm.get('TPR'))} "
                f"CI=[{fmt(ci[0])}, {fmt(ci[1])}] "
                f"TPR_succ={fmt(confirm.get('TPR_on_successful_attacks'))} "
                f"undetected_success={fmt(confirm.get('undetected_success_rate'))} "
                f"FPR={fmt(confirm.get('FPR'))} n={confirm.get('n_adv')}"
            )
        return d

    lambdas = {}
    for s in SEEDS:
        f = PROJECT / "experiments" / "evaluation" / f"results_adaptive_pgd_seed{s}.json"
        if not f.exists():
            continue
        d = load(f)
        if s == 42:
            print(f"  top-level keys: {list(d.keys())[:8]}")
        for lam_key, lam_data in d.items():
            if not isinstance(lam_data, dict) or "lambda" not in lam_key.lower():
                continue
            tpr = lam_data.get("TPR")
            fpr = lam_data.get("FPR")
            asr = lam_data.get("ASR", lam_data.get("attack_success_rate"))
            ci = lam_data.get("TPR_CI_95")
            lambdas.setdefault(lam_key, []).append({"TPR": tpr, "FPR": fpr, "ASR": asr, "CI": ci})

    for k, vals in lambdas.items():
        tprs = [v["TPR"] for v in vals if v["TPR"] is not None]
        fprs = [v["FPR"] for v in vals if v.get("FPR") is not None]
        if tprs:
            cis_lo = [v["CI"][0] for v in vals if v.get("CI")]
            cis_hi = [v["CI"][1] for v in vals if v.get("CI")]
            print(f"  {k:<28} TPR={fmt(mean(tprs))}±{fmt(std(tprs)):<6} "
                  f"FPR={fmt(mean(fprs)) if fprs else 'NA':<6} "
                  f"CI=[{fmt(mean(cis_lo)) if cis_lo else 'NA'}, "
                  f"{fmt(mean(cis_hi)) if cis_hi else 'NA'}] n={len(tprs)}")
    return lambdas


# ──────────────────────────────────────────────────────────────────────────
def check_vit_cifar10_summary():
    print("\n=== TABLE: main_attacks_multi.tex / ViT-B/16 rows ===")
    summary = ROOT / "prism" / "experiments" / "evaluation" / "vit_cifar10_summary.json"
    if not summary.exists():
        print(f"  MISSING: {summary}")
        return None
    d = load(summary)
    print(f"  source: {d.get('source_download')}")
    print(f"  scope: standard attacks only; latency_skipped={d['protocol']['latency_skipped']}")
    print(
        f"  backbone test_acc={fmt(d['model']['test_acc'], 4)} "
        f"verify_acc={fmt(d['model']['measured_verify_acc_n1000'], 4)}"
    )
    print(f"  {'attack':<8} {'TPR':>7} {'CI':>18} {'FPR':>7} {'n_adv':>6} {'base_ASR':>9}")
    for attack in ("FGSM", "PGD", "Square"):
        row = d["aggregate"][attack]
        ci = row["TPR_CI_95_pooled"]
        print(
            f"  {attack:<8} {fmt(row['TPR_mean'], 4):>7} "
            f"[{fmt(ci[0], 4)}, {fmt(ci[1], 4)}] "
            f"{fmt(row['FPR_mean'], 4):>7} "
            f"{row['pool_TP'] + row['pool_FN']:>6} "
            f"{fmt(row['base_attack_success_rate'], 4):>9}"
        )
    gate = d["target_metric_gate"]
    print(f"  gate passed: {gate['passed']} failures={gate['failures']}")
    return d


def check_tex_sources():
    print("\n=== STRICT .tex SOURCE CHECKS ===")
    checks = {
        "tables/main_attacks.tex": [
            "$0.882 \\pm 0.004$",
            "$0.987 \\pm 0.003$",
            "$0.886 \\pm 0.011$",
            "$0.938 \\pm 0.008$",
            "$1.000 \\pm 0.000$",
        ],
        "tables/adaptive_pgd.tex": [
            "Confirm & 10.0 & 1000 & \\textbf{0.479}",
            "\\textbf{0.866} & 0.082",
        ],
        "tables/campaign.tex": [
            "sustained\\_$\\rho{=}1.00$ & $\\mathbf{2.6 \\pm 0.5}$",
            "clean\\_only         & ---  (no adversary)",
        ],
        "tables/recovery.tex": [
            "\\textbf{tamsh topology gate} & $\\mathbf{0.139}$",
            "$\\mathbf{+10.62 \\pm 1.45}$",
            "Oracle ceiling",
        ],
        "tables/main_attacks_multi.tex": [
            "CIFAR-10  & WRN-28-10  & FGSM       & 0.985",
            "CIFAR-100 & ResNet-18  & Square     & 0.672",
            "CIFAR-10  & ViT-B/16   & Square     & 0.9998",
        ],
    }
    for rel_path, needles in checks.items():
        require_tex_contains(rel_path, needles)
        print(f"  OK {rel_path}")


def main():
    print("=" * 70)
    print("PRISM Paper Number Cross-Check")
    print(f"  POSTFIX: {POST_FIX}")
    print("=" * 70)

    check_main_attacks()
    check_ablation()
    check_baselines()
    check_recovery()
    check_campaign()
    check_adaptive_pgd()
    check_vit_cifar10_summary()
    check_tex_sources()


if __name__ == "__main__":
    main()
