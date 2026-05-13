"""
build_paper_tables.py (P0.7)

Reads every result JSON under prism_results/experiments/ and emits LaTeX
\\input-able tables with Wilson 95% CIs. Writing the paper becomes a
copy-paste job; no manual numerical transcription.

Emitted tables (into --out-dir, default: paper/tables/):
  main_attacks.tex            — PRISM attacks (FGSM/PGD/Square/AutoAttack/CW) 5-seed pooled
  adaptive_pgd.tex            — λ sweep TPR/FPR with tier FPRs
  ablation.tex                — Full vs No-MoE vs Ensemble-no-TDA vs TDA-only
  baselines.tex               — LID / Mahalanobis / ODIN / Energy vs PRISM at matched FPR
  campaign.tex                — SACD scenarios, L0-on vs L0-off ASR gap
  recovery.tex                — TAMSH recovery_accuracy per strategy

Seed aggregation:
  If multiple `results_<family>_seed<N>.json` files exist, per-cell TPR/FPR
  counts are summed and a single Wilson CI is computed on the pooled
  contingency. Falls back gracefully if only one seed is present.

USAGE
-----
  cd prism/
  python scripts/build_paper_tables.py
  python scripts/build_paper_tables.py --results-dir ../prism_results --out-dir paper/tables/
"""
import json
import os
import re
import argparse
from glob import glob
from collections import defaultdict

import numpy as np


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _fmt(val, prec=3):
    if val is None:
        return '--'
    return f"{val:.{prec}f}"


def _fmt_ci(lo, hi, prec=3):
    return f"[{_fmt(lo, prec)}, {_fmt(hi, prec)}]"


def _load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _experiments_root(results_dir):
    """Accept project root, experiments/, or external prism_results/ roots."""
    rd = os.path.abspath(results_dir)
    if os.path.basename(rd) == 'experiments':
        return rd
    if os.path.isdir(os.path.join(rd, 'evaluation')) or os.path.isdir(os.path.join(rd, 'ablation')):
        return rd
    exp = os.path.join(rd, 'experiments')
    return exp if os.path.isdir(exp) else rd


def _dataset_label(path, data):
    meta = data.get('metadata') or data.get('_meta') or {}
    dataset = meta.get('dataset') or data.get('dataset')
    if dataset:
        ds = str(dataset).upper().replace('_', '-')
        if ds in {'CIFAR10', 'CIFAR-10'}:
            return 'CIFAR-10'
        if ds in {'CIFAR100', 'CIFAR-100'}:
            return 'CIFAR-100'
        return ds
    name = os.path.basename(path).lower()
    if 'cifar100' in name or 'c100' in name:
        return 'CIFAR-100'
    return 'CIFAR-10'


def _entry_from_aggregate(entry):
    if not isinstance(entry, dict):
        return None
    if all(k in entry for k in ('pool_TP', 'pool_FP', 'pool_FN', 'pool_TN')):
        return {
            'TP': entry['pool_TP'], 'FP': entry['pool_FP'],
            'FN': entry['pool_FN'], 'TN': entry['pool_TN'],
        }
    return None


def _collect_attack_entries(path):
    """Return [(dataset, attack, counts_dict)] from single or multi-seed files."""
    d = _load_json(path)
    dataset = _dataset_label(path, d)
    rows = []

    if isinstance(d.get('aggregate'), dict):
        for atk, entry in d['aggregate'].items():
            counts = _entry_from_aggregate(entry)
            if counts is not None:
                rows.append((dataset, atk, counts))
        if rows:
            return rows

    if isinstance(d.get('per_seed'), dict):
        for seed_res in d['per_seed'].values():
            if not isinstance(seed_res, dict):
                continue
            for atk, entry in seed_res.items():
                if isinstance(entry, dict) and 'TP' in entry:
                    rows.append((dataset, atk, entry))
        if rows:
            return rows

    for atk, entry in d.items():
        if atk.startswith('_') or not isinstance(entry, dict):
            continue
        if 'TP' in entry:
            rows.append((dataset, atk, entry))
    return rows


# ────────────────────────────────────────────────────────────────────────────
# Pool helpers
# ────────────────────────────────────────────────────────────────────────────

def _pool_counts(entries):
    """Sum TP/FP/FN/TN across entries; return dict + pooled Wilson CIs."""
    tp = sum(e.get('TP', 0) for e in entries)
    fp = sum(e.get('FP', 0) for e in entries)
    fn = sum(e.get('FN', 0) for e in entries)
    tn = sum(e.get('TN', 0) for e in entries)
    n_adv = tp + fn
    n_clean = fp + tn
    tpr = tp / max(n_adv, 1)
    fpr = fp / max(n_clean, 1)
    tpr_lo, tpr_hi = wilson_ci(tp, n_adv)
    fpr_lo, fpr_hi = wilson_ci(fp, n_clean)
    return {
        'TPR': tpr, 'TPR_CI': (tpr_lo, tpr_hi),
        'FPR': fpr, 'FPR_CI': (fpr_lo, fpr_hi),
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
        'n_adv': n_adv, 'n_clean': n_clean,
        'n_seeds': len(entries),
    }


# ────────────────────────────────────────────────────────────────────────────
# Table: main attacks (PRISM)
# ────────────────────────────────────────────────────────────────────────────

def table_main_attacks(results_dir):
    exp_root = _experiments_root(results_dir)
    eval_dir = os.path.join(exp_root, 'evaluation')
    paths = sorted(set(
        glob(os.path.join(eval_dir, 'results_paper_seed*.json'))
        + glob(os.path.join(eval_dir, 'results_*paper_seed*.json'))
        + glob(os.path.join(eval_dir, 'results_fast_n*_ms*.json'))
        + glob(os.path.join(eval_dir, 'results_*fast_n*_ms*.json'))
        + glob(os.path.join(eval_dir, 'results_cw_n*_ms*.json'))
        + glob(os.path.join(eval_dir, 'results_*cw_n*_ms*.json'))
        + glob(os.path.join(eval_dir, 'results_cw_seed*.json'))
        + glob(os.path.join(eval_dir, 'results_*cw_seed*.json'))
    ))
    paths = [
        p for p in paths
        if not ('_ms' in os.path.basename(p) and '_seed' in os.path.basename(p))
    ]

    per_attack = defaultdict(list)
    for p in paths:
        for dataset, atk, entry in _collect_attack_entries(p):
            if atk in {'FGSM', 'PGD', 'Square', 'AutoAttack', 'CW', 'CW_L2'} or 'CW' in atk:
                per_attack[(dataset, atk)].append(entry)

    if not per_attack:
        return f"% No main-attack results found under {eval_dir}.\n"

    lines = [
        "% Auto-generated by scripts/build_paper_tables.py — do not edit by hand.",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Dataset & Attack & TPR & 95\\% CI & FPR & $n$ \\\\",
        "\\midrule",
    ]
    order = ['FGSM', 'PGD', 'Square', 'AutoAttack', 'CW', 'CW_L2']
    keys = sorted(per_attack, key=lambda k: (k[0], order.index(k[1]) if k[1] in order else 99, k[1]))
    for dataset, atk in keys:
        pooled = _pool_counts(per_attack[(dataset, atk)])
        lo, hi = pooled['TPR_CI']
        lines.append(
            f"{dataset} & {atk} & {_fmt(pooled['TPR'])} & {_fmt_ci(lo, hi)} & "
            f"{_fmt(pooled['FPR'])} & {pooled['n_adv']} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return '\n'.join(lines) + '\n'


# ────────────────────────────────────────────────────────────────────────────
# Table: adaptive PGD
# ────────────────────────────────────────────────────────────────────────────

def table_adaptive_pgd(results_dir):
    exp_root = _experiments_root(results_dir)
    paths = sorted(glob(os.path.join(
        exp_root, 'evaluation', 'results*adaptive_pgd_seed*.json')))
    per_dataset_lambda = defaultdict(list)
    for p in paths:
        d = _load_json(p)
        dataset = _dataset_label(p, d)
        for key, entry in d.items():
            if key.startswith('_') or not isinstance(entry, dict): continue
            m = re.match(r'AdaptivePGD_lambda_([0-9.]+)', key)
            if m:
                per_dataset_lambda[(dataset, float(m.group(1)))].append(entry)
    if not per_dataset_lambda:
        return f"% No adaptive-PGD results found under {os.path.join(exp_root, 'evaluation')}.\n"

    lines = [
        "% Auto-generated by scripts/build_paper_tables.py",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Dataset & $\\lambda$ & TPR & 95\\% CI & FPR \\\\",
        "\\midrule",
    ]
    for dataset, lam in sorted(per_dataset_lambda):
        pooled = _pool_counts(per_dataset_lambda[(dataset, lam)])
        lo, hi = pooled['TPR_CI']
        lines.append(
            f"{dataset} & {lam:.1f} & {_fmt(pooled['TPR'])} & {_fmt_ci(lo, hi)} & "
            f"{_fmt(pooled['FPR'])} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return '\n'.join(lines) + '\n'


# ────────────────────────────────────────────────────────────────────────────
# Table: ablation
# ────────────────────────────────────────────────────────────────────────────

def table_ablation(results_dir):
    exp_root = _experiments_root(results_dir)
    paths = sorted(glob(os.path.join(exp_root, 'ablation', 'results*ablation_multiseed.json')))
    if not paths:
        return f"% No ablation multi-seed file found under {os.path.join(exp_root, 'ablation')}.\n"

    lines = [
        "% Auto-generated by scripts/build_paper_tables.py",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Dataset & Configuration & Mean TPR & $\\Delta$ vs Full & $p$-value \\\\",
        "\\midrule",
    ]
    for path in paths:
        d = _load_json(path)
        dataset = _dataset_label(path, d)
        agg = d.get('aggregate', {})
        stat = d.get('statistical_tests', {})
        if not agg:
            continue
        for cfg_name, cfg_agg in agg.items():
            tpr_mean = cfg_agg.get('mean_TPR', cfg_agg.get('TPR_mean'))
            tests = stat.get(cfg_name, {})
            deltas = [
                v.get('mean_delta') for v in tests.values()
                if isinstance(v, dict)
                and isinstance(v.get('mean_delta'), (int, float))
                and np.isfinite(v.get('mean_delta'))
            ]
            pvals = [
                v.get('p_value') for v in tests.values()
                if isinstance(v, dict)
                and isinstance(v.get('p_value'), (int, float))
                and np.isfinite(v.get('p_value'))
            ]
            delta = float(np.mean(deltas)) if deltas else None
            pval = float(np.mean(pvals)) if pvals else None
            delta_s = (f"{delta:+.4f}" if isinstance(delta, (int, float)) else "--")
            pval_s  = (f"{pval:.3f}" if isinstance(pval, (int, float)) else "--")
            lines.append(
                f"{dataset} & {cfg_name} & {_fmt(tpr_mean, 4)} & {delta_s} & {pval_s} \\\\"
            )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return '\n'.join(lines) + '\n'


# ────────────────────────────────────────────────────────────────────────────
# Table: baselines
# ────────────────────────────────────────────────────────────────────────────

def table_baselines(results_dir):
    exp_root = _experiments_root(results_dir)
    paths = sorted(glob(os.path.join(
        exp_root, 'evaluation', 'results*baselines*seed*.json')))
    paths += sorted(glob(os.path.join(
        exp_root, 'evaluation', 'results*baselines.json')))
    per_dataset_det_atk = defaultdict(list)
    for p in paths:
        d = _load_json(p)
        dataset = _dataset_label(p, d)
        for det, attacks in d.items():
            if det.startswith('_') or not isinstance(attacks, dict): continue
            for atk, entry in attacks.items():
                if isinstance(entry, dict) and 'TP' in entry:
                    per_dataset_det_atk[(dataset, det, atk)].append(entry)
    if not per_dataset_det_atk:
        return f"% No baseline results found under {os.path.join(exp_root, 'evaluation')}.\n"

    lines = [
        "% Auto-generated by scripts/build_paper_tables.py",
        "\\begin{tabular}{lllccc}",
        "\\toprule",
        "Dataset & Detector & Attack & TPR & 95\\% CI & FPR \\\\",
        "\\midrule",
    ]
    for dataset, det, atk in sorted(per_dataset_det_atk):
        pooled = _pool_counts(per_dataset_det_atk[(dataset, det, atk)])
        lo, hi = pooled['TPR_CI']
        lines.append(
            f"{dataset} & {det} & {atk} & {_fmt(pooled['TPR'])} & "
            f"{_fmt_ci(lo, hi)} & {_fmt(pooled['FPR'])} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return '\n'.join(lines) + '\n'


# ────────────────────────────────────────────────────────────────────────────
# Table: campaign (SACD)
# ────────────────────────────────────────────────────────────────────────────

def table_campaign(results_dir):
    exp_root = _experiments_root(results_dir)
    paths = sorted(glob(os.path.join(
        exp_root, 'campaign', 'results*campaign*seed*.json')))
    paths += sorted(glob(os.path.join(
        exp_root, 'evaluation', 'results*campaign*.json')))
    per_dataset_scen = defaultdict(lambda: {'l0_on': [], 'l0_off': [], 'l0_active_frac': []})
    for p in paths:
        d = _load_json(p)
        dataset = _dataset_label(p, d)
        for scen, v in d.items():
            if scen.startswith('_') or not isinstance(v, dict): continue
            if 'l0_on' in v and 'l0_off' in v:
                per_dataset_scen[(dataset, scen)]['l0_on'].append(v['l0_on']['ASR'])
                per_dataset_scen[(dataset, scen)]['l0_off'].append(v['l0_off']['ASR'])
                per_dataset_scen[(dataset, scen)]['l0_active_frac'].append(
                    v['l0_on'].get('l0_active_fraction', None))
    if not per_dataset_scen:
        return f"% No campaign results found under {exp_root}.\n"

    lines = [
        "% Auto-generated by scripts/build_paper_tables.py",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Dataset & Scenario & ASR (L0 off) & ASR (L0 on) & $\\Delta$ (pp) \\\\",
        "\\midrule",
    ]
    for dataset, scen in sorted(per_dataset_scen):
        entry = per_dataset_scen[(dataset, scen)]
        off = np.mean(entry['l0_off']) if entry['l0_off'] else 0
        on  = np.mean(entry['l0_on']) if entry['l0_on'] else 0
        delta = 100.0 * (off - on)
        scen_tex = scen.replace('_', '\\_')
        lines.append(
            f"{dataset} & {scen_tex} & {_fmt(off)} & {_fmt(on)} & {delta:+.2f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return '\n'.join(lines) + '\n'


# ────────────────────────────────────────────────────────────────────────────
# Table: recovery (TAMSH)
# ────────────────────────────────────────────────────────────────────────────

def table_recovery(results_dir):
    exp_root = _experiments_root(results_dir)
    paths = sorted(glob(os.path.join(
        exp_root, 'recovery', 'results*recovery*seed*.json')))
    paths += sorted(glob(os.path.join(
        exp_root, 'evaluation', 'results*recovery*.json')))
    per_dataset_strat = defaultdict(list)
    for p in paths:
        d = _load_json(p)
        dataset = _dataset_label(p, d)
        for k, v in d.items():
            if k.startswith('_') or not isinstance(v, dict): continue
            if 'recovery_accuracy' in v:
                per_dataset_strat[(dataset, k)].append(v)
    if not per_dataset_strat:
        return f"% No recovery results found under {exp_root}.\n"

    lines = [
        "% Auto-generated by scripts/build_paper_tables.py",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Dataset & Strategy & Recovery Acc. & 95\\% CI & Availability \\\\",
        "\\midrule",
    ]
    datasets = sorted({dataset for dataset, _ in per_dataset_strat})
    for dataset in datasets:
        for strat in ['reject', 'passthrough', 'tamsh']:
            if (dataset, strat) not in per_dataset_strat:
                continue
            entries = per_dataset_strat[(dataset, strat)]
            n_correct = sum(e.get('n_correct', 0) for e in entries)
            n_l3      = sum(e.get('n_l3', 0) for e in entries)
            acc = n_correct / max(n_l3, 1)
            lo, hi = wilson_ci(n_correct, n_l3)
            avail = np.mean([e.get('availability', 0) for e in entries])
            lines.append(
                f"{dataset} & {strat} & {_fmt(acc)} & {_fmt_ci(lo, hi)} & {_fmt(avail)} \\\\"
            )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description="Build LaTeX tables from prism_results (P0.7)")
    parser.add_argument('--config', default=None,
                        help='YAML config path (accepted for API symmetry; '
                             'build_paper_tables does not read src.config).')
    parser.add_argument('--results-dir', default='../prism_results',
                        help='Path to prism_results/ (relative to prism/ or absolute).')
    parser.add_argument('--out-dir', default='paper/tables',
                        help='Output directory for .tex files.')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    outputs = {
        'main_attacks.tex': table_main_attacks(args.results_dir),
        'adaptive_pgd.tex': table_adaptive_pgd(args.results_dir),
        'ablation.tex':     table_ablation(args.results_dir),
        'baselines.tex':    table_baselines(args.results_dir),
        'campaign.tex':     table_campaign(args.results_dir),
        'recovery.tex':     table_recovery(args.results_dir),
    }
    for fname, content in outputs.items():
        path = os.path.join(args.out_dir, fname)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  wrote {path}  ({len(content.splitlines())} lines)")


if __name__ == '__main__':
    main()
