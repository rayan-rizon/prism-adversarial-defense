"""
plot_paper_figures.py

Builds publication-grade figures from existing result JSONs:
  1. Score distribution: clean vs adversarial (per dataset, all attacks)
  2. ROC curves per attack (from per-tier TPR/FPR points + score quantiles)
  3. Per-attack TPR comparison bar chart (PRISM vs baselines)

Inputs are pure JSON; no GPU re-run required.
Output: paper/figures/*.png and *.pdf

USAGE
-----
  python scripts/plot_paper_figures.py \
      --wrn-results experiments/wrn \
      --c100-results "../Cifar 100/Cifar 100/project/experiments" \
      --out-dir paper/figures
"""
import argparse
import json
import os
from glob import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

ATTACK_ORDER = ['FGSM', 'PGD', 'Square', 'AutoAttack', 'CW']
ATTACK_COLORS = {
    'FGSM': '#1f77b4',
    'PGD': '#ff7f0e',
    'Square': '#2ca02c',
    'AutoAttack': '#d62728',
    'CW': '#9467bd',
}


def _load(p):
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Figure 1: score distributions (synthesized from quantiles in per-seed files)
# ---------------------------------------------------------------------------

def _gather_quantiles(paths, attack_filter=None):
    """Return dict atk -> {'clean': [list of dicts], 'adv': [list of dicts]}."""
    out = {}
    for p in paths:
        d = _load(p)
        for atk, entry in d.items():
            if atk.startswith('_') or not isinstance(entry, dict):
                continue
            if attack_filter and atk not in attack_filter:
                continue
            sq = entry.get('score_quantiles')
            if not sq:
                continue
            out.setdefault(atk, {'clean': [], 'adv': []})
            out[atk]['clean'].append(sq.get('clean', {}))
            out[atk]['adv'].append(sq.get('adversarial', {}))
    return out


def plot_score_distributions(quantiles_by_atk, dataset_label, out_path):
    """Stacked horizontal violin-like plot using percentile envelopes."""
    attacks = [a for a in ATTACK_ORDER if a in quantiles_by_atk]
    if not attacks:
        print(f"  [skip] no quantile data for {dataset_label}")
        return
    fig, ax = plt.subplots(figsize=(8, 1.0 + 0.75 * len(attacks)))

    def _avg(entries, key):
        vals = [e.get(key) for e in entries if isinstance(e.get(key), (int, float))]
        return float(np.mean(vals)) if vals else None

    y = 0
    yticks, ylabels = [], []
    for atk in attacks:
        c = quantiles_by_atk[atk]['clean']
        a = quantiles_by_atk[atk]['adv']
        # CLEAN row
        c_p05 = _avg(c, 'p05'); c_p25 = _avg(c, 'p25'); c_p50 = _avg(c, 'p50')
        c_p75 = _avg(c, 'p75'); c_p95 = _avg(c, 'p95')
        a_p05 = _avg(a, 'p05'); a_p25 = _avg(a, 'p25'); a_p50 = _avg(a, 'p50')
        a_p75 = _avg(a, 'p75'); a_p95 = _avg(a, 'p95')
        if None in (c_p05, c_p25, c_p50, c_p75, c_p95, a_p05, a_p25, a_p50, a_p75, a_p95):
            continue
        # CLEAN: blue box (thin), whiskers
        ax.hlines(y, c_p05, c_p95, color='#1f77b4', linewidth=1.0, alpha=0.6)
        ax.add_patch(plt.Rectangle((c_p25, y - 0.18), c_p75 - c_p25, 0.36,
                                    facecolor='#aec7e8', edgecolor='#1f77b4', linewidth=1.0))
        ax.plot(c_p50, y, '|', color='#1f77b4', markersize=10, markeredgewidth=2)
        # ADV: red box (thin), whiskers, slight y offset
        y2 = y - 0.5
        ax.hlines(y2, a_p05, a_p95, color='#d62728', linewidth=1.0, alpha=0.6)
        ax.add_patch(plt.Rectangle((a_p25, y2 - 0.18), a_p75 - a_p25, 0.36,
                                    facecolor='#ff9896', edgecolor='#d62728', linewidth=1.0))
        ax.plot(a_p50, y2, '|', color='#d62728', markersize=10, markeredgewidth=2)
        yticks.append(y - 0.25)
        ylabels.append(atk)
        y -= 1.4

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel('PRISM score (higher = more adversarial)')
    ax.set_title(f'Score distributions: clean (blue) vs adversarial (red) — {dataset_label}')
    ax.grid(axis='x', linestyle=':', alpha=0.4)
    ax.axvline(0, color='gray', linewidth=0.5)
    # legend (proxy artists)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor='#aec7e8', edgecolor='#1f77b4', label='Clean (p5–p95, IQR, median)'),
        Patch(facecolor='#ff9896', edgecolor='#d62728', label='Adversarial (p5–p95, IQR, median)'),
    ], loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path + '.png', dpi=180)
    fig.savefig(out_path + '.pdf')
    plt.close(fig)
    print(f"  wrote {out_path}.png + .pdf")


# ---------------------------------------------------------------------------
# Figure 2: per-tier TPR/FPR points (PRISM operating curve)
# ---------------------------------------------------------------------------

def plot_tier_operating_points(per_attack_tiers, dataset_label, out_path):
    """For each attack: 3 points (L1, L2, L3) showing (FPR, TPR) trade-off."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], '--', color='gray', alpha=0.5, label='random')

    for atk, tiers in per_attack_tiers.items():
        fprs = [t[0] for t in tiers]
        tprs = [t[1] for t in tiers]
        ax.plot(fprs, tprs, '-o', color=ATTACK_COLORS.get(atk, '#333'),
                label=atk, markersize=8, linewidth=1.5)
        # tier labels
        for (fpr, tpr), lvl in zip(tiers, ['L1', 'L2', 'L3']):
            ax.annotate(lvl, (fpr, tpr), textcoords='offset points',
                        xytext=(4, 4), fontsize=7,
                        color=ATTACK_COLORS.get(atk, '#333'))

    ax.set_xlim(0, 0.15)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.set_title(f'PRISM tier operating points — {dataset_label}')
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.legend(loc='lower right', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path + '.png', dpi=180)
    fig.savefig(out_path + '.pdf')
    plt.close(fig)
    print(f"  wrote {out_path}.png + .pdf")


def gather_tier_points(paths):
    """Return dict atk -> [(FPR_L1, TPR_L1), (FPR_L2, TPR_L2), (FPR_L3, TPR_L3)] averaged across seeds."""
    per = {}
    for p in paths:
        d = _load(p)
        for atk, e in d.items():
            if atk.startswith('_') or not isinstance(e, dict):
                continue
            tiers = e.get('per_tier_fpr')
            if not tiers:
                continue
            # Use stored L1 FPR and main TPR; L2/L3 TPRs not in this schema, derive from level dist
            adv_levels = e.get('adversarial_level_distribution', {})
            clean_levels = e.get('clean_level_distribution', {})
            n_adv = sum(adv_levels.values()) or e.get('n_adv', 1000)
            n_clean = sum(clean_levels.values()) or e.get('n_clean', 1000)
            # L1+ TPR = anything not PASS (level >= L1)
            tpr_l1 = (n_adv - adv_levels.get('PASS', 0)) / n_adv
            tpr_l2 = (adv_levels.get('L2', 0) + adv_levels.get('L3_REJECT', 0)) / n_adv
            tpr_l3 = adv_levels.get('L3_REJECT', 0) / n_adv
            fpr_l1 = tiers.get('FPR_L1_plus', 0)
            fpr_l2 = tiers.get('FPR_L2_plus', 0)
            fpr_l3 = tiers.get('FPR_L3_plus', 0)
            per.setdefault(atk, []).append([(fpr_l1, tpr_l1), (fpr_l2, tpr_l2), (fpr_l3, tpr_l3)])

    # Average across seeds
    avg = {}
    for atk, seed_lists in per.items():
        arr = np.array(seed_lists)  # shape (n_seeds, 3, 2)
        mean = arr.mean(axis=0)
        avg[atk] = [tuple(p) for p in mean]
    return avg


# ---------------------------------------------------------------------------
# Figure 3: PRISM vs baselines comparison
# ---------------------------------------------------------------------------

def gather_prism_vs_baselines(prism_paths, baseline_paths):
    """Return dict atk -> dict detector -> TPR (pooled across seeds at matched FPR)."""
    # PRISM
    prism_pool = {}  # atk -> [TPR, TPR, ...]
    for p in prism_paths:
        d = _load(p)
        for atk, e in d.items():
            if atk.startswith('_') or not isinstance(e, dict):
                continue
            if 'TPR' in e:
                prism_pool.setdefault(atk, []).append(e['TPR'])
    # Baselines
    baseline_pool = {}  # (det, atk) -> [TPR]
    for p in baseline_paths:
        d = _load(p)
        for det, atks in d.items():
            if det.startswith('_') or not isinstance(atks, dict):
                continue
            for atk, e in atks.items():
                if isinstance(e, dict) and 'TPR' in e:
                    baseline_pool.setdefault((det, atk), []).append(e['TPR'])

    out = {}
    for atk, tprs in prism_pool.items():
        out.setdefault(atk, {})['PRISM'] = float(np.mean(tprs))
    for (det, atk), tprs in baseline_pool.items():
        out.setdefault(atk, {})[det] = float(np.mean(tprs))
    return out


def plot_prism_vs_baselines(data, dataset_label, out_path):
    attacks = [a for a in ATTACK_ORDER if a in data]
    detectors = ['PRISM', 'LID', 'Mahalanobis', 'ODIN', 'Energy']
    det_colors = {
        'PRISM': '#1f77b4',
        'LID': '#ff7f0e',
        'Mahalanobis': '#2ca02c',
        'ODIN': '#d62728',
        'Energy': '#9467bd',
    }
    n_det = len(detectors)
    bar_w = 0.8 / n_det
    x = np.arange(len(attacks))

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, det in enumerate(detectors):
        vals = [data[atk].get(det, 0.0) for atk in attacks]
        ax.bar(x + i * bar_w - 0.4 + bar_w / 2, vals, bar_w,
               label=det, color=det_colors[det], edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(attacks)
    ax.set_ylabel('TPR (matched FPR ≤ 10%)')
    ax.set_ylim(0, 1.05)
    ax.axhline(0.85, linestyle='--', color='gray', alpha=0.6, label='paper target (0.85)')
    ax.set_title(f'PRISM vs baselines — {dataset_label}')
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path + '.png', dpi=180)
    fig.savefig(out_path + '.pdf')
    plt.close(fig)
    print(f"  wrote {out_path}.png + .pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wrn-results', default='experiments/wrn')
    ap.add_argument('--c100-results', default='../Cifar 100/Cifar 100/project/experiments')
    ap.add_argument('--out-dir', default='paper/figures')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print('=== CIFAR-10 (WRN-28-10) ===')
    wrn_psq = sorted(glob(os.path.join(args.wrn_results, 'evaluation', 'results_prism_square_wrn_seed*.json')))
    wrn_fast = sorted(glob(os.path.join(args.wrn_results, 'evaluation', 'results_fast_wrn_seed*.json')))
    wrn_paper = sorted(glob(os.path.join(args.wrn_results, 'evaluation', 'results_paper_seed*_wrn.json')))
    wrn_baseline = sorted(glob(os.path.join(args.wrn_results, 'evaluation', 'results_baselines_wrn_seed*.json')))
    wrn_attack_paths = wrn_psq + wrn_fast + wrn_paper
    # score distributions
    q = _gather_quantiles(wrn_attack_paths)
    plot_score_distributions(q, 'CIFAR-10 (WRN-28-10)', os.path.join(args.out_dir, 'fig_score_dist_cifar10'))
    # tier operating points
    tiers = gather_tier_points(wrn_attack_paths)
    plot_tier_operating_points(tiers, 'CIFAR-10 (WRN-28-10)', os.path.join(args.out_dir, 'fig_tiers_cifar10'))
    # PRISM vs baselines
    cmp = gather_prism_vs_baselines(wrn_attack_paths, wrn_baseline)
    plot_prism_vs_baselines(cmp, 'CIFAR-10 (WRN-28-10)', os.path.join(args.out_dir, 'fig_baselines_cifar10'))

    print('\n=== CIFAR-100 (ResNet-18) ===')
    c100_fast = sorted(glob(os.path.join(args.c100_results, 'evaluation', 'results_cifar100_fast_n1000_ms5_seed*.json')))
    c100_cw = sorted(glob(os.path.join(args.c100_results, 'evaluation', 'results_cifar100_cw_n1000_ms5_seed*.json')))
    c100_baseline = sorted(glob(os.path.join(args.c100_results, 'evaluation', 'results_cifar100_baselines_seed*.json')))
    c100_attack_paths = c100_fast + c100_cw
    q = _gather_quantiles(c100_attack_paths)
    plot_score_distributions(q, 'CIFAR-100 (ResNet-18)', os.path.join(args.out_dir, 'fig_score_dist_cifar100'))
    tiers = gather_tier_points(c100_attack_paths)
    plot_tier_operating_points(tiers, 'CIFAR-100 (ResNet-18)', os.path.join(args.out_dir, 'fig_tiers_cifar100'))
    cmp = gather_prism_vs_baselines(c100_attack_paths, c100_baseline)
    plot_prism_vs_baselines(cmp, 'CIFAR-100 (ResNet-18)', os.path.join(args.out_dir, 'fig_baselines_cifar100'))

    print(f'\n[done] figures in {args.out_dir}')


if __name__ == '__main__':
    main()
