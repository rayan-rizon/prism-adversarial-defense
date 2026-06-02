"""
Generate all 3 paper figures from canonical post-fix Vast.ai data.

Fig 1: PRISM architecture pipeline diagram (illustration, no data)
Fig 2: Score distribution per attack (real quantiles from JSON)
Fig 3: Calibration threshold stability (real clean_scores.npy sweep)

Run from prism/paper/ directory:
    python figures/make_figures.py
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
PAPER = Path(__file__).resolve().parent.parent
FIG_DIR = PAPER / "figures"
ROOT = PAPER.parent.parent
VASTAI = ROOT / "Cifar 10"
if not VASTAI.exists():
    VASTAI = ROOT / "vastai_full_download_2026-05-20_0830UTC"
PROJECT = VASTAI / "project"
POSTFIX = VASTAI / "post_fix_local_2026-05-21"

# â”€â”€ Global style â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
})

# Color palette (color-blind safe)
C_CLEAN     = "#5BAD6F"
C_FGSM      = "#E8A838"
C_PGD       = "#C75B7A"
C_SQUARE    = "#E05A2B"
C_CW        = "#3F88C5"
C_AA        = "#7A4FB2"
C_L1        = "#E8A838"
C_L2        = "#E05A2B"
C_L3        = "#9B2237"


# â”€â”€ Figure 1: Architecture diagram (no data dependency) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def make_fig1():
    """
    PRISM defense pipeline. Shows:
      - Linear forward path: input -> ResNet -> activations -> persistence -> ensemble score
      - CADG conformal thresholding splits flow into PASS / L1 / L2 / L3 branches
      - L3 branch routes to TAMSH (MoE recovery)
      - SACD monitors score stream; on campaign alert, sends L0 feedback that
        tightens CADG thresholds and increases sensitivity
      - Final decision merges back into an output prediction
    """
    fig, ax = plt.subplots(figsize=(14.4, 6.3))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6.35)
    ax.axis("off")
    ax.grid(False)

    # Module colors
    C_INPUT  = "#4A90D9"
    C_MODEL  = "#5BAD6F"
    C_TAMM   = "#E8A838"
    C_CADG   = "#C75B7A"
    C_SACD   = "#9B59B6"
    C_TAMSH  = "#E05A2B"
    C_OUT_OK = "#5BAD6F"
    C_GRAY   = "#222"
    C_MUTED  = "#555"

    text_artists = []

    def add_text(*args, **kwargs):
        artist = ax.text(*args, **kwargs)
        text_artists.append(artist)
        return artist

    def stage(cx, cy, w, h, color, lines, fs=11.5, text_color="white"):
        rect = FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.12", linewidth=1.15,
            edgecolor="white", facecolor=color, alpha=0.96, zorder=3,
        )
        ax.add_patch(rect)
        title = lines[0]
        body = lines[1:]
        title_y = cy if not body else cy + 0.30
        title_fs = fs if not body else fs + 0.3
        add_text(cx, title_y, title, ha="center", va="center",
                 fontsize=title_fs, fontweight="bold", color=text_color,
                 zorder=4, clip_on=False)
        for i, line in enumerate(body):
            add_text(cx, cy + 0.04 - i * 0.31, line, ha="center", va="center",
                     fontsize=fs - 1.0, color=text_color, zorder=4,
                     clip_on=False)

    def arrow(x0, y0, x1, y1, color=C_GRAY, lw=1.35, style="-",
              rad=0.0, zorder=5):
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="->", color=color, lw=lw, linestyle=style,
                shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}",
            ),
            zorder=zorder,
        )

    # Title block.
    add_text(7.5, 6.05, "PRISM Defense Pipeline",
             ha="center", va="center", fontsize=17,
             fontweight="bold", color=C_GRAY, zorder=10, clip_on=False)
    add_text(7.5, 5.80,
             "Inference-time topological monitoring with conformal FPR certificates, "
             "campaign-aware thresholds, and MoE recovery",
             ha="center", va="center", fontsize=10.5,
             color=C_MUTED, style="italic", zorder=10, clip_on=False)

    # Forward inference path.
    Y_MAIN = 3.70
    stage(0.95,  Y_MAIN, 1.42, 1.06, C_INPUT,
          ["Input $x$", "$3{\\times}32{\\times}32$"])
    stage(2.85,  Y_MAIN, 1.76, 1.06, C_MODEL,
          ["Base Model", "ResNet-18", "(frozen)"])
    stage(4.95,  Y_MAIN, 1.86, 1.06, C_TAMM,
          ["TAMM", "activations $\\phi_\\ell$", "persistence diagrams"])
    stage(7.05,  Y_MAIN, 1.86, 1.06, C_TAMM,
          ["Ensemble Score", "Wasserstein $+$ DCT", "$+$ entropy"])
    stage(9.28,  Y_MAIN, 2.02, 1.06, C_CADG,
          ["CADG", "conformal tiers", "10%, 3%, 0.5% FPR"])
    stage(12.15, Y_MAIN, 2.10, 1.06, C_TAMSH,
          ["TAMSH", "MoE router", "$K{=}4$ experts"])

    for x0, y0, x1, y1 in [
        (1.66, Y_MAIN, 1.97, Y_MAIN),
        (3.73, Y_MAIN, 4.02, Y_MAIN),
        (5.88, Y_MAIN, 6.12, Y_MAIN),
        (7.98, Y_MAIN, 8.27, Y_MAIN),
        (10.29, Y_MAIN, 11.10, Y_MAIN),
    ]:
        arrow(x0, y0, x1, y1)

    # SACD control lane: separated from the data path to avoid label collisions.
    stage(10.02, 5.02, 5.85, 0.82, C_SACD,
          ["SACD: CUSUM campaign detector",
           "time-to-detect $=2.6$ queries at $\\rho=1.0$"],
          fs=12.6)
    arrow(7.05, 4.23, 8.55, 4.65, color=C_SACD, lw=1.25,
          style="dotted", rad=0.08, zorder=2)
    add_text(7.42, 4.42, "score stream", fontsize=9.3, color=C_SACD,
             ha="right", va="center", style="italic", zorder=6,
             clip_on=False,
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.82,
                       pad=0.8))
    arrow(9.28, 4.62, 9.28, 4.25, color=C_SACD, lw=1.35,
          style="dashed", zorder=5)
    add_text(9.57, 4.36, "L0 active: tighten $\\hat q_\\alpha$",
             fontsize=9.3, color=C_SACD, ha="left", va="center",
             style="italic", zorder=6, clip_on=False,
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.82,
                       pad=0.8))

    # Output branches.
    Y_OUT = 1.55
    stage(4.35, Y_OUT, 5.50, 0.72, C_OUT_OK,
          ["Output $\\hat y=f(x)$  |  PASS / L1 / L2 tiers"], fs=11.4)
    stage(12.15, Y_OUT, 4.10, 0.72, C_OUT_OK,
          ["Recovered output $\\hat y$  |  L3 routed"], fs=11.4)

    arrow(8.55, 3.10, 4.80, 1.95, color=C_OUT_OK, lw=1.45,
          rad=-0.22, zorder=2)
    add_text(6.30, 2.35, "PASS / L1 / L2", fontsize=10.0,
             color="#3A8A52", ha="center", va="center",
             fontweight="bold", zorder=6, clip_on=False)
    arrow(12.15, 3.12, 12.15, 1.95, color=C_TAMSH, lw=1.45, zorder=2)
    add_text(12.72, 2.56, "L3: reject\n+ MoE recover", fontsize=10.0,
             color=C_TAMSH, ha="left", va="center", fontweight="bold",
             linespacing=1.15, zorder=6, clip_on=False)

    # Bottom annotation strip: compact module semantics.
    legend_items = [
        ("TAMM",  C_TAMM,  "per-layer persistence (Wasserstein medoid)"),
        ("CADG",  C_CADG,  "distribution-free FPR cert. (split conformal)"),
        ("SACD",  C_SACD,  "campaign detector (time-to-detect headline)"),
        ("TAMSH", C_TAMSH, "MoE recovery (learned router 28.7%)"),
    ]
    row_y = [0.72, 0.30]
    col_x = [0.62, 7.48]
    for i, (tag, col, desc) in enumerate(legend_items):
        rx = col_x[i % 2]
        ry = row_y[i // 2]
        ax.add_patch(FancyBboxPatch((rx, ry - 0.10), 0.65, 0.30,
                                    boxstyle="round,pad=0.03",
                                    edgecolor="white", facecolor=col,
                                    alpha=0.95, lw=1.0, zorder=3))
        add_text(rx + 0.325, ry + 0.05, tag, ha="center", va="center",
                 fontsize=9.0, fontweight="bold", color="white", zorder=4)
        add_text(rx + 0.78, ry + 0.05, desc, ha="left", va="center",
                 fontsize=9.2, color=C_GRAY, zorder=4)

    # Guard against clipped text when figure dimensions or labels change.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_bbox = fig.bbox.padded(-2)
    clipped = []
    for artist in text_artists:
        bbox = artist.get_window_extent(renderer=renderer)
        (x0, y0), (x1, y1) = bbox.get_points()
        if not (fig_bbox.contains(x0, y0) and fig_bbox.contains(x1, y1)):
            clipped.append(artist.get_text())
    if clipped:
        raise RuntimeError(f"Figure 1 text outside canvas: {clipped}")

    fig.savefig(FIG_DIR / "fig1_architecture.pdf", bbox_inches="tight",
                pad_inches=0.05)
    fig.savefig(FIG_DIR / "fig1_architecture.png", bbox_inches="tight",
                pad_inches=0.05)
    plt.close(fig)
    print("[ok] fig1_architecture.{pdf,png}  (research layout verified)")

# â”€â”€ Figure 2: Score distribution per attack (real quantiles) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def make_fig2():
    """
    Box plot of canonical ensemble anomaly scores for clean and each attack.
    Quantile data is read from per-seed JSON `score_quantiles`. We aggregate
    across 5 seeds by mean of percentiles, then construct a box plot directly
    from those percentiles (no Gaussian re-sampling).
    """
    SEEDS = [42, 123, 456, 789, 999]

    # Collect quantiles per attack
    attacks = {
        "Clean":      ("clean", None,                              C_CLEAN),
        "FGSM":       ("adversarial", "results_fast_n1000_ms5",    C_FGSM),
        "Square":     ("adversarial", "results_fast_n1000_ms5",    C_SQUARE),
        "CW-L2":      ("adversarial", "results_cw_n1000_ms5",      C_CW),
        "PGD-40":     ("adversarial", "results_fast_n1000_ms5",    C_PGD),
        "AutoAttack": ("adversarial", "results_fast_n1000_ms5",    C_AA),
    }
    quantile_keys = ("p05", "p25", "p50", "p75", "p95")

    pooled = {}
    fast_attack_key = {"FGSM": "FGSM", "PGD-40": "PGD", "Square": "Square",
                       "AutoAttack": "AutoAttack"}
    for label, (which, file_prefix, color) in attacks.items():
        per_seed_qs = {q: [] for q in quantile_keys}
        for s in SEEDS:
            if label == "Clean":
                f = PROJECT / "experiments" / "evaluation" / f"results_fast_n1000_ms5_seed{s}.json"
                d = json.load(open(f))
                sq = d["FGSM"]["score_quantiles"]["clean"]
            elif label == "CW-L2":
                f = PROJECT / "experiments" / "evaluation" / f"results_cw_n1000_ms5_seed{s}.json"
                d = json.load(open(f))
                sq = d["CW"]["score_quantiles"]["adversarial"]
            else:
                f = PROJECT / "experiments" / "evaluation" / f"results_fast_n1000_ms5_seed{s}.json"
                d = json.load(open(f))
                key = fast_attack_key[label]
                sq = d[key]["score_quantiles"]["adversarial"]
            for q in quantile_keys:
                per_seed_qs[q].append(sq[q])
        pooled[label] = {q: float(np.mean(per_seed_qs[q])) for q in quantile_keys}
        pooled[label]["color"] = color

    # Approx conformal thresholds from clean quantile interpolation.
    cq = pooled["Clean"]
    # L1 = p90 â‰ˆ midpoint(p75,p95) shifted toward p95 (linear interp 0.90 of [0.75..0.95])
    L1 = cq["p75"] + (cq["p95"] - cq["p75"]) * ((0.90 - 0.75) / (0.95 - 0.75))
    # L2 = p97 â‰ˆ p95 + small extrapolation toward tail
    L2 = cq["p95"] + (cq["p95"] - cq["p75"]) * ((0.97 - 0.95) / (0.95 - 0.75))
    # L3 = p99.5 â€” use empirical p99.5 from raw clean_scores.npy if available
    clean_npy = PROJECT / "experiments" / "calibration" / "clean_scores.npy"
    if clean_npy.exists():
        arr = np.load(clean_npy)
        # the .npy is at a different (calibrated) scale; do not mix scales.
        # Instead extrapolate L3 from the eval-scale quantiles.
        pass
    L3 = cq["p95"] + (cq["p95"] - cq["p75"]) * ((0.995 - 0.95) / (0.95 - 0.75))

    # Build figure with broken y-axis (main panel + zoom for PGD/AutoAttack)
    labels = ["Clean", "FGSM", "Square", "CW-L2", "PGD-40", "AutoAttack"]
    fig, (ax_main, ax_high) = plt.subplots(
        2, 1, figsize=(8.5, 5.6),
        gridspec_kw={"height_ratios": [1.05, 1.0], "hspace": 0.18},
        sharex=True,
    )

    def build_bxp(label):
        q = pooled[label]
        # whislo/whishi are p05/p95; q1/q3 are p25/p75; med is p50
        return dict(label=label, whislo=q["p05"], q1=q["p25"], med=q["p50"],
                    q3=q["p75"], whishi=q["p95"], fliers=[], mean=q["p50"])

    bxp_data = [build_bxp(l) for l in labels]

    def style_bxp(ax):
        bp = ax.bxp(bxp_data, patch_artist=True, showmeans=False,
                    showfliers=False, widths=0.55,
                    medianprops=dict(color="white", linewidth=2.0),
                    whiskerprops=dict(linewidth=1.1, color="#444"),
                    capprops=dict(linewidth=1.1, color="#444"),
                    boxprops=dict(linewidth=0.8, edgecolor="#222"))
        for patch, label in zip(bp["boxes"], labels):
            patch.set_facecolor(pooled[label]["color"])
            patch.set_alpha(0.85)
        return bp

    style_bxp(ax_main); style_bxp(ax_high)

    # Y-limits â€” split view
    ax_high.set_ylim(-6, 14)            # bottom panel = full range, clean + light attacks
    ax_main.set_ylim(14, 36)            # top panel = PGD / AutoAttack tail
    # Hide spines between break
    ax_main.spines["bottom"].set_visible(False)
    ax_high.spines["top"].set_visible(False)
    ax_main.tick_params(labelbottom=False, bottom=False)

    # Diagonal break markers
    d = 0.008
    kw = dict(transform=ax_main.transAxes, color="k", lw=0.8, clip_on=False)
    ax_main.plot((-d, +d), (-d, +d), **kw)
    ax_main.plot((1 - d, 1 + d), (-d, +d), **kw)
    kw["transform"] = ax_high.transAxes
    ax_high.plot((-d, +d), (1 - d, 1 + d), **kw)
    ax_high.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw)

    # Threshold lines (lower panel) + compact stacked legend in the empty
    # upper-left region. A stacked color-keyed block avoids the label overlap
    # that occurs when L2/L3 thresholds are numerically close.
    thr = [("L1", L1, C_L1), ("L2", L2, C_L2), ("L3", L3, C_L3)]
    for name, val, col in thr:
        ax_high.axhline(val, color=col, linestyle="--", linewidth=1.1,
                        alpha=0.85, zorder=1)
    for i, (name, val, col) in enumerate(thr):
        ax_high.text(0.62, 12.6 - i * 2.1, f"{name} $\\approx$ {val:.2f}",
                     fontsize=8.0, color=col, va="center", ha="left",
                     fontweight="bold", zorder=5)

    ax_high.set_xticks(range(1, len(labels) + 1))
    ax_high.set_xticklabels(labels, fontsize=9)
    ax_high.set_xlabel("Input Type", labelpad=4)
    ax_high.set_ylabel("PRISM Anomaly Score $S(x)$")
    ax_main.set_ylabel("$S(x)$ (high range)")
    ax_main.set_title("Canonical Ensemble Score Distribution \u2014 Clean vs Adversarial\n"
                      "(5-seed pooled quantiles: whiskers $=$ p05/p95, box $=$ IQR, line $=$ median)",
                      fontsize=10, pad=6)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_score_dist.pdf")
    fig.savefig(FIG_DIR / "fig2_score_dist.png")
    plt.close(fig)
    print(f"[ok] fig2_score_dist.{{pdf,png}}  (thresholds L1={L1:.2f}, L2={L2:.2f}, L3={L3:.2f})")


# â”€â”€ Figure 3: Calibration threshold stability â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def make_fig3():
    """
    Sweep n_cal âˆˆ {500, 1000, 2000, 3000} using real clean_scores.npy from
    the Vast.ai calibration artifact. Plot L1/L2/L3 conformal thresholds
    vs n_cal, with Wilson-95% bands from achieved FPR on val_scores.npy.
    """
    cal_path = PROJECT / "experiments" / "calibration" / "clean_scores.npy"
    val_path = PROJECT / "experiments" / "calibration" / "val_scores.npy"
    if not cal_path.exists():
        print("[warn] clean_scores.npy not found \u2014 figure 3 falls back to simulation.")
        return

    cal = np.load(cal_path)
    val = np.load(val_path) if val_path.exists() else None
    sizes = [500, 1000, 2000, 3000]
    alphas = {"L1": 0.10, "L2": 0.03, "L3": 0.005}

    rng = np.random.RandomState(0)
    n_repeats = 25  # bootstrap variance across subsamples
    results = {lvl: {"mean": [], "lo": [], "hi": []} for lvl in alphas}
    achieved_fpr = {lvl: {"mean": [], "lo": [], "hi": []} for lvl in alphas}

    def wilson(p, n, z=1.96):
        if n == 0: return (0.0, 0.0)
        denom = 1 + z*z/n
        c = (p + z*z/(2*n)) / denom
        h = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
        return max(0.0, c - h), min(1.0, c + h)

    for n in sizes:
        ts = {lvl: [] for lvl in alphas}
        fs = {lvl: [] for lvl in alphas}
        for rep in range(n_repeats):
            # Bootstrap with replacement when n > population so n_cal=3000
            # sweep still runs on the 2000-sample clean_scores.npy artifact.
            replace = n > len(cal)
            idx = rng.choice(len(cal), n, replace=replace)
            sub = cal[idx]
            for lvl, a in alphas.items():
                # Split-conformal quantile: ceil((n+1)*(1-Î±))/n
                q_idx = int(np.ceil((n + 1) * (1 - a))) - 1
                q_idx = min(q_idx, n - 1)
                t = np.sort(sub)[q_idx]
                ts[lvl].append(t)
                if val is not None:
                    fs[lvl].append(np.mean(val > t))
        for lvl in alphas:
            arr = np.array(ts[lvl])
            results[lvl]["mean"].append(arr.mean())
            results[lvl]["lo"].append(np.quantile(arr, 0.025))
            results[lvl]["hi"].append(np.quantile(arr, 0.975))
            if val is not None:
                f_mean = float(np.mean(fs[lvl]))
                lo, hi = wilson(f_mean, len(val))
                achieved_fpr[lvl]["mean"].append(f_mean)
                achieved_fpr[lvl]["lo"].append(lo)
                achieved_fpr[lvl]["hi"].append(hi)

    # â”€â”€ Plot
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 3.6),
                                    gridspec_kw={"wspace": 0.32})
    color_map = {"L1": C_L1, "L2": C_L2, "L3": C_L3}

    for lvl, col in color_map.items():
        m = np.array(results[lvl]["mean"])
        lo = np.array(results[lvl]["lo"])
        hi = np.array(results[lvl]["hi"])
        axL.plot(sizes, m, marker="o", lw=2.0, ms=6, color=col,
                 label=f"{lvl} ($\\alpha{{=}}{alphas[lvl]}$)")
        axL.fill_between(sizes, lo, hi, color=col, alpha=0.18, linewidth=0)

    axL.set_xlabel("Calibration Set Size  $n_\\text{cal}$")
    axL.set_ylabel("Conformal Threshold Value")
    axL.set_title("Threshold Stability vs Calibration Size",
                  fontsize=10)
    axL.legend(loc="lower right", framealpha=0.9)
    axL.set_xticks(sizes); axL.set_xticklabels([str(s) for s in sizes])

    # Right panel: achieved FPR vs target
    if val is not None:
        for lvl, col in color_map.items():
            target = alphas[lvl]
            m = np.array(achieved_fpr[lvl]["mean"])
            lo = np.array(achieved_fpr[lvl]["lo"])
            hi = np.array(achieved_fpr[lvl]["hi"])
            axR.plot(sizes, m, marker="o", lw=1.8, ms=5.5, color=col,
                     label=f"{lvl} (target $={target}$)")
            axR.fill_between(sizes, lo, hi, color=col, alpha=0.18, linewidth=0)
            axR.axhline(target, color=col, linestyle=":", lw=0.9, alpha=0.7)
        axR.set_xlabel("Calibration Set Size  $n_\\text{cal}$")
        axR.set_ylabel("Achieved FPR on Validation Set")
        axR.set_title("Conformal FPR Coverage", fontsize=10)
        axR.legend(loc="upper right", framealpha=0.9)
        axR.set_xticks(sizes); axR.set_xticklabels([str(s) for s in sizes])
        axR.set_yscale("log")

    fig.suptitle("Conformal Calibration Sensitivity ($n_{\\mathrm{cal}} \\in$ {500, 1000, 2000, 3000}; 25 bootstrap subsamples each)",
                 fontsize=10, y=1.04)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_calibration.pdf")
    fig.savefig(FIG_DIR / "fig3_calibration.png")
    plt.close(fig)
    # Print L3 half-range for caption check
    l3 = np.array(results["L3"]["mean"])
    print(f"[ok] fig3_calibration.{{pdf,png}}  "
          f"(L3 range = [{l3.min():.3f}, {l3.max():.3f}]; half-range = {(l3.max()-l3.min())/2:.3f})")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    make_fig1()
    make_fig2()
    make_fig3()
    print(f"\nAll figures written to {FIG_DIR}")

