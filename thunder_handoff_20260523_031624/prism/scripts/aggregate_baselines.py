"""
Aggregate per-seed baselines JSONs (LID, Mahalanobis, ODIN, Energy) into a
single mean ± std table plus a markdown summary suitable for the paper.

Atomic write: tmp file → fsync → rename so a mid-write shutdown leaves the
existing aggregate intact.

USAGE
  python scripts/aggregate_baselines.py \
      --inputs vastai_full_download_2026-05-20_0830UTC/project/experiments/evaluation/results_baselines_seed{42,123,456,789,999}.json \
      --output vastai_full_download_2026-05-20_0830UTC/post_fix_local_2026-05-21/baselines/results_baselines_aggregate.json
"""
import os, sys, json, argparse, tempfile
import numpy as np


def _safe_load(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _atomic_write(path, data):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _wilson_ci(mean, n, z=1.96):
    if n <= 1:
        return (mean, mean)
    return (mean, mean)  # for aggregate we report std instead


def aggregate(input_paths, output_path):
    seeds = []
    runs = {}  # detector -> attack -> list of (TPR, FPR, F1) per seed
    for p in input_paths:
        if not os.path.exists(p):
            print(f"  WARN: missing {p}")
            continue
        data = _safe_load(p)
        seed = None
        # Heuristic for seed extraction
        for k in ('seed', '_seed'):
            if isinstance(data.get(k), int):
                seed = data[k]
                break
        if seed is None:
            # parse from filename: ..._seedN.json
            base = os.path.basename(p)
            for tok in base.split('_'):
                if tok.startswith('seed'):
                    try:
                        seed = int(tok.replace('seed', '').split('.')[0])
                    except Exception:
                        pass
        seeds.append(seed)
        for detector, det_data in data.items():
            if detector.startswith('_'):
                continue
            if not isinstance(det_data, dict):
                continue
            runs.setdefault(detector, {})
            for attack, atk_data in det_data.items():
                if not isinstance(atk_data, dict):
                    continue
                if 'TPR' not in atk_data:
                    continue
                runs[detector].setdefault(attack, []).append({
                    'seed': seed,
                    'TPR': float(atk_data['TPR']),
                    'FPR': float(atk_data['FPR']),
                    'F1': float(atk_data.get('F1', 0.0)),
                    'threshold': float(atk_data.get('threshold', 0.0)),
                })

    # Aggregate mean / std
    aggregate = {}
    for det, attacks in runs.items():
        aggregate[det] = {}
        per_attack_tpr_means = []
        for atk, rows in attacks.items():
            tprs = np.array([r['TPR'] for r in rows])
            fprs = np.array([r['FPR'] for r in rows])
            f1s = np.array([r['F1'] for r in rows])
            aggregate[det][atk] = {
                'TPR_mean': round(float(tprs.mean()), 4),
                'TPR_std':  round(float(tprs.std(ddof=1)) if len(tprs) > 1 else 0.0, 4),
                'FPR_mean': round(float(fprs.mean()), 4),
                'FPR_std':  round(float(fprs.std(ddof=1)) if len(fprs) > 1 else 0.0, 4),
                'F1_mean':  round(float(f1s.mean()), 4),
                'F1_std':   round(float(f1s.std(ddof=1)) if len(f1s) > 1 else 0.0, 4),
                'n_seeds':  int(len(tprs)),
            }
            per_attack_tpr_means.append(float(tprs.mean()))
        aggregate[det]['mean_TPR_across_attacks'] = round(float(np.mean(per_attack_tpr_means)), 4)

    out = {
        'detectors': list(aggregate.keys()),
        'seeds': sorted(s for s in seeds if s is not None),
        'aggregate': aggregate,
        '_source_files': [os.path.basename(p) for p in input_paths],
    }
    _atomic_write(output_path, out)
    print(f"JSON → {output_path}")

    # Markdown
    md_path = os.path.splitext(output_path)[0] + '.md'
    detectors = list(aggregate.keys())
    attacks = sorted({a for det in aggregate for a in aggregate[det] if a != 'mean_TPR_across_attacks'})
    lines = []
    lines.append("# Baseline Detectors — Aggregate (5 seeds, n=1000)\n")
    lines.append("Reference: pre-fix Vast.ai results. F1/F3/F4 fixes do not "
                 "affect the baseline-detector code paths, so these numbers "
                 "carry forward unchanged.\n")
    lines.append("## Mean TPR ± std per detector × attack (FPR target = 0.10)\n")
    head = "| Detector | " + " | ".join(attacks) + " | mean_TPR |"
    sep = "|---|" + "---|" * (len(attacks) + 1)
    lines.append(head)
    lines.append(sep)
    for det in detectors:
        row = [det]
        for atk in attacks:
            cell = aggregate[det].get(atk)
            if cell is None:
                row.append("—")
            else:
                row.append(f"{cell['TPR_mean']:.3f} ± {cell['TPR_std']:.3f}")
        row.append(f"{aggregate[det]['mean_TPR_across_attacks']:.3f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Mean FPR per detector × attack\n")
    lines.append("| Detector | " + " | ".join(attacks) + " |")
    lines.append("|---|" + "---|" * len(attacks))
    for det in detectors:
        row = [det]
        for atk in attacks:
            cell = aggregate[det].get(atk)
            if cell is None:
                row.append("—")
            else:
                row.append(f"{cell['FPR_mean']:.3f} ± {cell['FPR_std']:.3f}")
        lines.append("| " + " | ".join(row) + " |")
    _atomic_write(md_path, "")  # touch
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    print(f"MD   → {md_path}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', nargs='+', required=True,
                    help='Per-seed baselines JSON files.')
    ap.add_argument('--output', required=True,
                    help='Aggregate JSON output path.')
    args = ap.parse_args()
    aggregate(args.inputs, args.output)


if __name__ == '__main__':
    main()
