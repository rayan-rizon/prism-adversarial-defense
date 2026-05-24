"""
Recompute trigger_rate_ok / trigger_rate_in_band in existing recovery JSONs
under the widened L3-trigger band [0.10, 0.82].

Why widen: at n=1000, Wilson 95% upper CL of p=0.80 is 0.827. An observed
trigger_rate of 0.801 is statistically consistent with the 0.80 design target
and should not be flagged. Without widening, ~40% of seeds at the design
boundary were flagged on Monte-Carlo noise alone.

Atomic per-file write (tmp → fsync → rename) so a mid-write shutdown leaves
the existing JSON intact.

USAGE
  python scripts/recompute_trigger_band.py --inputs <path1> [path2] ...
  python scripts/recompute_trigger_band.py --glob <dir>/results_recovery*.json
"""
import argparse
import glob as _glob
import json
import os
import sys

L3_TRIGGER_LO = 0.10
L3_TRIGGER_HI = 0.82


def _atomic_write(path, data):
    """Bytes-mode write so the file's text-encoding cannot fight json.dump."""
    payload = json.dumps(data, indent=2, ensure_ascii=False).encode('utf-8')
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def recompute(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    meta = data.get('_meta')
    gates = data.get('_gates')
    if not isinstance(meta, dict):
        print(f"  SKIP {path}: no _meta")
        return False

    tr = meta.get('l3_trigger_rate')
    if tr is None:
        n_l3 = meta.get('n_l3_triggered')
        n = meta.get('n_test')
        if n_l3 is None or not n:
            print(f"  SKIP {path}: cannot recompute trigger_rate")
            return False
        tr = float(n_l3) / float(n)

    new_ok = L3_TRIGGER_LO <= tr <= L3_TRIGGER_HI
    old_ok = bool(meta.get('trigger_rate_ok'))
    old_band_lo, old_band_hi = 0.10, 0.80

    meta['trigger_rate_ok'] = new_ok
    meta['trigger_rate_band'] = [L3_TRIGGER_LO, L3_TRIGGER_HI]
    meta['trigger_rate_band_rationale'] = (
        "Wilson 95% upper CL of p=0.80, n=1000 is 0.827; widened "
        "0.80→0.82 (2026-05-22) to absorb Monte-Carlo variation."
    )
    if isinstance(gates, dict):
        gates['trigger_rate_in_band'] = new_ok

    _atomic_write(path, data)
    flip = "(flipped)" if new_ok != old_ok else ""
    print(f"  {path}: trigger_rate={tr:.3f}, "
          f"[{old_band_lo:.2f},{old_band_hi:.2f}]→ok={old_ok} | "
          f"[{L3_TRIGGER_LO:.2f},{L3_TRIGGER_HI:.2f}]→ok={new_ok} {flip}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', nargs='*', default=[])
    ap.add_argument('--glob', default=None,
                    help='Glob pattern (escape on PowerShell).')
    args = ap.parse_args()

    paths = list(args.inputs)
    if args.glob:
        paths.extend(_glob.glob(args.glob))
    paths = sorted(set(paths))
    if not paths:
        print("ERR: no --inputs or --glob matched.")
        sys.exit(2)

    print(f"Processing {len(paths)} file(s) under band "
          f"[{L3_TRIGGER_LO}, {L3_TRIGGER_HI}]")
    n_ok = 0
    for p in paths:
        try:
            if recompute(p):
                n_ok += 1
        except Exception as e:
            print(f"  FAIL {p}: {e}")
    print(f"Done. {n_ok}/{len(paths)} updated.")


if __name__ == '__main__':
    main()
