"""
Follow-up to test_l0_long_clean_stream.py: sweep cusum_threshold over the
feasible grid cells (5.0, 8.0, 10.0, 12.0) reported in the calibration
pkl's `top_rows`, to confirm whether a higher threshold would have given
a more durable 0.000 result on long streams.

Output: per-(threshold, stream_length, stream_model) active fractions.
"""
import os, sys, json, pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PRISM_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, PRISM_ROOT)
from src.sacd.monitor import CampaignMonitor

PKL_PATH = os.path.join(PRISM_ROOT, 'models', 'l0_thresholds.pkl')
CLEAN_NPY = os.path.join(PRISM_ROOT, 'experiments', 'calibration', 'clean_scores.npy')

EMP_MEAN = 0.286576
EMP_STD = 1.925517
SEEDS = [42, 123, 456, 789, 999]
STREAM_LENS = [500, 1000, 2000, 5000]
THRESHOLDS = [5.0, 8.0, 10.0, 12.0]  # feasible cells from pkl.top_rows
DRIFTS = [1.0]  # deployed only


def make_monitor(pkl, cusum_threshold, cusum_drift):
    return CampaignMonitor(
        hazard_rate=pkl['hazard_rate'],
        alert_run_prob=pkl['alert_run_prob'],
        warmup_steps=pkl['warmup_steps'],
        l0_factor=pkl['l0_factor'],
        detection_mode=pkl['detection_mode'],
        score_center=pkl['score_center'],
        score_scale=pkl['score_scale'],
        cusum_drift=cusum_drift,
        cusum_threshold=cusum_threshold,
        cusum_k_consecutive=pkl['cusum_k_consecutive'],
    )


def run_stream(monitor, scores):
    n_active = 0
    first = None
    for i, s in enumerate(scores):
        st = monitor.process_score(float(s))
        if st['l0_active']:
            n_active += 1
            if first is None:
                first = i
    return n_active, first


def main():
    pkl = pickle.load(open(PKL_PATH, 'rb'))
    cs = np.load(CLEAN_NPY).astype(np.float64)
    cs_resc = (cs - cs.mean()) / cs.std() * EMP_STD + EMP_MEAN

    out = {'thresholds': THRESHOLDS, 'drifts': DRIFTS, 'lengths': STREAM_LENS,
           'seeds': SEEDS, 'results': []}

    for thr in THRESHOLDS:
        for drift in DRIFTS:
            for sm in ('gaussian', 'bootstrap'):
                for L in STREAM_LENS:
                    per_seed = []
                    trigs = []
                    for sd in SEEDS:
                        rng = np.random.default_rng(sd)
                        if sm == 'gaussian':
                            scores = rng.normal(EMP_MEAN, EMP_STD, size=L)
                        else:
                            scores = rng.choice(cs_resc, size=L, replace=True)
                        mon = make_monitor(pkl, thr, drift)
                        n_active, first = run_stream(mon, scores)
                        per_seed.append(n_active / L)
                        trigs.append(first)
                    mean_frac = float(np.mean(per_seed))
                    max_frac = float(np.max(per_seed))
                    n_trig = sum(t is not None for t in trigs)
                    out['results'].append({
                        'threshold': thr, 'drift': drift, 'stream_model': sm,
                        'length': L, 'per_seed_fraction': per_seed,
                        'mean_fraction': mean_frac, 'max_fraction': max_frac,
                        'seeds_with_trigger': n_trig, 'first_triggers': trigs,
                    })
                    print(f'  thr={thr:>4.1f}  drift={drift:>3.1f}  sm={sm:>9s}  '
                          f'L={L:>5d}  mean={mean_frac:.5f}  max={max_frac:.5f}  '
                          f'n_trig={n_trig}/5')
                print()

    out_path = os.path.join(HERE, 'results_l0_threshold_sweep.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print('wrote', out_path)


if __name__ == '__main__':
    main()
