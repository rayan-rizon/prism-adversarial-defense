"""
P0 stress test: long clean-stream L0 false-alarm verification.

Paper claim (experiments.tex line 422): "After the F1 CUSUM drift-floor fix,
L0 activation fraction on clean-only streams is 0.000 on all 5 seeds" — but
that was measured on 500-step streams. This script extends the test to
500 / 1000 / 2000 / 5000 / 10000-step clean-only streams across all 5
fixed seeds, using the deployed calibrated monitor.

Score model:
  - The L0 monitor consumes the post-fusion ensemble score S(x)
    = ell_ens(x) + beta * S_TDA(x).
  - The deployed calibrated pkl uses score_center=0.281, score_scale=1.639.
  - Empirical clean distribution (from per-seed JSON score_quantiles.clean
    in results_fast_n1000_ms5_seed42.json):  mean=0.286576, std=1.925517.
  - We use two complementary stream models:
      (a) Gaussian N(mean, std^2) — parametric
      (b) Bootstrap resampling from the 2000 TDA-side clean scores
          (calibration.clean_scores.npy) rescaled to ensemble-logit moments
          — robustness check against heavy tails
  Both should give 0.000 active-fraction if the paper claim holds.
"""
import os
import sys
import json
import pickle
import numpy as np

# Make `src` importable when run from prism/
HERE = os.path.dirname(os.path.abspath(__file__))
PRISM_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, PRISM_ROOT)

from src.sacd.monitor import CampaignMonitor


PKL_PATH = os.path.join(PRISM_ROOT, 'models', 'l0_thresholds.pkl')
CLEAN_NPY = os.path.join(
    PRISM_ROOT, 'experiments', 'calibration', 'clean_scores.npy'
)

# Empirical clean ensemble-logit moments from per-seed JSON
# (results_fast_n1000_ms5_seed42.json :: score_quantiles.clean).
EMP_MEAN = 0.286576
EMP_STD = 1.925517

SEEDS = [42, 123, 456, 789, 999]
STREAM_LENS = [500, 1000, 2000, 5000, 10000]


def make_monitor(pkl):
    return CampaignMonitor(
        hazard_rate=pkl['hazard_rate'],
        alert_run_prob=pkl['alert_run_prob'],
        warmup_steps=pkl['warmup_steps'],
        l0_factor=pkl['l0_factor'],
        detection_mode=pkl['detection_mode'],
        score_center=pkl['score_center'],
        score_scale=pkl['score_scale'],
        cusum_drift=pkl['cusum_drift'],
        cusum_threshold=pkl['cusum_threshold'],
        cusum_k_consecutive=pkl['cusum_k_consecutive'],
    )


def run_stream(monitor, scores):
    n_active = 0
    first_trigger = None
    for i, s in enumerate(scores):
        st = monitor.process_score(float(s))
        if st['l0_active']:
            n_active += 1
            if first_trigger is None:
                first_trigger = i
    return n_active, first_trigger


def main():
    pkl = pickle.load(open(PKL_PATH, 'rb'))
    print('Calibrated L0 thresholds (CIFAR-10 deployed):')
    for k in ('detection_mode', 'cusum_drift', 'cusum_threshold',
              'warmup_steps', 'cusum_k_consecutive',
              'score_center', 'score_scale'):
        print(f'  {k} = {pkl[k]}')
    print()

    # Empirical bootstrap pool (TDA-side clean scores rescaled to ensemble
    # moments — keeps tail shape, calibrates to L0 scale).
    cs = np.load(CLEAN_NPY).astype(np.float64)
    cs_resc = (cs - cs.mean()) / cs.std() * EMP_STD + EMP_MEAN

    out = {
        'pkl_path': PKL_PATH,
        'pkl_values': {
            k: pkl[k] for k in (
                'detection_mode', 'cusum_drift', 'cusum_threshold',
                'warmup_steps', 'cusum_k_consecutive',
                'score_center', 'score_scale', 'alert_run_prob',
                'alert_run_length', 'hazard_rate', 'l0_factor',
            )
        },
        'empirical_moments': {
            'mean': EMP_MEAN, 'std': EMP_STD,
            'source': 'results_fast_n1000_ms5_seed42.json::score_quantiles.clean',
        },
        'bootstrap_pool': {
            'source': 'experiments/calibration/clean_scores.npy',
            'n': int(cs.size),
            'orig_mean': float(cs.mean()),
            'orig_std': float(cs.std()),
            'rescaled_to_mean': EMP_MEAN,
            'rescaled_to_std': EMP_STD,
        },
        'gaussian': {}, 'bootstrap': {},
    }

    for stream_model in ('gaussian', 'bootstrap'):
        print(f'=== stream_model: {stream_model} ===')
        for L in STREAM_LENS:
            per_seed = []
            triggers = []
            for sd in SEEDS:
                rng = np.random.default_rng(sd)
                if stream_model == 'gaussian':
                    scores = rng.normal(EMP_MEAN, EMP_STD, size=L)
                else:
                    scores = rng.choice(cs_resc, size=L, replace=True)
                mon = make_monitor(pkl)
                n_active, first = run_stream(mon, scores)
                frac = n_active / L
                per_seed.append(frac)
                triggers.append(first)
            mean_frac = float(np.mean(per_seed))
            max_frac = float(np.max(per_seed))
            n_seeds_any_trigger = int(sum(t is not None for t in triggers))
            out[stream_model][L] = {
                'per_seed_fraction': per_seed,
                'mean_fraction': mean_frac,
                'max_fraction': max_frac,
                'n_seeds_with_any_trigger': n_seeds_any_trigger,
                'first_trigger_steps': triggers,
            }
            print(f'  len={L:>5d}  mean_active={mean_frac:.5f}  '
                  f'max_active={max_frac:.5f}  '
                  f'seeds_with_trigger={n_seeds_any_trigger}/{len(SEEDS)}  '
                  f'first_triggers={triggers}')
        print()

    out_path = os.path.join(
        HERE, 'results_l0_long_clean_stream.json'
    )
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'wrote {out_path}')

    # Paper-claim gate: all (stream_model, length, seed) cells must have
    # active fraction <= 0.001 (consistent with the 0.000 claim under
    # rounding to 3 dp).
    fails = []
    for sm in ('gaussian', 'bootstrap'):
        for L, blk in out[sm].items():
            if blk['max_fraction'] > 0.001:
                fails.append((sm, L, blk['per_seed_fraction']))
    if fails:
        print('FAIL: paper claim of 0.000 violated in:')
        for f in fails:
            print(' ', f)
        sys.exit(1)
    print('PASS: paper claim of 0.000 clean-only L0 active-fraction holds'
          ' for all stream models and lengths up to', max(STREAM_LENS))


if __name__ == '__main__':
    main()
