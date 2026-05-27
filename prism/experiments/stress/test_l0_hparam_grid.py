"""
P1 hyperparameter sweep: k_consec x t_warmup grid sensitivity for the
deployed CUSUM L0 monitor on synthetic adversarial + clean streams.

Appendix Table tab:hyper currently lists only point estimates. This grid
fills in the table: for each (k_consec, t_warmup) cell we measure:
  - clean_active_fraction (1000-step clean, 5 seeds)
  - sustained_rho1_time_to_detect (200-step sustained adversarial, 5 seeds)

Result: an appendix table that quantifies the LATENCY-vs-DURABILITY
tradeoff users see when tuning these knobs.

Adversarial-score model uses the empirical sustained_rho1 mean from
results_fast_n1000_ms5_seed42.json::score_quantiles.adversarial for PGD:
  PGD-40 adversarial: mean=11.903, std=5.216
Clean uses Gaussian N(0.287, 1.926^2).
"""
import os, sys, json, pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PRISM_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, PRISM_ROOT)
from src.sacd.monitor import CampaignMonitor

PKL_PATH = os.path.join(PRISM_ROOT, 'models', 'l0_thresholds.pkl')

CLEAN_MEAN, CLEAN_STD = 0.286576, 1.925517      # post-fusion clean
ADV_MEAN_PGD, ADV_STD_PGD = 11.903334, 5.215697  # PGD-40 adversarial

K_GRID = [1, 2, 3, 4, 5]
T_GRID = [10, 20, 35, 50, 75]
SEEDS = [42, 123, 456, 789, 999]
CLEAN_LEN = 1000
WARMUP_THEN_ADV_LEN = 200   # post-warmup adversarial segment


def make_monitor(pkl, k_consec, warmup):
    return CampaignMonitor(
        hazard_rate=pkl['hazard_rate'],
        alert_run_prob=pkl['alert_run_prob'],
        warmup_steps=warmup,
        l0_factor=pkl['l0_factor'],
        detection_mode=pkl['detection_mode'],
        score_center=pkl['score_center'],
        score_scale=pkl['score_scale'],
        cusum_drift=pkl['cusum_drift'],
        cusum_threshold=pkl['cusum_threshold'],
        cusum_k_consecutive=k_consec,
    )


def clean_only_frac(pkl, k_consec, warmup, length, seed):
    rng = np.random.default_rng(seed)
    scores = rng.normal(CLEAN_MEAN, CLEAN_STD, size=length)
    mon = make_monitor(pkl, k_consec, warmup)
    n_active = 0
    for s in scores:
        st = mon.process_score(float(s))
        n_active += int(st['l0_active'])
    return n_active / length


def time_to_detect(pkl, k_consec, warmup, seed):
    """Stream: `warmup` clean warmup steps, then 200 sustained PGD adv steps.
    Return queries from first adv step to first L0 trigger; np.inf if none."""
    rng = np.random.default_rng(seed)
    clean_pre = rng.normal(CLEAN_MEAN, CLEAN_STD, size=warmup)
    adv = rng.normal(ADV_MEAN_PGD, ADV_STD_PGD, size=WARMUP_THEN_ADV_LEN)
    stream = np.concatenate([clean_pre, adv])
    first_adv_idx = warmup
    mon = make_monitor(pkl, k_consec, warmup)
    for i, s in enumerate(stream):
        st = mon.process_score(float(s))
        if st['l0_active']:
            return max(0, i - first_adv_idx)
    return float('inf')


def main():
    pkl = pickle.load(open(PKL_PATH, 'rb'))
    print('Calibrated CUSUM:',
          {k: pkl[k] for k in ('cusum_drift','cusum_threshold','score_center','score_scale')})
    print()

    results = []
    print('Cell:  k_consec x t_warmup  ->  '
          'clean_frac (mean/max over 5 seeds)  |  t2d (median/max over 5 seeds)')
    for k in K_GRID:
        for t in T_GRID:
            cfracs = [clean_only_frac(pkl, k, t, CLEAN_LEN, sd) for sd in SEEDS]
            t2ds = [time_to_detect(pkl, k, t, sd) for sd in SEEDS]
            t2d_finite = [x for x in t2ds if x != float('inf')]
            row = {
                'k_consec': k, 't_warmup': t,
                'clean_frac_mean': float(np.mean(cfracs)),
                'clean_frac_max': float(np.max(cfracs)),
                'clean_frac_per_seed': cfracs,
                't2d_per_seed': [str(x) for x in t2ds],
                't2d_median': float(np.median(t2d_finite)) if t2d_finite else None,
                't2d_max': float(np.max(t2d_finite)) if t2d_finite else None,
                'n_detect': len(t2d_finite),
            }
            results.append(row)
            print(f'  k={k}  t={t:>2d}  ->  '
                  f'clean={row["clean_frac_mean"]:.4f}/{row["clean_frac_max"]:.4f}  |  '
                  f't2d={row["t2d_median"]}/{row["t2d_max"]}  (detect={row["n_detect"]}/5)')

    out_path = os.path.join(HERE, 'results_l0_hparam_grid.json')
    with open(out_path, 'w') as f:
        json.dump({
            'k_grid': K_GRID, 't_grid': T_GRID, 'seeds': SEEDS,
            'clean_len': CLEAN_LEN, 'adv_len': WARMUP_THEN_ADV_LEN,
            'clean_dist': {'mean': CLEAN_MEAN, 'std': CLEAN_STD},
            'adv_dist_pgd': {'mean': ADV_MEAN_PGD, 'std': ADV_STD_PGD},
            'pkl': {k: pkl[k] for k in ('cusum_drift','cusum_threshold',
                                         'score_center','score_scale',
                                         'alert_run_prob','hazard_rate','l0_factor')},
            'cells': results,
        }, f, indent=2, default=str)
    print('\nwrote', out_path)


if __name__ == '__main__':
    main()
