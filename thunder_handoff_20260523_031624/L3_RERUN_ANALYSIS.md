# L3 Fix Rerun Analysis

| Tier | Calibration FPR | Target | Status |
|---|---:|---:|---:|
| L1 | 0.0790 | 0.1000 | PASS |
| L2 | 0.0260 | 0.0300 | PASS |
| L3 | 0.0050 | 0.0050 | PASS |

| Seed | Attack | TPR | FPR/L1+ | L2+ | L3+ | Status |
|---:|---|---:|---:|---:|---:|---:|
| 123 | FGSM | 0.9810 | 0.0780 | 0.0280 | 0.0010 | PASS |
| 123 | PGD | 0.9960 | 0.0780 | 0.0280 | 0.0010 | PASS |
| 42 | FGSM | 0.9870 | 0.0680 | 0.0260 | 0.0030 | PASS |
| 42 | PGD | 0.9970 | 0.0680 | 0.0260 | 0.0030 | PASS |

| Attack | Completed Seeds | Mean TPR | Mean FPR | Max L3 | Pass Count |
|---|---:|---:|---:|---:|---:|
| FGSM | 2 | 0.9840 | 0.0730 | 0.0030 | 2/2 |
| PGD | 2 | 0.9965 | 0.0730 | 0.0030 | 2/2 |

Current remote progress at download time: seed `456` had started. Completed downloaded seeds: `42, 123`.

Downloaded files: `experiments/wrn/evaluation/results_fgsm_pgd_l3fix_wrn_seed42.json`, `results_fgsm_pgd_l3fix_wrn_seed123.json`, `experiments/wrn/calibration/ensemble_fpr_report.json`, `logs/eval_fgsm_pgd_l3fix_manual.out`, `logs/recalibrate_wrn_l3_rerun_now.out`.
