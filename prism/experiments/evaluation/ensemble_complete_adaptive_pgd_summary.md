# Ensemble-Complete Adaptive PGD Summary

Source: `Cifar 10/project/experiments/evaluation/ensemble_complete_adaptive_pgd/ensemble_complete_lambda_scan_n50` and `Cifar 10/project/experiments/evaluation/ensemble_complete_adaptive_pgd/ensemble_complete_worst_lambda_n200`.

## Calibration Gate
- L1: FPR=0.079, target=0.100, passed=True
- L2: FPR=0.026, target=0.030, passed=True
- L3: FPR=0.003, target=0.005, passed=True

## Lambda Scan (n=50 x 5 seeds, exploratory)
| lambda | n_adv | TPR | TPR on successful | undetected successful | FPR |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 250 | 0.568 | 0.910 | 0.056 | 0.100 |
| 0.5 | 250 | 0.560 | 0.903 | 0.060 | 0.100 |
| 1.0 | 250 | 0.508 | 0.847 | 0.092 | 0.100 |
| 2.0 | 250 | 0.520 | 0.873 | 0.076 | 0.100 |
| 5.0 | 250 | 0.492 | 0.848 | 0.088 | 0.100 |
| 10.0 | 250 | 0.440 | 0.821 | 0.096 | 0.100 |

## Worst-Lambda Confirmation (lambda=10, n=200 x 5 seeds)
Pooled: TPR=0.479, FPR=0.082, precision=0.854, F1=0.614, model ASR=0.552, TPR on successful attacks=0.866, undetected successful attack rate=0.074.

| seed | TPR | FPR | TPR on successful | undetected successful | tier L1/L2/L3 FPR |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.560 | 0.100 | 0.926 | 0.045 | 0.100/0.025/0.005 |
| 123 | 0.460 | 0.060 | 0.836 | 0.090 | 0.060/0.015/0.005 |
| 456 | 0.470 | 0.070 | 0.847 | 0.085 | 0.070/0.030/0.010 |
| 789 | 0.400 | 0.080 | 0.851 | 0.070 | 0.080/0.010/0.005 |
| 999 | 0.505 | 0.100 | 0.862 | 0.080 | 0.100/0.020/0.000 |

Note: seed 456 has L3 FPR=0.010 on 200 clean samples; pooled L3 FPR is 0.005, and the held-out calibration gate passes at L3 FPR=0.003.
