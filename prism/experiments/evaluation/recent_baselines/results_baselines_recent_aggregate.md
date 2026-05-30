# Baseline Detectors — Aggregate (5 seeds, n=1000)

Reference: pre-fix Vast.ai results. F1/F3/F4 fixes do not affect the baseline-detector code paths, so these numbers carry forward unchanged.

## Mean TPR ± std per detector × attack (FPR target = 0.10)

| Detector | CW | FGSM | PGD | Square | mean_TPR |
|---|---|---|---|---|---|
| SID | 0.601 ± 0.006 | 0.296 ± 0.008 | 0.519 ± 0.015 | 0.286 ± 0.018 | 0.426 |
| SpectralDefense | 0.246 ± 0.011 | 1.000 ± 0.000 | 0.999 ± 0.000 | 0.206 ± 0.020 | 0.613 |

## Mean FPR per detector × attack

| Detector | CW | FGSM | PGD | Square |
|---|---|---|---|---|
| SID | 0.082 ± 0.004 | 0.082 ± 0.004 | 0.082 ± 0.004 | 0.082 ± 0.004 |
| SpectralDefense | 0.096 ± 0.003 | 0.105 ± 0.005 | 0.101 ± 0.005 | 0.096 ± 0.008 |
