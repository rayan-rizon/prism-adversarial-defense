# Baseline Detectors — Aggregate (5 seeds, n=1000)

Reference: pre-fix Vast.ai results. F1/F3/F4 fixes do not affect the baseline-detector code paths, so these numbers carry forward unchanged.

## Mean TPR ± std per detector × attack (FPR target = 0.10)

| Detector | CW | FGSM | PGD | Square | mean_TPR |
|---|---|---|---|---|---|
| SpectralDefense | 0.244 ± 0.013 | 1.000 ± 0.000 | 0.999 ± 0.000 | 0.208 ± 0.018 | 0.613 |
| SpectralDefense-LayerMFS | 0.458 ± 0.015 | 0.861 ± 0.007 | 0.899 ± 0.022 | 0.711 ± 0.029 | 0.732 |

## Mean FPR per detector × attack

| Detector | CW | FGSM | PGD | Square |
|---|---|---|---|---|
| SpectralDefense | 0.108 ± 0.006 | 0.105 ± 0.005 | 0.103 ± 0.009 | 0.096 ± 0.011 |
| SpectralDefense-LayerMFS | 0.118 ± 0.009 | 0.113 ± 0.009 | 0.114 ± 0.008 | 0.110 ± 0.015 |
