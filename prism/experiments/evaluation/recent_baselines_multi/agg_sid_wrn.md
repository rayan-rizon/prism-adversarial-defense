# Baseline Detectors — Aggregate (5 seeds, n=1000)

Reference: pre-fix Vast.ai results. F1/F3/F4 fixes do not affect the baseline-detector code paths, so these numbers carry forward unchanged.

## Mean TPR ± std per detector × attack (FPR target = 0.10)

| Detector | FGSM | PGD | Square | mean_TPR |
|---|---|---|---|---|
| SID | 0.330 ± 0.013 | 0.621 ± 0.011 | 0.238 ± 0.005 | 0.396 |

## Mean FPR per detector × attack

| Detector | FGSM | PGD | Square |
|---|---|---|---|
| SID | 0.094 ± 0.005 | 0.094 ± 0.005 | 0.094 ± 0.005 |
