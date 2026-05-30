# Baseline Detectors — Aggregate (5 seeds, n=1000)

Reference: pre-fix Vast.ai results. F1/F3/F4 fixes do not affect the baseline-detector code paths, so these numbers carry forward unchanged.

## Mean TPR ± std per detector × attack (FPR target = 0.10)

| Detector | FGSM | PGD | Square | mean_TPR |
|---|---|---|---|---|
| SID | 0.150 ± 0.009 | 0.231 ± 0.009 | 0.082 ± 0.005 | 0.154 |

## Mean FPR per detector × attack

| Detector | FGSM | PGD | Square |
|---|---|---|---|
| SID | 0.109 ± 0.005 | 0.109 ± 0.005 | 0.109 ± 0.005 |
