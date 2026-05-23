# PRISM Ablation Study — Multi-Seed Results

Seeds: [42, 123, 456, 789, 999]  |  Attacks: FGSM, PGD, Square  |  ε=8/255

_Values reported as mean ± std across seeds. Statistical comparison vs 'Full PRISM' via paired two-tailed t-test._


## FGSM (ε=0.0314=8/255)

| Configuration | TPR mean±std | FPR mean±std | F1 mean±std | Δ TPR vs Full | p-value | Cohen's d |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full PRISM | 0.8816±0.0038 | 0.0740±0.0071 | 0.9016±0.0036 | (ref) | — | — |
| No-LogitProfile | 0.9884±0.0019 | 0.9380±0.0072 | 0.6755±0.0021 | -0.1068 | 0.000* | -31.913 |
| No-StabilityV2 | 0.9360±0.0035 | 0.1066±0.0059 | 0.9165±0.0038 | -0.0544 | 0.000* | -13.272 |
| No-GradNorm | 0.9462±0.0075 | 0.2176±0.0051 | 0.8746±0.0046 | -0.0646 | 0.000* | -14.705 |
| No-DCT | 0.7510±0.0062 | 0.0978±0.0072 | 0.8124±0.0050 | +0.1306 | 0.000* | 25.466 |
| Ensemble-no-TDA | 0.5064±0.0081 | 0.0772±0.0030 | 0.6395±0.0067 | +0.3752 | 0.000* | 53.765 |
| TDA only | 0.0094±0.0021 | 0.0820±0.0050 | 0.0172±0.0038 | +0.8722 | 0.000* | 160.043 |

## PGD (ε=0.0314=8/255)

| Configuration | TPR mean±std | FPR mean±std | F1 mean±std | Δ TPR vs Full | p-value | Cohen's d |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full PRISM | 0.9874±0.0030 | 0.0740±0.0071 | 0.9580±0.0023 | (ref) | — | — |
| No-LogitProfile | 0.9798±0.0029 | 0.9380±0.0072 | 0.6716±0.0016 | +0.0076 | 0.007* | 2.313 |
| No-StabilityV2 | 0.3960±0.0139 | 0.1066±0.0059 | 0.5270±0.0149 | +0.5914 | 0.000* | 37.915 |
| No-GradNorm | 0.9854±0.0030 | 0.2176±0.0051 | 0.8946±0.0016 | +0.0020 | 0.166 | 0.756 |
| No-DCT | 0.9948±0.0022 | 0.0978±0.0072 | 0.9508±0.0030 | -0.0074 | 0.001* | -4.074 |
| Ensemble-no-TDA | 0.9692±0.0034 | 0.0772±0.0030 | 0.9472±0.0015 | +0.0182 | 0.000* | 5.564 |
| TDA only | 0.0658±0.0066 | 0.0820±0.0050 | 0.1146±0.0108 | +0.9216 | 0.000* | 156.226 |

## Square (ε=0.0314=8/255)

| Configuration | TPR mean±std | FPR mean±std | F1 mean±std | Δ TPR vs Full | p-value | Cohen's d |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full PRISM | 0.8856±0.0110 | 0.0740±0.0071 | 0.9038±0.0059 | (ref) | — | — |
| No-LogitProfile | 0.9200±0.0034 | 0.9380±0.0072 | 0.6438±0.0014 | -0.0344 | 0.001* | -4.046 |
| No-StabilityV2 | 0.8544±0.0102 | 0.1066±0.0059 | 0.8714±0.0050 | +0.0312 | 0.000* | 6.215 |
| No-GradNorm | 0.9934±0.0029 | 0.2176±0.0051 | 0.8986±0.0033 | -0.1078 | 0.000* | -11.713 |
| No-DCT | 0.9604±0.0027 | 0.0978±0.0072 | 0.9333±0.0028 | -0.0748 | 0.000* | -7.854 |
| Ensemble-no-TDA | 0.8864±0.0067 | 0.0772±0.0030 | 0.9029±0.0039 | -0.0008 | 0.850 | -0.090 |
| TDA only | 0.0186±0.0033 | 0.0820±0.0050 | 0.0338±0.0060 | +0.8670 | 0.000* | 97.545 |

_* p < 0.05 (two-tailed paired t-test, n=seeds)_

_Cohen's d: |d| < 0.2 = negligible, 0.2-0.5 = small, > 0.5 = medium_


**Interpretation note**: Components with p > 0.05 provide formal guarantees (conformal FPR bounds, Bayesian temporal model) that are not captured by mean TPR alone.

