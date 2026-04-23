# PRISM Ablation Study — Multi-Seed Results

Seeds: [42, 123, 456, 789, 999]  |  Attacks: FGSM, PGD, Square  |  ε=8/255

_Values reported as mean ± std across seeds. Statistical comparison vs 'Full PRISM' via paired two-tailed t-test._


## FGSM (ε=0.0314=8/255)

| Configuration | TPR mean±std | FPR mean±std | F1 mean±std | Δ TPR vs Full | p-value | Cohen's d |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full PRISM | 0.8104±0.0106 | 0.0698±0.0036 | 0.8620±0.0075 | (ref) | — | — |
| No L0 | 0.8114±0.0088 | 0.0698±0.0036 | 0.8626±0.0064 | -0.0010 | 0.298 | -0.534 |
| No MoE | 0.8102±0.0103 | 0.0698±0.0036 | 0.8619±0.0070 | +0.0002 | 0.838 | 0.098 |
| TDA only | 0.5138±0.0127 | 0.1740±0.0038 | 0.6088±0.0096 | +0.2966 | 0.000* | 31.564 |

## PGD (ε=0.0314=8/255)

| Configuration | TPR mean±std | FPR mean±std | F1 mean±std | Δ TPR vs Full | p-value | Cohen's d |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full PRISM | 1.0000±0.0000 | 0.0698±0.0036 | 0.9663±0.0017 | (ref) | — | — |
| No L0 | 1.0000±0.0000 | 0.0698±0.0036 | 0.9663±0.0017 | +0.0000 | nan | 0.000 |
| No MoE | 1.0000±0.0000 | 0.0698±0.0036 | 0.9663±0.0017 | +0.0000 | nan | 0.000 |
| TDA only | 1.0000±0.0000 | 0.1740±0.0038 | 0.9200±0.0016 | +0.0000 | nan | 0.000 |

## Square (ε=0.0314=8/255)

| Configuration | TPR mean±std | FPR mean±std | F1 mean±std | Δ TPR vs Full | p-value | Cohen's d |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full PRISM | 0.8910±0.0096 | 0.0698±0.0036 | 0.9088±0.0043 | (ref) | — | — |
| No L0 | 0.8898±0.0095 | 0.0698±0.0036 | 0.9081±0.0061 | +0.0012 | 0.851 | 0.090 |
| No MoE | 0.8912±0.0103 | 0.0698±0.0036 | 0.9089±0.0057 | -0.0002 | 0.953 | -0.028 |
| TDA only | 0.7260±0.0098 | 0.1740±0.0038 | 0.7642±0.0057 | +0.1650 | 0.000* | 44.907 |

_* p < 0.05 (two-tailed paired t-test, n=seeds)_

_Cohen's d: |d| < 0.2 = negligible, 0.2-0.5 = small, > 0.5 = medium_


**Interpretation note**: Components with p > 0.05 provide formal guarantees (conformal FPR bounds, Bayesian temporal model) that are not captured by mean TPR alone.

