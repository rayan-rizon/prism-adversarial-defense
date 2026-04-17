# PRISM Ablation Results (Paper-Quality)

n=500 per config, attacks: FGSM, ε=8/255


## FGSM (ε=0.0314=8/255)

| Configuration | TPR | 95% CI | FPR | F1 |
| :--- | ---: | :---: | ---: | ---: |
| Full PRISM | 0.3200 | [0.208, 0.458] | 0.1000 | 0.4507 |
| No L0 | 0.3200 | [0.208, 0.458] | 0.1000 | 0.4507 |
| No MoE | 0.3200 | [0.208, 0.458] | 0.1000 | 0.4507 |
| TDA only | 1.0000 | [0.929, 1.000] | 1.0000 | 0.6667 |

## Mean TPR Across Attacks

| Configuration | Mean TPR | Mean FPR |
| :--- | ---: | ---: |
| Full PRISM | 0.3200 | 0.1000 |
| No L0 | 0.3200 | 0.1000 |
| No MoE | 0.3200 | 0.1000 |
| TDA only | 1.0000 | 1.0000 |
