# PRISM Ablation Results (Paper-Quality)

n=500 per config, attacks: FGSM, PGD, eps=8/255


## FGSM (eps=0.0314=8/255)

| Configuration | TPR | 95% CI | FPR | F1 |
| :--- | ---: | :---: | ---: | ---: |
| Full PRISM | 0.9200 | [0.812, 0.969] | 0.0000 | 0.9583 |
| No L0 | 0.9200 | [0.812, 0.969] | 0.0000 | 0.9583 |
| No MoE | 0.9200 | [0.812, 0.969] | 0.0000 | 0.9583 |
| TDA only | 0.3400 | [0.224, 0.478] | 0.0600 | 0.4857 |

## PGD (eps=0.0314=8/255)

| Configuration | TPR | 95% CI | FPR | F1 |
| :--- | ---: | :---: | ---: | ---: |
| Full PRISM | 1.0000 | [0.929, 1.000] | 0.0000 | 1.0000 |
| No L0 | 1.0000 | [0.929, 1.000] | 0.0000 | 1.0000 |
| No MoE | 1.0000 | [0.929, 1.000] | 0.0000 | 1.0000 |
| TDA only | 1.0000 | [0.929, 1.000] | 0.0600 | 0.9709 |

## Mean TPR Across Attacks

| Configuration | Mean TPR | Mean FPR |
| :--- | ---: | ---: |
| Full PRISM | 0.9600 | 0.0000 |
| No L0 | 0.9600 | 0.0000 |
| No MoE | 0.9600 | 0.0000 |
| TDA only | 0.6700 | 0.0600 |
