# PRISM Ablation Results (Paper-Quality)

n=500 per config, attacks: FGSM, eps=8/255


## FGSM (eps=0.0314=8/255)

| Configuration | TPR | 95% CI | FPR | F1 |
| :--- | ---: | :---: | ---: | ---: |
| Full PRISM | 0.9000 | [0.786, 0.957] | 0.1200 | 0.8911 |
| No-LogitProfile | 1.0000 | [0.929, 1.000] | 0.9400 | 0.6803 |
| No-StabilityV2 | 0.9400 | [0.838, 0.979] | 0.1200 | 0.9126 |
| No-GradNorm | 0.9800 | [0.895, 0.997] | 0.2600 | 0.8750 |
| No-DCT | 0.7800 | [0.648, 0.873] | 0.1400 | 0.8125 |
| Ensemble-no-TDA | 0.4400 | [0.312, 0.577] | 0.0800 | 0.5789 |
| TDA only | 0.0000 | [0.000, 0.071] | 0.1000 | 0.0000 |

## Mean TPR Across Attacks

| Configuration | Mean TPR | Mean FPR |
| :--- | ---: | ---: |
| Full PRISM | 0.9000 | 0.1200 |
| No-LogitProfile | 1.0000 | 0.9400 |
| No-StabilityV2 | 0.9400 | 0.1200 |
| No-GradNorm | 0.9800 | 0.2600 |
| No-DCT | 0.7800 | 0.1400 |
| Ensemble-no-TDA | 0.4400 | 0.0800 |
| TDA only | 0.0000 | 0.1000 |
