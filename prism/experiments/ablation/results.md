# PRISM Ablation Results

| Configuration | TPR | FPR | Campaign Gain | Recovery Rate |
| :--- | ---: | ---: | ---: | ---: |
| Full PRISM | 78.0% | 8.0% | +0.0% | 100.0% |
| No L0 | 78.0% | 8.0% | +0.0% | 100.0% |
| No MoE | 78.0% | 8.0% | +0.0% | 89.0% |
| TDA only | 78.0% | 8.0% | +0.0% | 89.0% |

_Campaign Gain = TPR[second half] − TPR[first half]. Full PRISM shows positive gain as L0 lowers thresholds after campaign detection._

_Recovery Rate = fraction of adversarial inputs with prediction ≠ None. Full PRISM and No-L0 retain predictions via MoE expert routing at L3; No-MoE and TDA-only reject at L3 (prediction = None)._
