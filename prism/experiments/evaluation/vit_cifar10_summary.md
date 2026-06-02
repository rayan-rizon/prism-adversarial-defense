# ViT-B/16 CIFAR-10 PRISM Summary

Source bundle: `ViT/project`.
Raw aggregate: `experiments/vit_cifar10/evaluation/results_vit_cifar10_multiseed.json`.

Scope: standard-attack transfer only. ViT CW, AutoAttack, adaptive PGD, and latency were not run in this bundle.

| Attack | TPR | 95% CI | FPR | n_adv | Base ASR | TPR on successful attacks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FGSM | 1.0000 | [0.9992, 1.0000] | 0.0828 | 5000 | 0.6062 | 1.0000 |
| PGD | 1.0000 | [0.9992, 1.0000] | 0.0828 | 5000 | 1.0000 | 1.0000 |
| Square | 0.9998 | [0.9989, 1.0000] | 0.0828 | 5000 | 0.9950 | 0.9998 |

Validation FPR gate: L1=0.089, L2=0.027, L3=0.004 on n=1000 validation clean samples; all pass configured targets 0.10/0.03/0.005.
Evaluation pooled FPR gate: L1=0.0828, L2=0.0248, L3=0.0010; all pass.
Backbone gate: ViT-B/16 ImageNet fine-tuned on CIFAR-10, recorded test_acc=0.9832, measured verify acc=0.9860 on n=1000, min gate=0.9000.
Paper-safe claim: PRISM transfers to ViT-B/16 on FGSM/PGD-50/Square with near-perfect TPR at FPR 0.0828. Do not claim ViT latency, CW, AutoAttack, or adaptive robustness from this run.
