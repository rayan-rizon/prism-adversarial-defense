# PRISM Ablation Results

This generated report is intentionally left as a placeholder. Regenerate it with
`python experiments/ablation/run_ablation_paper.py --multiseed --output experiments/ablation/results_ablation_multiseed.json`
after producing the current CIFAR-native artifacts.

The current ablation schema is:

| Configuration | Purpose |
| :--- | :--- |
| Full PRISM | Complete conformal ensemble detector with SACD/TAMSH response stack. |
| No MoE | Detection stack with TAMSH expert recovery disabled for response analysis. |
| Ensemble-no-TDA | Ensemble scorer calibrated without topological features; used for C1. |
| TDA only | Calibrated base topological Wasserstein score. |
