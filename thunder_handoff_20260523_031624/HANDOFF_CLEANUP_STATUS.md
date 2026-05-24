# Handoff Cleanup Status

Folder: `/Users/rayanrizon/Desktop/Research/Prism/thunder_handoff_20260523_031624`

## Kept Runtime Essentials

| Item | Status |
|---|---|
| WRN checkpoint `models/cifar_wrn28_10.pt` | present |
| WRN scorer/calibrator files `models/wrn/*.pkl` | present |
| WRN config `configs/wrn_cifar10.yaml` | present, L3 factor patched to 0.47 |
| CIFAR-10 data archive and extracted batches | present |
| Source code `src/` | present |
| Required scripts `scripts/` | present |
| Adaptive PGD script | present |
| Ablation script | present |
| Train experts script | present |
| Result JSONs and logs | present |
| `models/wrn/experts.pkl` | missing; must be trained before ablation |

## Downloaded Result Coverage

| Test | Coverage | Target Status |
|---|---|---|
| CW-L2 | 5/5 seeds plus aggregate | pass |
| FGSM L3-fix | 5/5 seeds plus aggregate | pass |
| PGD L3-fix | 5/5 seeds plus aggregate | pass |
| Square | seed 42 only from interrupted old fast run | partial |
| AutoAttack | seed 42 only from interrupted old fast run | partial |
| Adaptive PGD | partial logs only, no final JSON | remaining |
| Ablation | failed earlier due missing experts file | remaining |

## Final Passing Metrics

| Test | TPR | FPR | L2+ | L3+ | Status |
|---|---:|---:|---:|---:|---|
| CW-L2 pooled | 0.9328 | 0.0790 | 0.0218 | 0.0030 | pass |
| FGSM L3-fix pooled | 0.9854 | 0.0714 | 0.0244 | 0.0016 | pass |
| PGD L3-fix pooled | 0.9948 | 0.0714 | 0.0244 | 0.0016 | pass |

## Remaining Commands

```bash
cd /path/to/thunder_handoff_20260523_031624/prism
export PRISM_CONFIG=configs/wrn_cifar10.yaml

python experiments/evaluation/run_adaptive_pgd.py \
  --n-test 1000 \
  --multi-seed \
  --seeds 42 123 456 789 999 \
  2>&1 | tee logs/adaptive_pgd_new_gpu.log

python scripts/train_experts.py --device cuda --output models/wrn/experts.pkl \
  2>&1 | tee logs/train_experts_new_gpu.log

python experiments/ablation/run_ablation_paper.py \
  2>&1 | tee logs/ablation_new_gpu.log
```
