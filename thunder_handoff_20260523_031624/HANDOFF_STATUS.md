# Thunder PRISM Handoff Status

Local handoff folder: `/Users/rayanrizon/Desktop/Research/Prism/thunder_handoff_20260523_031624`

## Download Contents

| Item | Included | Notes |
|---|---:|---|
| Full remote `prism/` tree | yes | code, configs, scripts, paper, tests |
| WRN checkpoint | yes | `models/cifar_wrn28_10.pt` |
| WRN scorer/calibrators | yes | `models/wrn/*.pkl`; `experts.pkl` absent |
| CIFAR-10 data | yes | archive + extracted batches |
| Logs/results | yes | current snapshot at download time |
| L3 rerun queue script | yes | copied from remote `/tmp` if available |

## Current Calibration Snapshot

| Tier | FPR | Target | Status |
|---|---:|---:|---:|
| L1 | 0.0790 | 0.1000 | PASS |
| L2 | 0.0250 | 0.0300 | PASS |
| L3 | 0.0050 | 0.0050 | PASS |

Config L3 factor in handoff: `0.47`. This is patched for next recalibration; downloaded calibrator may still be pre-recalibration if queue had not run yet.

## Completed / Partial Tests

| Test | Seeds / Stage | Metrics | Status |
|---|---|---|---|
| CW-L2 | 5 seeds downloaded | mean TPR=0.9328, mean FPR=0.0790, max L3=0.0050 | PASS |
| FGSM | 1 seed(s) downloaded | mean TPR=0.9830, mean FPR=0.0770, max L3=0.0080 | CHECK |
| PGD | 1 seed(s) downloaded | mean TPR=0.9940, mean FPR=0.0770, max L3=0.0080 | CHECK |
| Square | 1 seed(s) downloaded | mean TPR=0.9480, mean FPR=0.0770, max L3=0.0080 | CHECK |
| AutoAttack | 1 seed(s) downloaded | mean TPR=1.0000, mean FPR=0.0770, max L3=0.0080 | CHECK |
| Adaptive PGD | 5 partial log(s), no final JSON | killed/deferred for high-power GPU | REMAINING |
| Ablation | failed | missing experts.pkl | REMAINING, train `models/wrn/experts.pkl` first |

## Next-GPU Commands

Run from new GPU after uploading/copying this handoff folder:

```bash
cd thunder_handoff_20260523_031624/prism
python -m pip install -r requirements.txt
export PRISM_CONFIG=configs/wrn_cifar10.yaml

# 1. Recalibrate with patched L3 factor 0.47
python scripts/calibrate_ensemble.py 2>&1 | tee logs/recalibrate_wrn_l3_new_gpu.log
python scripts/compute_ensemble_val_fpr.py 2>&1 | tee logs/recheck_wrn_l3_new_gpu.log

# 2. Rerun only FGSM + PGD to validate L3 fix
python experiments/evaluation/run_evaluation_full.py \
  --n-test 1000 \
  --attacks FGSM PGD \
  --multi-seed \
  --seeds 42 123 456 789 999 \
  --gen-chunk 128 \
  --pgd-max-iter 50 \
  --pgd-restarts 10 \
  --skip-latency \
  --checkpoint-interval 100 \
  --output experiments/wrn/evaluation/results_fgsm_pgd_l3fix_wrn.json \
  2>&1 | tee logs/eval_fgsm_pgd_l3fix_new_gpu.log

# 3. Adaptive PGD deferred full run
python experiments/evaluation/run_adaptive_pgd.py \
  --n-test 1000 \
  --multi-seed \
  --seeds 42 123 456 789 999 \
  2>&1 | tee logs/adaptive_pgd_new_gpu.log

# 4. Ablation prerequisite, then rerun ablation
python scripts/train_experts.py --device cuda --output models/wrn/experts.pkl 2>&1 | tee logs/train_experts_new_gpu.log
python experiments/ablation/run_ablation_paper.py 2>&1 | tee logs/ablation_new_gpu.log
```
