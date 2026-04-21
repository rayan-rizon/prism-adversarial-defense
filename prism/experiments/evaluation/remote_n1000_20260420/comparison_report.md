# n=1000 ThunderCompute Download Comparison

Downloaded artifacts:
- `C1_multiseed.log`
- `results_paper_seed42.json`
- `results_paper_seed123.json`
- `results_paper_seed456.json`
- `results_paper_seed789.json`
- `results_paper_seed999.json`
- `results_n1000_multiseed_20260420.json`

Reference local baseline:
- `prism/experiments/evaluation/run_report_n500_local_20260420.md`

## Target Gates
- FGSM TPR >= 0.85
- PGD TPR >= 0.95
- Square TPR >= 0.85
- L1 FPR <= 0.10
- L2 FPR <= 0.03
- L3 FPR <= 0.005
- Mean latency <= 100 ms

## Aggregate Result

| Attack | TPR mean | FPR mean | F1 mean | Target pass |
| --- | ---: | ---: | ---: | --- |
| FGSM | 0.8626 | 0.0642 | 0.8953 | pass |
| PGD | 1.0000 | 0.0642 | 0.9689 | pass |
| Square | 0.9256 | 0.0642 | 0.9303 | pass |

Mean latency across seeds: 63.57 ms.

## Comparison vs Local Baseline

Local baseline from the retained n=500 report:
- FGSM TPR 0.8440
- PGD TPR 1.0000
- Square TPR 0.9240
- Mean latency 73.68 ms
- Tier FPRs: L1 0.0620, L2 0.0180, L3 0.0020

Delta vs local baseline:
- FGSM TPR: +0.0186
- PGD TPR: +0.0000
- Square TPR: +0.0016
- Mean latency: -10.11 ms

## Tier FPR Check

| Tier | Mean FPR | Worst seed FPR | Target | Pass |
| --- | ---: | ---: | ---: | --- |
| L1 | 0.0642 | 0.0680 | 0.10 | pass |
| L2 | 0.0194 | 0.0230 | 0.03 | pass |
| L3 | 0.0072 | 0.0080 | 0.005 | fail |

Seed-level L3 values:
- seed 42: 0.0060
- seed 123: 0.0060
- seed 456: 0.0080
- seed 789: 0.0080
- seed 999: 0.0080

## Bottom Line

The downloaded n=1000 evaluation clears the TPR targets and latency target, and it stays within L1/L2 FPR bounds. It does not clear the L3 FPR target of 0.005 on the final evaluation artifacts.
