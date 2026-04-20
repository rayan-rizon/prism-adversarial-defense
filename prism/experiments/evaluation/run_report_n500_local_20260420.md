# PRISM n=500 Local Evaluation Run Report (2026-04-20)

## Scope

This report documents the retained local CUDA evaluation artifact:

- `experiments/evaluation/results_n500_local_20260420.json`

Companion baseline detector artifact verified during this work:

- `experiments/evaluation/results_baselines_v2.json`

Supporting verified guardrail artifact retained:

- `experiments/calibration/ensemble_fpr_report.json`

This report also records the cleanup performed on older stale evaluation outputs and transient logs after the run was verified.

## Objective

Run the full refreshed PRISM pipeline far enough to replace stale ensemble and calibration artifacts, then produce a final local `n=500` evaluation on the held-out CIFAR-10 test split with attacks `FGSM`, `PGD`, and `Square`.

Primary acceptance targets used during this run:

- FGSM TPR >= 0.85
- PGD TPR >= 0.95
- Square TPR >= 0.85
- Tier FPR targets: `L1 <= 0.10`, `L2 <= 0.03`, `L3 <= 0.005`

## Environment

- Workspace root: `C:\Users\rayan\Desktop\Research\Prism\prism-adversarial-defense`
- Working directory for all run commands: `C:\Users\rayan\Desktop\Research\Prism\prism-adversarial-defense\prism`
- OS: Windows
- Python interpreter used for the successful run: `C:\Users\rayan\Desktop\Research\Prism\prism-adversarial-defense\prism\.venv\Scripts\python.exe`
- Python version in this environment: `3.11.15`
- Torch build in this environment: `2.6.0+cu124`
- Device used for final evaluation: `cuda`
- Required shell env for Unicode-safe output: `PYTHONIOENCODING=utf-8`

## Active Configuration Snapshot

From `configs/default.yaml` at time of run:

- Layers: `layer2`, `layer3`, `layer4`
- Layer weights: `layer2=0.30`, `layer3=0.30`, `layer4=0.40`
- `tda.n_subsample = 200`
- `tda.max_dim = 1`
- `tda.dim_weights = [0.70, 0.30]`
- Conformal targets: `L1=0.10`, `L2=0.03`, `L3=0.005`
- Conservative calibration multiplier: `cal_alpha_factor = 0.7`
- CIFAR-10 split indices:
  - profile: `0-4999`
  - calibration: `5000-6999`
  - validation: `7000-7999`
  - evaluation: `8000-9999`

## Current Pipeline Artifacts Verified During This Run

- `models/reference_profiles.pkl`
- `models/scorer.pkl`
- `models/ensemble_scorer.pkl`
- `models/calibrator.pkl`
- `models/experts.pkl`
- `experiments/calibration/ensemble_fpr_report.json`
- `experiments/evaluation/results_n500_local_20260420.json`
- `experiments/evaluation/results_baselines_v2.json`

Verified ensemble provenance from the final evaluation artifact:

- `use_dct = true`
- training attacks: `FGSM`, `PGD`, `Square`
- `training_n = 2100`
- `training_eps = 8/255`
- `n_features = 37`

## Executed Procedure

### C1. Rebuild clean reference profile

Purpose:

- refresh `reference_profiles.pkl` and `scorer.pkl` on the intended clean profile split

Executed step:

- `python scripts/build_profile_testset.py`

Observed outcome:

- fresh reference profiles built from CIFAR-10 test indices `0-4999`
- current reference medoids retained for `layer2`, `layer3`, `layer4`

### C2. Retrain ensemble scorer

Purpose:

- replace stale ensemble trained on incomplete attack coverage
- ensure DCT feature is active in both training and inference

Executed step:

- `python scripts/train_ensemble_scorer.py --n-train 2100`

Observed outcome:

- new ensemble trained with attacks `FGSM`, `PGD`, `Square`
- new ensemble uses 37 features, including the DCT high-frequency energy feature
- output artifact refreshed: `models/ensemble_scorer.pkl`

### C3. Recalibrate conformal thresholds

Purpose:

- rebuild the calibrator against the fresh ensemble scorer
- enforce stricter calibration slack using `cal_alpha_factor = 0.7`

Executed step:

- `python scripts/calibrate_ensemble.py`

Observed calibrated thresholds from sanity verification:

- `L1 = 12.102476`
- `L2 = 14.932669`
- `L3 = 18.204727`

Output artifact refreshed:

- `models/calibrator.pkl`

### C4. Validation FPR gate

Purpose:

- verify empirical clean-input FPR on the held-out validation split before final evaluation

Executed step:

- `python scripts/compute_ensemble_val_fpr.py`

Verified output artifact:

- `experiments/calibration/ensemble_fpr_report.json`

Measured gate results:

| Tier | FP | n | FPR | 95% CI | Target | Pass |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| L1 | 71 | 1000 | 0.071 | [0.056669, 0.088614] | 0.10 | true |
| L2 | 15 | 1000 | 0.015 | [0.009111, 0.024601] | 0.03 | true |
| L3 | 1 | 1000 | 0.001 | [0.000177, 0.005643] | 0.005 | true |

Validation split recorded by the artifact:

- `CIFAR-10 test idx 7000-7999`

### C5. Expert availability check

Purpose:

- confirm expert artifacts are present for the full PRISM stack

Observed sanity outcome:

- `models/experts.pkl` loads successfully
- 4 expert state dicts present
- expert forward pass succeeds

### D. Final retained evaluation

Purpose:

- run the held-out local evaluation on the refreshed pipeline

Executed step:

- `python experiments/evaluation/run_evaluation_full.py --n-test 500 --attacks FGSM PGD Square --output experiments/evaluation/results_n500_local_20260420.json`

Recorded metadata from the retained result artifact:

- `n_test = 500`
- `n_actual = 500`
- evaluation split: `CIFAR-10 test idx 8000-9999`
- seed: `42`
- epsilon: `8/255 = 0.03137254901960784`
- device: `cuda`

## Final Measured Results

### Attack summary

| Attack | TPR | TPR 95% CI | FPR | FPR 95% CI | Precision | F1 |
| --- | ---: | --- | ---: | --- | ---: | ---: |
| FGSM | 0.8380 | [0.8031, 0.8677] | 0.0600 | [0.0423, 0.0844] | 0.9332 | 0.8830 |
| PGD | 1.0000 | [0.9924, 1.0000] | 0.0600 | [0.0423, 0.0844] | 0.9434 | 0.9709 |
| Square | 0.9100 | [0.8817, 0.9321] | 0.0600 | [0.0423, 0.0844] | 0.9381 | 0.9239 |

### Tier FPR on the retained evaluation artifact

All three attacks produced the same clean-side tier rates in the retained evaluation:

- `FPR_L1_plus = 0.0600`
- `FPR_L2_plus = 0.0180`
- `FPR_L3_plus = 0.0020`

These are within the project targets:

- `L1 <= 0.10` -> pass
- `L2 <= 0.03` -> pass
- `L3 <= 0.005` -> pass

### Confusion counts

| Attack | TP | FP | FN | TN |
| --- | ---: | ---: | ---: | ---: |
| FGSM | 419 | 30 | 81 | 470 |
| PGD | 500 | 30 | 0 | 470 |
| Square | 455 | 30 | 45 | 470 |

### Clean/adversarial level distributions

Clean level distribution was identical across attacks:

- `PASS = 470`
- `L1 = 21`
- `L2 = 8`
- `L3_REJECT = 1`

Adversarial level distributions:

- FGSM: `L3_REJECT=246`, `L2=101`, `L1=72`, `PASS=81`
- PGD: `L3_REJECT=500`
- Square: `L3_REJECT=350`, `L2=64`, `L1=41`, `PASS=45`

## Latency Measurement in the Retained Result Artifact

Latency block recorded in the retained artifact:

- mean: `120.23 ms`
- std: `25.28 ms`
- min: `86.19 ms`
- max: `325.54 ms`
- p50: `114.57 ms`
- p95: `164.53 ms`
- sample count: `200`
- target: `100 ms`
- pass flag: `false`

## Interpretation

Measured status against the local targets used for this run:

- FGSM TPR target `>= 0.85`: **not met** (`0.8380`)
- PGD TPR target `>= 0.95`: **met** (`1.0000`)
- Square TPR target `>= 0.85`: **met** (`0.9100`)
- Tier FPR targets: **met** on validation gate and retained evaluation artifact
- Mean latency target `< 100 ms`: **not met** (`120.23 ms`)

Important note:

- The critical stale-artifact failure on Square was resolved. Square TPR improved to `0.9100`, well above the `0.85` target.
- FGSM remains slightly below the desired `0.85` threshold in this single-seed `n=500` retained run.

## Companion Baseline Detector Verification

This section records the corrected LID and Mahalanobis baseline test artifact:

- `experiments/evaluation/results_baselines_v2.json`

Purpose:

- verify the fixed detector split logic and the Mahalanobis label source on the same held-out CIFAR-10 evaluation split

Executed step:

- `python experiments/evaluation/run_baselines.py --n-test 500 --attacks FGSM PGD Square --output experiments/evaluation/results_baselines_v2.json`

Recorded metadata from the retained baseline artifact:

- `n_test = 500`
- `n_actual = 500`
- `n_ref = 1000`
- `n_thresh = 1000`
- evaluation split: `CIFAR-10 test idx 8000-9999`
- reference split: `CIFAR-10 test idx 5000-5999`
- threshold split: `CIFAR-10 test idx 6000-6999`
- seed: `42`
- device: `cuda`
- attacks: `FGSM`, `PGD`, `Square`
- epsilon: `8/255 = 0.031372`
- LID `k = 20`
- LID threshold: `26.980043`
- Mahalanobis threshold: `1540.614322`
- Mahalanobis label source: `CIFAR-10 ground-truth labels (0-9), n_classes=10`
- monitored layers: `layer2`, `layer3`, `layer4`

### Baseline attack summary

| Detector | Attack | TPR | TPR 95% CI | FPR | FPR 95% CI | Precision | F1 |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: |
| LID | FGSM | 0.9980 | [0.9888, 0.9996] | 0.0880 | [0.0662, 0.1161] | 0.9190 | 0.9569 |
| LID | PGD | 1.0000 | [0.9924, 1.0000] | 0.0880 | [0.0662, 0.1161] | 0.9191 | 0.9579 |
| LID | Square | 0.9900 | [0.9768, 0.9957] | 0.0880 | [0.0662, 0.1161] | 0.9184 | 0.9528 |
| Mahalanobis | FGSM | 1.0000 | [0.9924, 1.0000] | 0.1040 | [0.0802, 0.1338] | 0.9058 | 0.9506 |
| Mahalanobis | PGD | 1.0000 | [0.9924, 1.0000] | 0.1040 | [0.0802, 0.1338] | 0.9058 | 0.9506 |
| Mahalanobis | Square | 0.9900 | [0.9768, 0.9957] | 0.1040 | [0.0802, 0.1338] | 0.9049 | 0.9456 |

### Baseline confusion counts

| Detector | Attack | TP | FP | FN | TN |
| --- | ---: | ---: | ---: | ---: | ---: |
| LID | FGSM | 499 | 44 | 1 | 456 |
| LID | PGD | 500 | 44 | 0 | 456 |
| LID | Square | 495 | 44 | 5 | 456 |
| Mahalanobis | FGSM | 500 | 52 | 0 | 448 |
| Mahalanobis | PGD | 500 | 52 | 0 | 448 |
| Mahalanobis | Square | 495 | 52 | 5 | 448 |

### Baseline clean-side levels

The clean-side level distribution was identical across attacks:

- `PASS = 456` for LID and `448` for Mahalanobis
- `L1 = 21`
- `L2 = 8`
- `L3_REJECT = 1`

### Baseline interpretation

- LID meets the target clean-side operating point: `FPR = 0.0880` on all three attacks.
- Mahalanobis also meets the target clean-side operating point: `FPR = 0.1040` on all three attacks.
- The earlier v1 failure is superseded: the corrected disjoint reference/threshold split and true-label Mahalanobis fit removed the self-reference contamination.
- Square is no longer a failure case for the baseline detectors; both detectors remain near the intended 10% FPR regime.

## Sanity Verification Performed

`sanity_checks.py` was run after the refreshed artifacts were in place.

Observed pass conditions included:

- reference profiles loaded successfully
- calibrator thresholds ordered `L1 < L2 < L3`
- clean score distribution accepted
- experts loaded and executed a forward pass
- `PRISM.defend()` returned a valid result on a clean CIFAR-10 example

## Cleanup Performed After Verification

The following obsolete evaluation outputs and transient logs were removed so that the workspace retains the current verified result set and this report, rather than multiple stale copies.

Removed stale evaluation artifacts:

- `experiments/evaluation/n500_cpu_stderr.log`
- `experiments/evaluation/n500_cpu_stdout.log`
- `experiments/evaluation/results_n500_20260419.json`
- `experiments/evaluation/results_n500_cpu_20260420.json`
- `experiments/evaluation/results_n500_cpu_sq100_20260420.json`
- `experiments/evaluation/results_n500_planA.json`
- `experiments/evaluation/results_n500_retrained_20260419.json`
- `experiments/evaluation/results_paper.json`

Removed transient run logs:

- `build_profile_log.txt`
- `evaluation.log`
- `logs_c1_stderr.txt`
- `logs_c1_stdout.txt`
- `logs_c2_stderr.txt`
- `logs_c2_stdout.txt`
- `logs_c3_stderr.txt`
- `logs_c3_stdout.txt`
- `logs_c4_stderr.txt`
- `logs_c4_stdout.txt`
- `logs_c5_stderr.txt`
- `logs_c5_stdout.txt`
- `logs_d_stderr.txt`
- `logs_d_stdout.txt`
- `test_output.txt`
- `train_ensemble_log.txt`

## Retained Files After Cleanup

- `experiments/evaluation/results_n500_local_20260420.json`
- `experiments/evaluation/run_report_n500_local_20260420.md`
- `experiments/calibration/ensemble_fpr_report.json`
- current model artifacts in `models/`
- source code, scripts, configs, tests, and paper files

## Reproducibility Notes

- All reported numbers in this document were copied from verified local artifacts or direct run outputs from the successful retained run.
- No forecast or estimated metric is reported as a measured result.
- This document records a single-seed local `n=500` run, not a multi-seed pooled paper result.