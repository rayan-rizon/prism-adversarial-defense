# PRISM Vast.ai Full Test Readiness

Date: 2026-05-19

## Active protocol

- Launcher: `bash run_vastai_full.sh`
- Default config: `configs/vastai_cw_full.yaml`
- Seeds: `42 123 456 789 999`
- Test samples: `n=1000` per attack per seed
- Attack set: FGSM, PGD, Square, CW-L2, AutoAttack
- Canonical Square budget: `5000` queries
- CW-L2: torch engine, `max_iter=100`, `bss=9`, `confidence=1.0`
- Feature contract: 55 raw features, logit-profile + stability-v2 + side-quadratic + grad-norm
- Training mix: balanced FGSM/PGD/Square; CW and AutoAttack are evaluation-only on Vast.ai
- Conformal validation gates: L1 `<=0.10`, L2 `<=0.03`, L3 `<=0.005`

## Current local evidence

- `experiments/calibration/ensemble_fpr_report.json`: L1 `0.074`, L2 `0.026`, L3 `0.003`; all pass
- `experiments/evaluation/results_fast_local_n300_seed42_square5000_l2tight.json`: FGSM `0.8833`, PGD `0.9867`, Square `0.8767`; all fast gates pass with canonical Square `5000`
- `experiments/evaluation/results_cw_local_n100_seed42_vastparams.json`: CW-L2 `0.9500` TPR with torch `max_iter=100`, `bss=9`, `confidence=1.0`; FPR tiers pass
- `experiments/evaluation/results_autoattack_local_n100_seed42_standard.json`: AutoAttack standard `1.0000` TPR; FPR tiers pass
- `experiments/evaluation/results_latency_standalone.json`: mean latency `72.91ms`; pass
- `sanity_checks.py`: all available artifact/contract checks pass

## Canonical local artifacts

- `models/ensemble_scorer.pkl`
- `models/calibrator.pkl`
- `models/calibrator_base.pkl`
- `models/reference_profiles.pkl`
- `models/cifar_resnet18.pt`
- `models/cifar_resnet18.acc.json`

The canonical scorer is promoted from the balanced fast-attack candidate and
has metadata: FGSM/PGD/Square training attacks, balanced counts, feature-space
`pixel-stability-v2+logitprofile+sidequad+gradnorm`. CW and AutoAttack remain
in the Vast.ai evaluation stages.

## Stale-data policy

Rejected candidate arms and old smoke outputs are diagnostic only. Vast.ai
will regenerate logs and full result JSONs from the current launcher; the gate
checker uses only the current configured artifacts and fresh Vast result files.

## Post-run gate command

After `run_vastai_full.sh` finishes on Vast.ai, run:

```bash
python scripts/check_vastai_full_gate.py
```

The checker exits non-zero if any required attack, FPR tier, latency file, or
metric target is missing or failing.
