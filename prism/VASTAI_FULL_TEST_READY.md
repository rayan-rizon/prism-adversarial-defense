# PRISM Vast.ai Full Test Readiness

Date: 2026-05-17

## Active protocol

- Launcher: `bash run_vastai_full.sh`
- Default config: `configs/vastai_cw_full.yaml`
- Seeds: `42 123 456 789 999`
- Test samples: `n=1000` per attack per seed
- Attack set: FGSM, PGD, Square, CW-L2, AutoAttack
- Canonical Square budget: `5000` queries
- CW-L2: torch engine, `max_iter=40`, `bss=5`
- Feature contract: 54 raw features, logit-profile + stability-v2 + side-quadratic, grad-norm off
- Training mix: FGSM `1.5x`, PGD `1.0x`, Square `1.0x`, CW `0.5x`
- Conformal validation gates: L1 `<=0.10`, L2 `<=0.03`, L3 `<=0.005`

## Local evidence kept

- `experiments/evaluation/cw_candidate_fgsm_l1/research_report_20260517.md`
- `experiments/calibration/cw_candidate_fgsm_l1/fpr_report_l1_085.json`
- `experiments/evaluation/cw_candidate_fgsm_l1/smoke_fast_fgsm_l1_n300_seed42_square500.json`
- `experiments/evaluation/cw_candidate_fgsm_l1/smoke_cw_fgsm_l1_n300_seed42_cw40_bss5.json`
- `experiments/evaluation/cw_candidate_fgsm_l1/latency_only_n200_seed42.json`

## Canonical local artifacts

- `models/ensemble_scorer.pkl`
- `models/calibrator.pkl`
- `models/calibrator_base.pkl`
- `models/reference_profiles.pkl`
- `models/cifar_resnet18.pt`
- `models/cifar_resnet18.acc.json`

The canonical scorer was promoted from the CW-aware candidate and has metadata:
FGSM/PGD/Square/CW training attacks, counts `937/625/625/313`, weights
`1.5/1.0/1.0/0.5`, feature-space `pixel-stability-v2+logitprofile+sidequad`.

## Removed as stale

Rejected candidate arms, old smoke outputs, local failed gate JSONs, cache
directories, and transient logs were removed. Vast.ai will regenerate logs and
full result JSONs from the current launcher.

## Post-run gate command

After `run_vastai_full.sh` finishes on Vast.ai, run:

```bash
python scripts/check_vastai_full_gate.py
```

The checker exits non-zero if any required attack, FPR tier, latency file, or
metric target is missing or failing.
