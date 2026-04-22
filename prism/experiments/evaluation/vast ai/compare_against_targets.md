# Comparison vs Targeted Metrics — Vast.ai run (2026-04-22)

**Summary**

- Run bundle: `prism/experiments/evaluation/vast ai`
- Outcome: final gate **FAILED** due to pooled FGSM TPR below target. All other published targets (PGD, Square, AutoAttack, per-tier FPRs, latency) passed.

**Pooled Results (fast attacks, n=1000 × 5 seeds)**

| Attack | Target TPR | Observed TPR (mean ± std) | Observed FPR (mean ± std) | Status |
|---|---:|---:|---:|---|
| FGSM | ≥ 0.85 | 0.8060 ± 0.0105 | 0.0706 ± 0.0046 | ❌ FAIL |
| PGD  | ≥ 0.90 | 1.0000 ± 0.0000 | 0.0706 ± 0.0046 | ✅ PASS |
| Square | ≥ 0.85 | 0.8908 ± 0.0069 | 0.0706 ± 0.0046 | ✅ PASS |
| AutoAttack | ≥ 0.90 | 0.9996 ± 0.0009 | 0.0706 ± 0.0046 | ✅ PASS |

Notes: pooled values taken from `results_fast_n1000_ms5.json` in this bundle (aggregate means/stds).

**Per-seed Breakdown (TPR / FPR)**

| Seed | FGSM TPR | FGSM FPR | PGD TPR | PGD FPR | Square TPR | Square FPR | AutoAttack TPR | AutoAttack FPR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42  | 0.8200 | 0.0700 | 1.0000 | 0.0700 | 0.9010 | 0.0700 | 1.0000 | 0.0700 |
| 123 | 0.7910 | 0.0780 | 1.0000 | 0.0780 | 0.8890 | 0.0780 | 1.0000 | 0.0780 |
| 456 | 0.8100 | 0.0710 | 1.0000 | 0.0710 | 0.8820 | 0.0710 | 1.0000 | 0.0710 |
| 789 | 0.8040 | 0.0660 | 1.0000 | 0.0660 | 0.8890 | 0.0660 | 1.0000 | 0.0660 |
| 999 | 0.8050 | 0.0680 | 1.0000 | 0.0680 | 0.8930 | 0.0680 | 0.9980 | 0.0680 |

Values sourced from `results_paper_seed{seed}.json` files included in this bundle.

**FPR Tier Validation (validation split: test[7000-7999])**

| Tier | Target FPR | Observed FPR | 95% CI | Status |
|---|---:|---:|---:|---|
| L1 | ≤ 0.10 | 0.0800 | [0.064748, 0.098467] | ✅ PASS |
| L2 | ≤ 0.03  | 0.0190 | [0.012197, 0.029485] | ✅ PASS |
| L3 | ≤ 0.005 | 0.0030 | [0.001021, 0.008783] | ✅ PASS |

Source: `ensemble_fpr_report.json` (validation split reported in the bundle).

**Latency**

- Measured: mean = 34.5 ms (p95 = 39.3 ms) — **PASS** (target < 100 ms).
- Source: `pipeline.log` entries captured in the bundle.

**CW (C&W L2) Status**

- CW generation was running as a separate slow process (ART/CW L2 with `max_iter=100, bss=9`). During finalization the CW worker was stopped after the gate failure and no final `results_cw_n1000_ms5.json` aggregate was produced for this run.
- Smoke-level CW artifacts are present (`results_cw_smoke.json`) but canonical CW multi-seed results are not included in the bundle.

**Final Gate**

- The run printed the automatic TARGET METRIC GATE summary and the **GATE RESULT: FAIL** due to pooled FGSM TPR = 0.8060 < 0.85 (95% CI [0.7948, 0.8167]).

**Interpretation & Likely Causes**

- FGSM underperformance is systematic across all five seeds (TPR range 0.7910–0.8200); this suggests a persistent mismatch between either the scorer/calibrator and FGSM adversarial examples or the ensemble training mix.
- Possible root causes to investigate (non-exhaustive):
  - Insufficient FGSM coverage in `scripts/train_ensemble_scorer.py` (check `--fgsm-oversample` and `training_attacks`).
  - Conformal thresholding may reduce FGSM detection sensitivity; verify thresholds and per-tier alpha factors.
  - Preprocessing or backbone mismatch between training and evaluation (feature mismatch). Verify model provenance and preprocessing pipeline.

**Recommended Next Steps (quick loop)**

1. Run an FGSM-only smoke (n=200) to reproduce FGSM shortfall quickly:

```bash
source .venv/bin/activate
python experiments/evaluation/run_evaluation_full.py \
  --n-test 200 --attacks FGSM \
  --multi-seed --seeds 42 123 \
  --output experiments/evaluation/results_fgsm_smoke.json
```

2. Confirm trainer used FGSM oversample (example):

```bash
python scripts/train_ensemble_scorer.py --help
# If retraining, try: --fgsm-oversample 2.0 --n-train 3000
```

3. If retraining, re-run calibration (`calibrate_ensemble.py`) and the FPR gate (`compute_ensemble_val_fpr.py`) before full evaluation.

4. If you want to preserve compute, consider skipping CW multi-seed until FGSM is fixed; CW is expensive (~16 s/img observed here) and dominates runtime.

**Files included in this bundle**

- [prism/experiments/evaluation/vast ai/results_fast_n1000_ms5.json](prism/experiments/evaluation/vast%20ai/results_fast_n1000_ms5.json)
- [prism/experiments/evaluation/vast ai/results_paper_seed42.json](prism/experiments/evaluation/vast%20ai/results_paper_seed42.json)
- [prism/experiments/evaluation/vast ai/results_paper_seed123.json](prism/experiments/evaluation/vast%20ai/results_paper_seed123.json)
- [prism/experiments/evaluation/vast ai/results_paper_seed456.json](prism/experiments/evaluation/vast%20ai/results_paper_seed456.json)
- [prism/experiments/evaluation/vast ai/results_paper_seed789.json](prism/experiments/evaluation/vast%20ai/results_paper_seed789.json)
- [prism/experiments/evaluation/vast ai/results_paper_seed999.json](prism/experiments/evaluation/vast%20ai/results_paper_seed999.json)
- [prism/experiments/evaluation/vast ai/ensemble_fpr_report.json](prism/experiments/evaluation/vast%20ai/ensemble_fpr_report.json)
- [prism/experiments/evaluation/vast ai/pipeline.log](prism/experiments/evaluation/vast%20ai/pipeline.log)

**Notes on provenance**

- All numbers in this document are read directly from the JSON artifacts and `pipeline.log` saved in this bundle; nothing is fabricated.
- The full run log is included for traceability (see `pipeline.log`).

**If you want me to proceed**

- I can run a quick FGSM smoke test now, or prepare a retraining command and orchestrate a new full run. Which would you prefer?

---

Generated on 2026-04-22 by automation driven from the local workspace.
