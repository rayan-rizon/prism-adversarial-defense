# VAST.AI Run Comparison Report
**Date:** 2026-04-22  
**Run:** Live Vast.ai RTX 5090 (58.224.7.136) — 38-feature Ensemble  
**Status:** Gate FAILED — FGSM and Square below publishable thresholds  

---

## Executive Summary

**Outcome:** ❌ **NOT PUBLISHABLE** — Two critical metrics failed the gate.

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| **FGSM TPR** | 63.04% ± 0.95% | ≥ 85% | ❌ **FAIL (-21.96pp)** |
| **PGD TPR** | 100% ± 0% | ≥ 90% | ✅ **PASS** |
| **Square TPR** | 79.78% ± 0.46% | ≥ 85% | ❌ **FAIL (-5.22pp)** |
| **AutoAttack TPR** | 99.78% ± 0.13% | ≥ 90% | ✅ **PASS** |
| **L1 FPR** | 1.9% [CI: 1.2%-2.9%] | ≤ 10% | ✅ **PASS** |
| **L2 FPR** | 0.3% [CI: 0.1%-0.9%] | ≤ 3% | ✅ **PASS** |
| **L3 FPR** | 0.0% [CI: 0.0%-0.4%] | ≤ 0.5% | ✅ **PASS** |
| **Latency (mean)** | 49.9 ms | < 100 ms | ✅ **PASS** |
| **CW-L2 TPR** | Incomplete (256/1000 at 19.6 s/img) | ≥ 85% | ⏳ **INCOMPLETE** |

---

## Detailed Metrics

### Attack Detection Rates (Adversarial TPR)

#### FGSM: 63.04% ± 0.95% [CI: 61.69% – 64.37%] — ❌ FAIL
- **Failure magnitude:** 21.96 percentage points below 85% target
- **Per-seed breakdown:**
  - Seed 42: 63.9%
  - Seed 123: 61.5%
  - Seed 456: 63.6%
  - Seed 789: 62.8%
  - Seed 999: 63.4%
- **Interpretation:** Systemic weakness across all seeds; not seed-dependent variance
- **Pooled count:** 3,152 TP / 1,848 FN (out of 5,000 total FGSM examples)

#### PGD: 100% ± 0% [CI: 99.92% – 100%] — ✅ PASS
- **Margin over target:** +10 percentage points
- **Pooled count:** 5,000 TP / 0 FN
- **Note:** Perfect detection across all seeds

#### Square: 79.78% ± 0.46% [CI: 78.64% – 80.87%] — ❌ FAIL
- **Failure magnitude:** 5.22 percentage points below 85% target
- **Per-seed breakdown:**
  - Seed 42: 80.1%
  - Seed 123: 78.9%
  - Seed 456: 78.2%
  - Seed 789: 78.9%
  - Seed 999: 79.3%
- **Interpretation:** Consistent underperformance; not calibration noise
- **Pooled count:** 3,989 TP / 1,011 FN (out of 5,000)

#### AutoAttack: 99.78% ± 0.13% [CI: 99.61% – 99.88%] — ✅ PASS
- **Margin over target:** +9.78 percentage points
- **Pooled count:** 4,989 TP / 11 FN
- **Note:** Near-perfect detection; only 11 missed examples across all seeds

---

### False Positive Rates (Clean Data Safety)

#### Validation Split FPRs (CIFAR-10 test 7000–7999)

| Tier | FPR | Target | CI (95%) | Status |
|------|-----|--------|----------|--------|
| **L1** | 1.9% | ≤ 10% | [1.2%, 2.9%] | ✅ **PASS** |
| **L2** | 0.3% | ≤ 3% | [0.1%, 0.9%] | ✅ **PASS** |
| **L3** | 0.0% | ≤ 0.5% | [0.0%, 0.4%] | ✅ **PASS** |

**Interpretation:** Extremely tight FPR across all tiers. L2/L3 are well-controlled, suggesting the ensemble is correctly calibrated for safety. FPR uniformity (same 1.76% across all attack types per seed) indicates calibration is not attack-dependent.

---

### Latency

#### Per-Seed and Aggregate (Evaluation split: 8000–9999, 200 images/seed)

| Seed | Mean (ms) | Std (ms) | p95 (ms) | Pass |
|------|-----------|----------|----------|------|
| **42** | 54.61 | 19.05 | 62.01 | ✅ |
| **123** | 47.76 | — | — | ✅ |
| **456** | 47.29 | — | — | ✅ |
| **789** | 47.51 | — | — | ✅ |
| **999** | 47.43 | — | — | ✅ |
| **Aggregate** | **49.9** | — | **~55** | ✅ |

**Target:** < 100 ms — **✅ All seeds well under target**  
**Note:** Seed 42 shows higher std (19.05ms) and max latency (310.39ms), likely GPU warmup. Other seeds are stable at ~47–48ms.

---

### CW-L2 Status

**Status:** ⏳ **INCOMPLETE** — Still running at time of download  
**Progress:** 256/1000 samples completed  
**Current throughput:** 19.6 seconds/image  
**ETA:** ~80 hours for full 1000-image evaluation at current rate  
**Expected completion:** Off-instance download pending

**Note:** CW was not halted after fast-attack gate failure. Current throughput (19.6 s/img) is significantly above the guide's claimed 2–3 s/img expectation, indicating either:
- Suboptimal GPU utilization in the attack generation loop
- Inaccurate runtime model in the guide

---

## Ensemble Configuration

### Active Artifact Provenance

```json
{
  "ensemble": {
    "architecture": "PersistenceEnsembleScorer + LogisticRegression",
    "n_features": 38,
    "use_dct": true,
    "use_grad_norm": true,
    "training_attacks": ["FGSM", "PGD", "Square", "CW", "AutoAttack"],
    "training_n": 4000,
    "training_eps": 8/255 (0.031373),
    "fgsm_oversample": 1.5,
    "training_mix_computed": {
      "FGSM": "1.5 / (1.5 + 1 + 1 + 1 + 1) = 27.27%",
      "PGD": "1 / 5.5 = 18.18%",
      "Square": "1 / 5.5 = 18.18%",
      "CW": "1 / 5.5 = 18.18%",
      "AutoAttack": "1 / 5.5 = 18.18%"
    }
  }
}
```

### Critical Feature: Grad-Norm Enabled (38th Feature)

This run **differs from previous archived bundles** which used 37 features without gradient norm. The 38-feature ensemble adds the input-gradient L2 norm as an adversarial discriminator, with an expected +5ms latency overhead. However:

- **Observed latency increase:** Seed 42 = 54.61ms vs. historical ~52ms (only ~2.6ms increase, not the expected ~5ms)
- **Implication:** Grad-norm feature may be partially optimized or the historical baseline is higher than expected

---

## Root-Cause Summary

### Why FGSM Failed

The 38-feature grad-norm ensemble was trained on a 5-attack mix with only 27% FGSM representation (due to `--fgsm-oversample 1.5` being insufficient for a 5-attack regime). Contributing factors:

1. **Training-mix dilution:** FGSM share drops from ~33% (3-attack regime) to 27% when CW+AutoAttack are added without proportional oversample increase
2. **Calibration tightening:** With `use_grad_norm=True`, the ensemble exhibits tighter per-tier FPR (observed 0.019–0.003 vs. historical 0.07), which suppresses FGSM TPR more than other attacks
3. **Feature-space shift:** The 38th feature (grad-norm) may have altered the logistic regression decision boundary unfavorably for FGSM patterns

### Why Square Also Failed

Square suffers from the same training-mix dilution effect but to a lesser degree (5.22pp vs. FGSM's 21.96pp). The attack relies more on features robust across the training-attack distribution, making it partially tolerant to mix imbalance.

### Why PGD and AutoAttack Passed

Both attacks are well-represented in the 5-attack training mix and appear to have strong feature signatures that the ensemble learned robustly.

---

## Comparison with Previous Baseline

For context, an earlier run using a **3-attack ensemble (FGSM, PGD, Square, 37 features, NO grad-norm)** achieved:
- FGSM TPR: 86.76% ✅ (passed)
- Square TPR: 92.56% ✅ (passed)

**Degradation with current 5-attack, 38-feature ensemble:**
- FGSM: -23.72 pp (86.76% → 63.04%)
- Square: -12.78 pp (92.56% → 79.78%)

This large difference highlights the compounding effects of training-mix dilution and the grad-norm feature addition.

---

## Recommendations

### Phase 1: Remediation Factor Study (Recommended)

1. **Test grad-norm necessity:** Compare 37-feature vs 38-feature ensembles on the same 5-attack mix to isolate grad-norm's impact
2. **Validate calibration:** Ensure calibration is rerun after any ensemble architecture change
3. **Multi-seed gate on improvement candidate:** Only promote to Phase 2 if improved candidate passes FGSM ≥85%, Square ≥85%, with no regression on PGD/AutoAttack/FPR

### Phase 2: CW Profiling and Speedup

1. Profile CW generation separately from detection overhead
2. Sweep `gen_chunk` parameter (current: 8) to balance GPU utilization vs. visibility
3. Measure actual generation-only and total latency on RTX 5090 to validate guide's 2–3 s/img claim

### Reproducibility & Documentation

- Document that this run used **38-feature ensemble with grad-norm**, making it **NOT directly comparable** to older 37-feature artifacts without explicit methodology disclosure
- Update `run_vastai_full.sh` Step 2b verification to ensure artifact regeneration and feature count confirmation
- Fix CW stopping behavior: auto-halt CW generation on fast-attack gate failure to avoid wasted compute

---

## Artifact Metadata

| Field | Value |
|-------|-------|
| **Eval Split** | CIFAR-10 test indices 8000–9999 (n=1000/attack/seed) |
| **Validation Split** | CIFAR-10 test indices 7000–7999 (n=1000) |
| **n_seeds** | 5 (seeds: 42, 123, 456, 789, 999) |
| **ε (L∞)** | 8/255 ≈ 0.0314 (RobustBench standard) |
| **Device** | CUDA (RTX 5090, 32GB GDDR7) |
| **Gate Status** | ❌ **FAILED** (FGSM, Square) |
| **Publication Ready** | ❌ **NO** — Metrics below publishable thresholds |

---

## Files Reference

- **Evaluation results:** `/prism/experiments/evaluation/vast ai/results_fast_n1000_ms5.json`
- **Per-seed results:** `/prism/experiments/evaluation/vast ai/results_paper_seed{42,123,456,789,999}.json`
- **Validation FPR report:** `/prism/experiments/evaluation/vast ai/ensemble_fpr_report.json`
- **Run guide & targets:** `/prism/VASTAI_RUN_GUIDE.md`
- **Remediation plan:** Session memory (plan.md)

---

**Next steps:** Execute Phase 1 of the controlled factor study, starting with FGSM oversample sweep (1.5 → 2.5–3.0) on identical hardware.
