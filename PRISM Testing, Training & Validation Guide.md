# PRISM: Testing, Training & Validation Guide

## Overview

This guide answers three questions after your code is written:

1. **What data do I use?**
2. **How do I train the components?**
3. **How do I know PRISM is actually working?**

---

## 1. Datasets to Use

### Primary Benchmarks

| **Dataset** | **Purpose** | **Why PRISM uses it** | **Download** |
| --- | --- | --- | --- |
| CIFAR-10 | Main dev + calibration | Standard 10-class, fast iteration, widely used in adversarial robustness papers | `torchvision.datasets.CIFAR10` |
| CIFAR-100 | Generalization test | Harder classification — confirms PRISM isn't overfit to CIFAR-10 topology | `torchvision.datasets.CIFAR100` |
| RobustBench-style attacks | Standardized evaluation | AutoAttack-style stress tests on the same CIFAR-native backbone | `autoattack`, `robustbench` utilities |

### Data Split Strategy

```
Clean Data — active CIFAR test set (10,000 images), partitioned by index range.
Source of truth: active YAML config + src.config.{PROFILE_IDX, CAL_IDX, VAL_IDX, EVAL_IDX}.

├── Profile Set     [    0,  5000) — 5,000 images → reference-profile medoids (TAMM)
├── Calibration Set [5000,  7000) — 2,000 images → conformal thresholds (CADG)
├── Validation Set  [7000,  8000) — 1,000 images → FPR verification (strict)
└── Eval Set        [8000, 10000) — 2,000 images → held out, sampled per seed

Adversarial Data
├── Generated at test time using ART / AutoAttack
└── RobustBench pre-computed sets (for reproducibility)
```

Any script that needs these splits **must import from `src.config`** rather than
hardcoding ranges — see `experiments/ablation/run_ablation_paper.py` for the
canonical pattern.

<aside>
⚠️

**Never let adversarial examples contaminate the profile or calibration sets.** Those must be 100% clean inputs. Contamination breaks the conformal guarantee.

</aside>

---

## 2. Training Each Component

PRISM has **no single training loop** — each module is fitted/trained separately.

### 2.1 TAMM — Topological Self-Profile

**What "training" means here:** Running the 5,000 profile-set images through the frozen backbone, computing their persistence diagrams, and selecting medoid representatives. No gradient updates.

```bash
# scripts/build_profile_testset.py — run ONCE on the profile set
python scripts/build_profile_testset.py
# Output: models/reference_profiles.pkl  (or models/cifar100/... under PRISM_CONFIG=configs/cifar100.yaml)
```

**Validation check:**

```python
import pickle
from src.config import PATHS
profiles = pickle.load(open(PATHS['reference_profiles'], 'rb'))
for layer, dgms in profiles.items():
    print(f"{layer}: {len(dgms)} diagrams collected")
    # Reference set is 50 medoid diagrams per layer (configurable via tda.n_reference)
```

### 2.2 CADG — Conformal Calibration

**What "training" means here:** Computing anomaly scores on the 2,000-image calibration set (`cal_idx = [5000, 7000)`) and setting the quantile thresholds. Validation FPR is then verified on the disjoint 1,000-image validation set (`val_idx = [7000, 8000)`).

```bash
# scripts/calibrate_ensemble.py — run AFTER train_ensemble_scorer.py
python scripts/calibrate_ensemble.py
# Output: models/calibrator.pkl  (or models/cifar100/... under PRISM_CONFIG)
```

**Validation check — the conformal guarantee:**

```python
# On the held-out validation set of 1,000 clean images
fpr_actual = np.mean(val_scores > calibrator.thresholds['L1'])
print(f"Actual FPR: {fpr_actual:.4f}")
print(f"Target FPR (α): 0.10")
assert fpr_actual <= 0.10 + 0.02, "Conformal guarantee violated!"
# +0.02 tolerance for finite-sample deviation
```

### 2.3 SACD — BOCPD Campaign Monitor

**What "training" means here:** Fitting BOCPD/L0 hyperparameters on real clean
and PGD score streams from the validation split. No gradient updates.

```bash
python scripts/calibrate_l0_thresholds.py
# CIFAR-100:
PRISM_CONFIG=configs/cifar100.yaml python scripts/calibrate_l0_thresholds.py --config configs/cifar100.yaml
```

**Selection rule:** maximize sustained-stream ASR reduction subject to
clean-only `l0_active_frac <= 0.01`.

### 2.4 TAMSH — Expert Sub-Networks

**What "training" means here:** Actual PyTorch training. Each expert MLP maps
the final monitored layer activation to the active dataset's class logits.
The four experts are trained on clean, FGSM, PGD, and mixed perturbation pools.

**Validation check:**

```bash
python scripts/train_experts.py
python experiments/evaluation/run_recovery_eval.py --n-test 1000
# Gate: recovery_accuracy(tamsh) - recovery_accuracy(passthrough) >= 15pp
```

---

## 3. Testing — How to Know PRISM is Working

### 3.1 Unit Tests — Test Each Module in Isolation

```python
# tests/test_tamm.py
import numpy as np
from src.tamm.tda import TopologicalProfiler

def test_anomaly_score_increases_with_noise():
    profiler = TopologicalProfiler(n_subsample=100)
    clean_act = np.random.randn(200, 64)  # Simulated clean activations
    adv_act = clean_act + np.random.randn(200, 64) * 2.0  # Noisy adversarial
    
    clean_dgm = profiler.compute_diagram(clean_act)
    adv_dgm = profiler.compute_diagram(adv_act)
    
    ref_dgm = clean_dgm  # Reference is the clean diagram
    clean_score = profiler.anomaly_score(clean_dgm, ref_dgm)
    adv_score = profiler.anomaly_score(adv_dgm, ref_dgm)
    
    assert adv_score > clean_score, (
        f"Adversarial score ({adv_score:.3f}) should exceed "
        f"clean score ({clean_score:.3f})"
    )
    print(f"PASS — clean: {clean_score:.3f}, adversarial: {adv_score:.3f}")
```

```python
# tests/test_cadg.py
from src.cadg.calibrate import ConformalCalibrator
import numpy as np

def test_conformal_coverage():
    cal = ConformalCalibrator()
    # Fit on 1000 clean scores drawn from N(0.1, 0.02)
    clean_scores = np.random.normal(0.1, 0.02, 1000)
    cal.calibrate(clean_scores)
    
    # Validate on 1000 fresh clean scores
    val_scores = np.random.normal(0.1, 0.02, 1000)
    passed, fpr = cal.verify_coverage(val_scores, alpha=0.10, level='L1')
    assert passed, f"Coverage failed: FPR={fpr:.4f} > 0.10"
```

```python
# tests/test_sacd.py
from src.sacd.monitor import CampaignMonitor
import numpy as np

def test_campaign_detection():
    monitor = CampaignMonitor(hazard_rate=1/200, cp_threshold=0.3)
    clean = np.random.normal(0.1, 0.02, 100)
    attack = np.random.normal(0.7, 0.1, 50)
    
    l0_triggered = False
    trigger_time = None
    for t, s in enumerate(np.concatenate([clean, attack])):
        state = monitor.process_score(s, timestamp=t)
        if state['l0_active'] and not l0_triggered:
            l0_triggered = True
            trigger_time = t
    
    assert l0_triggered, "L0 never activated during simulated attack"
    assert trigger_time >= 100, "False positive: L0 triggered during clean phase"
    print(f"PASS — L0 triggered at t={trigger_time} (expected >=100)")
```

Run all tests:

```bash
pip install pytest
pytest tests/ -v
```

### 3.2 Integration Test — Full PRISM on One Image

```python
# tests/test_integration.py
import torch
from src.prism import PRISM
from src.models import load_backbone
from src.config import BACKBONE_INPUT_SIZE, LAYER_NAMES, PATHS

device = torch.device('cuda')
model = load_backbone(device)
prism = PRISM.from_saved(
    model=model,
    layer_names=LAYER_NAMES,
    calibrator_path=PATHS['calibrator'],
    profile_path=PATHS['reference_profiles'],
    ensemble_path=PATHS['ensemble_scorer'],
)

# Test 1: Clean image should PASS
x_clean = torch.randn(1, 3, BACKBONE_INPUT_SIZE, BACKBONE_INPUT_SIZE).to(device)
_, level, meta = prism.defend(x_clean)
print(f"Clean input → level: {level} (expected: PASS or L1)")

# Test 2: Heavily perturbed image should be flagged
x_adv = x_clean + torch.randn_like(x_clean) * 0.5
_, level_adv, meta_adv = prism.defend(x_adv)
print(f"Perturbed input → level: {level_adv} (expected: L2 or L3)")
print(f"Score delta: clean={meta['anomaly_score']:.4f}, adv={meta_adv['anomaly_score']:.4f}")
assert meta_adv['anomaly_score'] > meta['anomaly_score'], "Adversarial score must be higher!"
```

### 3.3 Quantitative Evaluation — The Core Metrics

These are the numbers your paper reports. Run the full evaluation script from Phase 6 of the implementation guide.

| **Metric** | **Formula** | **Target** | **Fail threshold** |
| --- | --- | --- | --- |
| True Positive Rate (TPR) | Detected adversarials / Total adversarials | > 90% | < 70% |
| False Positive Rate (FPR) | Flagged clean inputs / Total clean inputs | < 10% (L1), < 3% (L2), < 0.5% (L3) | > 15% |
| Campaign Detection Lag | Probe queries until L0 activates | < 20 queries | > 50 queries |
| Latency overhead | PRISM defend() time vs. bare model | < 100ms per image | > 500ms |
| TAMSH recovery gap | Recovery accuracy(tamsh) - passthrough on L3 adversarials | > 15pp | < 5pp |

### 3.4 Sanity Checks — Quick Smoke Tests

Run these after each major phase is complete:

```bash
python sanity_checks.py
PRISM_CONFIG=configs/cifar100.yaml python sanity_checks.py
```

### 3.5 Ablation Tests — Prove Each Component Contributes

For the paper, you must show that each module adds value. Test four configurations:

| **Configuration** | **TAMM** | **Ensemble** | **TDA features** | **TAMSH** | **Purpose** |
| --- | --- | --- | --- | --- | --- |
| Full PRISM | yes | yes | yes | yes | Reference |
| No MoE | yes | yes | yes | no | Recovery-only effect |
| Ensemble-no-TDA | no | yes | no | yes | C1 marginal topology gate |
| TDA only | yes | no | yes | no | Base topology baseline |

```bash
python experiments/ablation/run_ablation_paper.py \
  --n 1000 --multi-seed --seeds 42 123 456 789 999 \
  --attacks FGSM PGD Square CW \
  --output experiments/ablation/results_ablation_multiseed.json
```

---

## 4. Debugging — When Results Look Wrong

| **Symptom** | **Likely cause** | **Fix** |
| --- | --- | --- |
| TPR < 50% on all attacks | Anomaly scores not separating clean vs adversarial | Check subsample size — try n=500; check that hooks are on the right layers |
| FPR > 30% on clean data | Profile set was contaminated, or too few calibration samples | Rebuild profile with strictly clean data; increase calibration set to 2000+ |
| L0 never triggers | hazard_rate too low or cp_threshold too high | Lower cp_threshold to 0.2; increase hazard_rate to 1/50 |
| TDA takes > 500ms per image | Too many subsample points | Reduce n_subsample to 50; switch to ripser++ (GPU); use cubical complexes |
| Expert MSE > 0.3 | Expert architecture too small or not enough epochs | Increase hidden_dim to 512; train for 100 epochs; check cluster quality |

---

## 5. Recommended Evaluation Order

1. **Smoke tests** — `python sanity_checks.py` after each phase
2. **Unit tests** — `pytest tests/ -v` after each module is coded
3. **Integration test** — single clean + single adversarial image through full PRISM
4. **Small-scale eval** — 100 images per attack locally (no cloud cost)
5. **Full eval** — 1000 images per attack on CamberCloud A100
6. **Ablation** — 4 configurations × 5 attacks on CamberCloud
7. **Adaptive attack** — adversary who knows PRISM's architecture (the hard test)

<aside>
✅

**You know PRISM is working when:** TPR > 90% on PGD/AutoAttack, FPR < 10% on clean data, L0 detects probe campaigns within 20 queries, and ablation shows each module contributes measurable improvement.

</aside>
