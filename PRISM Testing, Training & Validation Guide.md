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
| ImageNet (subset) | Scale test | Larger images → bigger activation maps → stress-tests TDA speed | RobustBench provides pre-curated 5000-image subsets |
| RobustBench test sets | Standardized evaluation | Pre-generated adversarial examples under AutoAttack — used by all SOTA papers | `robustbench.data` |

### Data Split Strategy

```
Clean Data (total: ~12,000 images per dataset)
├── Profile Set     — 10,000 images → build topological self-profile (TAMM)
├── Calibration Set —  1,000 images → fit conformal thresholds (CADG)
└── Validation Set  —  1,000 images → verify FPR guarantees

Adversarial Data
├── Generated at test time using ART / AutoAttack
└── RobustBench pre-computed sets (for reproducibility)
```

<aside>
⚠️

**Never let adversarial examples contaminate the profile or calibration sets.** Those must be 100% clean inputs. Contamination breaks the conformal guarantee.

</aside>

---

## 2. Training Each Component

PRISM has **no single training loop** — each module is fitted/trained separately.

### 2.1 TAMM — Topological Self-Profile

**What "training" means here:** Running 10,000 clean images through the frozen backbone and computing their persistence diagrams. No gradient updates.

```python
# scripts/build_profile.py (already in Phase 1)
# Run this ONCE on the profile set
python scripts/build_profile.py
# Output: models/reference_profiles.pkl
```

**Validation check:**

```python
import pickle, numpy as np
profiles = pickle.load(open('models/reference_profiles.pkl', 'rb'))
for layer, dgms in profiles.items():
    print(f"{layer}: {len(dgms)} diagrams collected")
    # Should be 10,000 for each layer
```

### 2.2 CADG — Conformal Calibration

**What "training" means here:** Computing anomaly scores on 1,000 clean images and setting the quantile thresholds.

```python
# scripts/calibrate_thresholds.py (already in Phase 2)
python scripts/calibrate_thresholds.py
# Output: models/calibrator.pkl
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

**What "training" means here:** Setting the hyperparameters (`hazard_rate`, `cp_threshold`). No gradient updates. Tune on a synthetic simulation.

```python
# Tune via simulation:
from src.sacd.monitor import CampaignMonitor
import numpy as np

# Grid search over hazard rates
for hazard in [1/50, 1/100, 1/200, 1/500]:
    monitor = CampaignMonitor(hazard_rate=hazard, cp_threshold=0.3)
    # Simulate: 100 clean queries, then 20 probe queries
    clean = np.random.normal(0.1, 0.02, 100)
    probes = np.random.normal(0.3, 0.05, 20)
    scores = np.concatenate([clean, probes])
    
    detected_at = None
    for t, s in enumerate(scores):
        state = monitor.process_score(s, timestamp=t)
        if state['l0_active'] and detected_at is None:
            detected_at = t
    
    print(f"hazard=1/{int(1/hazard)}: detected at t={detected_at}")
    # Target: detected_at between t=100 and t=120 (within 20 probe queries)
```

**Pick the hazard rate that detects within 20 probe queries with the fewest false positives.**

### 2.4 TAMSH — Expert Sub-Networks

**What "training" means here:** Actual PyTorch training. Each expert MLP is trained to replicate the function of a backbone sub-span on a cluster of clean inputs.

```python
# scripts/train_experts.py
import torch, torch.nn as nn
from src.tamsh.experts import ExpertSubNetwork

# For each cluster k of clean activations:
for k in range(K):  # K=4 experts
    expert = ExpertSubNetwork(
        input_dim=cluster_input_dims[k],
        output_dim=cluster_output_dims[k],
    ).cuda()
    
    optimizer = torch.optim.Adam(expert.parameters(), lr=1e-3)
    criterion = nn.MSELoss()  # Reconstruct target activations
    
    for epoch in range(50):
        for x_in, x_target in cluster_loaders[k]:
            pred = expert(x_in.cuda())
            loss = criterion(pred, x_target.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Expert {k}, Epoch {epoch}: loss={loss.item():.4f}")
    
    torch.save(expert.state_dict(), f'models/expert_{k}.pt')
```

**Validation check:**

```python
# Check expert reconstruction quality on held-out clean activations
for k in range(K):
    expert.load_state_dict(torch.load(f'models/expert_{k}.pt'))
    val_loss = evaluate_expert(expert, val_loader_k)
    print(f"Expert {k} val MSE: {val_loss:.4f}")
    # Target: < 0.05 MSE (adjust based on activation scale)
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
import torch, torchvision
from src.prism import PRISM

model = torchvision.models.resnet18(pretrained=True).cuda().eval()
prism = PRISM(
    model=model,
    layer_names=['layer1','layer2','layer3','layer4'],
    calibrator_path='models/calibrator.pkl',
    profile_path='models/reference_profiles.pkl',
)

# Test 1: Clean image should PASS
x_clean = torch.randn(1, 3, 224, 224).cuda()
_, level, meta = prism.defend(x_clean)
print(f"Clean input → level: {level} (expected: PASS or L1)")

# Test 2: Heavily perturbed image should be flagged
x_adv = x_clean + torch.randn_like(x_clean) * 0.5
_, level_adv, meta_adv = prism.defend(x_adv)
print(f"Perturbed input → level: {level_adv} (expected: L2 or L3)")
print(f"Score delta: clean={meta['score']:.4f}, adv={meta_adv['score']:.4f}")
assert meta_adv['score'] > meta['score'], "Adversarial score must be higher!"
```

### 3.3 Quantitative Evaluation — The Core Metrics

These are the numbers your paper reports. Run the full evaluation script from Phase 6 of the implementation guide.

| **Metric** | **Formula** | **Target** | **Fail threshold** |
| --- | --- | --- | --- |
| True Positive Rate (TPR) | Detected adversarials / Total adversarials | > 90% | < 70% |
| False Positive Rate (FPR) | Flagged clean inputs / Total clean inputs | < 10% (L1), < 3% (L2), < 0.5% (L3) | > 15% |
| Campaign Detection Lag | Probe queries until L0 activates | < 20 queries | > 50 queries |
| Latency overhead | PRISM defend() time vs. bare model | < 100ms per image | > 500ms |
| Expert val MSE | Reconstruction loss on clean held-out set | < 0.05 | > 0.20 |

### 3.4 Sanity Checks — Quick Smoke Tests

Run these after each major phase is complete:

```python
# sanity_checks.py
import numpy as np, pickle, torch

# Check 1: Reference profiles exist and have content
profiles = pickle.load(open('models/reference_profiles.pkl','rb'))
for layer, dgms in profiles.items():
    assert len(dgms) > 1000, f"{layer} profile too small: {len(dgms)}"
    print(f"✅ {layer}: {len(dgms)} diagrams")

# Check 2: Calibrator thresholds are ordered correctly
cal = pickle.load(open('models/calibrator.pkl','rb'))
assert cal.thresholds['L1'] < cal.thresholds['L2'] < cal.thresholds['L3'], \
    "Thresholds out of order!"
print(f"✅ Thresholds: L1={cal.thresholds['L1']:.4f}, "
      f"L2={cal.thresholds['L2']:.4f}, L3={cal.thresholds['L3']:.4f}")

# Check 3: Score distribution is sensible
clean_scores = np.load('experiments/calibration/clean_scores.npy')
print(f"✅ Clean score stats: mean={clean_scores.mean():.4f}, "
      f"std={clean_scores.std():.4f}")
# If mean is > 0.5, something is wrong (clean inputs shouldn't be flagged)
assert clean_scores.mean() < 0.5, "Clean score mean too high — check profiler"
```

### 3.5 Ablation Tests — Prove Each Component Contributes

For the paper, you must show that each module adds value. Test four configurations:

| **Configuration** | **TAMM** | **CADG** | **SACD (L0)** | **TAMSH** | **Expected TPR** |
| --- | --- | --- | --- | --- | --- |
| Full PRISM | ✅ | ✅ | ✅ | ✅ | Highest |
| No L0 | ✅ | ✅ | ❌ | ✅ | Lower on campaign attacks |
| No MoE | ✅ | ✅ | ✅ | ❌ | Lower on L3 recovery |
| TDA only (no conformal) | ✅ | ❌ | ❌ | ❌ | No FPR guarantee |

```python
# experiments/ablation/run_ablation.py
configurations = {
    'Full PRISM':        dict(use_l0=True,  use_moe=True,  use_conformal=True),
    'No L0':             dict(use_l0=False, use_moe=True,  use_conformal=True),
    'No MoE':            dict(use_l0=True,  use_moe=False, use_conformal=True),
    'TDA only':          dict(use_l0=False, use_moe=False, use_conformal=False),
}

for name, config in configurations.items():
    prism = build_prism(**config)
    tpr, fpr = evaluate(prism, test_loader, attack='PGD')
    print(f"{name}: TPR={tpr:.3f}, FPR={fpr:.3f}")
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