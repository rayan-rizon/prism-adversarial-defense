# PRISM — Vast.ai RTX 5090 Run Guide (Paper-Quality, All Attacks)

## Purpose

Run the **full publishable n=1000 × 3-seed evaluation for all five attacks**
(FGSM, PGD, Square, CW-L2, AutoAttack) against a **single retrained ensemble**
that includes CW + AutoAttack in its training mix.

**Why we re-run everything (not just CW + AutoAttack):**
The retrain changes the logistic decision boundary and the calibrated L1/L2/L3
thresholds. Any FGSM/PGD/Square numbers from the prior Thundercompute runs were
computed against the **old** ensemble (`training_attacks=['FGSM','PGD','Square']`).
Mixing old + new numbers in one paper table would mean two different decision
functions reported as one system and would break the conformal FPR guarantee
that holds *per fitted ensemble*. For paper consistency we re-evaluate all five
attacks against the **same** retrained ensemble.

GTX 1060 locally is too slow (~30 s/sample for CW). RTX 5090 drops this to
~1 s/sample.

---

## 1. Local prerequisites (do before renting)

```bash
cd prism/

# Confirm all model artifacts exist
ls models/calibrator.pkl models/reference_profiles.pkl models/ensemble_scorer.pkl

# Quick smoke: both new pipelines run end-to-end
python experiments/evaluation/run_evaluation_full.py \
  --n-test 4 --attacks CW --gen-chunk 4 \
  --output experiments/evaluation/results_cw_smoke.json

python experiments/evaluation/run_evaluation_full.py \
  --n-test 4 --attacks AutoAttack --aa-chunk 4 \
  --output experiments/evaluation/results_aa_smoke.json
```

---

## 2. Vast.ai instance config

**GPU:** RTX 5090 (32 GB GDDR7, Blackwell sm_100)

| Setting | Value |
|---------|-------|
| GPU | RTX 5090 |
| vCPU | 8+ |
| RAM | 32 GB+ |
| Disk | 50 GB |
| Docker image | `pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel` |

> **Note:** RTX 5090 is Blackwell architecture (sm_100). Requires CUDA 12.6+ and
> PyTorch 2.6+. The image above is the correct one. Do not use older cuda11.x images.

**RTX 5090 throughput estimates:**

| Stage | Rate | Per-seed time (n=1000) |
|-------|------|------------------------|
| FGSM | ~0.05 s/sample | ~1 min |
| PGD-40 | ~0.1 s/sample | ~2 min |
| Square (5000 queries) | ~0.4 s/sample | ~7 min |
| CW-L2 (max_iter=50, bss=5, batch=64) | ~1 s/sample | ~17 min |
| AutoAttack (standard, aa-chunk=64) | ~0.3 s/sample | ~5 min |
| Retrain n=4000 (all 5 attacks) | — | ~35–45 min |
| calibrate + val-fpr | — | ~5 min |

---

## 3. Instance setup

```bash
# Clone the repo
git clone https://github.com/rayan-rizon/prism-adversarial-defense.git prism-repo
cd prism-repo/prism

# Install PyTorch for Blackwell (CUDA 12.6, PyTorch 2.6+)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Install PRISM dependencies
pip install adversarial-robustness-toolbox autoattack certifi tqdm scikit-learn ripser persim

# Verify GPU is detected
python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.version.cuda)"
```

Upload local model artifacts from your machine:

```bash
# Run from your local machine (replace <instance-ip> and <port>)
scp -P <port> \
  models/calibrator.pkl \
  models/reference_profiles.pkl \
  models/ensemble_scorer.pkl \
  root@<instance-ip>:/workspace/prism-repo/prism/models/

# Upload CIFAR-10 data if not already downloaded (saves re-download time)
scp -P <port> -r data/cifar-10-batches-py \
  root@<instance-ip>:/workspace/prism-repo/prism/data/
```

---

## 4a. Retrain ensemble with CW + AutoAttack in training mix

**Why:** CW currently scores 3.3% TPR because the prior ensemble was trained
only on L∞ attacks. Adding CW-L2 and AutoAttack-APGD-CE to the logistic
training mix calibrates the boundary for both L2 and L∞ attack signatures.
This ensemble becomes the **single decision function** used for all five
attacks in the paper table.

On RTX 5090, this takes ~35–45 min for n=4000.

```bash
mkdir -p logs

python scripts/train_ensemble_scorer.py \
  --n-train 4000 \
  --fgsm-oversample 1.5 \
  --use-grad-norm \
  --include-cw \
  --include-autoattack \
  --cw-max-iter 30 \
  --cw-bss 3 \
  --output models/ensemble_scorer.pkl \
  2>&1 | tee logs/retrain_n4000.log
```

Expected console at end:
```
training_attacks=['FGSM', 'PGD', 'Square', 'CW', 'AutoAttack']
Held-out validation AUC (logistic component): 0.97+
```

If AUC drops below 0.92, re-run with `--n-train 6000` for more training signal.

---

## 4b. Re-run the FPR gate (mandatory after retrain)

```bash
python scripts/calibrate_ensemble.py 2>&1 | tee logs/calibrate.log
python scripts/compute_ensemble_val_fpr.py 2>&1 | tee logs/val_fpr.log
```

Check `logs/val_fpr.log` — all three tiers must pass before proceeding:

| Tier | Target | Action if fail |
|------|--------|----------------|
| L1 FPR | ≤ 0.10 | Lower `tier_cal_alpha_factors.L1` by 0.05 in `configs/default.yaml`, re-run calibrate only |
| L2 FPR | ≤ 0.03 | Lower `tier_cal_alpha_factors.L2` by 0.05, re-run calibrate only |
| L3 FPR | ≤ 0.005 | Lower `tier_cal_alpha_factors.L3` by 0.05, re-run calibrate only |

After this step, `models/ensemble_scorer.pkl` + `models/calibrator.pkl` are the
**locked artifacts** used for every attack in the paper. Do not re-train or
re-calibrate between attack runs — that would invalidate cross-attack comparison.

---

## 4c. Full n=1000 × 3-seed evaluation — **all five attacks**

Run in `screen` or `tmux` so SSH disconnect doesn't kill the job. The two L2/AA
runs are the slow ones; the L∞ trio finishes in ~30 min combined.

### Single-attack sanity (seed 42 only) — confirm the new ensemble works

Before burning time on multi-seed, verify the retrained ensemble produces the
expected shape on the slow attacks:

```bash
mkdir -p logs experiments/evaluation

# CW — single seed, confirms TPR uplift from 3.3% baseline
screen -S cw_42
python experiments/evaluation/run_evaluation_full.py \
  --n-test 1000 \
  --attacks CW \
  --gen-chunk 64 \
  --checkpoint-interval 100 \
  --seed 42 \
  --output experiments/evaluation/results_cw_n1000_seed42.json \
  2>&1 | tee logs/cw_seed42.log

# AutoAttack — single seed
screen -S aa_42
python experiments/evaluation/run_evaluation_full.py \
  --n-test 1000 \
  --attacks AutoAttack \
  --aa-chunk 64 \
  --checkpoint-interval 100 \
  --seed 42 \
  --output experiments/evaluation/results_aa_n1000_seed42.json \
  2>&1 | tee logs/aa_seed42.log
```

Acceptance gate before continuing:
- CW seed-42 TPR ≥ 0.50 (we expect ≥ 0.90; ≥ 0.50 confirms the retrain landed)
- AutoAttack seed-42 TPR ≥ 0.90
- Both: per-tier FPR within targets

If CW seed-42 is still ≤ 0.10, **stop**. Re-run step 4a with `--n-train 6000`
or add a higher-confidence CW variant (`confidence=5.0`) before continuing.

### Full multi-seed sweep (paper-quality, all five attacks, 3 seeds each)

These are the numbers that go in the paper. Run them in one command per attack
using the built-in `--multi-seed` aggregator (it handles per-seed runs and
pooled aggregation automatically; the multi-seed flag passthrough was fixed in
`run_evaluation_full.py`).

```bash
# 1) FGSM — fast, ~3 min total
screen -S fgsm_ms
python experiments/evaluation/run_evaluation_full.py \
  --n-test 1000 --attacks FGSM \
  --multi-seed --seeds 42 123 456 \
  --checkpoint-interval 100 \
  --output experiments/evaluation/results_fgsm_n1000_multiseed.json \
  2>&1 | tee logs/fgsm_multiseed.log

# 2) PGD — ~6 min total
screen -S pgd_ms
python experiments/evaluation/run_evaluation_full.py \
  --n-test 1000 --attacks PGD \
  --multi-seed --seeds 42 123 456 \
  --checkpoint-interval 100 \
  --output experiments/evaluation/results_pgd_n1000_multiseed.json \
  2>&1 | tee logs/pgd_multiseed.log

# 3) Square — ~21 min total
screen -S sq_ms
python experiments/evaluation/run_evaluation_full.py \
  --n-test 1000 --attacks Square \
  --multi-seed --seeds 42 123 456 \
  --checkpoint-interval 100 \
  --output experiments/evaluation/results_square_n1000_multiseed.json \
  2>&1 | tee logs/square_multiseed.log

# 4) CW — ~50 min total
screen -S cw_ms
python experiments/evaluation/run_evaluation_full.py \
  --n-test 1000 --attacks CW \
  --multi-seed --seeds 42 123 456 \
  --gen-chunk 64 --checkpoint-interval 100 \
  --output experiments/evaluation/results_cw_n1000_multiseed.json \
  2>&1 | tee logs/cw_multiseed.log

# 5) AutoAttack — ~15 min total
screen -S aa_ms
python experiments/evaluation/run_evaluation_full.py \
  --n-test 1000 --attacks AutoAttack \
  --multi-seed --seeds 42 123 456 \
  --aa-chunk 64 --checkpoint-interval 100 \
  --output experiments/evaluation/results_aa_n1000_multiseed.json \
  2>&1 | tee logs/aa_multiseed.log
```

> **Optional one-shot variant** (single screen, all five attacks back-to-back —
> only do this if you don't need to monitor each independently):
> ```bash
> python experiments/evaluation/run_evaluation_full.py \
>   --n-test 1000 --attacks FGSM PGD Square CW AutoAttack \
>   --multi-seed --seeds 42 123 456 \
>   --gen-chunk 64 --aa-chunk 64 --checkpoint-interval 100 \
>   --output experiments/evaluation/results_all5_n1000_multiseed.json \
>   2>&1 | tee logs/all5_multiseed.log
> ```

---

## 5. Progress monitoring

The script prints per-chunk and per-checkpoint lines. Watch them live:

```bash
# CW generation + PRISM evaluation progress
tail -f logs/cw_multiseed.log | grep -E "\[gen\]|\[Checkpoint\]|TPR=|Traceback|Error|elapsed"

# AutoAttack generation + PRISM evaluation progress
tail -f logs/aa_multiseed.log | grep -E "\[AA gen\]|\[AA eval\]|TPR=|Traceback|Error|elapsed"

# Quick pass/fail snapshot for any finished run
grep -E "TPR=|FPR=|pass_L" logs/*_multiseed.log | tail -50
```

Healthy CW output looks like:
```
[gen] 64/1000 elapsed=64.1s  rate=1.00s/img  eta=936.0s
[gen] 128/1000 elapsed=128.3s  rate=1.00s/img  eta=871.7s
[Checkpoint 100/1000] TPR=0.8700 ✅ | FPR=0.0600 ✅ | F1=0.9147
```

---

## 6. Download results

```bash
# From your local machine — pull every JSON + log:
scp -P <port> \
  root@<instance-ip>:/workspace/prism-repo/prism/experiments/evaluation/results_*_n1000_multiseed.json \
  experiments/evaluation/

scp -P <port> \
  root@<instance-ip>:/workspace/prism-repo/prism/models/ensemble_scorer.pkl \
  models/ensemble_scorer_vastai_retrained.pkl

scp -P <port> \
  root@<instance-ip>:/workspace/prism-repo/prism/models/calibrator.pkl \
  models/calibrator_vastai_retrained.pkl

scp -P <port> \
  "root@<instance-ip>:/workspace/prism-repo/prism/logs/*.log" \
  /tmp/vastai_logs/
```

Keep both `ensemble_scorer.pkl` and `calibrator.pkl` together — they form the
single locked decision function reported in the paper.

---

## 7. Target metrics (all five attacks, single ensemble)

| Attack | TPR target | FPR L1+ target | Prior status (n=1000 Thundercompute, OLD ensemble) |
|--------|------------|----------------|-----------------------------------------------------|
| FGSM | ≥ 0.90 | ≤ 0.10 | (re-eval with new ensemble) |
| PGD | ≥ 0.90 | ≤ 0.10 | (re-eval with new ensemble) |
| Square | ≥ 0.90 | ≤ 0.10 | (re-eval with new ensemble) |
| CW-L2 | ≥ 0.90 | ≤ 0.10 | 0.033 ❌ on old ensemble — retrain required |
| AutoAttack | ≥ 0.90 | ≤ 0.10 | 1.0 ✅ on n=100 local (with old ensemble + label fix) |

Per-tier FPR (pooled over all clean samples in all five runs) must satisfy:
L1+ ≤ 0.10, L2+ ≤ 0.03, L3+ ≤ 0.005.

---

## 8. Pipeline gate — order is mandatory

```
train_ensemble_scorer.py        (step 4a — once)
        ↓
calibrate_ensemble.py           (step 4b — once)
        ↓
compute_ensemble_val_fpr.py     (step 4b — once)
        ↓
run_evaluation_full.py × 5      (step 4c — one per attack, all against the
                                 same locked ensemble + calibrator)
```

Skipping any step invalidates the conformal FPR guarantee. Re-running 4a or 4b
**between** attack evaluations would mean different attacks were scored against
different decision functions — which is the exact pitfall this guide is
written to avoid.

---

## 9. Estimated total wall-clock on RTX 5090

| Step | Time |
|------|------|
| Retrain n=4000 (incl. CW + AutoAttack features) | ~40 min |
| Calibrate + val-FPR gate | ~5 min |
| FGSM × 3 seeds | ~3 min |
| PGD × 3 seeds | ~6 min |
| Square × 3 seeds | ~21 min |
| CW × 3 seeds | ~50 min |
| AutoAttack × 3 seeds | ~15 min |
| **Total** | **~2 h 20 min** |

Runs can overlap on a single GPU only marginally (PRISM eval is GPU-bound for
the backbone forward pass). Easiest is to run them sequentially in one screen
with the optional one-shot variant in §4c, or in five separate screens
back-to-back.

---

## 10. Reproducibility manifest (include in paper appendix)

After all runs finish, capture:

```bash
python -c "
import pickle, json, hashlib
e = pickle.load(open('models/ensemble_scorer.pkl','rb'))
c = pickle.load(open('models/calibrator.pkl','rb'))
def h(p): return hashlib.sha256(open(p,'rb').read()).hexdigest()[:16]
print(json.dumps({
  'ensemble_training_attacks': list(e.training_attacks),
  'ensemble_training_n': int(e.training_n),
  'ensemble_n_features': int(e.n_features) if hasattr(e,'n_features') else None,
  'ensemble_sha256_16': h('models/ensemble_scorer.pkl'),
  'calibrator_sha256_16': h('models/calibrator.pkl'),
  'reference_profiles_sha256_16': h('models/reference_profiles.pkl'),
}, indent=2))
" 2>&1 | tee logs/manifest.json
```

This file (training mix + artifact hashes) is what reviewers need to verify
that every per-attack number in your table came from the same locked decision
function.
