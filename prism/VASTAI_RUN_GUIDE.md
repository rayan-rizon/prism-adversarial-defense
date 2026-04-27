# PRISM — Vast.ai RTX 5090 Full Pipeline Guide

> ⚠️ **SOURCE OF TRUTH:** For the NeurIPS/ICLR/ICML research-plan campaign,
> **follow Appendix §A1 only** (14-step pipeline). The 7-step body below is
> historical and retained for forensic continuity. Where body and appendix
> conflict, **Appendix wins**. All pre-research-plan result JSONs have been
> archived under `prism_results/archive_pre_research_plan/` — do not reference
> them.

> **Goal:** Run the complete publishable pipeline from scratch on a single Vast.ai
> RTX 5090 instance. Retrain → Calibrate → Gate → Evaluate (5 attacks × 5 seeds,
> parallel) → Adaptive PGD → Ablation → Campaign → Recovery → Baselines → Paper Tables → Download.
>
> **Every evaluation script now prints a TARGET METRIC GATE** at the end of
> each multi-seed run — explicit ✅/❌ per metric, so you know immediately
> whether to continue or stop the instance.

---

## Target Metrics

| Metric | Target |
|--------|--------|
| FGSM TPR | ≥ 85 % |
| PGD TPR | ≥ 90 % |
| Square TPR | ≥ 85 % |
| CW-L2 TPR | ≥ 85 % (stretch ≥ 90 %) |
| AutoAttack TPR | ≥ 90 % |
| L1 FPR | ≤ 10 % |
| L2 FPR | ≤ 3 % |
| L3 FPR | ≤ 0.5 % |
| GPU latency (mean, per image) | < 100 ms |
| Ablation paired-bootstrap p | < 0.05 (each component) |

All TPR/FPR figures reported with Wilson 95 % CIs pooled across 5 seeds.

> **What "passing" looks like in the log (verified locally):**
> ```
> =================================================================
> TARGET METRIC GATE
> ─────────────────────────────────────────────────────────────────
>   ✅ CW           TPR=0.8900  target≥0.85  CI=[0.87, 0.91]
>   ✅ L1           FPR=0.0790  target≤0.100
>   ✅ L2           FPR=0.0220  target≤0.030
>   ✅ L3           FPR=0.0030  target≤0.005
>   ✅ Latency      mean=52.1ms  target<100ms
> ─────────────────────────────────────────────────────────────────
> ✅ GATE RESULT: ALL TARGETS MET — results are publishable.
> ```

---

## Pipeline Overview

```
PHASE 0 — TRAINING (all three launchers start simultaneously after Step 1)
─────────────────────────────────────────────────────────────────────────────
build_profile_testset.py         → reference_profiles.pkl  [Step 1, ~10 min]
        │
        ├──[foreground]──► train_ensemble_scorer.py         [Step 2,  ~30 min]
        │                    (n=4000, CW+AA, fgsm-os=2.5)
        │                    Step 2b: post-retrain verify gate
        ├──[background]──► train_ensemble_scorer.py --no-tda-features  [Step 2c, ~30 min]
        │                    → models/ensemble_no_tda.pkl
        └──[background]──► train_experts.py                [Step 2d, ~25 min]
                             → models/experts.pkl

        ← all three join before LOCK (wall-clock ≈ max(30,30,25) = 30 min) →

LOCK: calibrate_ensemble.py      → calibrator.pkl          [Step 3,   ~3 min]
      compute_ensemble_val_fpr.py → FPR gate               [Step 4,   ~2 min]
        │
        ▼   ARTIFACTS LOCKED — all below are read-only consumers
        │
PHASE 1 — ATTACKS (all launched simultaneously, ~35m ceiling on CW)
─────────────────────────────────────────────────────────────────────────────
┌─────────────────────────────────────────────────────┐
│ 5A CW-L2          (torch engine, ~35m)              │
│ 5B FGSM+PGD+Square+AutoAttack                       │ ← all launched at once
│ 6  Adaptive PGD × 5 seeds in parallel               │
│ 7  Ablation (FGSM+PGD+Square+CW)                    │
│ 6b L0 calibration (background)                      │
└─────────────────────────────────────────────────────┘
        │  wait 5A+5B → combined gate check
        │  wait 6     → STEP6_FAIL gate
        │  wait 7     → ablation done
        │  wait 6b    → L0 thresholds locked
        ▼
PHASE 2 — SECONDARY EVAL (all launched simultaneously, ~1h)
─────────────────────────────────────────────────────────────────────────────
┌─────────────────────────────────────────────────────┐
│ 7a Campaign eval   (P0.4)                            │
│ 7b Recovery eval   (P0.5)                            │ ← all launched at once
│ 7c Baseline detectors (P0.2)                        │
└─────────────────────────────────────────────────────┘
        │  wait all three
        ▼
Step 8: paper tables (P0.7) → manifest.json (SHA256)
```

**Why full parallelism is safe:** All concurrent jobs are pure read-only
consumers of frozen artifacts. No inter-step write contention.
CUDA SM time-slicing handles ≤8 concurrent light-batch processes on the RTX 5090.

**GPU memory budget (RTX 5090, 32 GB GDDR7):**

| Phase | Active jobs | Peak VRAM | Headroom |
|-------|-------------|-----------|---------|
| Phase 0 training (2+2c+2d overlap) | 3 × ResNet-18 + ART | ~8+8+3 = **19 GB** | 13 GB |
| Phase 1 attacks (5A+5B+6+7+6b) | CW dominant | ~22 GB | 10 GB |
| Phase 2 secondary eval (7a+7b+7c) | 3 × scorer inference | ~6.5 GB | 25 GB |

GPU utilization: Phase 0 ~70 %, Phase 1 ~85–90 %, Phase 2 ~40 %.

---

## 0. Prerequisites

Before starting your Vast.ai instance:

1.  **Push Local Changes:** The instance will use `git clone`. Ensure all current local fixes (logging updates, master scripts, etc.) are **pushed to your GitHub repository**.
2.  **Private Repo Access:** If your repository is private, ensure you have a [Personal Access Token (PAT)](https://github.com/settings/tokens) ready to use during the `git clone` step.
3.  **No Data/Model Upload Needed:** Do not worry about `data/` or `models/` folders. The pipeline will automatically download CIFAR-10 and build all required model artifacts on the RTX 5090 to ensure hardware-specific optimization.

---

## 1. Vast.ai Instance Configuration

| Setting | Value |
|---------|-------|
| GPU | RTX 5090 (32 GB GDDR7) |
| vCPU | 8+ |
| RAM | 32 GB+ |
| Disk | 60 GB |
| Template | **pytorch_cuda** |
| Open ports | 22 (SSH) |

> **RTX 5090 requires** CUDA ≥ 12.6 and PyTorch ≥ 2.6.

---

## 2. Instance Setup

### 2.1 Boot-time check

```bash
python -c "
import torch
print('torch:', torch.__version__)
print('cuda:', torch.version.cuda)
print('device:', torch.cuda.get_device_name(0))
x = torch.randn(1024, 1024, device='cuda')
(x @ x.T).sum(); torch.cuda.synchronize()
print('GPU matmul: OK')
"
```

Required: `torch >= 2.6.0`, `cuda >= 12.6`. If PyTorch is too old:
```bash
pip install --upgrade --index-url https://download.pytorch.org/whl/cu126 \
  torch torchvision
```

### 2.2 Clone + install

```bash
git clone https://github.com/rayan-rizon/prism-adversarial-defense.git prism-repo
cd prism-repo/prism
pip install -r requirements.txt
pip install git+https://github.com/fra31/auto-attack   # if autoattack fails from pip
apt-get update && apt-get install -y screen
```

### 2.3 Performance flags

```bash
cat >> ~/.bashrc <<'EOF'
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVIDIA_TF32_OVERRIDE=1
export TORCH_CUDNN_V8_API_ENABLED=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
EOF
source ~/.bashrc
```

---

## 3. Run the Full Pipeline (one command)

```bash
screen -S prism_run
bash run_vastai_full.sh
# Ctrl-A D to detach and let it run
```

**Steps executed by `run_vastai_full.sh`:**

| Step | Script | Wall-clock | Notes |
|------|--------|-----------|-------|
| 0 | GPU + PyTorch verification + determinism flags | <1 min | |
| 1 | Build reference profiles | ~10 min | |
| 2+2c+2d | **Parallel training**: ensemble + no-TDA variant + experts | ~30 min | Was 85 min sequential; saves ~55 min |
| 2b | Post-retrain verification gate | <1 min | |
| 3 | Calibrate conformal thresholds | ~3 min | |
| 4 | FPR gate check (abort if fail) | ~2 min | |
| 5A+5B+6+7+6b | **All parallel**: CW + Fast + Adaptive PGD + Ablation + L0 cal | ~1 h | Adaptive PGD is now the bottleneck; CW finished in ~35m |
| 7a+7b+7c | **All parallel**: Campaign + Recovery + Baselines | ~1 h | |
| 8 | Paper tables + reproducibility manifest (SHA256) | <5 min | |
| **Total** | | **~2 h 35 min** | Faster CW engine saves ~2 hours |

---

## 3b. Local Pre-Flight Validation (run this BEFORE Vast.ai)

`run_local_full.sh` is a CPU-scale dry-run that exercises the entire pipeline
end-to-end on one seed at reduced n — validating every algorithmic component
(train → calibrate → L0 cal → campaign → recovery → ablation → gate check)
before you spend the ~8h Vast.ai budget.

### Coverage vs. the other scripts

| Phase | Smoke test | **Local full** | Vast.ai |
|-------|:----------:|:-----------:|:-------:|
| Build reference profiles | ✅ | ✅ | ✅ |
| Train ensemble scorer | ✅ | ✅ | ✅ |
| Calibrate conformal thresholds | ✅ | ✅ | ✅ |
| Train experts (TAMSH) | ❌ | ✅ | ✅ |
| Validation FPR gate | ✅ | ✅ | ✅ |
| L0 threshold calibration (P0.4) | ❌ | ✅ | ✅ |
| Standard eval (FGSM TPR, etc.) | ✅ | ✅ | ✅ |
| Campaign eval (P0.4) | ❌ | ✅ | ✅ |
| Recovery eval (P0.5) | ❌ | ✅ | ✅ |
| Ablation (P0.6) | ❌ | ✅ | ✅ |
| Gate check with exit code | partial | ✅ | ✅ |

### Usage

```bash
cd prism/

# Default — n=100, seed=42, trains from scratch (~2.5–3.5 h on M-series)
bash run_local_full.sh

# Higher-confidence dry run (n=200, ~5 h)
bash run_local_full.sh 200

# Skip training if models/*.pkl already exist from a previous run (~45 min)
bash run_local_full.sh 100 --skip-train
```

### Local deviations (algorithm identical; only scale reduced)

| Parameter | Local | Vast.ai |
|-----------|-------|---------|
| n-train (ensemble) | 500 | 4000 |
| n-test (eval) | 100 | 1000 |
| Seeds | 1 (seed=42) | 5 |
| Attacks | FGSM + PGD + Square | + CW + AutoAttack |
| Campaign scenarios | 2 (clean + sustained_ρ=1.0) | 6 |
| Ablation n | 50 / attack | 500 / attack |

### Exit codes

| Code | Meaning | Action |
|------|---------|--------|
| 0 | All P0.4 / P0.5 / P0.6 gates pass | Safe to proceed to Vast.ai |
| 1 | Setup / training / calibration error | Fix error; re-run from scratch |
| 3 | Gate miss (artifacts written) | Review gate_check output; retune config before Vast.ai |

> **Rule of thumb:** if exit code is 3 locally (gate miss at n=100), do NOT
> launch Vast.ai. Fix the underlying issue first — gates are soft at n=100 but
> the direction of the miss tells you where the problem is.

---

## 4. Manual Step-by-Step (if you prefer granular control)

### 4.1 Build profiles

```bash
mkdir -p logs models experiments/calibration experiments/evaluation
python scripts/build_profile_testset.py 2>&1 | tee logs/build_profile.log
```

### 4.2 Retrain ensemble (CW + AutoAttack in training mix)

```bash
python scripts/train_ensemble_scorer.py \
  --n-train 4000 \
  --fgsm-oversample 2.5 \
  --include-cw \
  --include-autoattack \
  --cw-max-iter 30 \
  --cw-bss 3 \
  --output models/ensemble_scorer.pkl \
  2>&1 | tee logs/retrain.log
```

**Verify the retrain worked** — this is critical for FGSM and CW TPR:
```bash
python -c "
import pickle, sys
d = pickle.load(open('models/ensemble_scorer.pkl', 'rb'))
# save() serialises a dict — use .get(), not getattr()
assert isinstance(d, dict), f'wrong pkl format: {type(d).__name__}'
ta = list(d.get('training_attacks', []))
ng = bool(d.get('use_grad_norm', False))
# n_features now saved directly; fallback computes from flags for old pkls
nf = d.get('n_features')
if nf is None:
    base = len(d.get('layer_names', [])) * len(d.get('dims', [])) * 6
    nf   = base + int(d.get('use_dct', False)) + int(d.get('use_grad_norm', False))
print('training_attacks:', ta)
print('use_grad_norm:', ng)
print('n_features:', nf)
assert 'CW' in ta, 'ERROR: CW not in training mix!'
assert 'AutoAttack' in ta, 'ERROR: AutoAttack not in training mix!'
assert not ng, 'ERROR: grad-norm must be OFF (regression risk)'
assert nf == 37, f'ERROR: expected 37 features, got {nf}'
print('RETRAIN CHECK: PASS')
"
```

Expected output:
```
training_attacks: ['FGSM', 'PGD', 'Square', 'CW', 'AutoAttack']
use_grad_norm: False
n_features: 37
RETRAIN CHECK: PASS
```

> [!IMPORTANT]
> **FGSM Oversample:** **2.5** is the locked research-plan value (P0.3). Earlier commits tested 1.8/2.0 and dropped FGSM TPR from 0.87 to 0.806 — this is a regression, not an acceptable operating point. `sanity_checks.py` Check 6 enforces `fgsm_oversample >= 2.5`.
> **Grad-Norm:** This feature was tested and **REVERTED** (April 22). It inflated calibration thresholds by 20%, dropping FGSM TPR to 63%. Do not enable it.

### 4.3 Calibrate + gate

```bash
python scripts/calibrate_ensemble.py 2>&1 | tee logs/calibrate.log
python scripts/compute_ensemble_val_fpr.py 2>&1 | tee logs/val_fpr.log
```

All three tiers must read `passed: true`. If any fail:

| Tier | Target FPR | Fix |
|------|------------|-----|
| L1 | ≤ 0.10 | Lower `tier_cal_alpha_factors.L1` by 0.05 in `configs/default.yaml` |
| L2 | ≤ 0.03 | Lower `tier_cal_alpha_factors.L2` by 0.05 |
| L3 | ≤ 0.005 | Lower `tier_cal_alpha_factors.L3` by 0.05 |

**After this step, `ensemble_scorer.pkl` + `calibrator.pkl` are LOCKED.
Do NOT retrain or recalibrate between attack runs.**

### 4.4 Full parallel evaluation (Steps 5 + 6 + 7 simultaneously)

> ⚠️ This section is historical (pre-research-plan). For the submission campaign,
> use `bash run_vastai_full.sh` which wraps the 14-step Appendix §A1 pipeline
> including parallel 7a/7b/7c phases. The commands below are kept for manual
> debugging only.

All three evaluation phases read only locked frozen artifacts — launch them all
at once to overlap Step 6 + 7 with the CW bottleneck.

```bash
# Step 5A: CW — research-plan canonical (max_iter=40, bss=5, bs=256 via chunk=128)
python experiments/evaluation/run_evaluation_full.py \
  --n-test 1000 --attacks CW \
  --multi-seed --seeds 42 123 456 789 999 \
  --cw-max-iter 40 --cw-bss 5 --cw-chunk 128 \
  --checkpoint-interval 100 \
  --output experiments/evaluation/results_cw_n1000_ms5.json \
  2>&1 | tee logs/cw_ms5.log &
PID_CW=$!

# Step 5B: FGSM + PGD + Square + AutoAttack
python experiments/evaluation/run_evaluation_full.py \
  --n-test 1000 --attacks FGSM PGD Square AutoAttack \
  --multi-seed --seeds 42 123 456 789 999 \
  --gen-chunk 128 --square-max-iter 5000 \
  --aa-version standard --aa-chunk 64 \
  --checkpoint-interval 100 \
  --output experiments/evaluation/results_fast_n1000_ms5.json \
  2>&1 | tee logs/fast_ms5.log &
PID_FAST=$!

# Step 6: Adaptive PGD — all 5 seeds in parallel
STEP6_PIDS=""; STEP6_SEEDS=""
for s in 42 123 456 789 999; do
  python experiments/evaluation/run_adaptive_pgd.py \
    --n-test 1000 --seed $s \
    --output experiments/evaluation/results_adaptive_pgd_seed${s}.json \
    2>&1 | tee logs/adaptive_pgd_seed${s}.log &
  STEP6_PIDS="$STEP6_PIDS $!"; STEP6_SEEDS="$STEP6_SEEDS $s"
done

# Step 7: Ablation
python experiments/ablation/run_ablation_paper.py \
  --n 1000 --multi-seed --seeds 42 123 456 789 999 \
  --attacks FGSM PGD Square CW \
  2>&1 | tee logs/ablation.log &
PID_ABLATION=$!

echo "All processes running — monitoring:"
echo "  tail -f logs/cw_ms5.log"
echo "  tail -f logs/adaptive_pgd_seed42.log"

# Wait Step 5 first (provenance check needs its JSON output)
wait $PID_CW && wait $PID_FAST
echo "Step 5 complete"

# Wait Step 6
set -- $STEP6_PIDS
for s in $STEP6_SEEDS; do
  pid=$1; shift; wait $pid && echo "  Seed $s: done" || echo "  Seed $s: FAILED"
done

# Wait Step 7
wait $PID_ABLATION && echo "Ablation: done" || echo "Ablation: FAILED"
```

> **Note:** Ablation uses FGSM/PGD/Square/CW. The script does not support
> AutoAttack internally. AutoAttack coverage is in Table 1 (§4.4).

---

## 4b. Resuming from a Failed Step (without restarting from scratch)

### Step 2b failed — can I skip Step 2 and continue?

**Yes — if Step 2 printed `[OK] Ensemble scorer trained and saved.` in its log**, the pkl is correct. Step 2b is a verification gate that reads the pkl but writes nothing. The failure was a code bug in the verification script, not a training failure.

**To resume:**

```bash
# 1. Pull the fix (the getattr → isinstance/d.get correction)
cd /workspace/prism-repo && git pull && cd prism

# 2. Re-run Step 2b check only (30 seconds)
python -c "
import pickle, sys
d = pickle.load(open('models/ensemble_scorer.pkl', 'rb'))
assert isinstance(d, dict), f'wrong pkl format: {type(d).__name__}'
ta = list(d.get('training_attacks', []))
ng = bool(d.get('use_grad_norm', False))
nf = d.get('n_features')
if nf is None:
    base = len(d.get('layer_names', [])) * len(d.get('dims', [])) * 6
    nf   = base + int(d.get('use_dct', False)) + int(d.get('use_grad_norm', False))
errors = []
if 'CW' not in ta:          errors.append(f'CW missing: {ta}')
if 'AutoAttack' not in ta:  errors.append(f'AutoAttack missing: {ta}')
if ng:                       errors.append('use_grad_norm=True — must be OFF')
if nf != 37:                 errors.append(f'n_features={nf}, expected 37')
if errors:
    print('FAIL:', errors); sys.exit(1)
print(f'PASS  training_attacks={ta}  n_features={nf}')
"

# 3. If PASS → continue from Step 3 directly
python scripts/calibrate_ensemble.py 2>&1 | tee logs/step3_calibrate.log
```

If Step 2b still fails after pulling (e.g. `training_attacks: []`): the pkl on
the instance is from before `training_attacks` was added to `save()`. Delete it
and re-run Step 2 (~40 min):

```bash
rm models/ensemble_scorer.pkl
python scripts/train_ensemble_scorer.py \
  --n-train 4000 --fgsm-oversample 2.5 \
  --include-cw --include-autoattack \
  --cw-max-iter 30 --cw-bss 3 \
  --output models/ensemble_scorer.pkl \
  2>&1 | tee logs/step2_retrain.log
```

### General resume guide

| Failed step | Safe to skip | Resume from |
|-------------|-------------|-------------|
| Step 2b only | ✅ Yes — pkl is fine | Step 3, after `git pull` + manual 2b check above |
| Step 2 (train crashed) | ❌ No | Re-run Step 2 from scratch |
| Step 3 (calibrate) | ❌ No | Re-run Step 3 |
| Step 4 (gate fail) | ❌ No | Lower `tier_cal_alpha_factors`, re-run Steps 3–4 |
| Step 5/6/7 (eval crash) | ✅ Partially | Re-run only the failed seed/attack; artifacts are still locked |

---

## 5. Monitoring Progress

### 5.1 Live log tailing

```bash
# CW progress — prints every ~8 images (~30-60s per line on GPU)
tail -f logs/cw_ms5.log | grep -E "\[gen\]|\[Checkpoint\]|TPR=|FPR=|TARGET|GATE"

# Fast attacks progress
tail -f logs/fast_ms5.log | grep -E "\[gen\]|\[AA gen\]|\[Checkpoint\]|TPR=|FPR=|GATE"

# GPU utilization (open a second SSH session)
watch -n 2 nvidia-smi
```

### 5.2 What healthy CW output looks like

```
============================================================
Attack: CW
============================================================
  CW running on CUDA (ART classifier on GPU)
  CW params: max_iter=100, bss=9, estimated ~45 min per seed
  CW generation: chunk=64, ~150s per chunk, ~45 min total
  Generating 1000 adversarial examples (chunk=64)...
    [gen] 64/1000 elapsed=150.3s  rate=2.35s/img  eta=2199.3s
    [gen] 128/1000 elapsed=300.1s  rate=2.34s/img  eta=2048.5s
  ...
  [Checkpoint 100/1000] TPR=0.8700 (✅) | FPR=0.0790 (✅) | F1=0.9231
```

> **If rate > 15 s/img:** CW is running on CPU, not GPU. Check that CUDA is
> available (`nvidia-smi` must show the process). On GPU, expect ~2–3 s/img.

### 5.3 Target metric gate (printed automatically after each multi-seed run)

After all 5 seeds finish for an attack group, the script automatically prints:

```
=================================================================
TARGET METRIC GATE
─────────────────────────────────────────────────────────────────
  ✅ FGSM         TPR=0.8676  target≥0.85  CI=[0.8579, 0.8767]
  ✅ PGD          TPR=1.0000  target≥0.90  CI=[0.9992, 1.0000]
  ✅ Square       TPR=0.9256  target≥0.85  CI=[0.9180, 0.9326]
  ✅ L1           FPR=0.0794  target≤0.100
  ✅ L2           FPR=0.0220  target≤0.030
  ✅ L3           FPR=0.0030  target≤0.005
  ✅ Latency      mean=52.1ms  target<100ms
─────────────────────────────────────────────────────────────────
✅ GATE RESULT: ALL TARGETS MET — results are publishable.
=================================================================
```

**If you see `❌ GATE RESULT: FAIL`:** Stop the CW process (which is still
running in parallel) with `kill $PID_CW`, check the failure reason, and do
not continue to adaptive PGD or ablation until fixed.

---

## 6. Download Results

From your laptop:

```bash
mkdir -p experiments/evaluation experiments/ablation logs models

# Table 1: multi-seed evaluation JSONs
scp -P <port> \
  root@<ip>:/workspace/prism-repo/prism/experiments/evaluation/results_*_ms5.json \
  experiments/evaluation/

# Adaptive PGD
scp -P <port> \
  root@<ip>:/workspace/prism-repo/prism/experiments/evaluation/results_adaptive_pgd_*.json \
  experiments/evaluation/

# Table 2: ablation
scp -P <port> \
  "root@<ip>:/workspace/prism-repo/prism/experiments/ablation/results_*" \
  experiments/ablation/

# Locked artifacts (needed for reproducing locally)
scp -P <port> \
  root@<ip>:/workspace/prism-repo/prism/models/ensemble_scorer.pkl \
  models/ensemble_scorer_vastai.pkl
scp -P <port> \
  root@<ip>:/workspace/prism-repo/prism/models/calibrator.pkl \
  models/calibrator_vastai.pkl

# Logs + manifest
scp -P <port> \
  "root@<ip>:/workspace/prism-repo/prism/logs/*" logs/
```

---

## 7. Post-Run Verification Checklist

> **Note:** Checklist applies to the output of the current run (fresh JSONs
> produced by `run_vastai_full.sh`). Any file under
> `prism_results/archive_pre_research_plan/` is frozen pre-research-plan state
> and is **not** authoritative.

- [ ] Retrain check passed: `training_attacks` includes `CW` and `AutoAttack`
- [ ] Sanity Check 6 (FGSM-oversample regression gate): `fgsm_oversample >= 2.5`, `use_grad_norm == False`
- [ ] FPR gate: all three tiers PASS in `experiments/calibration/ensemble_fpr_report.json`
- [ ] CW results — pooled TPR ≥ 0.85 across 5 seeds
- [ ] Fast-attack results — FGSM ≥ 0.85, PGD ≥ 0.90, Square ≥ 0.85, AA ≥ 0.90
- [ ] Pooled per-tier FPR: L1 ≤ 0.10, L2 ≤ 0.03, L3 ≤ 0.005
- [ ] Latency `_meta.latency.pass == true` (expect ~50 ms on RTX 5090)
- [ ] Adaptive PGD results exist for all 5 seeds (λ ∈ {0, 0.5, 1, 2, 5, 10})
- [ ] Ablation JSON includes ensemble-no-TDA arm (P0.6)
- [ ] Campaign-stream JSONs — `sustained_rho100.asr_gap_pp ≥ 10`, `clean_only.l0_on.l0_active_fraction ≤ 0.01` (P0.4)
- [ ] Recovery JSONs — `tamsh.recovery_accuracy − passthrough.recovery_accuracy ≥ 0.15` (P0.5)
- [ ] Baselines JSONs — LID / Mahalanobis / ODIN / Energy × 5 seeds (P0.2)
- [ ] `paper/tables/*.tex` — 6 LaTeX files generated (P0.7)
- [ ] `logs/manifest.json` SHA256 matches across all result files

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Step 2b: `training_attacks: []` | Stale code on instance (used `getattr` on dict pkl) | `git pull`, then re-run Step 2b check — see §4b |
| Step 2b: `n_features=0, expected 37` | `n_features` not saved in old pkl (pre-fix) | `git pull` (fix adds fallback computation) — see §4b |
| Step 2b: `pkl is not a dict` | Very old pkl pickled the class object directly | Delete pkl, re-run Step 2 |
| FGSM TPR ~63 % | grad-norm enabled | **REVERT: remove --use-grad-norm.** Feature is non-discriminative but inflates thresholds. See `regression_analysis_20260422.md`. |
| FGSM TPR ~80 % | Training mix dilution | Ensure `--fgsm-oversample 2.5` (locked research-plan value). 1.8/2.0 is a regression and fails sanity Check 6. |
| CW TPR ~10 % | Ensemble not retrained with CW | Run retrain verify check (§4.2); re-run `train_ensemble_scorer.py --include-cw` |
| CW rate > 15 s/img | Running on CPU | Check `nvidia-smi`; ensure CUDA available; `--device cuda` |
| CW shows no output | gen_chunk was too large | Fixed: defaults to 64 for CW for full GPU occupancy (batch_size=128). Use `--cw-chunk` to configure. |
| Latency ❌ in log | CPU-only run (expected ~140 ms on CPU) | On GPU, latency is ~50 ms ✅ — this is a non-issue on Vast.ai |
| FPR gate fails | Cal thresholds too tight | Lower `tier_cal_alpha_factors` by 0.05 in `configs/default.yaml`, re-run §4.3 |
| GATE: FAIL after fast attacks | TPR below target | Check if retrain was done; check FPR gate passed first |
| OOM on parallel eval | Too many models in VRAM | Use sequential: remove `&` from process launches |
| Screen not found | Not in Docker image | `apt-get install -y screen` |
| autoattack import error | Not installed | `pip install git+https://github.com/fra31/auto-attack` |
| `PRISM.from_saved` raises `FileNotFoundError` for ensemble_path | Expected — this is now a hard error when an explicit path is given but the file is missing. Silently falling back to baseline TopologicalScorer was masking misconfigured runs. | Either train `ensemble_scorer.pkl` first (`scripts/train_ensemble_scorer.py`) or pass `ensemble_path=None` to use the Wasserstein-only baseline. |
| Recovery accuracy 0 % for all strategies in Step 7b | `experts.pkl` is stale (trained before differentiated expert fix) | Re-train experts: `python scripts/train_experts.py --n-train 4000 --epochs 5`. Step 2d in `run_vastai_full.sh` does this automatically. |
| Medoid expert index always 0 (silent) | Pre-fix bug: gudhi inf-valued features caused `NaN` in argmin, which silently resolved to index 0, giving all experts the same medoid | Pull latest — `train_experts.py` now filters non-finite lifetimes before summing total persistence. |
| Gate-check prints MISS but pipeline exits 0 | Pre-fix bug: gate-check had no `sys.exit(1)`, so bash never saw a non-zero code | Pull latest — gate-check now calls `sys.exit(1)` on miss; `run_vastai_full.sh` captures this as exit code 3. |
| P0.4 ASR gap = 0 at n=100 (local run) | CADG catches 100 % of PGD-40 at small n → both l0_on and l0_off show 0 % ASR | Expected at n=100; run at n ≥ 200 locally or rely on Vast.ai n=1000. Gap is measurable when some adversarials slip through. |

---

# Appendix — Research-Plan Pipeline Update (v2)

This appendix **supersedes the 7-step pipeline in §Pipeline Overview** for the
NeurIPS/ICLR/ICML research-standard campaign. The original 7 steps are
preserved above for reference, but the full submission campaign executes the
14-step pipeline below. Work items are tagged with their plan IDs (P0.1–P1.4).

## A1. Expanded 14-Step Pipeline

Steps marked **[BG]** run as background subshells (parallel). Steps marked **[FG]**
are foreground and gate the next sequential constraint. Wall-clock: ~2 h 35 min
total on RTX 5090 with the current parallelism map.

```
PHASE 0 — TRAINING (wall-clock ≈ 30 min; was 85 min sequential)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 1.    [FG]  Build reference profiles            scripts/build_profile_testset.py
 2.    [FG]  Train ensemble scorer               scripts/train_ensemble_scorer.py
                                                   --fgsm-oversample 2.5 --include-cw --include-autoattack
 2c.   [BG]  Train ensemble-no-TDA variant       scripts/train_ensemble_scorer.py --no-tda-features
                                                   → models/ensemble_no_tda.pkl   (P0.6)
 2d.   [BG]  Train differentiated experts        scripts/train_experts.py
                                                   → models/experts.pkl            (P0.5)
         └─ all three join before Step 3 (2+2c+2d gate together)
 2b.   [FG]  Post-retrain verification gate      (inline python -c check)

PHASE 0 LOCK — calibration runs sequentially; artifacts sealed after Step 4
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 3.    [FG]  Calibrate conformal thresholds      scripts/calibrate_ensemble.py
 4.    [FG]  FPR gate check (abort on fail)      scripts/compute_ensemble_val_fpr.py

PHASE 1 — ATTACKS + L0 CAL (wall-clock ≈ 2 h; CW bottleneck)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 5A.   [BG]  CW-L2 (P0.1)                       run_evaluation_full.py  40 iter×5 bss×bs=256, 5 seeds
 5B.   [BG]  FGSM+PGD+Square+AutoAttack         run_evaluation_full.py  5 seeds
 6.    [BG]  Adaptive PGD × 5 seeds in parallel  run_adaptive_pgd.py     λ sweep, 100 steps×10 restarts
 7.    [BG]  Ablation (FGSM+PGD+Square+CW)      run_ablation_paper.py   5 seeds
 6b.   [BG]  L0 threshold calibration (P0.4)    calibrate_l0_thresholds.py
                Grid-searches (hazard_rate, alert_run_prob, warmup_steps) on real scorer streams.
                Writes models/l0_thresholds.pkl; CampaignMonitor auto-discovers + overlays it.
                Aborts non-zero if no feasible cell (clean FPR ≤ 1 %) found.
                ← runs hidden behind the ~2h CW bottleneck; adds ~0 wall-clock
         └─ 5A+5B+6+7+6b all join after CW finishes

PHASE 2 — SECONDARY EVAL (wall-clock ≈ 1 h)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 7a.   [BG]  Campaign-stream eval (P0.4)         run_campaign_eval.py    6 scenarios × 5 seeds
 7b.   [BG]  L3-recovery eval (P0.5)             run_recovery_eval.py    3 strategies × 5 seeds
 7c.   [BG]  Baseline detectors (P0.2)           run_baselines.py        LID/Mahalanobis/ODIN/Energy
         └─ 7a+7b+7c join → combined gate-check (sys.exit(1) on miss → bash exit 3)
 13.   [FG]  Paper tables (P0.7)                 scripts/build_paper_tables.py --out-dir paper/tables
 14.   [FG]  Manifest + SHA256                   (inline, Step 8 of run_vastai_full.sh)
```

### Parallel Launch Map (background PID assignments)

```
PHASE 0 training simultaneous launch:
  PID_2C (ensemble-no-TDA) ──┐
  PID_2D (experts)          ──┼── joined before Step 2b; crash-safe: kill on Step 2 fail
  Step 2 foreground (gates) ──┘

PHASE 1 attacks simultaneous launch (after Step 4 LOCK):
  PID_CW   (Step 5A CW)           ──┐
  PID_FAST (Step 5B fast attacks) ──┤
  PID_6x   (Step 6, 5 seeds)      ──┼── joined in wait loop; STEP6_FAIL captured
  PID_ABL  (Step 7 ablation)      ──┤
  PID_6B   (Step 6b L0 cal)       ──┘

PHASE 2 simultaneous launch (after Phase 1 join):
  PID_7A (campaign  → experiments/campaign/)  ──┐
  PID_7B (recovery  → experiments/recovery/)  ──┼── joined → gate-check → paper tables
  PID_7C (baselines → experiments/baselines/) ──┘
```

Peak VRAM on RTX 5090 (32 GB): ~19 GB Phase 0, ~22 GB Phase 1, ~6.5 GB Phase 2.
Seeds remain serial within each suite to keep logs readable.

All 14 steps must execute against `configs/default.yaml` (CIFAR-10). For
P1.1, re-run steps 1–13 against `configs/cifar100.yaml` (see §A4 below);
artifacts land in `models/cifar100/*.pkl` and `prism_results/cifar100/...`
so the CIFAR-10 run is not clobbered.

## A2. Locked Attack & Training Configurations

| Config | Value | Rationale |
|---|---|---|
| FGSM oversample | **2.5** | Restores pre-regression value (commits `dadf2cf` → `cf854f0` dropped to 1.8/2.0; FGSM TPR fell from 0.87 to 0.806) |
| CW iter | **40** | RobustBench detector-eval norm; balances ℓ₂ attack strength vs. GPU-hours |
| CW binary search steps | **5** | Standard Carlini 2017 parameterization |
| CW batch size | **256** | Fills a single 24 GB card; reduces wall-clock ~6× vs. bs=32 |
| Adaptive PGD λ sweep | **{0.0, 0.5, 1.0, 2.0, 5.0, 10.0}** | Athalye/Carlini; confirms no collapse at high λ |
| Adaptive PGD steps × restarts | **100 × 10** | Strong-attack standard for detector papers |
| Adaptive PGD EOT samples | **1** (verified) | Hash subsample is deterministic; EOT should be no-op, verify once |
| Seeds | **{42, 123, 456, 789, 999}** | 5-seed pooled, Wilson CI, paired t-test across seeds |
| Eval n per seed | **1000** | Pooled n=5000, width of 95% Wilson CI ≈ ±1pp at TPR=0.90 |

## A3. Go/No-Go Gates (extended)

Existing gates (§5 of main guide) remain in force. New gates:

| Gate | Threshold | Result file | Action on miss |
|---|---|---|---|
| **P0.1** CW-L2 TPR (pooled 5-seed) | ≥ 0.85 | `results_cw_n1000_ms5.json` | Report honestly; discuss in FGSM-shortfall section. Do **not** drop ℓ₂ from threat model. |
| **P0.2** Baselines complete | 4 methods × 5 seeds × N attacks | `results_baselines_*.json` | Block submission — no detection paper ships without reproduced baselines on matched splits. |
| **P0.3** FGSM TPR (pooled) | ≥ 0.85 | `results_fast_n1000_ms5.json` | **Blocks NeurIPS/ICLR submission.** If the new 2.5-oversample run misses 0.85, investigate training-mix / calibration regression — do **not** accept the old 0.806 number or cherry-pick seed 42. Fallback is the AAAI/UAI venue path with an honest FGSM-shortfall section. |
| **P0.4** Campaign ASR gap (sustained ρ=1.0, L0-off − L0-on) | ≥ 10 pp | `results_campaign_*.json` | Demote C3 (SACD) to appendix; cut from contributions list. |
| **P0.4** Clean-stream L0 false-alarm | ≤ 1 % | `results_campaign_clean_only_*.json` | Retune BOCPD priors (`mu0`, `beta0`) in `configs/default.yaml::campaign`; re-run. |
| **P0.5** TAMSH recovery gap (TAMSH − passthrough, L3-triggered) | ≥ 15 pp | `results_recovery_*.json` | Demote C4 (TAMSH) to appendix; cut from contributions list. |
| **P0.6** Ensemble-no-TDA Δ (Full − no-TDA, TPR) | ≥ 3 pp | `results_ablation_multiseed.json` | Cut C1 (TAMM) novelty claim; reframe ensemble as primary detector. |

Gates that **block submission**: P0.2, the FPR tier gate (existing), and the
coverage regression test (`tests/test_research_gates.py::TestConformalCoverageRegression`).
All other gates determine **venue**, not submission (see §A5).

## A4. CIFAR-100 Variant (P1.1)

Config: `configs/cifar100.yaml` (already in tree). Same ResNet-18, same
`layer_weights`, same 5000/2000/1000/2000 split shape as CIFAR-10; artifacts
at `models/cifar100/*.pkl`. Execution order matches §A1 steps 1–13, each
invoked with `--config configs/cifar100.yaml`. Launch via the sibling script:

```bash
bash run_vastai_cifar100.sh
```

Expected budget: ~1 GPU-week on a 5090-class card (bulk: CW + campaign +
recovery + baselines all need re-fitting against the CIFAR-100 clean-score
distribution). If cal→val FPR overruns target by >1 pp, tighten
`conformal.tier_cal_alpha_factors.L3` from 0.50 → 0.45 before re-running.

## A5. Venue Decision Matrix (end of week 7)

| Gates passed | Venue target | Story |
|---|---|---|
| P0.1 + P0.2 + P0.3 + P0.4 + P0.5 + P0.6 + P1.1 | **NeurIPS / ICLR** | 4-contribution story (C1–C4) supported by evidence; soften "architecture-agnostic" to "evaluated on ResNet-18 across CIFAR-10/100; architecture generalization left as future work." |
| P0.1 + P0.2 + P0.3 + P0.6 + P1.1, but P0.4 **or** P0.5 miss | **ICLR / ICML / CVPR** | Conformal-ensemble reframe; demote failing C3 or C4 to appendix; 2–3 contributions. |
| P0.1 + P0.2 + P0.3 clear, but P0.4 + P0.5 + P0.6 + P1.1 slip | **AAAI / UAI 2026** | Original 4-contribution story; broader ML audience; honest ablation framing. |

## A6. Compute Budget (added work on top of existing Step 6 budget)

| New step | GPU-days (est., 5090-class) | Notes |
|---|---|---|
| CW-L2 (P0.1, 5 seeds × n=1000 × 40 iter × 5 bss) | 1–2 | Largest single add; verify `--cw-chunk=64`, bs=256 |
| Campaign-stream (P0.4, 6 scenarios × 5 seeds) | 0.5–1 | PGD-40 adversarial pool dominates; cache pool across seeds |
| L3-recovery (P0.5, 3 strategies × 5 seeds) | 0.5–1 | TAMSH routing is single forward pass; pool re-used from step 10 |
| Baselines (P0.2, 4 methods × N attacks × 5 seeds) | 0.5–1 | Mahalanobis fit dominates; reuse per-seed activation cache |
| Adaptive PGD expanded (P1.4, 6 λ × 5 seeds × 100 steps × 10 restarts) | 1 | EOT-verify pass is cheap |
| Ensemble-no-TDA arm (P0.6, 1 retrain + 5-seed eval) | 0.25 | Retrain only step 2b; reuse existing adv pool |
| CIFAR-100 full repeat (P1.1) | 5–7 | Includes all above, top to bottom |

**Updated parallelism schedule (run_vastai_full.sh):**

| Phase | Jobs | Wall-clock | vs. old sequential |
|-------|------|-----------|-------------------|
| Phase 0 training (2+2c+2d) | 3 parallel | ~30 min | was ~85 min; saves **55 min** |
| Phase 0 lock (calibrate+gate) | sequential | ~5 min | unchanged |
| Phase 1 attacks + L0 cal | 5+1 parallel | ~2 h | L0 cal hidden in CW gap; saves **10–15 min** |
| Phase 2 secondary eval | 3 parallel | ~1 h | saves **40–50 % of phase** |
| **Total** | | **~2 h 35 min** | was ~3 h 45 min; **~30 % faster** |

**Serial wall-clock for the full CIFAR-10 campaign (Vast.ai):** ~2.5–3 h.
**With CIFAR-100 repeat (P1.1):** add ~5–7 GPU-days.
All scripts respect the `&` launch pattern and screen-session logging convention documented in §4.

## A7. Reproducibility Endpoint

`scripts/build_paper_tables.py` (P0.7) is the end-point that downstream users
and reviewers should run. It reads every `prism_results/experiments/**/*.json`
produced by steps 6–12 and emits six LaTeX-ready tables with pooled Wilson
95% CIs:

```
paper/tables/main_attacks.tex     (step  6 + 7 + 12)
paper/tables/adaptive_pgd.tex     (step  8)
paper/tables/ablation.tex         (step  9)
paper/tables/baselines.tex        (step 12)
paper/tables/campaign.tex         (step 10)
paper/tables/recovery.tex         (step 11)
```

A reviewer reproducing numbers should: `bash run_vastai_full.sh` →
`python scripts/build_paper_tables.py` → `sha256sum -c logs/manifest.json`.
Any divergence from paper numbers is a bug to triage, not a retraining excuse.
