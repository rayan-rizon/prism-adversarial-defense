# PRISM — Vast.ai RTX 5090 Full Pipeline Guide

> **Goal:** Run the complete publishable pipeline from scratch on a single Vast.ai
> RTX 5090 instance. Retrain → Calibrate → Gate → Evaluate (5 attacks × 5 seeds,
> parallel) → Adaptive PGD → Ablation → Download.
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
build_profile_testset.py         → reference_profiles.pkl
        │
        ▼
train_ensemble_scorer.py         → ensemble_scorer.pkl
  (--include-cw --include-autoattack)
  Step 2b: post-retrain verification gate ← catches stale pkl early
        │
        ▼
calibrate_ensemble.py            → calibrator.pkl
        │
        ▼
compute_ensemble_val_fpr.py      → FPR gate (abort if any tier fails)
        │
        ▼   ARTIFACTS LOCKED — all below are read-only consumers
        │
┌───────┼────────────────────────────────┐  ← ALL LAUNCHED SIMULTANEOUSLY
│       │                                │
│ 5A CW │ 5B FGSM+PGD+Square+AA │ 6 Adaptive PGD × 5 seeds │ 7 Ablation
│       │                                │
└───────┼────────────────────────────────┘
        │  wait Step 5 → provenance check
        │  wait Step 6 → STEP6_FAIL gate
        │  wait Step 7
        ▼
manifest.json                    → reproducibility record
```

**Why full parallelism is safe:** Steps 5, 6, and 7 are all pure read-only
consumers of the locked frozen artifacts. No inter-step file dependency.
CUDA SM time-slicing handles ≤8 concurrent light-batch processes on the 5090.
GPU utilization rises from ~25 % (sequential) to ~80–90 %. Saves ~35–40 %
wall-clock time vs the old sequential 5→6→7 schedule.

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

| Step | Script | Time |
|------|--------|------|
| 0 | GPU + PyTorch verification + determinism flags | <1 min |
| 1 | Build reference profiles | ~10 min |
| 2 | Retrain ensemble (n=4000, CW+AA, **FGSM-os=1.8**) | ~40 min |
| 2b | Post-retrain verification gate | <1 min |
| 3 | Calibrate conformal thresholds | ~3 min |
| 4 | FPR gate check (abort if fail) | ~2 min |
| 5+6+7 | **ALL PARALLEL**: CW + Fast + Adaptive PGD × 5 seeds + Ablation | ~150 min ← CW bottleneck |
| 8 | Reproducibility manifest (SHA256) | <1 min |
| **Total** | | **~3.5 h** (same ceiling; 5→6→7 now overlap) |

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
  --fgsm-oversample 1.8 \
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
> **FGSM Oversample:** 1.8 is used to restore FGSM's training share (~31.0%) and maintain TPR $\ge$ 85% with the 5-attack mix.
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

All three evaluation phases read only locked frozen artifacts — launch them all
at once to overlap Step 6 + 7 with the CW bottleneck.

```bash
# Step 5A: CW — paper-canonical (max_iter=100, bss=9), the bottleneck
python experiments/evaluation/run_evaluation_full.py \
  --n-test 1000 --attacks CW \
  --multi-seed --seeds 42 123 456 789 999 \
  --cw-max-iter 100 --cw-bss 9 \
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
  --attacks FGSM PGD Square \
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

> **Note:** Ablation uses FGSM/PGD/Square only — the script does not support
> CW/AutoAttack internally. CW and AutoAttack coverage is in Table 1 (§4.4).

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
  --n-train 4000 --fgsm-oversample 1.8 \
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
  CW generation: chunk=8, ~22s per chunk, ~45 min total
  Generating 1000 adversarial examples (chunk=8)...
    [gen] 8/1000 elapsed=21.3s  rate=2.66s/img  eta=2639.3s
    [gen] 16/1000 elapsed=43.1s  rate=2.69s/img  eta=2648.5s
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

- [ ] Retrain check passed: `training_attacks` includes `CW` and `AutoAttack`
- [ ] FPR gate: all three tiers `passed: true` in `ensemble_fpr_report.json`
- [ ] `results_cw_n1000_ms5.json` — CW TPR ≥ 0.85 across 5 seeds
- [ ] `results_fast_n1000_ms5.json` — FGSM ≥ 0.85, PGD ≥ 0.90, Square ≥ 0.85, AA ≥ 0.90
- [ ] Pooled per-tier FPR: L1 ≤ 0.10, L2 ≤ 0.03, L3 ≤ 0.005
- [ ] Latency `_meta.latency.pass == true` (expect ~50 ms on RTX 5090)
- [ ] Adaptive PGD results exist for all 5 seeds
- [ ] Ablation JSON + MD saved
- [ ] `logs/manifest.json` SHA256 matches across all result files

---

## 8. Troubleshooting

| Step 2b: `training_attacks: []` | Stale code on instance (used `getattr` on dict pkl) | `git pull`, then re-run Step 2b check — see §4b |
| Step 2b: `n_features=0, expected 37` | `n_features` not saved in old pkl (pre-fix) | `git pull` (fix adds fallback computation) — see §4b |
| Step 2b: `pkl is not a dict` | Very old pkl pickled the class object directly | Delete pkl, re-run Step 2 |
| FGSM TPR ~63 % | grad-norm enabled | **REVERT: remove --use-grad-norm.** Feature is non-discriminative but inflates thresholds |
| FGSM TPR ~80 % | Training mix dilution | Increase `--fgsm-oversample` to 1.8 |
| CW TPR ~10 % | Ensemble not retrained with CW | Run retrain verify check (§4.2); re-run `train_ensemble_scorer.py --include-cw` |
| CW rate > 15 s/img | Running on CPU | Check `nvidia-smi`; ensure CUDA available; `--device cuda` |
| CW shows no output | gen_chunk was too large | Fixed: auto-caps to 8 for CW — update from repo |
| Latency ❌ in log | CPU-only run (expected ~140 ms on CPU) | On GPU, latency is ~50 ms ✅ — this is a non-issue on Vast.ai |
| FPR gate fails | Cal thresholds too tight | Lower `tier_cal_alpha_factors` by 0.05 in `configs/default.yaml`, re-run §4.3 |
| GATE: FAIL after fast attacks | TPR below target | Check if retrain was done; check FPR gate passed first |
| OOM on parallel eval | Too many models in VRAM | Use sequential: remove `&` from process launches |
| Screen not found | Not in Docker image | `apt-get install -y screen` |
| autoattack import error | Not installed | `pip install git+https://github.com/fra31/auto-attack` |
