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
| 2 | Retrain ensemble (n=4000, CW+AA, **FGSM-os=2.5**) | ~40 min |
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

| Step 2b: `training_attacks: []` | Stale code on instance (used `getattr` on dict pkl) | `git pull`, then re-run Step 2b check — see §4b |
| Step 2b: `n_features=0, expected 37` | `n_features` not saved in old pkl (pre-fix) | `git pull` (fix adds fallback computation) — see §4b |
| Step 2b: `pkl is not a dict` | Very old pkl pickled the class object directly | Delete pkl, re-run Step 2 |
| FGSM TPR ~63 % | grad-norm enabled | **REVERT: remove --use-grad-norm.** Feature is non-discriminative but inflates thresholds |
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

---

# Appendix — Research-Plan Pipeline Update (v2)

This appendix **supersedes the 7-step pipeline in §Pipeline Overview** for the
NeurIPS/ICLR/ICML research-standard campaign. The original 7 steps are
preserved above for reference, but the full submission campaign executes the
14-step pipeline below. Work items are tagged with their plan IDs (P0.1–P1.4).

## A1. Expanded 14-Step Pipeline

```
 1. Build reference profiles            (scripts/build_profile_testset.py)
 2. Train ensemble scorer               (scripts/train_ensemble_scorer.py --fgsm-oversample 2.5)
 2b.Train ensemble-no-TDA variant       (P0.6:  --no-tda-features --output models/ensemble_no_tda.pkl)
 3. Train differentiated experts        (P0.5:  scripts/train_experts.py   — only if audit shows homogeneity)
 4. Calibrate conformal thresholds      (scripts/calibrate_ensemble.py)
 5. FPR gate check                      (scripts/compute_ensemble_val_fpr.py — abort on fail)
 6. Fast attacks (FGSM/PGD/Square/AA)   (experiments/evaluation/run_evaluation_full.py, 5 seeds, parallel)
 7. CW-L2                               (P0.1:  40 iter × 5 bss × bs=256 via --cw-chunk=128, 5 seeds)
 8. Adaptive PGD                        (P1.4:  λ ∈ {0, 0.5, 1, 2, 5, 10}, 100 steps × 10 restarts, EOT=1)
 9. Ablation                            (existing arms + P0.6 ensemble-no-TDA)
**6b. L0 threshold calibration**        (P0.4 lever: scripts/calibrate_l0_thresholds.py — runs AFTER steps 6/7/8/9 join, BEFORE 10/11/12)
   - Grid-searches (hazard_rate, alert_run_prob, warmup_steps) on real scorer streams
   - Writes models/l0_thresholds.pkl; CampaignMonitor auto-discovers and overlays it
   - Aborts with non-zero exit if no feasible cell (clean FPR ≤ 1%) found
10. Campaign-stream eval                (P0.4:  run_campaign_eval.py — 6 scenarios × 5 seeds)
11. L3-recovery eval                    (P0.5:  run_recovery_eval.py — 3 strategies × 5 seeds)
12. Baseline detectors                  (P0.2:  run_baselines.py --methods lid mahalanobis odin energy)
13. Paper tables                        (P0.7:  scripts/build_paper_tables.py --out-dir paper/tables)
14. Manifest + SHA256                   (embedded in run_vastai_full.sh Step 8 — covers all artifacts)
```

### Parallel Launch Map (phases 10+11+12)

Steps 10 (campaign), 11 (recovery), and 12 (baselines) all read-only the frozen
`ensemble_scorer.pkl` + `calibrator.pkl` + `experts.pkl` + `reference_profiles.pkl`.
They have zero write contention (separate output dirs). `run_vastai_full.sh`
launches them concurrently via background subshells:

```
PID_7A (campaign  → experiments/campaign/)    ──┐
PID_7B (recovery  → experiments/recovery/)    ──┼── wait all three
PID_7C (baselines → experiments/evaluation/)  ──┘   → run combined gate-check
                                                    → then step 13 (paper tables)
```

Peak VRAM on RTX 5090 (32 GB): ~6.5 GB with all three live (ResNet-18 activations
cached once per process). Wall-clock saving vs. sequential: ~40–50 % of phase-B.
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

**Serial wall-clock:** ~3 weeks end-to-end. **With parallelism (§Run Full Pipeline):**
~1.5–2 weeks. All new scripts respect the existing `&` launch pattern and
screen-session logging convention documented in §4.

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
