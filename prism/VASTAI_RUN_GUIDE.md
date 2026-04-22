# PRISM — Vast.ai RTX 5090 Run Guide (Paper-Quality, Full Pipeline)

> **Venue target (per `PRISM Implementation.md`):** UAI / AAAI 2026 with arXiv
> fallback. Publishable-standard metrics, 5-seed multi-seed evaluation,
> paper-canonical attack parameters.

## Purpose

Run the full publishable evaluation pipeline against a **single retrained
ensemble** that includes CW-L2 and AutoAttack in its training mix.

**Scope covered by this guide:**
1. Retrain ensemble with CW + AutoAttack in the training mix (one-time).
2. Re-run the conformal calibration + FPR gate.
3. **Paper Table 1:** n=1000 × **5 seeds** evaluation for all five attacks
   (FGSM, PGD, Square, CW-L2, AutoAttack) with paper-canonical params.
4. **Paper Table 2 (ablation):** full vs no-L0 vs no-MoE vs TDA-only, same
   eval pool, same 5 seeds.
5. **Optional — reviewer-required extras:** adaptive-PGD (§7.6) and LID +
   Mahalanobis baselines (§7.7) — see `PRISM Implementation.md` for the
   paper-level rationale.

**Why we re-evaluate everything (not just CW + AA):**
The retrain changes the logistic decision boundary and the calibrated L1/L2/L3
thresholds. Any FGSM/PGD/Square numbers from prior runs were computed against
the **old** ensemble (`training_attacks=['FGSM','PGD','Square']`). Mixing old
+ new numbers in one paper table would mean two different decision functions
reported as one system and would break the conformal FPR guarantee (which
holds *per fitted ensemble*). For paper consistency we re-evaluate all five
attacks against the **same** retrained ensemble.

---

## 0. Target metrics (from `PRISM Implementation.md`)

| Metric | Target |
|--------|--------|
| Mean TPR across FGSM/PGD/Square | ≥ 85 % |
| TPR on PGD + AutoAttack | ≥ 90 % |
| CW-L2 TPR | ≥ 90 % (stretch; ≥ 85 % acceptable after retrain) |
| L1 FPR | ≤ 10 % |
| L2 FPR | ≤ 3 % |
| L3 FPR | ≤ 0.5 % |
| GPU latency (mean, per image) | < 100 ms |
| Ablation paired-bootstrap p | < 0.05 (each component) |

All TPR/FPR figures reported with Wilson 95 % CIs pooled across 5 seeds.

---

## 1. Local prerequisites (do before renting)

```bash
cd prism/

# 1. Confirm all required model artifacts exist
ls -lh models/calibrator.pkl models/reference_profiles.pkl models/ensemble_scorer.pkl

# 2. Quick end-to-end smoke on the two new pipelines
python experiments/evaluation/run_evaluation_full.py \
  --n-test 4 --attacks CW --gen-chunk 4 \
  --output experiments/evaluation/results_cw_smoke.json

python experiments/evaluation/run_evaluation_full.py \
  --n-test 4 --attacks AutoAttack --aa-chunk 4 \
  --output experiments/evaluation/results_aa_smoke.json

# 3. Clean stale result JSONs so the Vast.ai output is unambiguous
#    (per Implementation.md Appendix A — results_n500_*.json and the n=300
#    results_paper.json are NOT publishable and should be deleted)
rm -f experiments/evaluation/results_n500_20260419.json \
      experiments/evaluation/results_n500_retrained_20260419.json \
      experiments/evaluation/results_paper.json
```

---

## 2. Vast.ai instance configuration

**Template:** pytorch_cuda (Vast.ai's preset — this is what you selected).

| Setting | Value |
|---------|-------|
| GPU | RTX 5090 (Blackwell sm_100, 32 GB GDDR7) |
| vCPU | 8+ |
| RAM | 32 GB+ |
| Disk | 60 GB (50 GB work + 10 GB headroom for the CUDA runtime) |
| Template | **pytorch_cuda** |
| CUDA runtime shipped by template | 12.x (verify on boot — §3.1) |
| Open ports | 22 (SSH) |

> **RTX 5090 compatibility check.** Blackwell (sm_100) requires CUDA 12.6+ and
> PyTorch 2.6+. The `pytorch_cuda` template on Vast.ai generally ships a recent
> CUDA + PyTorch, but you **must** verify on boot. If the shipped PyTorch is
> < 2.6, upgrade per §3.1 — otherwise CUDA kernels will fail with
> `no kernel image is available for execution on the device`.

### RTX 5090 throughput estimates (single-GPU, max-chunk settings from §3.3)

| Stage | Per-seed time (n=1000) |
|-------|------------------------|
| FGSM | ~1 min |
| PGD-40 | ~2 min |
| Square (5000 queries) | ~7 min |
| CW-L2 (max_iter=100, bss=9, bs=64) — **paper-canonical** | ~30 min |
| CW-L2 (max_iter=50, bss=5, bs=64) — fast fallback | ~17 min |
| AutoAttack Standard (chunk=64) | ~5 min |
| Retrain n=4000 (FGSM+PGD+Square+CW+AA mix) | ~40 min (one-time) |
| Calibrate + val-fpr gate | ~5 min (one-time) |
| Ablation n=1000 × 5 seeds | ~15 min (reuses existing adv scores) |

**Wall-clock totals, single RTX 5090:**
- With paper-canonical CW (100/9): **~4 h 30 min**
- With fast CW (50/5): **~3 h 20 min**
- Parallel 3-instance deployment (§8): **~2 h 30 min** (limited by CW)

---

## 3. Instance setup

### 3.1 Boot-time compatibility check (do this first)

```bash
# Verify PyTorch + CUDA versions shipped by the pytorch_cuda template
python -c "
import torch
print('torch:', torch.__version__)
print('cuda:', torch.version.cuda)
print('device:', torch.cuda.get_device_name(0))
print('compute capability:', torch.cuda.get_device_capability(0))
# Smoke test a tiny tensor op on GPU
x = torch.randn(1024, 1024, device='cuda')
y = (x @ x.T).sum()
torch.cuda.synchronize()
print('smoke GPU matmul: OK, sum=', float(y))
"
```

Required outputs:
- `torch >= 2.6.0`
- `cuda >= 12.6`
- `device = NVIDIA GeForce RTX 5090`
- `compute capability = (10, 0)` (Blackwell sm_100)
- Smoke matmul prints without `no kernel image` errors

If the template's PyTorch is < 2.6.0 or the smoke op fails:

```bash
pip install --upgrade --index-url https://download.pytorch.org/whl/cu126 \
  torch torchvision
```

### 3.2 Clone + install PRISM deps

```bash
git clone https://github.com/rayan-rizon/prism-adversarial-defense.git prism-repo
cd prism-repo/prism

pip install adversarial-robustness-toolbox autoattack certifi tqdm scikit-learn \
            ripser persim ruptures pot scipy
```

### 3.3 Max-GPU env vars (enable before every run)

Drop this once into `~/.bashrc` on the instance (or `source` it before the
big runs):

```bash
cat >> ~/.bashrc <<'EOF'
# PRISM / RTX 5090 performance flags
export CUBLAS_WORKSPACE_CONFIG=:4096:8           # TF32 workspace (required by CUDA 12.6)
export NVIDIA_TF32_OVERRIDE=1                    # allow TF32 on matmul + conv (2–3x on 5090)
export TORCH_CUDNN_V8_API_ENABLED=1              # cuDNN v8 frontend (better perf on sm_100)
export PYTHONUNBUFFERED=1                        # flush stdout immediately to tee
export OMP_NUM_THREADS=4                         # cap CPU threads (avoid oversubscription)
EOF
source ~/.bashrc
```

These give ~30 % additional throughput over default settings on Blackwell,
especially for the gradient-heavy CW + AA runs.

### 3.4 Upload local artifacts (run from your laptop)

```bash
# Replace <instance-ip> and <port> with the SSH target Vast.ai shows you
scp -P <port> \
  models/calibrator.pkl \
  models/reference_profiles.pkl \
  models/ensemble_scorer.pkl \
  root@<instance-ip>:/workspace/prism-repo/prism/models/

# Upload CIFAR-10 data if you have it locally (saves re-download on the instance)
scp -P <port> -r data/cifar-10-batches-py \
  root@<instance-ip>:/workspace/prism-repo/prism/data/
```

---

## 4. Retrain the ensemble (one-time)

**Why:** CW scored 3.3 % TPR on the old ensemble because the logistic was
trained only on L∞ attacks. Adding CW-L2 and AutoAttack-APGD-CE to the training
mix calibrates the boundary for both L2 and L∞ attack signatures. This
ensemble is the **single locked decision function** used for every attack in
the paper table.

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

Training CW params are deliberately lighter than eval params (`cw-max-iter 30
bss 3` vs eval `100/9`) — the goal is to teach the logistic the *shape* of CW
perturbations, not to run paper-quality CW on the training slice.

**Expected console at end:**
```
training_attacks=['FGSM', 'PGD', 'Square', 'CW', 'AutoAttack']
Held-out validation AUC (logistic component): 0.97+
```

If AUC < 0.92, re-run with `--n-train 6000`.

---

## 5. Recalibrate (one-time, mandatory after retrain)

```bash
python scripts/calibrate_ensemble.py 2>&1 | tee logs/calibrate.log
python scripts/compute_ensemble_val_fpr.py 2>&1 | tee logs/val_fpr.log
```

All three tiers must read `passed: true`:

| Tier | Target FPR | Action if fail |
|------|------------|----------------|
| L1 | ≤ 0.10 | Lower `tier_cal_alpha_factors.L1` by 0.05 in `configs/default.yaml`, re-run calibrate only |
| L2 | ≤ 0.03 | Lower `tier_cal_alpha_factors.L2` by 0.05, re-run calibrate only |
| L3 | ≤ 0.005 | Lower `tier_cal_alpha_factors.L3` by 0.05, re-run calibrate only |

**After this step, `models/ensemble_scorer.pkl` + `models/calibrator.pkl` are
LOCKED.** Do not retrain or recalibrate between attack runs — that would
invalidate cross-attack comparison and break the conformal guarantee.

---

## 6. Sanity gate — seed-42 dry run on the two slow attacks

Before burning 5× time on multi-seed, prove CW and AutoAttack work on the
retrained ensemble at full n=1000 with one seed. This is the fast-fail point.

```bash
mkdir -p logs experiments/evaluation

# CW — paper-canonical params (max_iter=100, bss=9)
screen -S cw_42
python experiments/evaluation/run_evaluation_full.py \
  --n-test 1000 \
  --attacks CW \
  --cw-max-iter 100 --cw-bss 9 \
  --gen-chunk 128 \
  --checkpoint-interval 100 \
  --seed 42 \
  --output experiments/evaluation/results_cw_n1000_seed42.json \
  2>&1 | tee logs/cw_seed42.log
# Ctrl-A D to detach

# AutoAttack — standard version
screen -S aa_42
python experiments/evaluation/run_evaluation_full.py \
  --n-test 1000 \
  --attacks AutoAttack \
  --aa-version standard --aa-chunk 64 \
  --checkpoint-interval 100 \
  --seed 42 \
  --output experiments/evaluation/results_aa_n1000_seed42.json \
  2>&1 | tee logs/aa_seed42.log
# Ctrl-A D
```

### Acceptance gate

- **CW seed-42 TPR ≥ 0.50** (we target ≥ 0.90; ≥ 0.50 confirms the retrain
  landed the L2 signature into the ensemble)
- **AutoAttack seed-42 TPR ≥ 0.90**
- **Both:** per-tier FPR within targets (L1 ≤ 0.10, L2 ≤ 0.03, L3 ≤ 0.005)
- **Latency mean < 100 ms**

If CW seed-42 TPR ≤ 0.10, **stop**:
1. Check `training_attacks` in the JSON `_meta.ensemble` block includes `CW`.
2. Re-run step 4 with `--n-train 6000` (more training signal).
3. Optionally add a higher-confidence CW variant (`confidence=5.0`) to the
   training mix.

---

## 7. Paper Table 1 — full 5-attack × 5-seed multi-seed sweep

**Canonical configuration:**
- Seeds: `42 123 456 789 999` (per `PRISM Implementation.md` §6.2)
- n=1000 per seed → 5000 pooled observations for Wilson CI
- Max-GPU chunking: `--gen-chunk 128` (L∞ attacks), `--aa-chunk 64`
- CW paper-canonical: `--cw-max-iter 100 --cw-bss 9`
- AutoAttack: `--aa-version standard`

Each attack is one command; run them back-to-back in one screen. Total
wall-clock on single 5090: ~3 h 40 min (paper-canonical CW) / ~2 h 30 min
(fast CW 50/5).

### 7.1 Single-instance sequential run (recommended)

```bash
screen -S paper_run

# 1) FGSM — ~5 min total for 5 seeds
python experiments/evaluation/run_evaluation_full.py \
  --n-test 1000 --attacks FGSM \
  --multi-seed --seeds 42 123 456 789 999 \
  --gen-chunk 128 --checkpoint-interval 100 \
  --output experiments/evaluation/results_fgsm_n1000_ms5.json \
  2>&1 | tee logs/fgsm_ms5.log

# 2) PGD-40 — ~10 min total
python experiments/evaluation/run_evaluation_full.py \
  --n-test 1000 --attacks PGD \
  --multi-seed --seeds 42 123 456 789 999 \
  --gen-chunk 128 --checkpoint-interval 100 \
  --output experiments/evaluation/results_pgd_n1000_ms5.json \
  2>&1 | tee logs/pgd_ms5.log

# 3) Square (5000 queries) — ~35 min total
python experiments/evaluation/run_evaluation_full.py \
  --n-test 1000 --attacks Square \
  --multi-seed --seeds 42 123 456 789 999 \
  --gen-chunk 128 --square-max-iter 5000 --checkpoint-interval 100 \
  --output experiments/evaluation/results_square_n1000_ms5.json \
  2>&1 | tee logs/square_ms5.log

# 4) CW-L2 (paper-canonical 100/9) — ~150 min total
python experiments/evaluation/run_evaluation_full.py \
  --n-test 1000 --attacks CW \
  --multi-seed --seeds 42 123 456 789 999 \
  --cw-max-iter 100 --cw-bss 9 \
  --gen-chunk 128 --checkpoint-interval 100 \
  --output experiments/evaluation/results_cw_n1000_ms5.json \
  2>&1 | tee logs/cw_ms5.log

# 5) AutoAttack Standard — ~25 min total
python experiments/evaluation/run_evaluation_full.py \
  --n-test 1000 --attacks AutoAttack \
  --multi-seed --seeds 42 123 456 789 999 \
  --aa-version standard --aa-chunk 64 --checkpoint-interval 100 \
  --output experiments/evaluation/results_aa_n1000_ms5.json \
  2>&1 | tee logs/aa_ms5.log
```

### 7.2 One-shot variant (all five attacks, single command)

Trades per-attack log granularity for fewer moving pieces:

```bash
python experiments/evaluation/run_evaluation_full.py \
  --n-test 1000 \
  --attacks FGSM PGD Square CW AutoAttack \
  --multi-seed --seeds 42 123 456 789 999 \
  --gen-chunk 128 --aa-chunk 64 \
  --cw-max-iter 100 --cw-bss 9 \
  --aa-version standard --square-max-iter 5000 \
  --checkpoint-interval 100 \
  --output experiments/evaluation/results_paper_table1.json \
  2>&1 | tee logs/paper_table1.log
```

Output JSON contains both `per_seed` and `aggregate` blocks — the latter is
what goes in Table 1.

### 7.3 Fast-CW fallback (if budget is tight)

Replace `--cw-max-iter 100 --cw-bss 9` with `--cw-max-iter 50 --cw-bss 5` —
saves ~80 min but deviates from RobustBench canonical params. Report the
deviation explicitly in the paper's threat-model block (§7.5 of
Implementation.md).

---

## 8. Parallel deployment — 3 instances (optional, ~2× faster)

If budget allows, rent **three** RTX 5090 instances simultaneously and split
the workload. Wall-clock collapses from ~4 h to ~2.5 h (limited by CW).

> **Prerequisite:** do steps 4 + 5 (retrain + calibrate) on **one** instance
> first, then `scp` the locked `ensemble_scorer.pkl` + `calibrator.pkl` +
> `reference_profiles.pkl` to the other two. All three instances must run the
> **same** locked artifacts or the paper table mixes decision functions.

| Instance | Workload | ~Wall-clock |
|----------|----------|-------------|
| A | Retrain + Calibrate + FGSM + PGD + Square (5-seed) | ~90 min |
| B | CW (5-seed, paper-canonical 100/9) | **~150 min** ← limiter |
| C | AutoAttack (5-seed, standard) + Ablation (step 9) | ~40 min |

Deployment sequence:

```bash
# On instance A (does retrain + calibrate + L∞ attacks):
# — follow §4 + §5 + §7.1 steps 1–3 —

# After §5 completes on A, copy locked artifacts to B and C:
scp -P <portA> root@<ipA>:/workspace/prism-repo/prism/models/*.pkl /tmp/prism_locked/
scp -P <portB> /tmp/prism_locked/*.pkl root@<ipB>:/workspace/prism-repo/prism/models/
scp -P <portC> /tmp/prism_locked/*.pkl root@<ipC>:/workspace/prism-repo/prism/models/

# On instance B — only CW:
python experiments/evaluation/run_evaluation_full.py \
  --n-test 1000 --attacks CW \
  --multi-seed --seeds 42 123 456 789 999 \
  --cw-max-iter 100 --cw-bss 9 --gen-chunk 128 \
  --output experiments/evaluation/results_cw_n1000_ms5.json \
  2>&1 | tee logs/cw_ms5.log

# On instance C — AutoAttack then ablation:
python experiments/evaluation/run_evaluation_full.py \
  --n-test 1000 --attacks AutoAttack \
  --multi-seed --seeds 42 123 456 789 999 \
  --aa-version standard --aa-chunk 64 \
  --output experiments/evaluation/results_aa_n1000_ms5.json \
  2>&1 | tee logs/aa_ms5.log
# then §9 ablation
```

> **Single-GPU parallelism is NOT worth it.** Running two attack processes on
> the same RTX 5090 time-slices the SMs and actually ends up ~20 % slower in
> aggregate than serial execution — every attack here is compute-bound, not
> memory-bound. Use multiple instances, not multiple processes per instance.

---

## 9. Paper Table 2 — ablation (n=1000 × 5 seeds)

Per `PRISM Implementation.md` §6.4. Ablation reuses the adversarial scores
already produced by step 7, so it's fast (~15 min).

```bash
python experiments/ablation/run_ablation_paper.py \
  --n-test 1000 \
  --seeds 42 123 456 789 999 \
  --output experiments/ablation/results_ablation_n1000_ms5.json \
  2>&1 | tee logs/ablation_n1000.log
```

Required configurations (all should already be in `run_ablation_paper.py`):
Full PRISM / No-L0 / No-MoE / No-Ensemble (TDA-only). Each must have paired-
bootstrap p < 0.05 vs Full to claim a measurable contribution.

> **Narrative recalibration (Implementation.md).** The ablation is the
> evidence that *the ensemble is the detector*, not TDA alone (Full 0.8847 vs
> TDA-only 0.6213). Paper abstract + intro must reflect this framing.

---

## 10. Optional reviewer-required add-ons

Per `PRISM Implementation.md` §7.6 and §7.7, for venue credibility:

### 10.1 Adaptive PGD against the ensemble score (§7.6)

Run on **any one** of the three instances after step 7 completes:

```bash
# NOTE: script path per Implementation.md Week-27 checklist;
# create it if it doesn't yet exist (reuses the PGD harness in run_evaluation_full.py).
python experiments/evaluation/run_adaptive_pgd.py \
  --n-test 1000 --seeds 42 123 456 789 999 \
  --output experiments/evaluation/results_adaptive_pgd_ms5.json \
  2>&1 | tee logs/adaptive_pgd.log
```

Reports TPR under an attacker that minimises the PRISM ensemble score
(constrained by the backbone-misclassification objective).

### 10.2 LID + Mahalanobis baselines (§7.7)

Reproduce on the same `EVAL_IDX` pool, same ResNet-18 backbone:

```bash
python experiments/baselines/run_lid.py \
  --n-test 1000 --seeds 42 123 456 789 999 \
  --output experiments/baselines/results_lid_ms5.json

python experiments/baselines/run_mahalanobis.py \
  --n-test 1000 --seeds 42 123 456 789 999 \
  --output experiments/baselines/results_mahalanobis_ms5.json
```

Head-to-head TPR at matched L1/L2/L3 tiers goes in a comparison table.

---

## 11. Progress monitoring

All attack generators now emit timestamped chunked progress with `flush=True`
(UTF-8 console reconfigure is set at the top of `run_evaluation_full.py`).

```bash
# Live attack-gen + PRISM-eval progress for any run
tail -f logs/cw_ms5.log | grep -E "\[gen\]|\[AA gen\]|\[Checkpoint\]|TPR=|FPR=|elapsed|Traceback|Error"

# Pass/fail snapshot across all finished runs
grep -E "TPR=|FPR=|pass_L|Held-out" logs/*_ms5.log | tail -80

# GPU utilization check (run in a separate SSH session)
watch -n 2 nvidia-smi
```

Healthy CW output:
```
[gen] 128/1000 elapsed=128.3s  rate=1.00s/img  eta=871.7s
[Checkpoint 100/1000] TPR=0.8700 ✅ | FPR=0.0600 ✅ | F1=0.9147
```

GPU should sit at **95–100 % utilization** during attack generation with
`--gen-chunk 128` and the §3.3 env vars. If it drops below 70 %, CPU
preprocessing is the bottleneck — raise `OMP_NUM_THREADS` to 8.

---

## 12. Download results

From your laptop:

```bash
mkdir -p experiments/evaluation experiments/ablation /tmp/vastai_logs

# All multi-seed result JSONs (Table 1)
scp -P <port> \
  root@<instance-ip>:/workspace/prism-repo/prism/experiments/evaluation/results_*_n1000_ms5.json \
  experiments/evaluation/
scp -P <port> \
  root@<instance-ip>:/workspace/prism-repo/prism/experiments/evaluation/results_paper_table1.json \
  experiments/evaluation/ 2>/dev/null || true  # only exists if §7.2 was used

# Ablation (Table 2)
scp -P <port> \
  root@<instance-ip>:/workspace/prism-repo/prism/experiments/ablation/results_ablation_n1000_ms5.json \
  experiments/ablation/

# Locked artifacts — keep these together forever
scp -P <port> \
  root@<instance-ip>:/workspace/prism-repo/prism/models/ensemble_scorer.pkl \
  models/ensemble_scorer_vastai_retrained.pkl
scp -P <port> \
  root@<instance-ip>:/workspace/prism-repo/prism/models/calibrator.pkl \
  models/calibrator_vastai_retrained.pkl

# All logs
scp -P <port> \
  "root@<instance-ip>:/workspace/prism-repo/prism/logs/*.log" \
  /tmp/vastai_logs/
```

---

## 13. Pipeline gate — order is mandatory

```
train_ensemble_scorer.py         (§4 — ONCE, creates ensemble_scorer.pkl)
           │
           ▼
calibrate_ensemble.py            (§5 — ONCE, creates calibrator.pkl)
           │
           ▼
compute_ensemble_val_fpr.py      (§5 — ONCE, verifies FPR gate)
           │
           ▼
run_evaluation_full.py × 5       (§7 — one per attack, ALL against the
                                  same locked ensemble + calibrator)
           │
           ▼
run_ablation_paper.py            (§9 — reuses locked artifacts)
```

Re-running §4 or §5 **between** attack runs means different attacks were
scored against different decision functions. Reviewers will catch it by
diffing `_meta.ensemble` across result JSONs.

---

## 14. Total wall-clock estimates (RTX 5090)

| Configuration | Wall-clock |
|---------------|------------|
| Single instance, paper-canonical CW (100/9) | **~4 h 30 min** |
| Single instance, fast CW (50/5) | **~3 h 20 min** |
| 3-instance parallel (§8), paper-canonical CW | **~2 h 30 min** |
| 3-instance parallel (§8), fast CW | **~1 h 45 min** |

Retrain + calibrate are one-time and already included in the above.

---

## 15. Reproducibility manifest (paper appendix)

After all runs finish, capture artifact provenance + hashes:

```bash
python -c "
import pickle, json, hashlib, os
def h(p): return hashlib.sha256(open(p,'rb').read()).hexdigest()[:16] if os.path.exists(p) else None
e = pickle.load(open('models/ensemble_scorer.pkl','rb'))
out = {
  'ensemble_training_attacks': list(getattr(e, 'training_attacks', [])),
  'ensemble_training_n':       int(getattr(e, 'training_n', 0)),
  'ensemble_n_features':       int(getattr(e, 'n_features', 0)) if hasattr(e, 'n_features') else None,
  'ensemble_use_dct':          getattr(e, 'use_dct', None),
  'ensemble_sha256_16':        h('models/ensemble_scorer.pkl'),
  'calibrator_sha256_16':      h('models/calibrator.pkl'),
  'reference_profiles_sha256_16': h('models/reference_profiles.pkl'),
  'seeds':                     [42, 123, 456, 789, 999],
  'eval_split':                'CIFAR-10 test idx 8000-9999',
  'eps_linf':                  8.0/255,
  'cw_eval_params':            {'max_iter': 100, 'bss': 9, 'batch_size': 64, 'confidence': 0.0},
  'aa_eval_params':            {'version': 'standard', 'chunk': 64},
  'square_eval_params':        {'max_iter': 5000, 'nb_restarts': 1},
}
print(json.dumps(out, indent=2))
" 2>&1 | tee logs/manifest.json
```

This JSON (training mix + artifact hashes + eval params + seeds) is what
reviewers need to verify that every per-attack number in your table came from
the same locked decision function under the paper-canonical attack
parameters.

---

## 16. Post-run verification checklist

Before calling the run finished:

- [ ] Every `results_*_n1000_ms5.json` `_meta.ensemble.training_attacks` list
      reads `['FGSM', 'PGD', 'Square', 'CW', 'AutoAttack']`
- [ ] All five attack aggregates satisfy the §0 TPR targets (Wilson CI lower
      bound ≥ target where applicable)
- [ ] Pooled per-tier FPR satisfies L1 ≤ 0.10, L2 ≤ 0.03, L3 ≤ 0.005
- [ ] Latency `_meta.latency.pass == true` on every seed's JSON
- [ ] Ablation paired-bootstrap p < 0.05 for each ablated component
- [ ] `logs/manifest.json` captured and `ensemble_sha256_16` matches across
      all result JSONs (they should — single locked ensemble)
- [ ] Stale artifacts deleted locally (`results_n500_*.json`, n=300
      `results_paper.json`) so future greps don't surface them
