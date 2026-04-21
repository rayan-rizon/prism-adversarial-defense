# PRISM — Full n=1000 Thundercompute Evaluation Guide

**Purpose:** Run the paper-grade n=1000 evaluation on a Thundercompute A100/H100 GPU
instance, including CW, AutoAttack, strengthened adaptive attack, ablation, baselines,
cross-dataset generalization, and campaign detection.

**Canonical local baseline:** `prism/experiments/evaluation/run_report_n500_local_20260420.md`
(retained single-seed n=500 CUDA run from 2026-04-21; FGSM TPR 0.844, PGD 1.000, Square 0.924, latency 73.68 ms).

**Workflow lock (must match local validated run):**
- scorer training: `python scripts/train_ensemble_scorer.py --n-train 3000 --fgsm-oversample 1.5`
- ensemble blend: `alpha=0.4` in `scripts/train_ensemble_scorer.py`
- scorer features: `n_features=37` (`use_grad_norm=False`)
- TDA subsample: `n_subsample=150`
- calibration: `cal_alpha_factor=0.7`
- local reference artifact: `prism/experiments/evaluation/results_n500_optimized_20260421.json`

---

## 0. Instance provisioning

| Item | Requirement |
|---|---|
| GPU | A100 80GB (preferred) or H100 80GB. A100 40GB is sufficient. |
| Storage | ≥80 GB (CIFAR-10+100 auto-download + checkpoints + logs) |
| Python | 3.11.x |
| CUDA toolkit | ≥12.1 |
| Budget | ~14 GPU-hours (see Phase time table) |

---

## 1. Repo setup

```bash
# Clone and enter the working directory
git clone <repo-url> prism-run
cd prism-run/prism-adversarial-defense/prism

# Virtual environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel

# Dependencies
pip install -r requirements.txt
pip install adversarial-robustness-toolbox autoattack   # in case not in requirements.txt

# Environment
export PYTHONIOENCODING=utf-8
export SSL_CERT_FILE=$(python -c 'import certifi; print(certifi.where())')
export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE

# Confirm GPU
python -c "import torch; assert torch.cuda.is_available(), 'NO GPU'; \
           print('GPU:', torch.cuda.get_device_name(0)); \
           print('CUDA:', torch.version.cuda)"

# Make logs directory
mkdir -p logs
```

---

## 2. Phase A — Code changes (apply before any run)

These Phase A changes must be present in your checkout before running:

1. **`scripts/train_ensemble_scorer.py`** — run with `--n-train 3000 --fgsm-oversample 1.5`; retained local run also uses `alpha=0.4`.
2. **`src/cadg/ensemble_scorer.py`** — logger hygiene and canonical 37-feature scorer path.
3. **`experiments/evaluation/run_adaptive_pgd.py`** — `--through-scorer` flag with
  differentiable DCT energy matching term.

Canonical training/evaluation path for this guide uses **no grad-norm feature**
(`n_features=37`, `use_grad_norm=False`).

> If running on a fresh checkout without the Phase A commits, apply those changes first.

### Phase A preflight verification (required)

```bash
# Record exact revision used for the run report
git rev-parse HEAD

# Verify locked scorer defaults and alpha=0.4 are present in source
grep -n "default=3000" scripts/train_ensemble_scorer.py
grep -n "default=1.5" scripts/train_ensemble_scorer.py
grep -n "alpha=0.4" scripts/train_ensemble_scorer.py

# Verify locked TDA/calibration config
grep -n "n_subsample: 150" configs/default.yaml
grep -n "cal_alpha_factor: 0.7" configs/default.yaml
```

If any preflight check does not match, stop and fix code/config before Phase B.

---

## 3. Phase B — Rebuild the artifact chain (MANDATORY ORDER)

> **Critical:** Any change to scorer/features/calibration MUST run all four rebuild steps
> in order. Skipping causes silent regressions (documented: FGSM TPR 0.832 → 0.622).

```bash
# B1. Reference profiles from test[0:5000].
#     Builds models/reference_profiles.pkl and models/scorer.pkl
#     Time: ~30-40 min on A100
python scripts/build_profile_testset.py \
  2>&1 | tee logs/B1_build_profile.log
echo "B1 exit: $?"

# B2. Retrain ensemble scorer with FGSM oversampling (canonical 37-feature path).
#     Builds models/ensemble_scorer.pkl
#     n_train=3000 -> FGSM:1285, PGD:857, Square:858 (fgsm_oversample=1.5)
#     Time: ~20-30 min on A100
python scripts/train_ensemble_scorer.py \
  --n-train 3000 \
  --fgsm-oversample 1.5 \
  2>&1 | tee logs/B2_train_ensemble.log
echo "B2 exit: $?"

# B3. Conformal calibration against the refreshed ensemble.
#     Rebuilds models/calibrator.pkl using test[5000:7000]
#     Time: ~3-5 min
python scripts/calibrate_ensemble.py \
  2>&1 | tee logs/B3_calibrate.log
echo "B3 exit: $?"

# B4. Validation-split FPR gate — MUST pass before any evaluation.
#     Writes experiments/calibration/ensemble_fpr_report.json (test[7000:8000])
#     Time: ~3 min
python scripts/compute_ensemble_val_fpr.py \
  2>&1 | tee logs/B4_val_fpr.log
echo "B4 exit: $?"

# --- GATE CHECK ---
# Inspect B4 output. Required:
#   L1 FPR  <= 0.10
#   L2 FPR  <= 0.03
#   L3 FPR  <= 0.005
# If L2 or L3 fails: edit configs/default.yaml → reduce cal_alpha_factor
# (e.g. 0.7 → 0.6), then re-run B3 + B4 only.
python -c "
import json
with open('experiments/calibration/ensemble_fpr_report.json') as f:
    r = json.load(f)
for tier, tgt in [('L1', 0.10), ('L2', 0.03), ('L3', 0.005)]:
    fpr = r.get(f'FPR_{tier}_plus', r.get(tier, {}).get('FPR', 999))
    status = 'PASS' if fpr <= tgt else 'FAIL'
    print(f'{tier} FPR={fpr:.4f} target={tgt}  [{status}]')
"

# B5. Sanity checks — validates all artifacts end-to-end.
python sanity_checks.py 2>&1 | tee logs/B5_sanity.log
echo "B5 exit: $?"
```

---

## 4. Phase C — Main n=1000 evaluation

### C1. Multi-seed evaluation: FGSM, PGD, Square (5 seeds × n=1000)

```bash
# Time: ~3-4 h on A100
python experiments/evaluation/run_evaluation_full.py \
  --n-test 1000 \
  --attacks FGSM PGD Square \
  --multi-seed \
  --seeds 42 123 456 789 999 \
  --output experiments/evaluation/results_n1000_multiseed_$(date +%Y%m%d).json \
  --checkpoint-interval 200 \
  2>&1 | tee logs/C1_multiseed.log
echo "C1 exit: $?"
```

### C2. CW-L2 (seed 42, n=1000)

```bash
# Time: ~60-90 min on A100 (GPU ART classifier: ~3s/sample)
python experiments/evaluation/run_evaluation_full.py \
  --n-test 1000 \
  --attacks CW \
  --seed 42 \
  --output experiments/evaluation/results_n1000_cw_seed42_$(date +%Y%m%d).json \
  2>&1 | tee logs/C2_cw.log
echo "C2 exit: $?"
```

### C3. AutoAttack (seed 42, n=1000)

```bash
# Time: ~4-6 h on A100 (standard version: apgd-ce + apgd-t + fab + square)
python experiments/evaluation/run_evaluation_full.py \
  --n-test 1000 \
  --attacks AutoAttack \
  --seed 42 \
  --output experiments/evaluation/results_n1000_autoattack_seed42_$(date +%Y%m%d).json \
  2>&1 | tee logs/C3_autoattack.log
echo "C3 exit: $?"
```

---

## 5. Phase D — Companion experiments

### D1. Baselines (LID, Mahalanobis) at n=1000 across 3 seeds

```bash
# Time: ~40-50 min total
for S in 42 123 456; do
  python experiments/evaluation/run_baselines.py \
    --n-test 1000 \
    --attacks FGSM PGD Square \
    --seed $S \
    --output experiments/evaluation/results_baselines_n1000_seed${S}.json \
    2>&1 | tee -a logs/D1_baselines.log
  echo "D1 seed=$S exit: $?"
done
```

### D2. Ablation at n=1000

```bash
# Time: ~60 min (4 configs × 3 attacks × 1000 samples)
python experiments/ablation/run_ablation_paper.py \
  --n-test 1000 \
  --output experiments/ablation/results_ablation_n1000_$(date +%Y%m%d).json \
  2>&1 | tee logs/D2_ablation.log
echo "D2 exit: $?"
```

### D3. Strengthened adaptive PGD (--through-scorer)

```bash
# Time: ~50-70 min (5 lambdas × 1000 samples, per-sample backward pass)
python experiments/evaluation/run_adaptive_pgd.py \
  --n-test 1000 \
  --pgd-steps 40 \
  --lambdas 0.0 0.5 1.0 2.0 5.0 \
  --through-scorer \
  --output experiments/evaluation/results_adaptive_pgd_n1000_$(date +%Y%m%d).json \
  2>&1 | tee logs/D3_adaptive.log
echo "D3 exit: $?"
```

### D4. Cross-dataset generalization (CIFAR-100)

```bash
# Time: ~30 min
python experiments/generalization/run_cifar100.py \
  --n-clean 500 \
  --n-adv 500 \
  --seed 42 \
  --output experiments/generalization/results_cifar100_n500_$(date +%Y%m%d).json \
  2>&1 | tee logs/D4_cifar100.log
echo "D4 exit: $?"
```

### D5. Campaign detection (multi-trial)

```bash
# Time: ~30 min (10 trials)
python experiments/campaign/run_campaign_real.py \
  --n-trials 10 \
  --output experiments/campaign/results_campaign_n1000_$(date +%Y%m%d).json \
  2>&1 | tee logs/D5_campaign.log
echo "D5 exit: $?"
```

---

## 6. Phase E — Aggregation and report

```bash
# Collect all result paths into a variable for readability
EVAL_DATE=$(date +%Y%m%d)
RESULTS=(
  experiments/evaluation/results_n1000_multiseed_${EVAL_DATE}.json
  experiments/evaluation/results_n1000_cw_seed42_${EVAL_DATE}.json
  experiments/evaluation/results_n1000_autoattack_seed42_${EVAL_DATE}.json
  experiments/evaluation/results_baselines_n1000_seed42.json
  experiments/evaluation/results_baselines_n1000_seed123.json
  experiments/evaluation/results_baselines_n1000_seed456.json
  experiments/ablation/results_ablation_n1000_${EVAL_DATE}.json
  experiments/evaluation/results_adaptive_pgd_n1000_${EVAL_DATE}.json
  experiments/generalization/results_cifar100_n500_${EVAL_DATE}.json
  experiments/campaign/results_campaign_n1000_${EVAL_DATE}.json
)

python scripts/analyze_results.py \
  --inputs "${RESULTS[@]}" \
  --out experiments/evaluation/run_report_n1000_thundercompute_${EVAL_DATE}.md \
  2>&1 | tee logs/E1_aggregate.log
```

> The produced report must follow the same section structure as
> `run_report_n500_local_20260420.md` (Scope → Environment → Config snapshot →
> Procedure → Results → Tier FPR → Latency → Multi-seed validation → Baselines →
> Ablation → Adaptive → Cross-dataset → Campaign → Reproducibility notes).

---

## 7. Time budget

| Phase | Item | Estimated time |
|---|---|---|
| B | Artifact rebuild (B1–B5) | ~60 min |
| C1 | Multi-seed FGSM/PGD/Square, 5 seeds × n=1000 | ~3.5 h |
| C2 | CW, seed 42, n=1000 | ~1.5 h |
| C3 | AutoAttack, seed 42, n=1000 | ~5 h |
| D1 | Baselines, 3 seeds × n=1000 | ~50 min |
| D2 | Ablation n=1000 | ~60 min |
| D3 | Adaptive PGD --through-scorer, n=1000 | ~60 min |
| D4 | CIFAR-100 generalization | ~30 min |
| D5 | Campaign detection | ~30 min |
| E | Aggregation | ~5 min |
| **Total** | | **~14 h** |

Each phase checkpoints to disk. A mid-run disconnect restarts at phase granularity.

---

## 8. Acceptance criteria

The run is accepted for paper reporting only if **all** of the following hold:

| # | Criterion | Source |
|---|---|---|
| 1 | Validation-split FPR gate: L1 ≤ 0.10, L2 ≤ 0.03, L3 ≤ 0.005 | B4 output |
| 2 | Pooled 5-seed FGSM TPR 95% CI **lower bound ≥ 0.85** | C1 multiseed JSON |
| 3 | Pooled 5-seed PGD TPR ≥ 0.95 | C1 multiseed JSON |
| 4 | Pooled 5-seed Square TPR ≥ 0.85 | C1 multiseed JSON |
| 5 | All 5 seeds individually pass L1/L2/L3 tier FPR (pooled n=5000 expected to stabilise) | C1 multiseed JSON |
| 6 | CW (seed 42, n=1000) TPR reported with Wilson CI | C2 JSON |
| 7 | AutoAttack (seed 42, n=1000) TPR reported with Wilson CI | C3 JSON |
| 8 | Adaptive PGD --through-scorer TPR across λ ∈ {0, 0.5, 1, 2, 5} reported | D3 JSON |
| 9 | Baselines at n=1000 reported (fair comparison) | D1 JSONs |
| 10 | Latency table (mean/p50/p95) across 5 seeds committed to one `tda.n_subsample` point | C1 |
| 11 | `sanity_checks.py` exits cleanly after run | B5 log |
| 12 | All JSON artifacts reproducible from logged `logs/` commands; no manual editing | audit |

**If criterion 2 (FGSM CI lower bound) still fails after the retained local config (`n_train=3000`, `fgsm_oversample=1.5`, `alpha=0.4`, `n_subsample=150`):**
- Option A: Increase `--fgsm-oversample` to 2.0 and re-run from Phase B.
- Option B (last resort): Lower `L1 alpha` from 0.10 to 0.12 in `configs/default.yaml`
  (trades FPR for TPR — must be clearly disclosed in the paper).
- Option C (reframe): Reframe the paper narrative per publishability verdict in
  `PRISM_ANALYSIS_PLAN.md` Part 1 — FGSM moves from "win" to "comparable at lower FPR"
  column; document it inline in the run report.

---

## 9. Config snapshot at run time

From `configs/default.yaml` — confirm these values match before running:

```yaml
model:
  layer_names: [layer2, layer3, layer4]
  layer_weights: {layer2: 0.30, layer3: 0.30, layer4: 0.40}
tda:
  n_subsample: 150
  max_dim: 1
  dim_weights: [0.70, 0.30]
conformal:
  alphas: {L1: 0.10, L2: 0.03, L3: 0.005}
  cal_alpha_factor: 0.7
splits:
  profile_idx: [0, 5000]
  cal_idx: [5000, 7000]
  val_idx: [7000, 8000]
  eval_idx: [8000, 10000]
```

---

## 10. Files modified by Phase A (reference)

| File | Change |
|---|---|
| `src/cadg/ensemble_scorer.py` | logger hygiene and canonical scorer path |
| `scripts/train_ensemble_scorer.py` | `--fgsm-oversample` CLI flag |
| `experiments/evaluation/run_adaptive_pgd.py` | `--through-scorer` flag + DCT loss term |
