# Stretch A — What's Next (Higher-GPU Resume)

The ThunderCompute A6000 run finished the **core detection table** for WRN-28-10
(CW, FGSM, PGD multi-seed PASS; Square + AA seed42 PASS). Three things did not
finish on the A6000 and need a bigger GPU:

1. Square + AutoAttack for seeds 123/456/789/999 (the A6000 script aborted on a
   pre-recalibration L3 FPR miss before reaching them)
2. Adaptive PGD sweep (the A6000 was hitting ~120 sec/img — full original plan
   would take ~190 hours per λ)
3. Ablation (the original script forgot to train `models/wrn/experts.pkl`)
4. `build_paper_tables.py` (Step 7) — never ran because Step 6 errored

`run_stretch_a_finish.sh` (sitting next to this file) does all four.

---

## GPU sizing chart

| GPU              | Step B (Square+AA, 4 seeds) | Step C (ablation) | Step D (adaptive, default) | Total est. |
|------------------|-----------------------------|--------------------|----------------------------|------------|
| H200 141 GB      | ~30 min                     | ~45 min            | ~5–6 h                     | **~7 h**   |
| H100 80 GB       | ~40 min                     | ~60 min            | ~7–9 h                     | **~10 h**  |
| A100 80 GB       | ~50 min                     | ~90 min            | ~12–14 h                   | **~16 h**  |
| RTX 5090 32 GB   | ~40 min                     | ~70 min            | ~9–10 h (batch=8)          | **~12 h**  |

Cost rough estimates: H100 ≈ $25–40, H200 ≈ $30–50, A100 ≈ $20–30,
5090 (Vast.ai) ≈ $5–10.

**Recommendation:** H100 80 GB. Best perf/$ for adaptive PGD, plenty of VRAM
headroom for ablation's parallel attack generation. H200 is fastest but ~30%
more for ~30% time savings.

---

## How to run on the higher GPU

```bash
# 1. Clone or unzip the project (use the merged-back main repo)
cd prism-adversarial-defense/prism

# 2. Install deps
pip install -r requirements.txt

# 3. Make sure these existing artifacts are uploaded (the A6000 work product):
#    models/cifar_wrn28_10.pt          (143 MB)
#    models/cifar_wrn28_10.acc.json
#    models/wrn/calibrator.pkl
#    models/wrn/ensemble_scorer.pkl
#    models/wrn/ensemble_no_tda.pkl
#    models/wrn/reference_profiles.pkl
#    models/wrn/scorer.pkl

# 4. Run the finish script
bash run_stretch_a_finish.sh 2>&1 | tee logs/stretch_a_finish/run.log
```

### Want a faster run?

Pare down the adaptive PGD budget further:

```bash
# minimal: 2 λ values, 500 images, 1 seed (~2 h on H100)
ADAPTIVE_LAMBDAS="0.0 5.0" ADAPTIVE_N=500 bash run_stretch_a_finish.sh

# original spec (slow, only do this on H200): 6 λ × 5 seeds × 1000 imgs
ADAPTIVE_LAMBDAS="0.0 0.5 1.0 2.0 5.0 10.0" \
ADAPTIVE_SEEDS="42 123 456 789 999" \
ADAPTIVE_STEPS=100 ADAPTIVE_RESTARTS=10 \
bash run_stretch_a_finish.sh
```

---

## After the finish run — local steps

1. Download the new `experiments/wrn/` and `models/wrn/experts.pkl` artifacts.
2. Verify the paper tables in `paper/tables/wrn/` look right.
3. Add the WRN table to `paper/sections/experiments.tex` (architecture-agnostic
   section). Cite results vs ResNet-18 within noise → backs up the C5
   contribution.
4. Commit + push.

---

## What is already paper-ready (do NOT re-run)

- WRN backbone @ 96.36% (`models/cifar_wrn28_10.pt`)
- Calibration with three-tier FPR PASS
- CW (5 seeds aggregated): TPR=0.933
- FGSM (5 seeds aggregated): TPR=0.985
- PGD (5 seeds aggregated): TPR=0.995
- Square (seed42): TPR=0.948
- AutoAttack (seed42): TPR=1.000

These results live in `experiments/wrn/`. See `experiments/wrn/STATUS.md` for
the per-file breakdown.
