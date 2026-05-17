# PRISM — Predictive Runtime Immune System with Manifold Monitoring

PRISM is an architecture-agnostic adversarial-example detector for pre-trained
PyTorch image classifiers. It monitors intermediate-layer activations with
topological data analysis (persistent homology), scores each inference against
a calibrated clean-data profile, and applies split conformal prediction to
produce **distribution-free false-positive-rate certificates** at three
severity tiers — without retraining or modifying the base model.

The associated paper draft is at `prism/paper/main.tex` (NeurIPS 2025 preprint
style).

---

## Pipeline

```
input x ──► classifier f ──► prediction
              │
              ├─► TAMM   ─ persistence diagrams from monitored layers
              ├─► CADG   ─ conformal scoring at L1 / L2 / L3 tiers
              ├─► SACD   ─ BOCPD campaign / sequential-attack monitor
              ├─► TAMSH  ─ topology-aware MoE recovery on L3-tier inputs
              └─► ImmuneMemory + Federation
```

| Tier | Conformal α | Action |
|------|-------------|--------|
| L1   | 0.10        | Log & continue |
| L2   | 0.03        | Input purification |
| L3   | 0.005       | Route through topology-matched MoE expert (or reject) |

---

## Repo layout

```
prism-adversarial-defense/
├── README.md                       (this file)
├── PRISM Implementation.md         (full architectural spec)
├── PRISM Testing, Training & Validation Guide.md
└── prism/
    ├── run_vastai_full.sh          CIFAR-10 multi-seed run (5 seeds, n=1000/attack)
    ├── run_vastai_cifar100.sh      CIFAR-100 generalization run
    ├── run_local_full.sh           Single-seed local validation
    ├── requirements.txt
    ├── configs/
    │   ├── vastai_cw_full.yaml     Canonical config used in the paper
    │   └── default.yaml, cifar100.yaml
    ├── src/                        Pipeline modules (sacd, tamm, cadg, tamsh, memory, federation)
    ├── scripts/                    Trainers, calibrators, gate checkers, table builders
    ├── experiments/
    │   ├── evaluation/             Main attacks + adaptive PGD + baselines
    │   ├── ablation/               Full / No-MoE / Ensemble-no-TDA / TDA-only
    │   ├── campaign/               SACD sustained / burst / low-rate streams
    │   ├── recovery/               TAMSH L3-recovery evaluation
    │   └── generalization/         CIFAR-100 second-dataset evaluation
    ├── tests/                      Unit tests for SACD / CADG / TAMM / gates
    └── paper/                      LaTeX source, figures, auto-generated tables
```

---

## Installation

```bash
cd prism-adversarial-defense/prism
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Notes:
- PyTorch ≥ 2.1 with CUDA build matched to your driver
  (`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`
  if `pip install -r requirements.txt` does not pick the right wheel).
- `auto-attack` is installed from GitHub (see `requirements.txt`); no PyPI release.
- TDA stack: `ripser`, `gudhi`, `persim`, `pot`.

---

## Quick start (local validation)

```bash
cd prism/
bash run_local_full.sh        # ~30–45 min on a single mid-range GPU
```

This drives the full pipeline end-to-end on a single seed:

1. Pre-train CIFAR-10 ResNet-18 backbone (or reuse existing checkpoint).
2. Build reference profiles on `test[0:5000]`.
3. Train ensemble scorer (CW-aware weighted attack mix; FGSM/PGD/Square/CW).
4. Train MoE experts.
5. Calibrate L1/L2/L3 conformal thresholds on `test[5000:7000]`.
6. Validate FPR gate on `test[7000:8000]`.
7. Run attack evaluation on `test[8000:10000]`.
8. Run ablation and gate-check the result.

The script exits non-zero on any gate miss (FPR or attack TPR).

---

## Full evaluation (Vast.ai / multi-seed)

```bash
cd prism/
bash run_vastai_full.sh
```

Defaults (overridable via env vars or `configs/vastai_cw_full.yaml`):

- 5 seeds: `42 123 456 789 999`
- n=1000 test images per attack per seed
- Attacks: FGSM, PGD-40, CW-L2 (max_iter=40, bss=5), Square (5000 queries), AutoAttack-standard
- Adaptive PGD: λ ∈ {0, 0.5, 1, 2, 5, 10}, 100 steps × 10 restarts, through-scorer BPDA
- Baselines (matched FPR): LID, Mahalanobis, ODIN, Energy
- Tier calibration: L1=0.85, L2=0.70, L3=0.50 alpha factors
- Wall-clock ~5–7h on an RTX 5090

After the run, validate gates:

```bash
python scripts/check_vastai_full_gate.py
```

Exit-code contract:
- `0` — all gates pass (FGSM ≥ 0.85, PGD ≥ 0.90, Square ≥ 0.85, CW ≥ 0.85, AA ≥ 0.90; val FPR L1/L2/L3 ≤ 0.10/0.03/0.005; latency < 100 ms)
- `1` — FPR or training failure
- `2` — eval failure
- `3` — Phase-2 gate miss (P0.4 campaign / P0.5 recovery)

Generate the paper tables from the result JSONs:

```bash
python scripts/build_paper_tables.py \
  --results-dir experiments \
  --out-dir paper/tables
```

Run the CIFAR-100 generalization study:

```bash
bash run_vastai_cifar100.sh
```

---

## Reproducibility

- Splits are strict, disjoint, and identical across all runs:
  - Profile: `test[0:5000]`
  - Calibration: `test[5000:7000]` (n_cal = 2000)
  - Validation: `test[7000:8000]` (n_val = 1000)
  - Evaluation: `test[8000:10000]` (n_eval = 2000)
- TDA point-cloud subsampling is **deterministic** (MD5-hash-based), so scoring
  is reproducible across calibration and inference.
- `torch.use_deterministic_algorithms(True, warn_only=True)` and
  `cudnn.deterministic = True` are set in the launcher.
- All seeds, configs, and artifact SHA-256 are recorded in `logs/manifest.json`
  after `run_vastai_full.sh`.
- All paper tables are auto-generated from result JSONs by
  `scripts/build_paper_tables.py` — no values are hand-transcribed.

---

## Tests

```bash
cd prism/
pytest tests/
```

Eight test files cover SACD priors, CADG calibration & coverage, TAMM feature
extraction, federation merging, gate definitions, integration, feature-space
contracts, and core utilities.

---

## Documentation pointers

- Full architecture spec: `PRISM Implementation.md`
- Training/validation walkthrough: `PRISM Testing, Training & Validation Guide.md`
- Vast.ai run protocol: `prism/VASTAI_RUN_GUIDE.md`, `prism/VASTAI_FULL_TEST_READY.md`
- Paper draft: `prism/paper/main.tex`
- Research-rules: `.github/instructions/publishable-research.instructions.md`

---

## Citation

```bibtex
@misc{prism2026,
  title = {PRISM: Predictive Runtime Immune System with Manifold Monitoring
           for Architecture-Agnostic Adversarial Defense},
  author = {Anonymous Author(s)},
  year = {2026},
  note = {Under review at NeurIPS 2026}
}
```

---

## License

See `LICENSE` (to be added).
