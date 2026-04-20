# PRISM: Implementation Plan (Publishable-Paper Edition)

> **Purpose.** This is the working plan for taking PRISM from a reproducible
> prototype to a publishable paper. It contains *what* to build, *why*, *how
> to verify it*, and *how to close the remaining metric gaps* — but not source
> code. Every implementation pointer is a link to an already-checked-in module
> in the repo. Change the module, don't change the plan.

> **Venue target — honest.** Given the current evaluation scope (CIFAR-10 only,
> ResNet-18 backbone, no adaptive attacks, Square TPR below 0.70), this paper
> realistically targets **UAI / AAAI 2026** with a workshop/arXiv fallback.
> NeurIPS / ICLR / ICML / CVPR / USENIX Security require strictly more: ImageNet
> + ViT, adaptive-attack evaluation, reproduced baselines (RAILS, LID,
> Mahalanobis, feature-squeezing), multi-dataset generalisation (CIFAR-100).
> The Weeks 26–34 schedule below delivers the UAI-tier paper; the NeurIPS
> extension is a post-submission stretch.

> **One-line thesis.** A **conformal-calibrated ensemble over persistence-diagram
> statistics**, escalated by session-level BOCPD and routed by a topology-aware
> MoE, provides tiered adversarial-input detection with a strict per-tier FPR
> guarantee. TDA is the feature source, not the detector.

---

## Critical Validation Report

An honest assessment of the PRISM research plan based on cross-checking
against the current literature (April 2026).

> ⚠️ **Novelty corrections — read before starting.** The original plan
> overstated novelty in several areas. The corrections below are critical for
> honest paper positioning.

### Contribution 1: TDA for Adversarial Detection — Claimed ~95 %, Actual ~60–70 %

**What the plan missed:**

- **Gebhart & Schrater** — "Adversary Detection in Neural Networks via Persistent Homology" — directly uses persistent homology for adversarial detection.
- **LANL** — `tda-adversarial-detection` — open-source persistent-homology + two-sample testing on CLIP/CIFAR/ImageNet.
- **Goibert, Ricatte & Dohmatob** — "An Adversarial Robustness Perspective on the Topology of Neural Networks" — shows clean vs. adversarial graph-topology differences.
- **Ballester et al. (2023)** — Survey "TDA for Neural Network Analysis" covers adversarial detection as a known application.
- **Czernek (2025)** — Master thesis on persistent homology for image classification robustness.
- **WGTL (2025)** — Witness Graph Topological Layer for GNN adversarial robustness with stability guarantees.

**What IS still novel:** Using TDA as *one input* to a multi-tiered,
architecture-agnostic runtime defense that is conformal-calibrated and
BOCPD-escalated. The integration is novel; the individual TDA component is
not.

### Contribution 2: Conformal Prediction for Detection — Claimed ~85 %, Actual ~65–75 %

- **Gendler et al. (ICLR 2022)** — "Adversarially Robust Conformal Prediction."
- **VRCP (2025)** — "Verifiably Robust Conformal Prediction."
- **ScienceDirect (2026)** — Conformal ML framework for anomaly detection in CPS.

**What IS still novel:** Using split-conformal to calibrate an adversarial
DETECTOR (not a prediction set). The Stutz differentiation in the plan is
valid.

### Contribution 3: Sequential Campaign Detection — Claimed ~96 %, Actual ~90–96 %

**This claim holds up well.** No direct competitor for session-level adversarial
campaign detection in NN inference pipelines using BOCPD. Strongest contribution.

### Contribution 4: Topology-Aware MoE — Claimed ~78 %, Actual ~70–78 %

Reasonable claim. Wasserstein-gated expert selection over persistence diagrams is genuinely novel.

### Baselines

- **RAILS** (Wang et al., IEEE Access 2022) — confirmed valid. Our differentiation (architecture-coupled DkNN vs. architecture-agnostic PRISM) is accurate.
- **Croce et al. (ICML 2022)** — the plan's three-angle rebuttal is solid. Key experiment to run: PRISM + adversarially-trained backbone > backbone alone.

> ✅ **Bottom line.** PRISM is still a publishable, novel contribution — but
> you MUST cite the papers above and reframe novelty from "first to use TDA
> for adversarial defense" to "first to integrate **conformal-calibrated
> ensemble scoring** over persistence features with **campaign-level
> monitoring** and **topology-aware MoE recovery** in an architecture-agnostic
> runtime system."

> 🔁 **Narrative recalibration (mandatory).** See §Narrative Recalibration at
> the end of this doc. Ablation data (Appendix A.2) shows the learned ensemble
> over persistence statistics is the detector; base-TDA Wasserstein alone
> cannot carry the paper. The paper must credit that ensemble, not TDA in the
> abstract.

---

## Phase 0: Environment Setup (Week 0)

### 0.1 Hardware Requirements

| Component | Minimum | Recommended |
| --- | --- | --- |
| GPU | RTX 3060 12 GB (local dev) | A100 80 GB (remote GPU) |
| RAM | 16 GB | 32 GB+ |
| Storage | 100 GB SSD | 500 GB SSD |
| CPU | 8-core | 16-core+ |

### 0.2 Python Environment

Install the stack in the repo's `.venv` (conda or venv). Dependency groups:

- **Core ML**: `torch`, `torchvision`, `torchaudio` (CUDA 12.1 wheels).
- **TDA**: `ripser`, `gudhi`, `giotto-tda`, `persim`.
- **Adversarial**: `adversarial-robustness-toolbox` (ART), `autoattack`, `robustbench`.
- **Changepoint**: `ruptures`.
- **Conformal**: `mapie`, `crepes`.
- **Tracking**: `wandb`.
- **Utilities**: `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `pandas`, `tqdm`, `pot`.

The frozen list lives in `requirements.txt` — install from there, not from this doc.

### 0.3 Project Structure

```
prism/
├── configs/            YAML configs; see src/config.py loader
├── data/               CIFAR-10, ImageNet subset
├── models/             Pretrained weights + reference_profiles.pkl, calibrator.pkl, ensemble_scorer.pkl
├── src/
│   ├── tamm/           Topological Activation Manifold Monitor
│   ├── cadg/           Conformal Adversarial Detection Guarantee
│   ├── sacd/           Sequential Adversarial Campaign Detection
│   ├── tamsh/          Topology-Aware MoE Self-Healing
│   ├── memory/         Persistence-diagram immune memory
│   ├── federation/     Federated sharing protocol
│   └── prism.py        Main wrapper class
├── experiments/{feasibility, calibration, campaign, evaluation, ablation}/
├── scripts/            Build-profile / calibrate / train scripts
├── notebooks/
├── paper/              LaTeX source
└── tests/
```

### 0.4 Compute strategy — local-first, remote-later

| Task | Where | Why |
| --- | --- | --- |
| Writing code, debugging | Local | Free, fast iteration |
| TDA feasibility benchmark | Local GPU | One-off |
| Ensemble training (`train_ensemble_scorer.py`) | Local GPU | ~1–2 h on RTX 3060 |
| Conformal calibration + FPR verification | Local | Minutes |
| FGSM / PGD / Square evaluation (n=1000 × 5 seeds) | **Local GPU** | Target window before cloud |
| Expert sub-network training | Local GPU | Seconds–minutes |
| CW-L2 evaluation | **Remote GPU (Thundercompute)** — later | Minutes/sample; only run after local targets pass |
| AutoAttack evaluation | **Remote GPU (Thundercompute)** — later | 30–60 min per 1000 samples |
| Paper writing | Local (Overleaf) | No GPU needed |

**Rule.** The local three-attack result (FGSM + PGD + Square, n=1000, 5 seeds)
must hit all publishable targets before any remote run. No point paying to
validate a broken pipeline.

---

## Data Splits & Split Hygiene (READ THIS BEFORE ANY RUN)

All splits use a **single source of truth**, CIFAR-10 test indices
(`torchvision.datasets.CIFAR10(train=False)`). The logistic ensemble scorer
is the only component trained on CIFAR-10 *train* images — everything
topological lives on the test split.

| Purpose | Source | Index range | Size | Built by |
| --- | --- | --- | --- | --- |
| Topological self-profile (medoid per layer) | CIFAR-10 test | 0 – 4 999 | 5 000 | [`scripts/build_profile_testset.py`](prism/scripts/build_profile_testset.py) |
| Conformal calibration | CIFAR-10 test | 5 000 – 6 999 | 2 000 | [`scripts/calibrate_ensemble.py`](prism/scripts/calibrate_ensemble.py) |
| FPR verification (sanity gate) | CIFAR-10 test | 7 000 – 7 999 | 1 000 | [`scripts/compute_ensemble_val_fpr.py`](prism/scripts/compute_ensemble_val_fpr.py) |
| Attack evaluation (paper) | CIFAR-10 test | 8 000 – 9 999 | 2 000 pool | [`experiments/evaluation/run_evaluation_full.py`](prism/experiments/evaluation/run_evaluation_full.py) |
| Ensemble logistic training | CIFAR-10 **train** | — | up to 50 000 | [`scripts/train_ensemble_scorer.py`](prism/scripts/train_ensemble_scorer.py) |

These index ranges **are** the live values in `configs/default.yaml` and
`src/config.py` (`PROFILE_IDX`, `CAL_IDX`, `VAL_IDX`, `EVAL_IDX`). The 2 000-item
eval pool is sampled at n=1 000 per seed × 5 seeds = 5 000 pooled Wilson-CI
observations; cross-seed variance is real because each seed draws a distinct
1 000-image subset.

> **Note on a prior proposal.** An earlier draft of this plan proposed
> cal n=3 000 / val n=1 000 / eval n=1 000 (indices 5 000–7 999 / 8 000–8 999 /
> 9 000–9 999). That proposal was **not** adopted — the 2 000-sample cal is
> statistically sufficient at `{L1: 0.10, L2: 0.03, L3: 0.005}` with the
> 0.7 `CAL_ALPHA_FACTOR` slack, and a 2 000-item eval pool is required for
> independent multi-seed subsampling. Any future change to these splits must
> update the YAML + `src/config.py` + this table in the same commit.

### Hard pipeline gate

Any change to the ensemble scorer **requires** re-running the following, in
order, or the conformal guarantee is void:

1. `python scripts/train_ensemble_scorer.py` → `models/ensemble_scorer.pkl`
2. `python scripts/calibrate_ensemble.py` → `models/calibrator.pkl`
3. `python scripts/compute_ensemble_val_fpr.py` → `experiments/calibration/ensemble_fpr_report.json` — all three tiers must be `passed: true`
4. `python experiments/evaluation/run_evaluation_full.py --multi-seed --attacks FGSM PGD Square`

Skipping any step silently breaks the FPR claim. Two of the four existing
result files (`results_n500_20260419.json`, `results_n500_retrained_20260419.json`)
were produced by skipping step 2 or 3; their numbers are *not* publishable.

> ⚠️ **Never let adversarial examples contaminate the profile or calibration
> sets.** Those must be 100 % clean inputs. Contamination breaks the conformal
> guarantee.

---

## Phase 1: TDA Foundation (Weeks 1–5)

### Step 1.1: Literature Deep-Dive (Week 1–2)

Reading order (mandatory):

1. Munch (2017) — *A User's Guide to TDA*. 20 pages, from scratch.
2. Edelsbrunner & Harer (2010) — *Computational Topology*, ch. 1–4.
3. Carlsson (2009) — *Topology and Data*.
4. Angelopoulos & Bates (2023) — *A Gentle Introduction to Conformal Prediction*.
5. Adams & MacKay (2007) — *Bayesian Online Changepoint Detection*.
6. Croce et al. (ICML 2022) — *Evaluating Adversarial Robustness of Adaptive Test-time Defenses*.
7. Wang et al. (IEEE Access 2022) — **RAILS** (closest competitor).
8. Gebhart & Schrater — *Adversary Detection via Persistent Homology* (closest TDA+adversarial work).
9. LANL `tda-adversarial-detection` — GitHub repo.

Output: a 2-page positioning document that explicitly differentiates PRISM
from each paper above.

### Step 1.2: TDA Feasibility Benchmark (Week 2–3) — critical gate

> 🚨 **Do this before building anything.** If TDA is too slow, the project
> pivots to approximations.

Implementation: [`experiments/feasibility/tda_benchmark.py`](prism/experiments/feasibility/tda_benchmark.py) — hooks ResNet-18 layers, sweeps subsample sizes, reports ripser + Wasserstein time.

| TDA time (200 points, one layer) | Action |
| --- | --- |
| < 10 ms | Proceed as planned |
| 10–50 ms | Proceed with subsampling strategy |
| 50–200 ms | Switch to cubical complexes / landmarks |
| > 200 ms | STOP. Consider `ripser++` (GPU) |

**Observed on RTX / A100 (n_subsample=200, 3 layers):** ~30–40 ms per image.
Proceed with current plan.

### Step 1.3: Build the Topological Self-Profile (Week 3–5)

**What "training" means here.** Pass thousands of clean images through the
frozen backbone, compute a persistence diagram per layer, save the Wasserstein
**medoid** per layer as the reference "this is what clean looks like."

- Activation extractor: [`src/tamm/extractor.py`](prism/src/tamm/extractor.py) — `ActivationExtractor` with forward hooks on `layer2/3/4` of ResNet-18. `layer1` is intentionally skipped (too shallow).
- Persistence computation: [`src/tamm/tda.py`](prism/src/tamm/tda.py) — `TopologicalProfiler` wraps `ripser` (max_dim=1, n_subsample=200) and exposes `compute_diagram`, `compute_reference_medoid`, `anomaly_score`.
- Per-layer weighted scorer: [`src/tamm/scorer.py`](prism/src/tamm/scorer.py) — `TopologicalScorer` aggregates per-dim Wasserstein distances with layer weights {L2: 0.15, L3: 0.30, L4: 0.55} and dim weights [0.5, 0.5] (H0 + H1).
- Driver: [`scripts/build_profile_testset.py`](prism/scripts/build_profile_testset.py) — runs the extractor + profiler over CIFAR-10 test idx 0–4999 and saves `models/reference_profiles.pkl` plus precomputed clean scores to `experiments/calibration/clean_scores.npy` for downstream calibration.

**Empty-diagram handling.** Shallow layers sometimes return empty H1 diagrams.
Both `TopologicalProfiler.anomaly_score` and `TopologicalScorer.score` treat
an empty diagram as a zero contribution (not a skip). Silent skipping would
inflate the final score by shrinking the denominator.

---

## Phase 2: Conformal Calibration (Weeks 6–9)

### Step 2.1: Split-Conformal Calibrator

Implementation: [`src/cadg/calibrate.py`](prism/src/cadg/calibrate.py) — `ConformalCalibrator`.

Key formulas:

- Quantile: `q_idx = ceil((n + 1) * (1 - α))`, clamped to [1, n], 0-indexed lookup. Strict `>` in `classify()` — matches the standard exchangeability proof.
- Published tiers (`self.alphas`):
    - **L1** — 10 % FPR — monitor/log tier.
    - **L2** — 3 % FPR — purification tier.
    - **L3** — 0.5 % FPR — expert-route / reject tier.
- Conservative calibration option: `calibrate(scores, alphas=cal_alphas)` computes thresholds at `cal_alphas` (e.g. 70 % of published) but records `self.alphas = published`. `get_coverage_report()` always verifies against `self.alphas`. The 30 % slack absorbs distribution shift between cal and val splits.

### Step 2.2: Ensemble Scorer (the actual detector)

Implementation:

- [`src/tamm/persistence_stats.py`](prism/src/tamm/persistence_stats.py) — extracts a fixed-length feature vector (per-layer, per-dim summary statistics) from a persistence diagram.
- [`src/cadg/ensemble_scorer.py`](prism/src/cadg/ensemble_scorer.py) — `PersistenceEnsembleScorer` combines the base Wasserstein score with a logistic-regression probability fit on those features. Parameters `logit_shift` and `w_score_mean` are data-derived (no magic numbers).

Training driver: [`scripts/train_ensemble_scorer.py`](prism/scripts/train_ensemble_scorer.py) — trains the logistic on CIFAR-10 *train* images (2000 clean + 2000 adversarial). The adversarial mix has been broadened for publishability:

| Component | Share | Parameters |
| --- | --- | --- |
| FGSM (L∞) | 34 % | eps = 8/255 |
| PGD (L∞) | 33 % | eps = 8/255, 20 steps, step = 2/255 |
| Square (L∞) | 33 % | eps = 8/255, 1000 queries |

CW-L2 is **not** in the training mix; it is evaluated on remote GPU only.

Recalibration driver: [`scripts/calibrate_ensemble.py`](prism/scripts/calibrate_ensemble.py) — runs the ensemble on cal/val splits, fits thresholds at `cal_alphas = {L1: 0.07, L2: 0.021, L3: 0.0035}` (70 % of published), verifies empirical FPR ≤ published α on the val split with `tolerance=0.0`. Saves `models/calibrator.pkl`.

### Step 2.3: Verify the conformal guarantee

Driver: [`scripts/compute_ensemble_val_fpr.py`](prism/scripts/compute_ensemble_val_fpr.py) — produces `experiments/calibration/ensemble_fpr_report.json` with Wilson 95 % CIs for each tier. Required to read `passed: true` for all of L1/L2/L3 before proceeding to evaluation.

---

## Phase 3: Sequential Campaign Monitor — L0 (Weeks 10–13)

### Step 3.1: BOCPD

Implementation: [`src/sacd/bocpd.py`](prism/src/sacd/bocpd.py) — `BayesianOnlineChangepoint` implements Adams & MacKay 2007 with a Gaussian predictive. Exposes `update(score) → cp_prob` and `reset()`.

**Bounded memory.** The run-length distribution and observation buffer are
both truncated at `max_run_length=500` and renormalized on overflow. Without
this, a long inference stream would leak memory. The truncation makes the
update O(T²) in the instantaneous window, not O(T²) globally — important for
throughput claims in the paper.

### Step 3.2: L0 Monitor

Implementation: [`src/sacd/monitor.py`](prism/src/sacd/monitor.py) — `CampaignMonitor`:

- Rolling score buffer (`window_size=100`).
- `hazard_rate=1/200` (expected run length); tune on a synthetic grid.
- On `cp_prob > cp_threshold` (default 0.3–0.5), sets `l0_active=True`.
- When L0 active, `calibrator.classify` applies `l0_factor=0.8` to tier thresholds — single-input escalation during a detected campaign.

**Hazard tuning protocol.** Simulate 100 clean queries (∼ N(0.1, 0.02)) then 20
probes (∼ N(0.3, 0.05)); grid-search `hazard ∈ {1/50, 1/100, 1/200, 1/500}`
and pick the smallest hazard that detects within 20 probe queries with no
false positives on the clean prefix.

**Publishable-paper caveat (QUARANTINED).** The existing
`experiments/campaign/results.json` uses synthetic score streams and **is not
publishable as-is**. Before the paper can include any campaign-detection
result, BOCPD must be rerun on *real* score streams emitted by
`run_evaluation_full.py` on FGSM / PGD / Square. Until then, the synthetic
result is a sanity test only; do not cite it in Table 1 or the abstract.

---

## Phase 4: Topology-Aware MoE Self-Healing (Weeks 14–17)

### Step 4.1: Experts and Gating

Implementation: [`src/tamsh/experts.py`](prism/src/tamsh/experts.py) — `ExpertSubNetwork` (3-layer MLP, `hidden_dim=256`) and `TopologyAwareMoE` (Wasserstein-nearest-reference expert selection). BatchNorm inside the MLP requires `eval()` at single-image inference time; `TopologyAwareMoE.select_expert` calls `.eval()` defensively.

Gating details: [`src/tamsh/gating.py`](prism/src/tamsh/gating.py) — Wasserstein-based K-medoids clustering over persistence diagrams.

### Step 4.2: Training

Driver: [`scripts/train_experts.py`](prism/scripts/train_experts.py) — clusters clean activations (K=4) by Wasserstein distance on persistence diagrams; one expert per cluster, trained to reconstruct the activations of the span it replaces (MSE, Adam 1e-3, 50 epochs). Each expert's "reference diagram" is the cluster medoid.

Validation target: held-out cluster MSE < 0.05 on clean activations (adjust
for activation scale).

---

## Phase 5: Immune Memory + Federation (Weeks 18–20)

- Persistence-diagram memory store: [`src/memory/immune_memory.py`](prism/src/memory/immune_memory.py) — `ImmuneMemory.store(diagram, attack_type, level)`, `.match(input_diagram)` with Wasserstein-threshold matching.
- Federation: [`src/federation/protocol.py`](prism/src/federation/protocol.py), [`src/federation/manager.py`](prism/src/federation/manager.py) — peers share persistence-diagram attack signatures (not activations, not weights). Smoke test in [`experiments/federation/run_federation_demo.py`](prism/experiments/federation/run_federation_demo.py).

---

## Phase 6: Full PRISM Integration + Evaluation (Weeks 21–29)

### Step 6.1: The `PRISM` Class

Implementation: [`src/prism.py`](prism/src/prism.py) — wraps any frozen backbone. `PRISM.defend(x)` returns `(prediction, response_level, metadata)` with `response_level ∈ {PASS, L1, L2, L3, L3_REJECT}`.

Defend pipeline (per input):

1. Extract activations from `layer2/3/4` via `ActivationExtractor`.
2. Compute persistence diagrams via `TopologicalProfiler`.
3. Ensemble score via `PersistenceEnsembleScorer` (logistic + base Wasserstein).
4. Check `ImmuneMemory.match` — fast-path for known attack signatures.
5. `CampaignMonitor.process_score` — updates BOCPD, returns `l0_active`.
6. `ConformalCalibrator.classify(score, l0_active)` — emits tier.
7. Execute per-tier response (PASS / log / purify / route-through-expert / reject).

**Pickle safety.** `PRISM._load` gates on an allow-list of paths
(`calibrator.pkl`, `reference_profiles.pkl`, `experts.pkl`, `ensemble_scorer.pkl`).
Pickle executes arbitrary code on load — never accept external pkl files.

### Step 6.2: Attack Evaluation

Driver: [`experiments/evaluation/run_evaluation_full.py`](prism/experiments/evaluation/run_evaluation_full.py) — current canonical evaluator. Supports single-seed and `--multi-seed` modes, per-attack Wilson 95 % CIs, per-tier FPR breakdown, and a latency benchmark (n=200 clean images).

Attack parameters (RobustBench convention):

| Attack | Norm | eps | Steps / budget | Run locally? |
| --- | --- | --- | --- | --- |
| FGSM | L∞ | 8/255 | 1 | Yes |
| PGD | L∞ | 8/255 | 40 steps, step = 2/255, 1 random init | Yes |
| Square | L∞ | 8/255 | 5 000 queries, 1 restart | Yes |
| CW-L2 | L2 | c=1.0 | 100 iter, bs=9 bin-search, bs=64 | **No — remote GPU only** |
| AutoAttack | L∞ | 8/255 | Standard (APGD-CE + APGD-T + FAB + Square) | **No — remote GPU only** |

Canonical local command (paper table 1):

```
python experiments/evaluation/run_evaluation_full.py \
  --multi-seed --seeds 42 123 456 789 999 \
  --n-test 1000 --attacks FGSM PGD Square \
  --output experiments/evaluation/results_paper.json
```

> **Note on the existing `results_paper.json` (n=300, single-seed).** That file
> is stale and will be overwritten by the canonical multi-seed run above. It
> also **fails the latency gate** (`mean = 107.98 ms`, `pass = false`) and must
> not be cited. The only currently-valid latency baseline is
> `results_n500_planA.json` at **92 ms**.

Canonical remote command (after local targets pass):

```
python experiments/evaluation/run_evaluation_full.py \
  --multi-seed --seeds 42 123 456 789 999 \
  --n-test 1000 --attacks FGSM PGD Square CW AutoAttack \
  --output experiments/evaluation/results_paper_remote.json
```

### Step 6.3: Campaign Detection Test

Driver: [`experiments/campaign/run_campaign.py`](prism/experiments/campaign/run_campaign.py) — simulates a probe → full-attack sequence and asserts that L0 activates within the 20-query target. For publication, see the "real score stream" note in Phase 3.

### Step 6.4: Ablation

Driver: [`experiments/ablation/run_ablation_paper.py`](prism/experiments/ablation/run_ablation_paper.py). Configurations:

| Configuration | TAMM | Ensemble | SACD (L0) | TAMSH | Expected TPR |
| --- | --- | --- | --- | --- | --- |
| Full PRISM | ✅ | ✅ | ✅ | ✅ | Highest |
| No L0 | ✅ | ✅ | ❌ | ✅ | Lower on campaign attacks |
| No MoE | ✅ | ✅ | ✅ | ❌ | Lower on L3 recovery |
| No Ensemble (TDA only) | ✅ | ❌ | ❌ | ❌ | Baseline — exposes ensemble's contribution |

Ablation must run at **n=1000 × 5 seeds** (current file at n=500 has CIs that
clip the L1 cap).

---

## Phase 7: Paper Writing (Weeks 30–34)

### 7.1 Setup

- Overleaf NeurIPS 2027 template.
- Paper skeleton already present under [`prism/paper/`](prism/paper/) — `main.tex`, `sections/intro.tex`, `sections/related.tex`, `prism.bib`, `neurips_2025.sty`.

### 7.2 Writing Order

| Order | Section | Why |
| --- | --- | --- |
| 1 | Experiments | Numbers first — they constrain claims. |
| 2 | Method | Describe only what produced the numbers. |
| 3 | Related Work | Position against what's published. |
| 4 | Introduction | Sharpen after everything else exists. |
| 5 | Discussion + Conclusion | Reflect on what worked. |
| 6 | Abstract | Last — summarize actual contributions. |

### 7.3 Key Figures

- [`paper/figures/fig1_architecture.png`](prism/paper/figures/fig1_architecture.png) — system architecture.
- [`paper/figures/fig2_persistence_viz.py`](prism/paper/figures/fig2_persistence_viz.py) — clean vs. adversarial activation cloud + persistence diagram (2×2 panel).
- **Figure 3 (TO ADD)** — per-tier FPR bars with Wilson CIs; built from `ensemble_fpr_report.json`.

### 7.4 Claim recalibration (see §Narrative Recalibration)

**Do not** write "TDA detects adversarial inputs." **Do** write
"A conformal-calibrated ensemble of persistence-diagram statistics, escalated
by BOCPD and routed by topology-gated experts, provides multi-tier adversarial
**detection** with a strict FPR guarantee and **availability preservation**
under L3 rejection."

### 7.5 Threat model block (paste into Experiments §)

Every adversarial-defense paper since 2019 is expected to state this
explicitly. Missing any line is a desk-reject risk.

| Dimension | Value |
| --- | --- |
| Norm / budget | L∞, ε = 8/255 (FGSM, PGD, Square, AutoAttack); L2, c=1.0 (CW) |
| Gradient access | White-box (FGSM, PGD, CW, APGD variants of AutoAttack); black-box (Square) |
| Attacker knowledge | **Unaware** — attacker does not see the PRISM score. **Aware/adaptive** results in §7.6. |
| Optimisation target | Misclassification of the backbone. Adaptive variant: PRISM ensemble-score loss. |
| Compute budget per sample | FGSM ≤ 1 grad; PGD ≤ 40 grads; Square ≤ 5000 queries; CW ≤ 100 iter; AutoAttack standard |
| Defence type | **Detection + rejection**; not certified robustness, not input purification |

### 7.6 Adaptive attack evaluation (REQUIRED for credibility)

Implement one adaptive attacker: PGD with the loss replaced by the ensemble's
scalar score (i.e. the attacker *minimises* the PRISM anomaly probability
while keeping the backbone misclassification objective as a constraint).
Report TPR degradation vs. the standard PGD. This is the minimum-viable
adaptive-attack bar for a robustness venue; skipping it is a reviewer red
flag. Plan: 1 week, one new script at
`experiments/evaluation/run_adaptive_pgd.py`, reuses the existing PGD harness.

### 7.7 Reproduced baselines (REQUIRED — do not cite literature numbers)

Reproduce at least two comparable detectors on the same n=1000 eval pool and
the same backbone (ResNet-18 on CIFAR-10):

1. **LID** (Ma et al. 2018) — local intrinsic dimensionality detector. Small,
   fast, a standard baseline.
2. **Mahalanobis** (Lee et al. 2018) — class-conditional Gaussian in feature
   space. Ubiquitous in adversarial-detection papers.
3. *(Optional)* **Feature squeezing** (Xu et al. 2018) — cheap, widely cited.

Report head-to-head TPR at matched L1/L2/L3 FPR tiers (or at each method's
default operating point if tier-matching is infeasible). Without this, the
paper's comparison table is literature-cited not reproduced and reviewers
will discount every number.

---

## Reaching Publication Metrics

This is the current-state → target mapping, with the exact moves that close
each gap.

### Target (from `PRISM Testing, Training & Validation Guide.md`)

- Mean TPR ≥ 85 % across FGSM / PGD / Square (≥ 90 % on PGD + AutoAttack).
- Per-tier FPR: L1 ≤ 10 %, L2 ≤ 3 %, L3 ≤ 0.5 %.
- Latency < 100 ms / image on GPU.
- Ablation: each component measurably contributes (paired-bootstrap p < 0.05).

### Current (best local run: `results_n500_planA.json`, n=500, GPU)

| Metric | Observed | Target | Gap |
| --- | --- | --- | --- |
| FGSM TPR | 0.832 | 0.85 | −1.8 pp |
| PGD TPR | 1.000 | 0.95 | +5.0 pp ✓ |
| Square TPR | 0.654 | 0.85 | **−19.6 pp** |
| L1 FPR | 0.032 | 0.10 | ✓ |
| L2 FPR | 0.016 | 0.03 | ✓ |
| L3 FPR | 0.008 | 0.005 | **+0.3 pp** |
| Latency (mean) | 96 ms | < 100 ms | ✓ |

### Moves to close the TPR gap (Square is the critical one) — FORECASTS, NOT RESULTS

> ⚠️ **These numbers are projections, not measurements.** A reviewer will
> reject any claim that is not post-hoc measured. Report **pre- and post-fix**
> numbers for every move below; do not cite the projected lifts in the paper.

1. **Broaden the ensemble training mix.** Add Square to the training set alongside FGSM and PGD (tri-split 34/33/33). Pure diversity in feature space; the logistic is a linear classifier and cannot overfit in the way a deep model would. **Projected lift (unmeasured): +10–20 pp Square TPR.** Touches [`scripts/train_ensemble_scorer.py`](prism/scripts/train_ensemble_scorer.py). **Caveat:** if post-fix Square TPR < 0.80 this is the canary that TDA+logistic cannot generalise to gradient-free attacks, and feature design must change.
2. **Add a DCT-high-frequency-energy feature.** Square perturbs pixel statistics more than latent topology. Append a 37th feature (log high-frequency DCT energy on the original pixel image) to the persistence-feature vector. Touches [`src/tamm/persistence_stats.py`](prism/src/tamm/persistence_stats.py). Negligible latency cost. **Projected lift (unmeasured): +5–10 pp Square TPR.**
3. **Per-attack isotonic recalibration (optional).** Apply a monotone map from raw logistic probability to a recalibrated score per attack family on a held-out TRAIN slice. Tightens separation without breaking conformal guarantees (we re-run `calibrate_ensemble.py` afterward). **Projected lift (unmeasured): +2–5 pp on the weakest attack.**

### Moves to close the L3 FPR gap

4. **Tighten `cal_targets` from 80 % → 70 % of published.** Set in [`scripts/calibrate_ensemble.py`](prism/scripts/calibrate_ensemble.py) — new targets `{L1: 0.07, L2: 0.021, L3: 0.0035}`. On current val distribution this pushes L3 FPR to ≈ 0.003. **Projected cost: ~1–2 pp TPR across the board**, expected to be absorbed by moves 1–3. ✅ **Already implemented in code** (`CAL_ALPHA_FACTOR=0.7`); awaiting measured effect.
5. ~~**Increase calibration set to n=3000.**~~ **Rejected** — the 2 000-item cal set is statistically sufficient at the target alphas and the 2 000-item eval pool is required for independent multi-seed subsampling. See the Note in §Data Splits.

### Verification after the moves

Run, in order:

1. `python scripts/train_ensemble_scorer.py --n-train 2000`
2. `python scripts/calibrate_ensemble.py`
3. `python scripts/compute_ensemble_val_fpr.py` → expect all three tiers `passed: true`, L3 empirical FPR in [0.002, 0.005].
4. `python experiments/evaluation/run_evaluation_full.py --multi-seed --seeds 42 123 456 789 999 --n-test 1000 --attacks FGSM PGD Square`
5. `python experiments/ablation/run_ablation_paper.py --n-test 1000 --seeds 42 123 456 789 999`

Reject the run and investigate if any of:

- Step 3 reports `passed: false` on any tier.
- Step 4 pooled TPR < 0.85 on any attack.
- Step 4 per-tier FPR CI upper bound exceeds the target.
- Latency mean > 100 ms.

### What goes to remote GPU after local targets pass

Only CW-L2 and AutoAttack, using the same multi-seed harness and n=1000.
Merge outputs into `results_paper_remote.json`; Table 1 in the paper is the
union of local + remote seeds.

> **Gate.** The Thundercompute CW + AutoAttack run is **contingent**, not
> scheduled. Trigger only if all three of the local attacks (FGSM, PGD, Square)
> satisfy the pooled Wilson 95 % CI lower bound ≥ 0.80 TPR **and** all three
> FPR tiers pass on the val split. Running CW/AutoAttack on a broken local
> pipeline wastes money.

---

## Scope & Limitations (paste into Discussion §)

Honest statement of what this paper covers and what it does not. Fill this
section in the paper — reviewers who find undisclosed limitations discount
harder than reviewers who find disclosed ones.

- **Single dataset.** CIFAR-10 only. No CIFAR-100, no ImageNet. Scaling to
  larger-resolution inputs requires re-evaluating TDA subsample strategy;
  the medoid computation is O(N²) in diagram count.
- **Single backbone.** ResNet-18 only. No ViT, no larger ResNet, no
  adversarially-trained backbone. The architecture-agnostic claim rests on
  the forward-hook abstraction, not on empirical cross-architecture
  generalisation.
- **Detection-only defence.** PRISM rejects or routes; it does not purify or
  certify. No robustness certificate is produced — the guarantee is on
  clean-input FPR, not on adversarial TPR.
- **Unaware attacker.** Results are against attackers that do not see the
  PRISM score. An adaptive-PGD result (§7.6) must accompany these numbers;
  without it the threat model is incomplete.
- **Synthetic campaign test.** The existing `experiments/campaign/results.json`
  is a simulated score stream. Paper version must use real attack scores or
  be labelled "simulated campaign, sanity test only."
- **GPU latency 96 ms mean.** Acceptable for server inference; marginal for
  real-time systems (autonomous vehicles, robotics); slow for interactive
  inference (< 50 ms expectation).
- **TDA is features, not the detector.** The ablation (Full 0.8847 vs
  TDA-only 0.6213) makes this explicit. Claims must reflect it.
- **Baselines are reproduced only for LID and Mahalanobis** (§7.7). RAILS is
  architecturally incompatible (it replaces layers) and is discussed, not
  run head-to-head.

---

## Narrative Recalibration

The base-TDA Wasserstein channel alone cannot carry the paper. The ablation
table (`experiments/ablation/results_ablation_paper.md`) is the primary
evidence:

| Configuration | Mean TPR |
| --- | --- |
| Full PRISM | 0.8847 |
| No L0 | 0.8867 |
| No MoE | 0.8860 |
| **TDA only (no ensemble)** | **0.6213** |

The −27 pp drop from Full → TDA-only is almost entirely the ensemble. L0 and
MoE contribute < 1 pp each in the current configuration. The paper must say
this.

### Reframed contributions (for abstract + intro)

1. A **conformal-calibrated ensemble over persistence-diagram statistics** — a tiered detector with a strict published-FPR guarantee, verified on a held-out val split. This is the primary contribution; the ensemble is the detector, and the paper's framing must reflect that.
2. A **BOCPD-based session-level escalation** (L0) — temporally-aware threshold tightening; the strongest standalone novelty claim, with no direct prior art in the adversarial-defense setting.
3. A **topology-aware MoE routing for availability preservation** (TAMSH) — Wasserstein-gated expert selection over persistence diagrams. Note: this **preserves availability** under L3 rejection (every input gets a prediction path); it does **not** claim to restore correct classification on adversarial inputs.
4. A **persistence-diagram immune memory** — fast-path recall of known attack signatures without storing raw inputs.

> **Mandatory global edit.** Every occurrence of "topology-aware PRISM",
> "topological defense", or similar phrasing must be replaced in the abstract,
> intro, and contribution list with "conformal-calibrated persistence-feature
> ensemble" (or the equivalent phrase above). Reason: the TDA-only ablation
> (0.6213 vs Full 0.8847) directly contradicts the "topological defense"
> framing; a reviewer reading both the abstract and the ablation will
> immediately flag it.

### Honest limitations paragraph

- **Square under-detection (pre-broadening).** Gradient-free attacks leave weaker latent-topology signatures than gradient-based ones. Mitigated by training-mix broadening and the pixel-space DCT feature; report both pre- and post-fix numbers.
- **Medoid computation is O(N²).** Current profile uses 5000 diagrams per layer. Scaling to ImageNet requires landmark subset or sliced-Wasserstein barycenter. Flagged as scalability limitation; not claimed as solved.
- **Campaign test is synthetic.** Current `experiments/campaign/results.json` uses synthetic scores. The paper version must run BOCPD on a real-attack score stream, or label the result explicitly.
- **Base-TDA weakness.** We do not claim TDA alone is sufficient. The ensemble is the detector; TDA provides the features.
- **PGD at 100 % across all configs** (including TDA-only) likely reflects PGD-induced activation shifts large enough for any scorer to detect. Claim must be "PRISM preserves PGD detection at the architecture-agnostic backbone" — not "PRISM solves PGD."

---

## Sanity / Debugging Reference

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| TPR < 50 % on all attacks | Anomaly scores not separating | Check subsample size (n=200); confirm hooks on the right layers |
| FPR > 30 % on clean | Profile contaminated or cal set too small | Rebuild profile on strictly clean data; cal ≥ 2000 |
| L0 never triggers | `hazard_rate` too low or `cp_threshold` too high | Lower `cp_threshold` to 0.2; raise `hazard_rate` to 1/50 |
| TDA > 500 ms / image | Too many subsample points | Reduce n_subsample to 50; switch to ripser++/cubical |
| Expert val MSE > 0.3 | Too-small expert or under-trained | Increase `hidden_dim` to 512; train 100 epochs; check cluster quality |
| Regression after ensemble change | Skipped recalibration | Re-run the full pipeline-gate (§Data Splits) |

---

## Phase Checklist (Weeks 0–34)

- [ ] **Week 0** — Environment, repo, data.
- [ ] **Week 1–2** — Literature + positioning document.
- [ ] **Week 2–3** — TDA feasibility gate.
- [ ] **Week 3–5** — Build topological self-profile.
- [ ] **Week 6–7** — Conformal calibration.
- [ ] **Week 8–9** — Verify conformal coverage; Pareto curve.
- [ ] **Week 10–11** — Implement BOCPD for L0.
- [ ] **Week 12–13** — Campaign detection validation (real score stream).
- [ ] **Week 14–15** — Cluster activations; design experts.
- [ ] **Week 16–17** — Train K=4 experts; topology-aware gating.
- [ ] **Week 18–19** — Immune memory store.
- [ ] **Week 20** — Federated protocol demo.
- [ ] **Week 21–23** — Full integration; smoke tests.
- [ ] **Week 24–26** — FGSM / PGD / Square evaluation locally (multi-seed).
- [ ] **Week 26** — Reproduce LID + Mahalanobis baselines on same eval pool (§7.7).
- [ ] **Week 27** — Adaptive PGD against ensemble score (§7.6).
- [ ] **Week 27** — CW + AutoAttack on remote GPU **only if** local pooled TPR ≥ 0.80 on all three attacks.
- [ ] **Week 28–29** — Ablation (n=1000, 5 seeds).
- [ ] **Week 30–31** — Experiments + Method sections.
- [ ] **Week 32** — Related Work + Introduction.
- [ ] **Week 33** — Discussion, Conclusion, Abstract.
- [ ] **Week 34** — Polish, internal review, submit.

---

## Appendix A: Audit Changelog (compressed)

One-paragraph summary of the April 2026 pipeline audit. All items are
resolved in code; detail lives in git history, not here. Retain this section
as provenance for reviewers who ask how the pipeline was hardened.

The audit found eleven silent mismatches between plan, code, and results.
Training-set composition (Square missing from the ensemble mix), calibration
hygiene (hard-coded splits duplicated across four scripts, dead `_STD`
override, undisclosed 80→70 % cal-alpha slack), scoring arithmetic
(`TopologicalScorer.score` is an average, not a sum), and pipeline
sequencing (two result files produced by skipping `calibrate_ensemble.py`
after ensemble retrain) were the main categories. All are now fixed at the
source: `src/config.py` is the single source of truth, the hard pipeline
gate in §Data Splits enforces the re-run order, and `CAL_ALPHA_FACTOR=0.7`
is a named constant in `configs/default.yaml`. Two residual methodological
notes belong in the paper rather than the code: (i) the BOCPD
`max_run_length=500` truncation is an approximation that must be disclosed
in the Method section, and (ii) PGD TPR=1.00 is a backbone property (any
scorer detects it), not a PRISM-specific claim. Current canonical results:
`results_n500_planA.json` (local) and `results_ablation_paper.json`
(ablation). Stale artefacts `results_n500_20260419.json`,
`results_n500_retrained_20260419.json`, and the n=300 `results_paper.json`
are **not publishable** and should be deleted once the canonical multi-seed
run overwrites them.
