# WRN-28-10 / CIFAR-10 Adaptive PGD Results

This folder contains the full WRN-28-10 adaptive-PGD λ-sweep results
referenced in `paper/sections/appendix.tex` Table~`tab:wrn_adaptive`.

## Protocol

- Backbone: WRN-28-10 (CIFAR-10)
- n_test = 1000, seed = 42, EVAL split
- PGD: 100 steps × 10 restarts, ε = 8/255 (Linf)
- BPDA: `--through-scorer` (attacks TAMM activation-matching + DCT-energy
  channels via straight-through estimator)
- λ sweep: {0.0, 0.5, 1.0, 2.0, 5.0, 10.0}
- FPR: 0.081 (calibrated, constant across λ)

## Files

| File | Lambdas | Provenance |
|---|---|---|
| `vastai_wrn_adaptive_pgd.jsonl`         | 0, 0.5, 1.0  | pid 4355 sequential, original sweep (`--lambdas 0 0.5 1 2 5 10` interrupted after λ=1.0). Append-only progress + lambda_done events. |
| `vastai_wrn_adaptive_pgd_l2.{json,jsonl}`  | 2.0    | pid 8213 parallel rerun (split job, GIL-bound workload, 4× wall-clock speedup) |
| `vastai_wrn_adaptive_pgd_l5.{json,jsonl}`  | 5.0    | pid 8214 parallel rerun |
| `vastai_wrn_adaptive_pgd_l10.{json,jsonl}` | 10.0   | pid 8215 parallel rerun |
| `vastai_wrn_adaptive_pgd_merged.json`   | all 6  | Consolidated paper-facing JSON: per-λ TPR + CI95 + Δ vs ResNet-18 |

## Headline numbers (TPR @ L1 with 95% CI, FPR = 0.081 throughout)

| λ   | WRN-28-10 TPR | ResNet-18 TPR (5-seed pool) | Δ (pp) |
|-----|---------------|------------------------------|--------|
| 0   | 0.842 [0.818, 0.863] | 0.832 | +1.0 |
| 0.5 | 0.799 [0.773, 0.823] | 0.796 | +0.3 |
| 1   | 0.745 [0.717, 0.771] | 0.753 | -0.8 |
| 2   | 0.664 [0.634, 0.693] | 0.660 | +0.4 |
| 5   | 0.488 [0.457, 0.519] | 0.490 | -0.2 |
| 10  | 0.426 [0.396, 0.457] | 0.429 | -0.3 |

WRN and ResNet-18 agree within ±1pp at every λ on a 0.84 → 0.43 collapse
curve. **Interpretation:** adaptive-PGD vulnerability on CIFAR-10 is
dataset-level (channel-coverage gap of the attack), not architecture-level.
CIFAR-100 holds at TPR ∈ [0.92, 0.95] across the same sweep because the
attack does not cover StabilityV2, the dominant C100 channel.
