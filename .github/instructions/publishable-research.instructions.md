now---
description: "Use when writing or modifying PRISM experiments, evaluation scripts, paper sections, result summaries, or claims. Enforces publication-grade correctness, reproducibility, and no fabricated data."
name: "PRISM Publishable Research Rules"
applyTo:
  - "prism/**/*.py"
  - "prism/**/*.md"
  - "prism/paper/**/*.tex"
  - "prism/**/*.json"
  - "PRISM*.md"
---

# PRISM Publishable Research Rules

## Non-Negotiable Integrity

- Never invent, estimate, or round into existence any metric, table value, confidence interval, or citation detail.
- Never present synthetic or placeholder outputs as real experimental results.
- If a value cannot be verified from code outputs or checked artifacts, mark it as `UNVERIFIED` and request confirmation.
- Keep novelty claims conservative and evidence-backed; avoid "first" claims unless explicitly validated against cited literature.

## Data and Split Hygiene

- Preserve strict separation between profile, calibration, and validation/test sets.
- Do not allow adversarial samples into profile/calibration clean sets.
- Do not evaluate on data used for threshold fitting.
- When changing data pipelines, add or update checks that assert split non-overlap.

## Reproducibility Requirements

- For any reported result, include: command used, dataset/split, random seed(s), key config values, and artifact path.
- Prefer writing results to new timestamped output files; do not silently overwrite canonical paper artifacts.
- Keep scripts deterministic where feasible (seed Python/NumPy/PyTorch; document nondeterministic components).
- When modifying experiment logic, update usage docs to reflect exact runnable commands.

## Metrics and Reporting Discipline

- Report metrics with sample counts and clear scope (attack set, dataset, tier).
- Include uncertainty when available (CI or std) and avoid overclaiming from single runs.
- Keep paper/report numbers synchronized with source JSON/NPY artifacts.
- If paper and artifacts disagree, treat artifact-derived values as source of truth until rerun confirms otherwise.

## Validation Before Claiming Success

- Run relevant tests for touched modules before finalizing changes.
- For evaluation changes, run a small smoke evaluation first, then full evaluation when required.
- Verify key guardrails from project docs (e.g., FPR tier ordering, threshold ordering, score sanity ranges).
- Do not declare a fix complete if tests/evaluations were not run; explicitly state what was and was not validated.

## Safe Editing of Research Assets

- Preserve comparability: do not change attack definitions, preprocessing, or metric formulas without documenting rationale.
- Keep backward-compatible output schema for result files used by analysis/paper scripts.
- When changing paper text, tie each quantitative claim to a concrete results artifact path.

## Communication Style for This Project

- Prefer precise, falsifiable wording over promotional language.
- Distinguish clearly between: implemented, tested, validated, and hypothesized.
- Surface risks and limitations explicitly when evidence is incomplete.
