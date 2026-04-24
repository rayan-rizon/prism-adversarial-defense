# archive_pre_research_plan/

Frozen copies of pre-research-plan artifacts (≤ 2026-04-24).
Moved here — **not deleted** — to preserve forensic continuity.

**These results do NOT reflect the final paper submission numbers.**

Authoritative results live under
`prism/experiments/{evaluation,ablation,campaign,recovery}/`
after the next Vast.ai run. See `prism/VASTAI_RUN_GUIDE.md` §Appendix A1
for the 14-step pipeline that produces them.

## Layout

- `run_vastai_full.sh` — superseded duplicate of the canonical script
  (`prism/run_vastai_full.sh`). Older config (`CW_MAX_ITER=100`, `CW_BSS=9`,
  `FGSM_OVERSAMPLE=2.0`); retained for audit only.
- `experiments/evaluation/` — pre-research-plan evaluation JSONs
  (n=1000 fast + n=500 CW/PGD, 5-seed set produced before the research-plan
  gates). Superseded by the next Vast.ai run.
- `experiments/ablation/` — pre-research-plan ablation (n=500 single- and
  multi-seed). The new run produces n=1000 multi-seed with
  paired t-tests.
- `experiments/calibration/` — pre-research-plan ensemble FPR report.
- `logs/` — step logs from the pre-research-plan run.
- `models/` — pickled reference profiles / calibrators / ensemble scorer
  from the pre-research-plan run. Kept so archived JSONs remain reproducible
  bit-for-bit; new run regenerates these under `prism/models/`.
- `vast_ai_run_20260423/` — the earlier Vast.ai run output
  (renamed from `prism/experiments/evaluation/vast ai/` — the space in the
  original folder name was a long-standing typo).
- `PRISM_Results_Report.md` — pre-research-plan written-up results.

## Why archive instead of delete?

- Forensic continuity — if a reviewer asks "where did the 0.8847 ablation
  number come from?", the audit trail must survive.
- `regression_analysis_20260422.md` (still active in the repo root) cites
  the archived numbers as the baseline for the FGSM oversample regression
  investigation.
- `sanity_checks.py` references the archived `step3_calibrate.log` line
  pattern when verifying the FGSM oversample gate.
