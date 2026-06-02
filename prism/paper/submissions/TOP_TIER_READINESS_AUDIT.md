# PRISM Top-Tier Readiness Audit

Date: 2026-06-02

## Proven Ready

- NeurIPS, ICLR, and arXiv paper packages build successfully.
- NeurIPS and ICLR main papers were expanded with a claim ledger, threat-model
  matrix, statistical-treatment discussion, and failure-mode discussion using
  only verified artifact data.
- arXiv source package has no NeurIPS checklist/style file, review marker,
  anonymous marker, or LaTeX build byproduct.
- All three rebuilt PDFs use embedded outline fonts; `pdffonts` reports no
  Type 3 fonts. arXiv also uses hidden hyperlink borders.
- NeurIPS package is anonymous and includes the checklist in the combined PDF.
- ICLR package is anonymous and under the submission page limit.
- Paper-number verifiers pass against the locked submitted artifacts.
- Local quick checks verify that the exact CIFAR-10 ResNet-18 checkpoint reaches
  95.16% accuracy on the official CIFAR-10 test split and that the strict
  train-split config has no profile/calibration/validation overlap.
- An anonymized artifact supplement exists and self-verifies.
- The strict train-split rerun scaffold executes end-to-end in `SMOKE=1`
  mode. Smoke metrics are intentionally not publishable evidence because the
  detector head is trained on only a tiny integration-test sample.

## Venue Rule Matrix

Authoritative sources checked on 2026-06-02:

- NeurIPS 2026 Main Track Handbook: anonymous submission, official style,
  anonymized supplementary material, and completed checklist are required.
  Public preprints should use preprint-style wording and must not say under
  review.
  Source: https://neurips.cc/Conferences/2026/MainTrackHandbook
- NeurIPS checklist guide: the checklist belongs in the NeurIPS PDF after the
  paper/references/optional appendices and does not count toward page limit.
  Source: https://nips.cc/public/guides/PaperChecklist
- ICLR 2026 Author Guide: submitted papers are double blind, main text is at
  most 9 pages at submission, references and appendix are outside that limit,
  and supplementary material is due with the paper.
  Source: https://iclr.cc/Conferences/2026/AuthorGuide
- arXiv submission process: a source package must compile to a usable PDF
  before the submission can leave working state.
  Source: https://arxiv.github.io/arxiv-submission-core/announcement_process.html

Current package evidence:

| Package | Evidence |
| --- | --- |
| NeurIPS | `PRISM_neurips_2026_submission.pdf` is 10 pages total: main body through page 7, references start page 8, protocol appendix page 9, checklist page 10; it is anonymous and includes `NeurIPS Paper Checklist`; `PRISM_neurips_2026_source.zip` has `checklist.tex` and `neurips_2026.sty`, with no build-log entries. |
| ICLR | `PRISM_iclr_2026_submission.pdf` is 9 pages total: main body through page 6, references start page 7, and protocol notes follow after references; it is anonymous, under review as ICLR 2026, and has no NeurIPS checklist; `PRISM_iclr_2026_source.zip` has ICLR style files and no NeurIPS/checklist entries. |
| arXiv | `PRISM_arxiv_preprint.pdf` is 34 pages, nonanonymous, no Type 3 fonts, and has no anonymous/review/checklist markers; `PRISM_arxiv_source.zip` has no NeurIPS/ICLR/checklist entries, no placeholder comments, and fresh-extracts to a 34-page PDF. |
| Artifact supplement | `PRISM_anonymized_artifact_supplement.zip` has no author/path leak matches, no build byproducts, no checkpoint/numpy/cache files, and includes strict rerun plus smoke configs. It also includes local quick-check JSONs under `prism/experiments/local_quick_checks/`. |

## Remaining Scientific Reruns

These are required before calling the scientific evidence fully top-tier, not just package-ready.

1. Strict CIFAR-10 train-split development protocol.
   - Run `bash run_top_tier_train_split_rerun.sh`.
   - This builds profile/calibration/validation from CIFAR train indices only, trains the scorer from a disjoint train window, and evaluates on the official test split.
   - Promote numbers only if FPR gates pass and main attack TPRs remain within the paper's stated claim scope.

2. Ensemble-complete adaptive PGD beyond CIFAR-10/ResNet-18.
   - After each backbone/dataset detector is trained/calibrated, run:
     - `CONFIG=configs/wrn_cifar10.yaml TAG=wrn_ensemble_complete bash run_vastai_ensemble_complete_adaptive.sh`
     - `CONFIG=configs/cifar100.yaml TAG=cifar100_ensemble_complete bash run_vastai_ensemble_complete_adaptive.sh`
     - `CONFIG=configs/vit_cifar10.yaml TAG=vit_ensemble_complete bash run_vastai_ensemble_complete_adaptive.sh`
   - Do not claim broad adaptive robustness unless these runs are present and reported.

3. Image-disjoint or clustered confidence intervals.
   - Required for CW, AutoAttack, WRN, CIFAR-100, and ViT before using pooled Wilson intervals as headline-strength evidence.

4. Config-matched recent-baseline CW comparison.
   - Run recent baselines with the same canonical CW settings used by PRISM: `max_iter=100`, `binary_search_steps=9`, and confidence `kappa=1.0`.
   - Do not headline lighter-CW baseline comparisons.

5. Long clean-stream L0 stress if claiming beyond a 1000-query horizon.
   - Current paper wording limits the clean-stream false-alarm claim to the tested horizon.

## Rule

Do not update the NeurIPS/ICLR/arXiv quantitative claims until the corresponding JSON artifacts, table regeneration, LaTeX rebuild, and verifier checks all pass.
