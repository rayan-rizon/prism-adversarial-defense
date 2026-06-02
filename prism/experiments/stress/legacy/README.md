# Legacy stress-test artifacts

Files in this folder are earlier-iteration experiments that have been
**superseded by canonical versions** referenced in the paper. They are
retained for reproducibility and historical record only.

| Legacy file | Superseded by | Reason |
|---|---|---|
| `vastai_stronger_cw.{json,py}` | `../vastai_stronger_cw_canonical.{json,py}` | Earlier exploration with weakened CW config (`max_iter=5`, `bss=5`). The paper uses canonical CW (`max_iter=100`, `bss=9`) from the canonical version. |
| `vastai_recovery_pgd.{json,py}` | `../vastai_recovery_multi.{json,py}` | Earlier PGD-only recovery test. The paper uses the multi-attack recovery (FGSM/PGD/Square/CW) from the multi version. |

None of these files are referenced in `paper/` or in current LaTeX
sources. They are kept here strictly for reproducibility audit; deleting
them does not affect the paper PDF or the deployed PRISM artifacts.
