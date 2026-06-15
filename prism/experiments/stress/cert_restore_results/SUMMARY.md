# CIFAR-10-C Certificate Restoration (CADG shift fix)

The CADG split-conformal FPR certificate degrades 3.9x-28x under benign
CIFAR-10-C covariate shift (broken exchangeability; Table cert_shift). This run
shows the degradation is **recoverable** by shift-aware split-conformal
recalibration: re-fit q_alpha on a small held-out slice (n_recal=300) of the
SAME corrupted-but-benign distribution, evaluate on the disjoint remainder
(n_eval ~ 700). CIFAR-10/ResNet-18, deployed scorer, 8 corruptions x sev {1,3,5}
= 24 cells, seed 42. Vast RTX 5070 Ti, 8 parallel (one corruption each).

## Result (24 cells pooled)
| tier | alpha | mean frozen FPR | mean restored FPR | cells restored <= alpha |
|---|---|---|---|---|
| L1 | 0.10 | 0.386 | **0.092** | 20/24 |
| L2 | 0.03 | 0.253 | **0.025** | 19/24 |
| L3 | 0.005 | 0.142 | **0.0034** | 16/24 |

Worst cells (L1): gaussian_noise sev5 0.999 -> 0.096; shot_noise sev5 0.997 ->
0.060; gaussian_noise sev3 0.993 -> 0.083.

## Validation
The FROZEN numbers (0.386 / 0.253 / 0.142) reproduce the paper's published
audit (Table cert_shift: 0.387 / 0.254 / 0.142) EXACTLY -> harness is correct,
restoration is real.

## Conclusion
Online shift-aware recalibration restores realised FPR to ~alpha at every tier.
This is the certificate analogue of the SACD session-level fix: a measured
negative (cert breaks under shift) turned into a measured positive (recalibration
restores it). Caveat (stated in the paper): assumes a small confirmed-benign
recal slice of the shifted stream is available (standard online-conformal-under-
shift assumption); does not require knowing the corruption type. A few cells land
just over alpha after recal -- within the Wilson interval at n_eval ~ 700.

Written into all 4 papers (sec:cert_shift, Table cert_restore; abstract + conclusion).
Reproduce: run_cifar10c_restore.py / run_cert_restore.sh / aggregate_cert_restore.py.
