# Latest Thunder L3 Rerun Download

Downloaded latest log/calibration/results snapshot. Seed `456` JSON is not available yet because PGD for seed `456` is still running; seed JSON is written after both FGSM and PGD finish.

| Seed | Attack | TPR | FPR | L2+ | L3+ | Source | Status |
|---:|---|---:|---:|---:|---:|---|---:|
| 123 | FGSM | 0.9810 | 0.0780 | 0.0280 | 0.0010 | JSON | PASS |
| 123 | PGD | 0.9960 | 0.0780 | 0.0280 | 0.0010 | JSON | PASS |
| 42 | FGSM | 0.9870 | 0.0680 | 0.0260 | 0.0030 | JSON | PASS |
| 42 | PGD | 0.9970 | 0.0680 | 0.0260 | 0.0030 | JSON | PASS |
| 456 | FGSM | 0.9840 | 0.0770 | 0.0240 | 0.0010 | log-final | PASS |