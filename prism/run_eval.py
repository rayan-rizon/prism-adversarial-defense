import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')
# run_evaluation_full.py is the paper-quality evaluation:
#   - n=1000 per attack (vs 300 in the legacy run_evaluation.py stub)
#   - Strictly uses EVAL_IDX = (8000, 10000) — zero overlap with cal/val splits
#   - 95% Wilson CI, per-tier FPR, latency measurement
#   - Outputs results_paper.json
from experiments.evaluation.run_evaluation_full import run_evaluation_full
run_evaluation_full()
