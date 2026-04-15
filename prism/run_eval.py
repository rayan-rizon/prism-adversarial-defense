import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')
from experiments.evaluation.run_evaluation import run_evaluation
run_evaluation(n_test=300)
