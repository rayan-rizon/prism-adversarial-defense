import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')
from experiments.campaign.run_campaign import run_campaign_experiment
# eps must match EPS_LINF_STANDARD = 8/255 used across all evaluation scripts
run_campaign_experiment(n_clean=50, n_adv=100, eps=8/255)
