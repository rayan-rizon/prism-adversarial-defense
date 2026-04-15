import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')
from experiments.campaign.run_campaign import run_campaign_experiment
run_campaign_experiment(n_clean=50, n_adv=100, eps=0.05)
