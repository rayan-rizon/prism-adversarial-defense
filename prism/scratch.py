import json
import numpy as np
from src.sacd.monitor import CampaignMonitor

with open('experiments/campaign/results.json') as f:
    res = json.load(f)

scores = res.get('scores', [])
if not scores:
    # Need to extract from another source if not stored, wait the script didn't store scores array, only means.
    pass
