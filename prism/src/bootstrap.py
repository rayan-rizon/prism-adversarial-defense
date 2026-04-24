"""
PRISM Bootstrap — Early --config CLI → PRISM_CONFIG env-var routing.

Import this BEFORE any `from src.config import …` so the module-level
constants (LAYER_NAMES, DATASET, PATHS, …) are populated from the YAML
named by --config rather than from configs/default.yaml.

Usage (at the top of every script that takes --config):

    from src import bootstrap  # noqa: F401 — must precede src.config import
    from src.config import LAYER_NAMES, DATASET, PATHS  # now config-aware

The bootstrap is a no-op when --config is absent, so default.yaml behavior
is unchanged for the primary CIFAR-10 pipeline.
"""
import os
import sys

# Scan sys.argv early (before argparse runs) for --config and route to env.
# Accepts both "--config path" and "--config=path" forms.
for _i, _tok in enumerate(sys.argv):
    if _tok == '--config' and _i + 1 < len(sys.argv):
        os.environ['PRISM_CONFIG'] = sys.argv[_i + 1]
        break
    if _tok.startswith('--config='):
        os.environ['PRISM_CONFIG'] = _tok.split('=', 1)[1]
        break
