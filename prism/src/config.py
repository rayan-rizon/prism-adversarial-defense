"""
PRISM Configuration Loader

Single source of truth for all shared constants.  Scripts should import
from this module instead of hardcoding values inline, so that a change to
configs/default.yaml propagates everywhere automatically.

Usage
-----
    from src.config import load_config, LAYER_NAMES, LAYER_WEIGHTS, DIM_WEIGHTS
    from src.config import CAL_IDX, VAL_IDX, EVAL_IDX, CAL_ALPHA_FACTOR

    cfg = load_config()                      # full config dict
    cfg = load_config('configs/custom.yaml') # override path
"""
import os
import yaml
from typing import Dict, List, Optional, Tuple

# Default config path relative to the prism/ project root
_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'configs', 'default.yaml'
)


def load_config(path: Optional[str] = None) -> dict:
    """Load and return the YAML config as a nested dict.

    Args:
        path: Path to the YAML file.  Defaults to configs/default.yaml
              relative to the prism/ project root.
    Returns:
        Nested dict matching the YAML structure.
    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    cfg_path = path or _DEFAULT_CONFIG_PATH
    cfg_path = os.path.abspath(cfg_path)
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"Config file not found: {cfg_path}\n"
            "Ensure you are running from the prism/ project root."
        )
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f)


def _get_defaults() -> dict:
    """Load the default config, with a graceful fallback for import-time use."""
    try:
        return load_config()
    except FileNotFoundError:
        # Fallback values used when the config file cannot be found (e.g. during
        # unit tests run from a different working directory).  These MUST match
        # the values in configs/default.yaml exactly.
        return {
            'model': {
                'layer_names':   ['layer2', 'layer3', 'layer4'],
                'layer_weights': {'layer2': 0.30, 'layer3': 0.30, 'layer4': 0.40},
            },
            'tda': {
                'n_subsample': 200,
                'max_dim':     1,
                'dim_weights': [0.70, 0.30],
            },
            'conformal': {
                'alphas': {'L1': 0.10, 'L2': 0.03, 'L3': 0.005},
                'cal_alpha_factor': 0.7,
            },
            'data': {
                'mean': [0.485, 0.456, 0.406],
                'std':  [0.229, 0.224, 0.225],
            },
            'splits': {
                'profile_idx': [0,    5000],
                'cal_idx':     [5000, 7000],
                'val_idx':     [7000, 8000],
                'eval_idx':    [8000, 10000],
            },
        }


_CFG = _get_defaults()

# ── Module-level constants for direct import ──────────────────────────────────
# All scripts that previously hardcoded these values should import from here.
# Do NOT duplicate these values elsewhere -- silent drift was the root cause of
# the results_n500_20260419.json regression (see Appendix A, items A-2/A-6/A-7).

LAYER_NAMES:   List[str]          = _CFG['model']['layer_names']
LAYER_WEIGHTS: Dict[str, float]   = _CFG['model']['layer_weights']
DIM_WEIGHTS:   List[float]        = _CFG['tda']['dim_weights']
N_SUBSAMPLE:   int                = _CFG['tda']['n_subsample']
MAX_DIM:       int                = _CFG['tda']['max_dim']
IMAGENET_MEAN: List[float]        = _CFG['data']['mean']
IMAGENET_STD:  List[float]        = _CFG['data']['std']
EPS_LINF_STANDARD: float          = 8 / 255  # standard CIFAR-10 evaluation budget
CONFORMAL_ALPHAS: Dict[str, float] = _CFG['conformal']['alphas']

# Calibration alpha multiplier: calibration thresholds are fitted at
# (alpha * CAL_ALPHA_FACTOR) to create a slack buffer, then verified against
# the published alpha.  0.7 replaces the previous 0.8 to close the L3 FPR gap
# (results_n500_planA.json: L3 FPR=0.008 > target 0.005).
CAL_ALPHA_FACTOR: float = _CFG.get('conformal', {}).get('cal_alpha_factor', 0.7)

# CIFAR-10 test-set split indices -- single source of truth.
# ANY script referencing split ranges must import from here, never hardcode.
_splits = _CFG.get('splits', {
    'profile_idx': [0,    5000],
    'cal_idx':     [5000, 7000],
    'val_idx':     [7000, 8000],
    'eval_idx':    [8000, 10000],
})
PROFILE_IDX: Tuple[int, int] = tuple(_splits['profile_idx'])
CAL_IDX:     Tuple[int, int] = tuple(_splits['cal_idx'])
VAL_IDX:     Tuple[int, int] = tuple(_splits['val_idx'])
EVAL_IDX:    Tuple[int, int] = tuple(_splits['eval_idx'])
