"""
Config-aware PRISM artifact sanity checks.

This script is intentionally artifact-light: it validates the files produced
by the CIFAR-native pipeline without downloading ImageNet weights or using
obsolete 224x224 assumptions.
"""
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from src.config import (  # noqa: E402
    BACKBONE_CHECKPOINT_PATH,
    BACKBONE_INPUT_SIZE,
    BACKBONE_NUM_CLASSES,
    DATASET,
    LAYER_NAMES,
    PATHS,
)
from src.models import load_backbone  # noqa: E402


failures = []


def check(name: str, condition: bool, detail: str = ''):
    if condition:
        print(f"  PASS {name}")
    else:
        print(f"  FAIL {name}" + (f": {detail}" if detail else ''))
        failures.append(name)


def skip(name: str, reason: str = ''):
    print(f"  SKIP {name}" + (f" ({reason})" if reason else ''))


def sibling(path: str, name: str) -> str:
    return os.path.join(os.path.dirname(path) or '.', name)


print(f"\n=== PRISM sanity checks: {DATASET.upper()} ===")
print(f"Backbone classes={BACKBONE_NUM_CLASSES}, input={BACKBONE_INPUT_SIZE}x{BACKBONE_INPUT_SIZE}")


print("\n=== Check 1: reference profiles ===")
profile_path = PATHS['reference_profiles']
if not os.path.exists(profile_path):
    skip(profile_path, "not built yet")
else:
    profiles = pickle.load(open(profile_path, 'rb'))
    check("reference profile is dict", isinstance(profiles, dict))
    for layer in LAYER_NAMES:
        check(f"{layer} profile present", layer in profiles)
        if layer in profiles:
            check(f"{layer} medoid diagram list", isinstance(profiles[layer], list) and len(profiles[layer]) > 0)


print("\n=== Check 2: calibrators ===")
for label, path in [
    ('ensemble', PATHS['calibrator']),
    ('base_tda', sibling(PATHS['calibrator'], 'calibrator_base.pkl')),
    ('ensemble_no_tda', sibling(PATHS['calibrator'], 'calibrator_no_tda.pkl')),
]:
    if not os.path.exists(path):
        skip(f"{label} calibrator", path)
        continue
    cal = pickle.load(open(path, 'rb'))
    thresholds = getattr(cal, 'thresholds', {})
    check(f"{label} thresholds include L1/L2/L3", all(k in thresholds for k in ('L1', 'L2', 'L3')))
    if all(k in thresholds for k in ('L1', 'L2', 'L3')):
        check(
            f"{label} thresholds ordered",
            thresholds['L1'] <= thresholds['L2'] <= thresholds['L3'],
            str(thresholds),
        )


print("\n=== Check 3: ensemble artifacts ===")
for label, path, expect_tda in [
    ('main ensemble', PATHS['ensemble_scorer'], True),
    ('no-TDA ensemble', sibling(PATHS['ensemble_scorer'], 'ensemble_no_tda.pkl'), False),
]:
    if not os.path.exists(path):
        skip(label, path)
        continue
    data = pickle.load(open(path, 'rb'))
    check(f"{label} stored as dict", isinstance(data, dict), f"type={type(data).__name__}")
    if isinstance(data, dict):
        check(f"{label} softmax entropy enabled", bool(data.get('use_softmax_entropy', False)))
        check(f"{label} grad norm disabled", not bool(data.get('use_grad_norm', False)))
        check(f"{label} use_tda={expect_tda}", bool(data.get('use_tda', True)) is expect_tda)
        if expect_tda:
            check(f"{label} n_features >= 38", int(data.get('n_features') or 0) >= 38)


print("\n=== Check 4: experts ===")
experts_path = PATHS['experts']
if not os.path.exists(experts_path):
    skip("experts", experts_path)
else:
    experts = pickle.load(open(experts_path, 'rb'))
    check("experts artifact is dict", isinstance(experts, dict))
    if isinstance(experts, dict):
        check("expert state dicts present", len(experts.get('experts', [])) > 0)
        check(
            "expert output_dim matches dataset classes",
            int(experts.get('output_dim', -1)) == BACKBONE_NUM_CLASSES,
            f"output_dim={experts.get('output_dim')}, expected={BACKBONE_NUM_CLASSES}",
        )


print("\n=== Check 5: clean-score cache ===")
scores_path = PATHS['clean_scores']
if not os.path.exists(scores_path):
    skip("clean scores", scores_path)
else:
    scores = np.load(scores_path)
    check("clean scores finite", np.all(np.isfinite(scores)))
    check("clean scores non-negative", np.all(scores >= 0))
    check("clean score count >= 1000", len(scores) >= 1000, f"n={len(scores)}")


print("\n=== Check 6: optional end-to-end load ===")
if not os.path.exists(BACKBONE_CHECKPOINT_PATH):
    skip("backbone checkpoint", BACKBONE_CHECKPOINT_PATH)
elif not (os.path.exists(PATHS['calibrator']) and os.path.exists(PATHS['reference_profiles'])):
    skip("E2E inference", "needs calibrator and reference profiles")
else:
    from src.prism import PRISM  # noqa: E402

    model = load_backbone(torch.device('cpu'))
    prism = PRISM.from_saved(
        model=model,
        layer_names=LAYER_NAMES,
        layer_weights=None,
        dim_weights=None,
        calibrator_path=PATHS['calibrator'],
        profile_path=PATHS['reference_profiles'],
        ensemble_path=PATHS['ensemble_scorer'] if os.path.exists(PATHS['ensemble_scorer']) else None,
    )
    x = torch.randn(1, 3, BACKBONE_INPUT_SIZE, BACKBONE_INPUT_SIZE)
    _, level, meta = prism.defend(x)
    check("PRISM.defend level valid", level in {'PASS', 'L1', 'L2', 'L3', 'L3_REJECT'})
    check("PRISM.defend metadata has anomaly_score", 'anomaly_score' in meta)


print()
if failures:
    print(f"FAILED: {len(failures)} sanity check(s)")
    for failure in failures:
        print(f"  - {failure}")
    sys.exit(1)

print("PASSED: all available sanity checks")
