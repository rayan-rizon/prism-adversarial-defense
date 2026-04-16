"""
Sanity Checks (Guide Section 3.4)

Run after each major phase to verify artifacts exist and are internally consistent.
Checks:
  1. Reference profiles exist + have sufficient diagrams
  2. Calibrator thresholds are ordered
  3. Clean score distribution is sane (mean << 0.5)
  4. Expert models exist and load
  5. PRISM can perform inference end-to-end

Usage:
    cd prism/
    python sanity_checks.py

Exit code 0 = all checks passed.
Exit code 1 = one or more checks failed.
"""
import os
import sys
import pickle
import numpy as np

# ── SSL fix ─────────────────────────────────────────────────────────────────
import ssl, certifi
os.environ.setdefault('SSL_CERT_FILE', certifi.where())
os.environ.setdefault('REQUESTS_CA_BUNDLE', certifi.where())
ssl._create_default_https_context = ssl.create_default_context

sys.path.insert(0, os.path.dirname(__file__))

PASS_SYMBOL = "✅"
FAIL_SYMBOL = "❌"
SKIP_SYMBOL = "⏭"

failures = []


def check(name: str, condition: bool, detail: str = ''):
    if condition:
        print(f"  {PASS_SYMBOL}  {name}")
    else:
        print(f"  {FAIL_SYMBOL}  {name}" + (f": {detail}" if detail else ''))
        failures.append(name)


def skip(name: str, reason: str = ''):
    print(f"  {SKIP_SYMBOL}  {name}" + (f" ({reason})" if reason else ''))


# ────────────────────────────────────────────────────────────────────────────
# Check 1: Reference profiles
# ────────────────────────────────────────────────────────────────────────────

print("\n=== Check 1: Reference profiles ===")
profile_path = 'models/reference_profiles.pkl'

if not os.path.exists(profile_path):
    skip("reference_profiles.pkl exists", "not built yet — run build_profile.py")
else:
    profiles = pickle.load(open(profile_path, 'rb'))
    check("reference_profiles.pkl loaded", True)

    expected_layers = ['layer2', 'layer3', 'layer4']
    for layer in expected_layers:
        check(
            f"{layer}: profile key present",
            layer in profiles,
            f"missing layer {layer}"
        )
        if layer in profiles:
            med = profiles[layer]
            check(
                f"{layer}: medoid is valid diagram list",
                isinstance(med, list) and len(med) > 0,
                f"got type={type(med)}"
            )


# ────────────────────────────────────────────────────────────────────────────
# Check 2: Calibrator thresholds ordered
# ────────────────────────────────────────────────────────────────────────────

print("\n=== Check 2: Calibrator thresholds ===")
cal_path = 'models/calibrator.pkl'

if not os.path.exists(cal_path):
    skip("calibrator.pkl exists", "not built yet — run calibrate_thresholds.py")
else:
    calibrator = pickle.load(open(cal_path, 'rb'))
    check("calibrator.pkl loaded", True)
    check(
        "L1 < L2 threshold",
        calibrator.thresholds.get('L1', 0) < calibrator.thresholds.get('L2', 0),
        f"L1={calibrator.thresholds.get('L1'):.4f}, L2={calibrator.thresholds.get('L2'):.4f}"
        if 'L1' in calibrator.thresholds else "thresholds missing"
    )
    check(
        "L2 < L3 threshold",
        calibrator.thresholds.get('L2', 0) < calibrator.thresholds.get('L3', 0),
        f"L2={calibrator.thresholds.get('L2'):.4f}, L3={calibrator.thresholds.get('L3'):.4f}"
        if 'L2' in calibrator.thresholds else "thresholds missing"
    )
    for level, thr in calibrator.thresholds.items():
        print(f"     {level}: threshold = {thr:.6f}  (α={calibrator.alphas[level]})")


# ────────────────────────────────────────────────────────────────────────────
# Check 3: Clean score distribution
# ────────────────────────────────────────────────────────────────────────────

print("\n=== Check 3: Clean score distribution ===")
scores_path = 'experiments/calibration/clean_scores.npy'

if not os.path.exists(scores_path):
    skip("clean_scores.npy exists", "not built yet — run build_profile.py")
else:
    scores = np.load(scores_path)
    mean_s = scores.mean()
    std_s = scores.std()
    print(f"     n={len(scores)}, mean={mean_s:.4f}, std={std_s:.4f}, "
          f"min={scores.min():.4f}, max={scores.max():.4f}")

    # Clean scores are raw Wasserstein distances (not normalized to [0,1]).
    # Sanity: mean must be well below L1 threshold (if calibrator exists).
    if os.path.exists(cal_path):
        _cal = pickle.load(open(cal_path, 'rb'))
        l1_thr = _cal.thresholds.get('L1', float('inf'))
        check(
            "Clean score mean well below L1 threshold",
            mean_s < l1_thr * 0.95,
            f"mean={mean_s:.4f}, L1_threshold={l1_thr:.4f} — gap too small"
        )
    else:
        # Calibrator not built yet — just check scores are positive and finite
        check(
            "Clean scores positive and finite",
            mean_s > 0 and np.isfinite(mean_s),
            f"mean={mean_s:.4f}"
        )
    check(
        "Clean scores non-negative",
        np.all(scores >= 0),
        f"{np.sum(scores < 0)} negative scores detected"
    )
    check(
        "Sufficient calibration data (>= 1000)",
        len(scores) >= 1000,
        f"only {len(scores)} scores; recommend 1000+"
    )


# ────────────────────────────────────────────────────────────────────────────
# Check 4: Expert models
# ────────────────────────────────────────────────────────────────────────────

print("\n=== Check 4: Expert models ===")
experts_path = 'models/experts.pkl'

if not os.path.exists(experts_path):
    skip("experts.pkl exists", "not trained yet — run train_experts.py")
else:
    experts_data = pickle.load(open(experts_path, 'rb'))
    check("experts.pkl loaded", True)
    check(
        "experts dict has required keys",
        all(k in experts_data for k in ['experts', 'k', 'input_dim', 'output_dim']),
    )
    k = experts_data.get('k', 0)
    n_saved = len(experts_data.get('experts', []))
    check(
        f"Correct number of expert state dicts ({k})",
        n_saved == k,
        f"expected {k}, got {n_saved}"
    )

    # Try loading expert
    import torch
    from src.tamsh.experts import ExpertSubNetwork
    try:
        expert = ExpertSubNetwork(
            experts_data['input_dim'],
            experts_data['output_dim'],
        )
        expert.load_state_dict(experts_data['experts'][0])
        expert.eval()
        check("Expert 0 loads and runs", True)
        dummy = torch.randn(1, experts_data['input_dim'])
        _ = expert(dummy)
        check("Expert 0 forward pass", True)
    except Exception as e:
        check("Expert loads and runs", False, str(e))


# ────────────────────────────────────────────────────────────────────────────
# Check 5: End-to-end PRISM inference
# ────────────────────────────────────────────────────────────────────────────

print("\n=== Check 5: End-to-end PRISM inference ===")

if not (os.path.exists(cal_path) and os.path.exists(profile_path)):
    skip("E2E inference", "needs calibrator.pkl + reference_profiles.pkl")
else:
    try:
        import torch
        import torchvision
        from torchvision.models import ResNet18_Weights
        from src.prism import PRISM

        model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.eval()

        prism = PRISM.from_saved(
            model=model,
            layer_names=['layer2', 'layer3', 'layer4'],
            layer_weights={'layer2': 0.15, 'layer3': 0.30, 'layer4': 0.55},
            dim_weights=[0.5, 0.5],
            calibrator_path=cal_path,
            profile_path=profile_path,
        )

        # --- build a clean test input ---
        # Prefer a real CIFAR-10 test image for a realistic clean baseline;
        # fall back to an ImageNet-mean gray image if data is unavailable.
        cifar_path = 'data/cifar-10-batches-py/test_batch'
        _normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        _resize = torchvision.transforms.Resize((224, 224), antialias=True)

        if os.path.exists(cifar_path):
            import pickle as _pkl
            with open(cifar_path, 'rb') as _f:
                _batch = _pkl.load(_f, encoding='bytes')
            # first test image: shape (3072,) uint8
            _img = _batch[b'data'][0].reshape(3, 32, 32).astype(np.float32) / 255.0
            _t = torch.from_numpy(_img).unsqueeze(0)          # (1,3,32,32)
            _t = _resize(_t)                                   # (1,3,224,224)
            x_clean = _normalize(_t)
            _source = "CIFAR-10 test image"
        else:
            # gray image = ImageNet mean → zero in normalized space → moderate TDA score
            x_clean = torch.zeros(1, 3, 224, 224)
            _source = "gray fallback"

        pred, level, meta = prism.defend(x_clean)
        check("PRISM.defend() returns valid level", level in {'PASS', 'L1', 'L2', 'L3', 'L3_REJECT'})
        check("anomaly_score >= 0", meta['anomaly_score'] >= 0)
        print(f"     clean input ({_source}): score={meta['anomaly_score']:.4f}, level={level}")

        # Perturbed-input ordering: use FGSM-style L∞ noise (eps=0.1 in [0,1])
        # which corresponds to ~0.44 std-devs in normalized space.
        eps_norm = 0.1 / np.mean([0.229, 0.224, 0.225])  # ~0.44

        clean_scores = []
        adv_scores = []
        rng = np.random.RandomState(0)
        for _ in range(5):
            _img_idx = rng.randint(0, 100)
            if os.path.exists(cifar_path):
                _img = _batch[b'data'][_img_idx].reshape(3, 32, 32).astype(np.float32) / 255.0
                _t = _resize(torch.from_numpy(_img).unsqueeze(0))
                xc = _normalize(_t)
            else:
                xc = torch.zeros(1, 3, 224, 224)
            # signed random perturbation in normalized space
            sign_noise = torch.sign(torch.from_numpy(
                rng.randn(1, 3, 224, 224).astype(np.float32)
            ))
            xa = xc + sign_noise * eps_norm
            _, _, mc = prism.defend(xc)
            _, _, ma = prism.defend(xa)
            clean_scores.append(mc['anomaly_score'])
            adv_scores.append(ma['anomaly_score'])

        print(f"     5-trial avg: clean={np.mean(clean_scores):.4f}, adv={np.mean(adv_scores):.4f}")
        check(
            "Perturbed inputs score higher than clean (avg)",
            np.mean(adv_scores) > np.mean(clean_scores),
            f"clean_mean={np.mean(clean_scores):.4f}, adv_mean={np.mean(adv_scores):.4f}"
        )

    except Exception as e:
        check("PRISM E2E", False, str(e))
        import traceback
        traceback.print_exc()


# ────────────────────────────────────────────────────────────────────────────
# Summary
# ────────────────────────────────────────────────────────────────────────────

print()
if failures:
    print(f"❌  {len(failures)} check(s) FAILED:")
    for f in failures:
        print(f"     - {f}")
    sys.exit(1)
else:
    print("✅  All checks PASSED")
    sys.exit(0)
