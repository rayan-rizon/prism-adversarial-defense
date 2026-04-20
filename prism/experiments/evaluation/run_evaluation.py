"""
Full Attack Evaluation (Phase 6, Weeks 24-26)

Evaluates PRISM against multiple attack types using IBM ART.
Computes TPR, FPR, and detection rates per attack.

Key fixes from plan:
1. Test dataset now has proper transforms (was missing in plan's code)
2. ART classifier wrapping handles CIFAR-10 → 224px properly
3. Attack generation uses correct input format
4. Results include per-tier breakdown
"""
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models import ResNet18_Weights
import numpy as np
import json
import os
import sys
import ssl
import certifi
from tqdm import tqdm

# Fix SSL certificate verification on macOS Python 3.11
os.environ.setdefault('SSL_CERT_FILE', certifi.where())
os.environ.setdefault('REQUESTS_CA_BUNDLE', certifi.where())
ssl._create_default_https_context = ssl.create_default_context

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# These imports require ART installed
try:
    from art.attacks.evasion import (
        FastGradientMethod,
        ProjectedGradientDescent,
        CarliniL2Method,
        SquareAttack,
        HopSkipJump,
    )
    from art.estimators.classification import PyTorchClassifier
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False
    print("WARNING: adversarial-robustness-toolbox not installed.")
    print("Install with: pip install adversarial-robustness-toolbox")

from src.prism import PRISM
from src.cadg.calibrate import ConformalCalibrator
from src.sacd.monitor import NoOpCampaignMonitor
from src.config import LAYER_NAMES, LAYER_WEIGHTS, DIM_WEIGHTS, IMAGENET_MEAN, IMAGENET_STD

# Normalization constants from src.config (backed by configs/default.yaml)
_MEAN = IMAGENET_MEAN
_STD  = IMAGENET_STD

# Pixel-space transform for ART: [0,1]-valued tensors, clip_values correct
_PIXEL_TRANSFORM = T.Compose([T.Resize(224), T.ToTensor()])
# PRISM normalization applied after attack generation
_NORMALIZE = T.Normalize(mean=_MEAN, std=_STD)


class _NormalizedResNet(torch.nn.Module):
    """ResNet wrapper with normalization baked in.
    Lets ART attack in pixel [0,1] space; clip_values=(0.0,1.0) is valid.
    """
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self._model = model
        self.register_buffer(
            '_mean', torch.tensor(_MEAN, dtype=torch.float32).view(3, 1, 1)
        )
        self.register_buffer(
            '_std',  torch.tensor(_STD,  dtype=torch.float32).view(3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model((x - self._mean) / self._std)


def run_evaluation(
    n_test: int = 1000,
    data_root: str = './data',
    output_path: str = 'experiments/evaluation/results.json',
    seed: int = 42,
    attacks_to_run: list = None,
    _fgsm_eps_override: float = None,  # Override FGSM eps for sweep; None = use default 0.03
):
    if not ART_AVAILABLE:
        print("Cannot run evaluation without ART. Exiting.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Fix seed for reproducible test-set sample selection
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    # --- Setup model ---
    model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model = model.to(device).eval()

    # Constants from configs/default.yaml via src.config
    layer_names   = LAYER_NAMES
    layer_weights = LAYER_WEIGHTS
    dim_weights   = DIM_WEIGHTS

    # --- Load PRISM (requires pre-built profiles and calibrator) ---
    calibrator_path = 'models/calibrator.pkl'
    profile_path = 'models/reference_profiles.pkl'

    if not os.path.exists(calibrator_path) or not os.path.exists(profile_path):
        print("ERROR: Pre-built models not found.")
        print("Run these first:")
        print("  1. python scripts/build_profile.py")
        print("  2. python scripts/calibrate_thresholds.py")
        sys.exit(1)

    prism = PRISM.from_saved(
        model=model,
        layer_names=layer_names,
        calibrator_path=calibrator_path,
        profile_path=profile_path,
        layer_weights=layer_weights,
        dim_weights=dim_weights,
    )

    # --- Setup ART classifier ---
    # ART operates in pixel [0,1] space; _NormalizedResNet applies normalization
    # internally so the model receives correctly-normalized activations.
    # clip_values=(0.0, 1.0) is valid because inputs are pixel-space.
    model_art = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device).eval()
    _wrapped_art = _NormalizedResNet(model_art).to(device).eval()

    device_type = 'gpu' if device.type == 'cuda' else 'cpu'
    classifier = PyTorchClassifier(
        model=_wrapped_art,
        loss=torch.nn.CrossEntropyLoss(),
        input_shape=(3, 224, 224),
        nb_classes=1000,  # ImageNet classes for ResNet-18
        clip_values=(0.0, 1.0),
        device_type=device_type,
    )

    # --- Define attacks ---
    # CW excluded for n_test>=100: ~285s/sample on CPU → impractical.
    # Square is fast (score-based, no gradient) and complementary to white-box attacks.
    attacks = {
        'FGSM': FastGradientMethod(classifier, eps=_fgsm_eps_override or 8/255),  # ε≈8/255 standard benchmark
        'PGD': ProjectedGradientDescent(
            classifier, eps=8/255, max_iter=40, eps_step=2/255
        ),
        'CW': CarliniL2Method(
            classifier, max_iter=100, confidence=0.0
        ),
        'Square': SquareAttack(classifier, eps=8/255, max_iter=1000),
    }

    # Filter attacks if a subset was requested
    if attacks_to_run is not None:
        attacks = {k: v for k, v in attacks.items() if k in attacks_to_run}

    # --- Load test data in pixel space (ART attacks in [0,1]) ---
    # After attack generation, we apply _NORMALIZE before passing to PRISM.
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=_PIXEL_TRANSFORM
    )

    # --- Evaluate each attack ---
    # IMPORTANT: Create a fresh PRISM instance per attack so that the
    # TopologicalProfiler RNG (used in point-cloud subsampling) is always
    # reset to seed=42 before evaluating each attack's clean images.
    # Without this, the RNG state changes after each attack, causing
    # different subsampling of clean images per attack → inconsistent FPR.
    results = {}

    for attack_name, attack in attacks.items():
        print(f"\n{'='*50}")
        print(f"Evaluating: {attack_name}")
        print(f"{'='*50}")

        # Fresh PRISM instance with NoOpCampaignMonitor:
        # Evaluation measures pure TAMM+CADG detection (TPR/FPR).
        # L0 campaign detection is evaluated separately in experiments/campaign/.
        # Using active L0 here inflates FPR because BOCPD fires false alarms
        # during the 300-sample clean phase, then lowers thresholds for all subsequent samples.
        prism_attack = PRISM.from_saved(
            model=model,
            layer_names=layer_names,
            calibrator_path=calibrator_path,
            profile_path=profile_path,
            layer_weights=layer_weights,
            dim_weights=dim_weights,
            campaign_monitor=NoOpCampaignMonitor(),
        )

        tp, fp, fn, tn = 0, 0, 0, 0
        level_counts_clean = {}
        level_counts_adv = {}

        # Test split for evaluation is strictly 8000-9999 to prevent data leakage
        n_samples = min(n_test, 2000)
        eval_indices = list(range(8000, 10000))
        sample_indices = rng.choice(eval_indices, n_samples, replace=False)

        for i in tqdm(sample_indices):
            img, label = test_dataset[int(i)]
            # img is in pixel [0,1] space (no normalization)
            x_pixel = img.unsqueeze(0)  # (1, 3, 224, 224) in [0,1]

            # Normalize for PRISM (same preprocessing as build_profile.py)
            x = _NORMALIZE(img).unsqueeze(0).to(device)

            # --- Test clean input ---
            _, level_clean, _ = prism_attack.defend(x)
            level_counts_clean[level_clean] = level_counts_clean.get(level_clean, 0) + 1

            if level_clean == 'PASS':
                tn += 1
            else:
                fp += 1

            # --- Generate adversarial in pixel space, then normalize for PRISM ---
            x_np = x_pixel.cpu().numpy()  # pixel space [0,1] — correct for ART
            try:
                x_adv_np = attack.generate(x_np)  # still in [0,1] after clipping
            except Exception as e:
                print(f"  Attack failed on sample {i}: {e}")
                continue

            # Apply normalization so PRISM receives the same distribution as training
            x_adv = _NORMALIZE(torch.tensor(x_adv_np[0])).unsqueeze(0).to(device)

            # --- Test adversarial input ---
            _, level_adv, _ = prism_attack.defend(x_adv)
            level_counts_adv[level_adv] = level_counts_adv.get(level_adv, 0) + 1

            if level_adv != 'PASS':
                tp += 1
            else:
                fn += 1

        # Compute metrics
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 2 * precision * tpr / max(precision + tpr, 1e-8)

        results[attack_name] = {
            'TPR': round(tpr, 4),
            'FPR': round(fpr, 4),
            'Precision': round(precision, 4),
            'F1': round(f1, 4),
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'clean_level_distribution': level_counts_clean,
            'adversarial_level_distribution': level_counts_adv,
        }

        print(f"  TPR={tpr:.4f}  FPR={fpr:.4f}  F1={f1:.4f}")
        print(f"  Clean levels:  {level_counts_clean}")
        print(f"  Adv levels:    {level_counts_adv}")

    # --- Save results ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # --- Summary table ---
    print(f"\n{'='*60}")
    print(f"{'Attack':>12} {'TPR':>8} {'FPR':>8} {'F1':>8}")
    print(f"{'-'*60}")
    for name, r in results.items():
        print(f"{name:>12} {r['TPR']:>8.4f} {r['FPR']:>8.4f} {r['F1']:>8.4f}")

    return results


def run_fgsm_sweep(
    epsilons: list = None,
    n_test: int = 300,
    data_root: str = './data',
    output_path: str = 'experiments/evaluation/results_fgsm_sweep.json',
    seed: int = 42,
):
    """
    FGSM epsilon sweep for detection sensitivity curve.
    Runs FGSM at multiple epsilon values to show TPR vs ε tradeoff.
    Essential for honest paper presentation of TDA's sensitivity limits.
    """
    if epsilons is None:
        epsilons = [0.01, 0.02, 0.03, 0.05, 0.10]

    # Run evaluation once per epsilon, collect results
    sweep_results = {}
    for eps in epsilons:
        print(f"\n{'='*50}")
        print(f"FGSM sweep: ε={eps:.3f}")
        r = run_evaluation(
            n_test=n_test,
            data_root=data_root,
            output_path=None,   # don't save per-epsilon
            seed=seed,
            attacks_to_run=['FGSM'],
            _fgsm_eps_override=eps,
        )
        if r and 'FGSM' in r:
            sweep_results[f'FGSM_eps_{eps:.3f}'] = r['FGSM']
            print(f"  ε={eps:.3f}: TPR={r['FGSM']['TPR']:.4f} FPR={r['FGSM']['FPR']:.4f}")

    # Print curve
    print(f"\n{'='*50}")
    print(f"{'ε':>8} {'TPR':>8} {'FPR':>8}")
    for key, v in sweep_results.items():
        eps = float(key.split('_')[-1])
        print(f"{eps:>8.3f} {v['TPR']:>8.4f} {v['FPR']:>8.4f}")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(sweep_results, f, indent=2)
        print(f"Sweep results saved to {output_path}")

    return sweep_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="PRISM full attack evaluation")
    parser.add_argument('--n-test',   type=int, default=1000,
                        help='Number of test images per attack (default: 1000)')
    parser.add_argument('--data-root', default='./data')
    parser.add_argument('--output',   default='experiments/evaluation/results.json')
    parser.add_argument('--attacks',  nargs='+',
                        default=['FGSM', 'PGD', 'Square'],
                        help='Attacks to run (default: FGSM PGD Square)')
    parser.add_argument('--sweep', action='store_true',
                        help='Run FGSM epsilon sweep instead of main eval')
    args = parser.parse_args()
    if args.sweep:
        run_fgsm_sweep(n_test=args.n_test, data_root=args.data_root)
    else:
        run_evaluation(n_test=args.n_test, data_root=args.data_root,
                       output_path=args.output, attacks_to_run=args.attacks)
