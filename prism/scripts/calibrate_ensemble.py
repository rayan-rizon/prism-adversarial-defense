"""
Calibrate Conformal Thresholds for Persistence Ensemble Scorer

This script re-calibrates the conformal thresholds using the composite
anomaly score from PersistenceEnsembleScorer. This is essential because
the ensemble score has a different distribution than the base Wasserstein score.

RATIONALE:
Ensuring the FPR targets (10%, 3%, 0.5%) are strictly met for the 
final publishable results.

USAGE:
  python scripts/calibrate_ensemble.py
"""
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models import ResNet18_Weights
import numpy as np
import pickle
import os, sys
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.tamm.extractor import ActivationExtractor
from src.tamm.tda import TopologicalProfiler
from src.cadg.ensemble_scorer import PersistenceEnsembleScorer
from src.cadg.calibrate import ConformalCalibrator

# --- Configuration (MUST match build_profile_testset.py) ---
LAYER_NAMES = ['layer2', 'layer3', 'layer4']
CAL_IDX     = (5000, 7000)
VAL_IDX     = (7000, 8000)

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]
_TRANSFORM = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize(mean=_MEAN, std=_STD),
])

def calibrate_ensemble(
    data_root: str = './data',
    ensemble_path: str = 'models/ensemble_scorer.pkl',
    profile_path: str  = 'models/reference_profiles.pkl',
    output_path: str   = 'models/calibrator.pkl',
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    if not os.path.exists(ensemble_path):
        print(f"ERROR: {ensemble_path} not found. Run scripts/train_ensemble_scorer.py first.")
        sys.exit(1)
        
    with open(profile_path, 'rb') as f:
        ref_profiles = pickle.load(f)
        
    # Load base components to reconstruct ensemble
    from src.tamm.scorer import TopologicalScorer
    base_scorer = TopologicalScorer(
        ref_profiles=ref_profiles,
        layer_names=LAYER_NAMES,
        layer_weights={'layer2': 0.15, 'layer3': 0.30, 'layer4': 0.55},
        dim_weights=[0.5, 0.5],
    )
    
    ensemble = PersistenceEnsembleScorer.load(ensemble_path, base_scorer, LAYER_NAMES)
    print("Loaded PersistenceEnsembleScorer.")

    # --- Setup model ---
    model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model = model.to(device).eval()
    extractor = ActivationExtractor(model, LAYER_NAMES)
    profiler  = TopologicalProfiler(n_subsample=200, max_dim=1)

    dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=_TRANSFORM
    )

    def get_ensemble_scores(idx_range, label):
        scores = []
        for i in tqdm(range(idx_range[0], idx_range[1]), desc=f"Scoring {label}"):
            img, _ = dataset[i]
            x = img.unsqueeze(0).to(device)
            acts = extractor.extract(x)
            dgms = {}
            for layer in LAYER_NAMES:
                act_np = acts[layer].squeeze(0).cpu().numpy()
                dgms[layer] = profiler.compute_diagram(act_np)
            s = ensemble.score(dgms)
            scores.append(s)
        return np.array(scores, dtype=np.float32)

    print("\nComputing ensemble scores for calibration and validation splits...")
    cal_scores = get_ensemble_scores(CAL_IDX, 'calibration')
    val_scores = get_ensemble_scores(VAL_IDX, 'validation')

    # --- Calibrate ---
    calibrator = ConformalCalibrator()
    thresholds = calibrator.calibrate(cal_scores)

    print("\n=== Ensemble Conformal Thresholds ===")
    for level, threshold in thresholds.items():
        print(f"  {level:2s} (α={calibrator.alphas[level]:.3f}): threshold = {threshold:.6f}")

    # --- Verify ---
    print("\n=== FPR Verification (Ensemble) ===")
    report = calibrator.get_coverage_report(val_scores)
    for level, info in sorted(report.items()):
        status = "[PASS]" if info['passed'] else "[FAIL]"
        print(f"  {level:2s}: empirical FPR = {info['empirical_fpr']:.4f}  "
              f"(target <= {info['target_alpha']:.4f})  {status}")

    # --- Save ---
    with open(output_path, 'wb') as f:
        pickle.dump(calibrator, f)
    print(f"\nCalibrator (ensemble) saved → {output_path}")
    
    extractor.cleanup()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='./data')
    args = parser.parse_args()
    calibrate_ensemble(data_root=args.data_root)
