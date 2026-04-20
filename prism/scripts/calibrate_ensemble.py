"""
Calibrate Conformal Thresholds for Persistence Ensemble Scorer

This script re-calibrates the conformal thresholds using the composite
anomaly score from PersistenceEnsembleScorer. This is essential because
the ensemble score has a different distribution than the base Wasserstein score.

RATIONALE:
Ensuring the FPR targets (10%, 3%, 0.5%) are strictly met for the
final publishable results.

PIPELINE GATE:
Any change to PersistenceEnsembleScorer or the feature vector MUST be
followed by re-running this script, then compute_ensemble_val_fpr.py,
then run_evaluation_full.py.  See Appendix A items A-6/A-7 in
PRISM Implementation.md for the silent failures this prevents.

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
# All shared constants come from src.config (backed by configs/default.yaml).
# Do not re-introduce hardcoded copies — silent drift caused results failures.
from src.config import (
    LAYER_NAMES, LAYER_WEIGHTS, DIM_WEIGHTS,
    IMAGENET_MEAN, IMAGENET_STD,
    CAL_IDX, VAL_IDX, CONFORMAL_ALPHAS, CAL_ALPHA_FACTOR,
    N_SUBSAMPLE, MAX_DIM,
)

_TRANSFORM = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def calibrate_ensemble(
    data_root: str = './data',
    ensemble_path: str = 'models/ensemble_scorer.pkl',
    profile_path: str  = 'models/reference_profiles.pkl',
    output_path: str   = 'models/calibrator.pkl',
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Cal split:  test idx {CAL_IDX[0]}-{CAL_IDX[1]-1}  (n={CAL_IDX[1]-CAL_IDX[0]})")
    print(f"Val split:  test idx {VAL_IDX[0]}-{VAL_IDX[1]-1}  (n={VAL_IDX[1]-VAL_IDX[0]})")
    print(f"Cal alpha factor: {CAL_ALPHA_FACTOR:.2f} x published targets")

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
        layer_weights=LAYER_WEIGHTS,
        dim_weights=DIM_WEIGHTS,
    )

    ensemble = PersistenceEnsembleScorer.load(ensemble_path, base_scorer, LAYER_NAMES)
    print("Loaded PersistenceEnsembleScorer.")

    # --- Setup model ---
    model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model = model.to(device).eval()
    extractor = ActivationExtractor(model, LAYER_NAMES)
    profiler  = TopologicalProfiler(n_subsample=N_SUBSAMPLE, max_dim=MAX_DIM)

    dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=_TRANSFORM
    )

    def get_ensemble_scores(idx_range, label):
        scores = []
        _use_dct = getattr(ensemble, 'use_dct', False)
        for i in tqdm(range(idx_range[0], idx_range[1]), desc=f"Scoring {label}"):
            img, _ = dataset[i]
            x = img.unsqueeze(0).to(device)
            acts = extractor.extract(x)
            dgms = {}
            for layer in LAYER_NAMES:
                act_np = acts[layer].squeeze(0).cpu().numpy()
                dgms[layer] = profiler.compute_diagram(act_np)
            img_np = img.numpy() if _use_dct else None
            s = ensemble.score(dgms, image=img_np)
            scores.append(s)
        return np.array(scores, dtype=np.float32)

    print("\nComputing ensemble scores for calibration and validation splits...")
    cal_scores = get_ensemble_scores(CAL_IDX, 'calibration')
    val_scores = get_ensemble_scores(VAL_IDX, 'validation')

    # --- Calibrate with conservative alphas (CAL_ALPHA_FACTOR of published) ---
    # Published targets:    L1=10 %,  L2=3 %,    L3=0.5 %
    # Calibration factor:   configs/default.yaml -> conformal.cal_alpha_factor
    #                       (default 0.7 -> L1=7 %, L2=2.1 %, L3=0.35 %)
    #
    # The previous 0.8 factor let L3 FPR=0.008 slip past the 0.005 target in
    # results_n500_planA.json.  0.7 brings L3 into compliance at the cost of
    # ~1-2 pp TPR.  Threshold is computed at 70% but verified against the
    # published alpha, so the conformal guarantee holds.
    pub_targets = dict(CONFORMAL_ALPHAS)
    cal_targets = {k: v * CAL_ALPHA_FACTOR for k, v in pub_targets.items()}

    calibrator = ConformalCalibrator(alphas=pub_targets)
    # Pass cal_targets explicitly -> stored in calibrator.calibration_alphas
    thresholds = calibrator.calibrate(cal_scores, alphas=cal_targets)

    print("\n=== Ensemble Conformal Thresholds ===")
    print(f"  [Cal alpha factor: {CAL_ALPHA_FACTOR:.2f} | "
          f"Published: L1<={pub_targets['L1']:.3f}, "
          f"L2<={pub_targets['L2']:.3f}, L3<={pub_targets['L3']:.4f}]")
    for level, threshold in thresholds.items():
        cal_a = calibrator.calibration_alphas[level]
        pub_a = pub_targets[level]
        print(f"  {level:2s}  cal_alpha={cal_a:.4f}  pub_alpha={pub_a:.4f}  threshold={threshold:.6f}")

    # --- Verify on val split against PUBLISHED alpha targets (strict, tol=0.0) ---
    # calibrator.alphas == pub_targets (set in __init__).
    # get_coverage_report() checks empirical_fpr <= self.alphas[level] + tolerance.
    print("\n=== FPR Verification on Val Split (strict, tol=0.0, target=published alpha) ===")
    report = calibrator.get_coverage_report(val_scores, tolerance=0.0)
    all_passed = True
    for level, info in report.items():
        status = "[PASS]" if info['passed'] else "[FAIL]"
        if not info['passed']:
            all_passed = False
        print(
            f"  {level:2s}: empirical FPR={info['empirical_fpr']:.4f}  "
            f"(pub target<={info['target_alpha']:.4f})  {status}"
        )

    if all_passed:
        print("\n  [OK] All ensemble FPR guarantees verified on val split.")
    else:
        factor_hint = round(CAL_ALPHA_FACTOR - 0.05, 2)
        print(f"\n  [WARN] Some guarantees still violated. "
              f"Lower conformal.cal_alpha_factor in configs/default.yaml "
              f"from {CAL_ALPHA_FACTOR:.2f} to {factor_hint:.2f} and rerun.")

    # --- Save ---
    with open(output_path, 'wb') as f:
        pickle.dump(calibrator, f)
    print(f"\nCalibrator (ensemble) saved -> {output_path}")

    extractor.cleanup()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='./data')
    args = parser.parse_args()
    calibrate_ensemble(data_root=args.data_root)
