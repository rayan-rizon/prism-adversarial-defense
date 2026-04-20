"""
Build Topological Self-Profile from CIFAR-10 TEST set (FPR fix)

RATIONALE
---------
The original build_profile.py used CIFAR-10 TRAINING data for profiling and
calibration, while run_evaluation.py evaluates on the TEST set.  CIFAR-10
train and test have a measurable distribution shift in activation space, which
causes the conformal guarantee to be violated:

  - L2 target α=0.03  -> observed FPR ≈ 3.3%  (excess 0.3%)
  - L3 target α=0.005 -> observed FPR ≈ 1.3%  (excess 0.8%)

FIX: Use the TEST set for both profiling and calibration.  This restores the
exchangeability assumption required by split conformal prediction.

TEST SET SPLIT (10,000 images):
  images 0–4999    -> profile set  (compute reference medoids + clean scores)
  images 5000–6999 -> calibration  (fit conformal thresholds)
  images 7000–7999 -> validation   (verify FPR guarantee)
  images 8000–9999 -> HELD OUT     (used by run_evaluation_full.py for final eval)

This clean separation ensures zero data leakage between:
  profiling / calibration / validation / evaluation.

USAGE
-----
  cd prism/
  python scripts/build_profile_testset.py          # builds from CIFAR-10 test
  python scripts/calibrate_testset.py              # calibrates using test cal split
"""
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models import ResNet18_Weights
import numpy as np
import pickle
import os
import sys
import ssl
import certifi
from tqdm import tqdm

# SSL fix
os.environ.setdefault('SSL_CERT_FILE', certifi.where())
os.environ.setdefault('REQUESTS_CA_BUNDLE', certifi.where())
ssl._create_default_https_context = ssl.create_default_context

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.tamm.extractor import ActivationExtractor
from src.tamm.tda import TopologicalProfiler
from src.tamm.scorer import TopologicalScorer
from src.config import (
    LAYER_NAMES, LAYER_WEIGHTS, DIM_WEIGHTS, N_SUBSAMPLE, MAX_DIM,
    IMAGENET_MEAN, IMAGENET_STD,
    PROFILE_IDX, CAL_IDX, VAL_IDX, EVAL_IDX,
)
# All shared constants imported from src.config (backed by configs/default.yaml).
# Split indices are the single source of truth -- do not hardcode here.

_TRANSFORM = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def build_profile_testset(
    data_root: str = './data',
    output_dir: str = './models',
    device: str = None,
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print("Building topological profile from CIFAR-10 TEST set...")
    print(f"  Profile range: indices {PROFILE_IDX[0]}-{PROFILE_IDX[1]-1}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model = model.to(device).eval()

    extractor = ActivationExtractor(model, LAYER_NAMES)
    profiler  = TopologicalProfiler(n_subsample=N_SUBSAMPLE, max_dim=MAX_DIM)

    # ── Dataset (test) ────────────────────────────────────────────────────────
    dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=_TRANSFORM
    )
    print(f"  Test set size: {len(dataset)}")

    # ── Phase 1: collect persistence diagrams for profile images ───────────────
    n_profile = PROFILE_IDX[1] - PROFILE_IDX[0]
    all_diagrams = {layer: [] for layer in LAYER_NAMES}
    print(f"\nPhase 1: Collecting diagrams from {n_profile} test images "
          f"(indices {PROFILE_IDX[0]}-{PROFILE_IDX[1]-1})...")
    for i in tqdm(range(PROFILE_IDX[0], PROFILE_IDX[1]), desc="Phase 1: diagrams"):
        img, _ = dataset[i]
        x = img.unsqueeze(0).to(device)
        acts = extractor.extract(x)
        for layer in LAYER_NAMES:
            act_np = acts[layer].squeeze(0).cpu().numpy()
            all_diagrams[layer].append(profiler.compute_diagram(act_np))

    # ── Phase 2: compute medoid reference diagrams ─────────────────────────────
    print("\nPhase 2: Computing medoid reference diagrams "
          f"(dims={[0,1]}, dim_weights={DIM_WEIGHTS})...")
    # Use the SAME dims and dim_weights as TopologicalScorer at inference time,
    # so the medoid minimises the same distance criterion used for scoring.
    ref_profiles = {}
    for layer in LAYER_NAMES:
        print(f"  {layer}: computing medoid from {len(all_diagrams[layer])} diagrams...")
        ref_profiles[layer] = profiler.compute_reference_medoid(
            all_diagrams[layer], dims=[0, 1], dim_weights=DIM_WEIGHTS
        )
        n_h0 = len(ref_profiles[layer][0]) if len(ref_profiles[layer]) > 0 else 0
        n_h1 = len(ref_profiles[layer][1]) if len(ref_profiles[layer]) > 1 else 0
        print(f"    Medoid: {n_h0} H0 features, {n_h1} H1 features")

    os.makedirs(output_dir, exist_ok=True)
    profile_path = os.path.join(output_dir, 'reference_profiles.pkl')
    with open(profile_path, 'wb') as f:
        pickle.dump(ref_profiles, f)
    print(f"\nReference profiles saved -> {profile_path} (built from TEST set)")

    # Free diagram memory before scoring phase
    del all_diagrams

    # ── Phase 3: compute anomaly scores for ALL splits ─────────────────────────
    scorer = TopologicalScorer(
        ref_profiles=ref_profiles,
        layer_names=LAYER_NAMES,
        layer_weights=LAYER_WEIGHTS,
        dim_weights=DIM_WEIGHTS,
    )

    # Save the scorer for consistent reuse
    scorer_path = os.path.join(output_dir, 'scorer.pkl')
    with open(scorer_path, 'wb') as f:
        pickle.dump(scorer, f)
    print(f"Scorer saved -> {scorer_path}")

    def compute_scores(idx_range, label):
        """Compute anomaly scores for a contiguous index range."""
        scores = []
        for i in tqdm(range(idx_range[0], idx_range[1]),
                      desc=f"Scoring {label}"):
            img, _ = dataset[i]
            x = img.unsqueeze(0).to(device)
            acts = extractor.extract(x)
            dgms = {}
            for layer in LAYER_NAMES:
                act_np = acts[layer].squeeze(0).cpu().numpy()
                dgms[layer] = profiler.compute_diagram(act_np)
            s = scorer.score(dgms)
            scores.append(s)
        return np.array(scores, dtype=np.float32)

    print(f"\nPhase 3: Computing anomaly scores for all splits...")

    cal_dir = os.path.join(output_dir, '..', 'experiments', 'calibration')
    os.makedirs(cal_dir, exist_ok=True)

    # Profile scores (used to verify profiler is sane)
    profile_scores = compute_scores(PROFILE_IDX, 'profile')
    np.save(os.path.join(cal_dir, 'profile_scores.npy'), profile_scores)
    print(f"  Profile : n={len(profile_scores)}, "
          f"mean={profile_scores.mean():.4f}, std={profile_scores.std():.4f}")

    # Calibration scores (for conformal threshold fitting)
    cal_scores = compute_scores(CAL_IDX, 'calibration')
    np.save(os.path.join(cal_dir, 'clean_scores.npy'), cal_scores)
    print(f"  Calibration: n={len(cal_scores)}, "
          f"mean={cal_scores.mean():.4f}, std={cal_scores.std():.4f}")

    # Validation scores (for FPR guarantee verification)
    val_scores = compute_scores(VAL_IDX, 'validation')
    np.save(os.path.join(cal_dir, 'val_scores.npy'), val_scores)
    print(f"  Validation:  n={len(val_scores)}, "
          f"mean={val_scores.mean():.4f}, std={val_scores.std():.4f}")

    extractor.cleanup()

    # ── Phase 4: sanity check ──────────────────────────────────────────────────
    print("\nPhase 4: Sanity checks...")
    assert profile_scores.mean() > 0, "Profile scores must be positive"
    assert np.all(np.isfinite(profile_scores)), "No NaN/inf in profile scores"
    assert cal_scores.mean() < 30, \
        f"Cal score mean={cal_scores.mean():.2f} suspiciously high"

    print("\n[OK] Profile built successfully from CIFAR-10 TEST set.")
    print(f"   Profile : {len(profile_scores)} images (test idx 0-4999)")
    print(f"   Cal     : {len(cal_scores)} images (test idx 5000-6999)")
    print(f"   Val     : {len(val_scores)} images (test idx 7000-7999)")
    print(f"   Eval    : 2000 images HELD OUT (test idx 8000-9999)")
    print(f"\nNext step: python scripts/calibrate_testset.py")

    return ref_profiles, cal_scores, val_scores


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='./data')
    parser.add_argument('--output-dir', default='./models')
    args = parser.parse_args()
    build_profile_testset(data_root=args.data_root, output_dir=args.output_dir)
