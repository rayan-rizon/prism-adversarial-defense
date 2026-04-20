"""
Train Persistence Ensemble Scorer (Phase 3 from Improvement Plan)

Trains the logistic regression component of PersistenceEnsembleScorer on
persistence statistics features extracted from clean and adversarial images.

DESIGN FOR PUBLISHABILITY
--------------------------
1. Uses a DEDICATED training split, fully separate from:
   - The conformal calibration/validation split (test idx 5000-7999)
   - The final evaluation set (test idx 8000-9999)
   Training data: CIFAR-10 TRAINING set (50,000 images).

2. Adversarial training mix -- tri-split (~34% FGSM / 33% PGD / 33% Square):
   - FGSM   (L-inf, eps=8/255, single-step)        -- gradient-based, one step
   - PGD    (L-inf, eps=8/255, 20 steps)           -- gradient-based, iterative
   - Square (L-inf, eps=8/255, 1000 queries)       -- GRADIENT-FREE black-box
   All three at the SAME epsilon as the evaluation (eps=8/255), ensuring the
   logistic boundary is calibrated for the correct perturbation budget.
   Square is included because results_n500_planA.json showed Square TPR
   trailing PGD by ~35 pp when the logistic was trained only on FGSM+PGD.
   Adding Square exposes the linear classifier to its gradient-free topology
   signature directly.  AutoAttack and CW-L2 are NOT trained on locally --
   they remain evaluation-only (run on Thundercompute after local targets pass).

3. The logistic regression is a linear classifier in feature space -- it has
   no hidden layers and cannot overfit in the way a deep model would.

4. Regularisation C=1.0 (L2). Validated by held-out 20% AUC. Reported
   transparently in the paper.

5. Data-derived normalisation constants (logit_shift, w_score_mean) replace
   ad-hoc magic numbers, making the composite score formula reproducible.

6. Training provenance (eps, attack list, n) stored in the pkl artifact.

PIPELINE GATE:
After running this script you MUST run:
  python scripts/calibrate_ensemble.py
  python scripts/compute_ensemble_val_fpr.py
  python experiments/evaluation/run_evaluation_full.py
Skipping any step invalidates the FPR guarantee.

USAGE
-----
  cd prism/
  python scripts/train_ensemble_scorer.py [--n-train 2100] [--fast]
"""
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models import ResNet18_Weights
import numpy as np
import pickle
import os, sys, ssl, certifi
from tqdm import tqdm

os.environ.setdefault('SSL_CERT_FILE', certifi.where())
os.environ.setdefault('REQUESTS_CA_BUNDLE', certifi.where())
ssl._create_default_https_context = ssl.create_default_context

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from art.attacks.evasion import (
        FastGradientMethod, ProjectedGradientDescent, SquareAttack,
    )
    from art.estimators.classification import PyTorchClassifier
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False
    print("WARNING: ART not available -- cannot generate adversarial features.")

from src.tamm.extractor import ActivationExtractor
from src.tamm.tda import TopologicalProfiler
from src.tamm.scorer import TopologicalScorer
from src.tamm.persistence_stats import extract_feature_vector
from src.cadg.ensemble_scorer import PersistenceEnsembleScorer
from src.config import (
    LAYER_NAMES, LAYER_WEIGHTS, DIM_WEIGHTS, IMAGENET_MEAN, IMAGENET_STD,
    EPS_LINF_STANDARD, N_SUBSAMPLE, MAX_DIM,
)

# All shared constants are imported from src.config (backed by default.yaml).

DIMS = (0, 1)  # homology dimensions; matches DIM_WEIGHTS length from src.config

_MEAN = IMAGENET_MEAN
_STD  = IMAGENET_STD
_PIXEL_TRANSFORM = T.Compose([T.Resize(224), T.ToTensor()])
_NORMALIZE       = T.Normalize(mean=_MEAN, std=_STD)


class _NormalizedResNet(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self._model = model
        self.register_buffer('_mean', torch.tensor(_MEAN).view(3, 1, 1))
        self.register_buffer('_std',  torch.tensor(_STD).view(3, 1, 1))

    def forward(self, x):
        return self._model((x - self._mean) / self._std)


def _extract_features(dataset_indices, dataset, model, extractor, profiler,
                      scorer, ref_profiles, device, art_attack=None,
                      label='clean', n_max=None):
    """
    Extract 37-dim persistence+DCT feature vectors AND base Wasserstein scores.

    Returns:
        features: (N, 37) float32 feature matrix (36 persistence + 1 DCT energy)
        w_scores:  (N,) float32 base Wasserstein composite scores
    """
    features = []
    w_scores = []
    indices = list(dataset_indices)
    if n_max is not None:
        indices = indices[:n_max]

    for i in tqdm(indices, desc=f"Extracting {label} features"):
        img_pixel, _ = dataset[int(i)]  # pixel [0,1] space

        if art_attack is not None:
            x_np = img_pixel.unsqueeze(0).numpy()
            try:
                x_adv_np = art_attack.generate(x_np)
                img_tensor = _NORMALIZE(torch.tensor(x_adv_np[0]))
                img_for_dct = torch.tensor(x_adv_np[0])  # (C, H, W) adversarial, [0,1]
            except Exception:
                img_tensor = _NORMALIZE(img_pixel)
                img_for_dct = img_pixel  # fallback to clean
        else:
            img_tensor = _NORMALIZE(img_pixel)
            img_for_dct = img_pixel  # (C, H, W) clean image, [0,1]

        x = img_tensor.unsqueeze(0).to(device)
        acts = extractor.extract(x)
        dgms = {}
        for layer in LAYER_NAMES:
            act_np = acts[layer].squeeze(0).cpu().numpy()
            dgms[layer] = profiler.compute_diagram(act_np)

        feat    = extract_feature_vector(dgms, ref_profiles, LAYER_NAMES, list(DIMS),
                                         image=img_for_dct.numpy() if hasattr(img_for_dct, 'numpy') else img_for_dct)
        w_score = scorer.score(dgms)   # base Wasserstein composite score

        features.append(feat)
        w_scores.append(w_score)

    if features:
        return np.stack(features, axis=0), np.array(w_scores, dtype=np.float32)
    n_feat = 37  # 36 persistence stats + 1 DCT energy feature
    return np.zeros((0, n_feat), dtype=np.float32), np.zeros(0, dtype=np.float32)


def train_ensemble_scorer(
    n_train: int = 2100,
    fgsm_eps: float = EPS_LINF_STANDARD,   # 8/255 -- matches evaluation eps
    data_root: str = './data',
    output_path: str = 'models/ensemble_scorer.pkl',
    calibrator_path: str = 'models/calibrator.pkl',
    profile_path: str = 'models/reference_profiles.pkl',
    device: str = None,
):
    if not ART_AVAILABLE:
        print("ERROR: adversarial-robustness-toolbox not installed.")
        print("  pip install adversarial-robustness-toolbox")
        sys.exit(1)

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Training ensemble scorer: n_train={n_train}, eps={fgsm_eps:.6f} (= 8/255 = {8/255:.6f})")
    print(f"Adversarial mix: ~1/3 FGSM + ~1/3 PGD + ~1/3 Square (all at eps=8/255)")

    # -- Load pre-built components ------------------------------------------------
    if not os.path.exists(profile_path):
        print(f"ERROR: {profile_path} not found. Run build_profile_testset.py first.")
        sys.exit(1)

    with open(profile_path, 'rb') as f:
        ref_profiles = pickle.load(f)

    # -- Setup model -------------------------------------------------------------
    model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model = model.to(device).eval()

    extractor = ActivationExtractor(model, LAYER_NAMES)
    profiler  = TopologicalProfiler(n_subsample=N_SUBSAMPLE, max_dim=MAX_DIM)
    base_scorer = TopologicalScorer(
        ref_profiles=ref_profiles,
        layer_names=LAYER_NAMES,
        layer_weights=LAYER_WEIGHTS,
        dim_weights=DIM_WEIGHTS,
    )

    # -- ART adversarial generation ---------------------------------------------
    # Training data uses CIFAR-10 TRAINING set to avoid any test-set leakage
    pixel_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=_PIXEL_TRANSFORM
    )
    rng = np.random.RandomState(42)
    all_idx = rng.permutation(len(pixel_dataset))
    clean_idx = all_idx[:n_train]
    # Pre-allocate a 3x budget pool; actual slices are taken per attack below
    adv_pool  = all_idx[n_train: n_train + 3 * n_train]

    # ART classifier for attack generation (normalisation wrapped inside model)
    norm_model = _NormalizedResNet(
        torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
    )
    art_clf = PyTorchClassifier(
        model=norm_model,
        loss=torch.nn.CrossEntropyLoss(),
        input_shape=(3, 224, 224),
        nb_classes=1000,
        clip_values=(0.0, 1.0),
    )

    # Attacks at the SAME epsilon as evaluation
    fgsm_attack   = FastGradientMethod(art_clf, eps=fgsm_eps)
    pgd_attack    = ProjectedGradientDescent(
        art_clf, eps=fgsm_eps, eps_step=fgsm_eps / 4, max_iter=20
    )
    # Square: gradient-free black-box; 1000 queries is enough for training
    # signal (eval uses 5000).  Lower budget keeps training time reasonable.
    square_attack = SquareAttack(
        art_clf, eps=fgsm_eps, max_iter=1000, batch_size=1, verbose=False,
    )

    # -- Extract features -------------------------------------------------------
    print(f"\nExtracting {n_train} clean feature vectors...")
    X_clean, W_clean = _extract_features(
        clean_idx, pixel_dataset, model, extractor, profiler,
        base_scorer, ref_profiles, device, art_attack=None, label='clean'
    )

    # Tri-split adversarial budget: ~1/3 FGSM + ~1/3 PGD + ~1/3 Square.
    # Tail imbalance (n_train mod 3) goes to Square so FGSM/PGD stay equal.
    n_fgsm   = n_train // 3
    n_pgd    = n_train // 3
    n_square = n_train - n_fgsm - n_pgd

    off = 0
    print(f"\nExtracting {n_fgsm} FGSM (eps={fgsm_eps:.6f}) adversarial features...")
    X_fgsm, _ = _extract_features(
        adv_pool[off: off + n_fgsm], pixel_dataset, model, extractor, profiler,
        base_scorer, ref_profiles, device, art_attack=fgsm_attack, label='FGSM adv'
    )
    off += n_fgsm

    print(f"\nExtracting {n_pgd} PGD (eps={fgsm_eps:.6f}, 20 steps) adversarial features...")
    X_pgd, _ = _extract_features(
        adv_pool[off: off + n_pgd], pixel_dataset, model, extractor, profiler,
        base_scorer, ref_profiles, device, art_attack=pgd_attack, label='PGD adv'
    )
    off += n_pgd

    print(f"\nExtracting {n_square} Square (eps={fgsm_eps:.6f}, 1000 queries) adversarial features...")
    X_square, _ = _extract_features(
        adv_pool[off: off + n_square], pixel_dataset, model, extractor, profiler,
        base_scorer, ref_profiles, device, art_attack=square_attack, label='Square adv'
    )

    X_adv = np.vstack([X_fgsm, X_pgd, X_square])
    # Shuffle so the 80/20 train/val split sees all three attack families --
    # without shuffle, X_adv[:80%] would contain almost no Square samples.
    perm  = np.random.RandomState(42).permutation(len(X_adv))
    X_adv = X_adv[perm]

    print(f"\nFeature matrix shapes: clean={X_clean.shape}, adv={X_adv.shape}")
    print(f"  (FGSM: {X_fgsm.shape[0]}, PGD: {X_pgd.shape[0]}, Square: {X_square.shape[0]})")

    # -- Train ensemble scorer --------------------------------------------------
    print("\nFitting logistic regression ensemble component...")
    ensemble = PersistenceEnsembleScorer(
        base_scorer=base_scorer,
        layer_names=LAYER_NAMES,
        dims=DIMS,
        alpha=0.5,   # equal blend; calibrated via conformal recalibration
        use_dct=True,  # 37-dim: 36 persistence stats + 1 DCT energy feature
    )

    # 80/20 split for training vs internal validation
    n_clean_train = int(0.8 * len(X_clean))
    n_adv_train   = int(0.8 * len(X_adv))
    ensemble.fit_logistic(
        clean_features=X_clean[:n_clean_train],
        adv_features=X_adv[:n_adv_train],
        C=1.0,
        clean_w_scores=W_clean[:n_clean_train],  # for data-derived w_score_mean
    )

    # -- Validation AUC on held-out 20% ----------------------------------------
    from sklearn.metrics import roc_auc_score, classification_report
    X_val = np.vstack([X_clean[n_clean_train:], X_adv[n_adv_train:]])
    y_val = np.array(
        [0] * len(X_clean[n_clean_train:]) + [1] * len(X_adv[n_adv_train:])
    )
    if len(X_val) > 0:
        probs = np.array([ensemble._logistic_prob(x) for x in X_val])
        auc   = roc_auc_score(y_val, probs)
        preds = (probs > 0.5).astype(int)
        print(f"\nHeld-out validation AUC (logistic component): {auc:.4f}")
        print(classification_report(y_val, preds, target_names=['clean', 'adv']))

    # -- Set provenance metadata ------------------------------------------------
    ensemble.training_eps     = fgsm_eps
    ensemble.training_attacks = ['FGSM', 'PGD', 'Square']
    ensemble.training_n       = len(X_adv)

    # -- Save -------------------------------------------------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ensemble.save(output_path)

    extractor.cleanup()

    print("\n[OK] Ensemble scorer trained and saved.")
    print(f"   Training eps = {fgsm_eps:.6f} (= 8/255, matches evaluation)")
    print(f"   Adversarial mix: FGSM({n_fgsm}) + PGD({n_pgd}) + Square({n_square})")
    print(f"   Trained on CIFAR-10 TRAINING set (no test-set leakage)")
    print(f"   logit_shift={ensemble.logit_shift:.4f}, w_score_mean={ensemble.w_score_mean:.4f}")
    print(f"\nNext step: python scripts/calibrate_ensemble.py")
    return ensemble


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-train',   type=int,   default=2100,
                        help='Total adversarial samples (~1/3 each: FGSM, PGD, Square). '
                             'Default 2100 -> 700 per attack.')
    parser.add_argument('--fgsm-eps',  type=float, default=8/255,
                        help='L-inf epsilon. Must match EPS_LINF_STANDARD=8/255.')
    parser.add_argument('--data-root', default='./data')
    parser.add_argument('--output',    default='models/ensemble_scorer.pkl')
    parser.add_argument('--profile',   default='models/reference_profiles.pkl')
    parser.add_argument('--fast',      action='store_true',
                        help='Quick smoke-test: n_train=150 (50 per attack)')
    args = parser.parse_args()

    if args.fast:
        args.n_train = 150
        print("[FAST MODE] n_train=150 (50 FGSM + 50 PGD + 50 Square)")

    train_ensemble_scorer(
        n_train=args.n_train,
        fgsm_eps=args.fgsm_eps,
        data_root=args.data_root,
        output_path=args.output,
        profile_path=args.profile,
    )
