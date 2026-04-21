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
    python scripts/train_ensemble_scorer.py --n-train 3000 --fgsm-oversample 1.5
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


def compute_input_grad_norm(model: torch.nn.Module,
                            x_norm: torch.Tensor,
                            device: str) -> float:
    """
    Compute L2 norm of the gradient of the predicted-class logit w.r.t. the input.

    Single-step attacks (FGSM) align the perturbation with this gradient, so its
    magnitude is a cheap discriminator: FGSM adversarials tend to have a larger
    gradient norm than clean inputs at the same manifold point.

    Uses torch.autograd.grad to avoid accumulating model parameter gradients.
    Requires one extra forward pass per sample (~5ms on GPU).
    """
    x_g = x_norm.detach().clone().to(device).requires_grad_(True)
    with torch.enable_grad():
        logits = model(x_g)
        pred_idx = int(logits.argmax(1).item())
        (grad_x,) = torch.autograd.grad(logits[0, pred_idx], x_g)
    return float(grad_x.norm().item())


def _extract_features(dataset_indices, dataset, model, extractor, profiler,
                      scorer, ref_profiles, device, art_attack=None,
                      label='clean', n_max=None, use_grad_norm=False):
    """
    Extract feature vectors AND base Wasserstein scores.

    Feature dimension:
      37 = 36 persistence + 1 DCT energy (use_grad_norm=False)
      38 = 36 persistence + 1 DCT + 1 grad_norm (use_grad_norm=True)

    Returns:
        features: (N, d) float32 feature matrix
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
                img_for_dct = torch.tensor(x_adv_np[0])
            except Exception:
                img_tensor = _NORMALIZE(img_pixel)
                img_for_dct = img_pixel
        else:
            img_tensor = _NORMALIZE(img_pixel)
            img_for_dct = img_pixel

        x = img_tensor.unsqueeze(0).to(device)
        acts = extractor.extract(x)
        dgms = {}
        for layer in LAYER_NAMES:
            act_np = acts[layer].squeeze(0).cpu().numpy()
            dgms[layer] = profiler.compute_diagram(act_np)

        gn = compute_input_grad_norm(model, x, device) if use_grad_norm else None

        feat = extract_feature_vector(
            dgms, ref_profiles, LAYER_NAMES, list(DIMS),
            image=img_for_dct.numpy() if hasattr(img_for_dct, 'numpy') else img_for_dct,
            grad_norm=gn,
        )
        w_score = scorer.score(dgms)

        features.append(feat)
        w_scores.append(w_score)

    if features:
        return np.stack(features, axis=0), np.array(w_scores, dtype=np.float32)
    n_feat = 38 if use_grad_norm else 37
    return np.zeros((0, n_feat), dtype=np.float32), np.zeros(0, dtype=np.float32)


def train_ensemble_scorer(
    n_train: int = 2100,
    fgsm_eps: float = EPS_LINF_STANDARD,
    fgsm_oversample: float = 1.0,
    use_grad_norm: bool = False,
    data_root: str = './data',
    output_path: str = 'models/ensemble_scorer.pkl',
    calibrator_path: str = 'models/calibrator.pkl',
    profile_path: str = 'models/reference_profiles.pkl',
    device: str = None,
):
    """
    Args:
        fgsm_oversample: Relative weight of FGSM in the adversarial training mix.
            Default 1.0 → equal tri-split (~1/3 each).
            1.5 → FGSM gets 1.5/(1.5+1+1)=37.5% of budget; helps FGSM TPR.
        use_grad_norm: If True, append input-gradient L2 norm as 38th feature.
            Adds ~5ms latency per inference; requires retraining after change.
    """
    if not ART_AVAILABLE:
        print("ERROR: adversarial-robustness-toolbox not installed.")
        print("  pip install adversarial-robustness-toolbox")
        sys.exit(1)

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    total_weight = fgsm_oversample + 1.0 + 1.0
    print(f"Device: {device}")
    print(f"Training ensemble scorer: n_train={n_train}, eps={fgsm_eps:.6f} (= 8/255 = {8/255:.6f})")
    print(f"Adversarial mix: fgsm_oversample={fgsm_oversample} "
          f"(FGSM:{fgsm_oversample/total_weight:.1%} PGD:{1/total_weight:.1%} "
          f"Square:{1/total_weight:.1%})")

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
        base_scorer, ref_profiles, device, art_attack=None, label='clean',
        use_grad_norm=use_grad_norm,
    )

    # Split adversarial budget according to fgsm_oversample ratio.
    # total_weight = fgsm_oversample + 1 + 1; tail remainder goes to Square.
    n_fgsm   = int(n_train * fgsm_oversample / total_weight)
    n_pgd    = int(n_train * 1.0 / total_weight)
    n_square = n_train - n_fgsm - n_pgd

    off = 0
    print(f"\nExtracting {n_fgsm} FGSM (eps={fgsm_eps:.6f}) adversarial features...")
    X_fgsm, _ = _extract_features(
        adv_pool[off: off + n_fgsm], pixel_dataset, model, extractor, profiler,
        base_scorer, ref_profiles, device, art_attack=fgsm_attack, label='FGSM adv',
        use_grad_norm=use_grad_norm,
    )
    off += n_fgsm

    print(f"\nExtracting {n_pgd} PGD (eps={fgsm_eps:.6f}, 20 steps) adversarial features...")
    X_pgd, _ = _extract_features(
        adv_pool[off: off + n_pgd], pixel_dataset, model, extractor, profiler,
        base_scorer, ref_profiles, device, art_attack=pgd_attack, label='PGD adv',
        use_grad_norm=use_grad_norm,
    )
    off += n_pgd

    print(f"\nExtracting {n_square} Square (eps={fgsm_eps:.6f}, 1000 queries) adversarial features...")
    X_square, _ = _extract_features(
        adv_pool[off: off + n_square], pixel_dataset, model, extractor, profiler,
        base_scorer, ref_profiles, device, art_attack=square_attack, label='Square adv',
        use_grad_norm=use_grad_norm,
    )

    X_adv = np.vstack([X_fgsm, X_pgd, X_square])
    # Shuffle so the 80/20 train/val split sees all three attack families --
    # without shuffle, X_adv[:80%] would contain almost no Square samples.
    perm  = np.random.RandomState(42).permutation(len(X_adv))
    X_adv = X_adv[perm]

    print(f"\nFeature matrix shapes: clean={X_clean.shape}, adv={X_adv.shape}")
    print(f"  (FGSM: {X_fgsm.shape[0]}, PGD: {X_pgd.shape[0]}, Square: {X_square.shape[0]})")

    # -- Train ensemble scorer --------------------------------------------------
    n_feat_desc = f"{37 + (1 if use_grad_norm else 0)}-dim"
    print(f"\nFitting logistic regression ensemble component ({n_feat_desc})...")
    ensemble = PersistenceEnsembleScorer(
        base_scorer=base_scorer,
        layer_names=LAYER_NAMES,
        dims=DIMS,
        alpha=0.4,
        use_dct=True,
        use_grad_norm=use_grad_norm,
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
    ensemble.training_eps          = fgsm_eps
    ensemble.training_attacks      = ['FGSM', 'PGD', 'Square']
    ensemble.training_n            = len(X_adv)
    ensemble.fgsm_oversample       = fgsm_oversample
    ensemble.training_fgsm_n       = n_fgsm
    ensemble.training_pgd_n        = n_pgd
    ensemble.training_square_n     = n_square

    # -- Save -------------------------------------------------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ensemble.save(output_path)

    extractor.cleanup()

    print("\n[OK] Ensemble scorer trained and saved.")
    print(f"   Training eps = {fgsm_eps:.6f} (= 8/255, matches evaluation)")
    print(f"   Adversarial mix: FGSM({n_fgsm}) + PGD({n_pgd}) + Square({n_square})")
    print(f"   fgsm_oversample={fgsm_oversample:.2f}, n_features={ensemble.n_features}")
    print(f"   use_grad_norm={use_grad_norm}")
    print(f"   Trained on CIFAR-10 TRAINING set (no test-set leakage)")
    print(f"   logit_shift={ensemble.logit_shift:.4f}, w_score_mean={ensemble.w_score_mean:.4f}")
    print(f"\nNext step: python scripts/calibrate_ensemble.py")
    return ensemble


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-train',   type=int,   default=3000,
                        help='Total adversarial samples. Default 3000.')
    parser.add_argument('--fgsm-eps',  type=float, default=8/255,
                        help='L-inf epsilon. Must match EPS_LINF_STANDARD=8/255.')
    parser.add_argument('--fgsm-oversample', type=float, default=1.5,
                        help='Relative FGSM weight in the training mix. '
                             '1.5 = FGSM gets 37.5%% (recommended for FGSM TPR gap).')
    parser.add_argument('--use-grad-norm', action='store_true',
                        help='Append input-gradient L2 norm as 38th feature. '
                             'Improves FGSM discrimination; adds ~5ms latency.')
    parser.add_argument('--data-root', default='./data')
    parser.add_argument('--output',    default='models/ensemble_scorer.pkl')
    parser.add_argument('--profile',   default='models/reference_profiles.pkl')
    parser.add_argument('--fast',      action='store_true',
                        help='Quick smoke-test: n_train=150')
    args = parser.parse_args()

    if args.fast:
        args.n_train = 150
        print("[FAST MODE] n_train=150")

    train_ensemble_scorer(
        n_train=args.n_train,
        fgsm_eps=args.fgsm_eps,
        fgsm_oversample=args.fgsm_oversample,
        use_grad_norm=args.use_grad_norm,
        data_root=args.data_root,
        output_path=args.output,
        profile_path=args.profile,
    )
