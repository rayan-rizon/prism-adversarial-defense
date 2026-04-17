"""
Train Persistence Ensemble Scorer (Phase 2A from Improvement Plan)

Trains the logistic regression component of PersistenceEnsembleScorer on
persistence statistics features extracted from clean and adversarial images.

DESIGN FOR PUBLISHABILITY
--------------------------
1. Uses a DEDICATED training split, fully separate from:
   - The conformal calibration/validation split (test idx 5000-7999)
   - The final evaluation set (test idx 8000-9999)
   Training data: CIFAR-10 TRAINING set (50,000 images); we use 2000 clean
   + 2000 FGSM adversarials from the training set for logistic regression.

2. The logistic regression is a linear classifier in feature space — it has
   no hidden layers and cannot overfit in the way a deep model would.

3. The final anomaly score remains a fixed-coefficient linear combination of
   Wasserstein distance and the logistic score — this is fully interpretable.

4. Validation AUC and per-class accuracy are reported for transparency.

AFTER THIS SCRIPT:
  python scripts/calibrate_ensemble.py  — re-calibrate thresholds for ensemble score

USAGE
-----
  cd prism/
  python scripts/train_ensemble_scorer.py [--n-train 2000] [--fgsm-eps 0.03]
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
    from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
    from art.estimators.classification import PyTorchClassifier
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False
    print("WARNING: ART not available — cannot generate adversarial features.")

from src.tamm.extractor import ActivationExtractor
from src.tamm.tda import TopologicalProfiler
from src.tamm.scorer import TopologicalScorer
from src.tamm.persistence_stats import extract_feature_vector
from src.cadg.ensemble_scorer import PersistenceEnsembleScorer

# ── Must match build_profile_testset.py exactly ───────────────────────────────
LAYER_NAMES   = ['layer2', 'layer3', 'layer4']
LAYER_WEIGHTS = {'layer2': 0.15, 'layer3': 0.30, 'layer4': 0.55}
DIM_WEIGHTS   = [0.5, 0.5]
DIMS          = (0, 1)

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]
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
    Extract 42-dim persistence feature vectors for a set of images.

    Args:
        dataset_indices: Indices into `dataset` to use.
        art_attack: If provided, generate adversarial images first.
    Returns:
        (N, 42) float32 feature matrix.
    """
    features = []
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
            except Exception:
                img_tensor = _NORMALIZE(img_pixel)
        else:
            img_tensor = _NORMALIZE(img_pixel)

        x = img_tensor.unsqueeze(0).to(device)
        acts = extractor.extract(x)
        dgms = {}
        for layer in LAYER_NAMES:
            act_np = acts[layer].squeeze(0).cpu().numpy()
            dgms[layer] = profiler.compute_diagram(act_np)

        feat = extract_feature_vector(dgms, ref_profiles, LAYER_NAMES, list(DIMS))
        features.append(feat)

    return np.stack(features, axis=0) if features else np.zeros((0, 42))


def train_ensemble_scorer(
    n_train: int = 2000,
    fgsm_eps: float = 0.03,
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
    print(f"Training ensemble scorer: n_train={n_train}, fgsm_eps={fgsm_eps}")

    # ── Load pre-built components ──────────────────────────────────────────────
    if not os.path.exists(profile_path):
        print(f"ERROR: {profile_path} not found. Run build_profile_testset.py first.")
        sys.exit(1)

    with open(profile_path, 'rb') as f:
        ref_profiles = pickle.load(f)

    # ── Setup model ────────────────────────────────────────────────────────────
    model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model = model.to(device).eval()

    extractor = ActivationExtractor(model, LAYER_NAMES)
    profiler  = TopologicalProfiler(n_subsample=200, max_dim=1)
    base_scorer = TopologicalScorer(
        ref_profiles=ref_profiles,
        layer_names=LAYER_NAMES,
        layer_weights=LAYER_WEIGHTS,
        dim_weights=DIM_WEIGHTS,
    )

    # ── ART adversarial generation ─────────────────────────────────────────────
    # Training data uses CIFAR-10 TRAINING set to avoid any test-set leakage
    pixel_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=_PIXEL_TRANSFORM
    )
    rng = np.random.RandomState(42)
    all_idx = rng.permutation(len(pixel_dataset))
    clean_idx = all_idx[:n_train]
    adv_idx   = all_idx[n_train:2 * n_train]

    # ART classifier for attack generation
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
    fgsm_attack = FastGradientMethod(art_clf, eps=fgsm_eps)
    pgd_attack  = ProjectedGradientDescent(art_clf, eps=fgsm_eps, eps_step=fgsm_eps/4, max_iter=20)

    # ── Extract features ───────────────────────────────────────────────────────
    print(f"\nExtracting {n_train} clean feature vectors...")
    X_clean = _extract_features(
        clean_idx, pixel_dataset, model, extractor, profiler,
        base_scorer, ref_profiles, device, art_attack=None, label='clean'
    )

    n_fgsm = n_train // 2
    n_pgd  = n_train - n_fgsm

    print(f"\nExtracting {n_fgsm} FGSM (ε={fgsm_eps}) adversarial feature vectors...")
    X_fgsm = _extract_features(
        adv_idx[:n_fgsm], pixel_dataset, model, extractor, profiler,
        base_scorer, ref_profiles, device, art_attack=fgsm_attack, label='FGSM adv'
    )

    print(f"\nExtracting {n_pgd} PGD (ε={fgsm_eps}) adversarial feature vectors...")
    X_pgd = _extract_features(
        adv_idx[n_fgsm:], pixel_dataset, model, extractor, profiler,
        base_scorer, ref_profiles, device, art_attack=pgd_attack, label='PGD adv'
    )

    X_adv = np.vstack([X_fgsm, X_pgd])

    print(f"\nFeature matrix shapes: clean={X_clean.shape}, adv={X_adv.shape}")

    # ── Train ensemble scorer ──────────────────────────────────────────────────
    print("\nFitting logistic regression ensemble component...")
    ensemble = PersistenceEnsembleScorer(
        base_scorer=base_scorer,
        layer_names=LAYER_NAMES,
        dims=DIMS,
        alpha=0.5,   # equal blend; will be tuned after conformal recalibration
    )

    # Use 80% for training, 20% for internal validation
    n_clean_train = int(0.8 * len(X_clean))
    n_adv_train   = int(0.8 * len(X_adv))
    ensemble.fit_logistic(
        clean_features=X_clean[:n_clean_train],
        adv_features=X_adv[:n_adv_train],
        C=1.0,
    )

    # ── Validation AUC on held-out 20% ────────────────────────────────────────
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

    # ── Save ───────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ensemble.save(output_path)

    extractor.cleanup()

    print("\n✅ Ensemble scorer trained and saved.")
    print(f"   ART FGSM ε={fgsm_eps}, n_train={n_train} per class")
    print(f"   Trained on CIFAR-10 TRAINING set (no test-set leakage)")
    print(f"\nNext step: python scripts/calibrate_ensemble.py")
    return ensemble


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-train',    type=int,   default=2000)
    parser.add_argument('--fgsm-eps',   type=float, default=0.03)
    parser.add_argument('--data-root',  default='./data')
    parser.add_argument('--output',     default='models/ensemble_scorer.pkl')
    parser.add_argument('--profile',    default='models/reference_profiles.pkl')
    args = parser.parse_args()
    train_ensemble_scorer(
        n_train=args.n_train,
        fgsm_eps=args.fgsm_eps,
        data_root=args.data_root,
        output_path=args.output,
        profile_path=args.profile,
    )
