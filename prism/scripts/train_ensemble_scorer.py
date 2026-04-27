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

2. Adversarial training mix -- default tri-split (~34% FGSM / 33% PGD / 33% Square),
   optionally extended with CW-L2 and/or AutoAttack-APGD-CE:
   - FGSM          (L-inf, eps=8/255, single-step)        -- gradient-based, one step
   - PGD           (L-inf, eps=8/255, 20 steps)           -- gradient-based, iterative
   - Square        (L-inf, eps=8/255, 1000 queries)       -- GRADIENT-FREE black-box
   - CW-L2         (L2, confidence=0, --include-cw)       -- optimization-based, L2 norm
   - AutoAttack    (L-inf APGD-CE, --include-autoattack)  -- strongest gradient attack
   All at the SAME epsilon as the evaluation (eps=8/255 for L-inf; CW optimises L2
   separately). Use --include-cw to fix the CW TPR miss (was 3.3% without training).

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

# Route --config CLI flag to PRISM_CONFIG env var BEFORE importing src.config.
from src import bootstrap  # noqa: F401

try:
    from art.attacks.evasion import (
        FastGradientMethod, ProjectedGradientDescent, SquareAttack,
        CarliniL2Method,
    )
    from art.estimators.classification import PyTorchClassifier
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False
    print("WARNING: ART not available -- cannot generate adversarial features.")

# AutoAttack APGD-CE slice (only needed if --include-autoattack).
try:
    from autoattack.autopgd_base import APGDAttack
    AUTOATTACK_AVAILABLE = True
except ImportError:
    AUTOATTACK_AVAILABLE = False

from src.tamm.extractor import ActivationExtractor
from src.tamm.tda import TopologicalProfiler
from src.tamm.scorer import TopologicalScorer
from src.tamm.persistence_stats import extract_feature_vector
from src.cadg.ensemble_scorer import PersistenceEnsembleScorer
from src.config import (
    LAYER_NAMES, LAYER_WEIGHTS, DIM_WEIGHTS, IMAGENET_MEAN, IMAGENET_STD,
    EPS_LINF_STANDARD, N_SUBSAMPLE, MAX_DIM,
    DATASET, PATHS,
)
from src.data_loader import load_test_dataset

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


class _APGDGenerator:
    """Wrap AutoAttack's APGDAttack so it exposes ART's .generate(x_np) API.

    The training extraction loop calls ``art_attack.generate(x_np)``; APGD's
    native interface is ``.perturb(x, y)``. Labels come from the backbone's own
    clean predictions (ImageNet-1000 space), not CIFAR-10, for the same reason
    we fixed the _run_autoattack label bug in the evaluation script.
    """
    def __init__(self, norm_model, eps, device, n_iter=40):
        self._attack = APGDAttack(
            norm_model, norm='Linf', eps=eps,
            n_iter=n_iter, n_restarts=1, loss='ce', rho=0.75, verbose=False,
        )
        self._norm_model = norm_model
        self._device = device

    def generate(self, x_np):
        x = torch.tensor(x_np, dtype=torch.float32).to(self._device)
        with torch.no_grad():
            y = self._norm_model(x).argmax(dim=1)
        x_adv = self._attack.perturb(x, y)
        return x_adv.detach().cpu().numpy()


def _extract_features(dataset_indices, dataset, model, extractor, profiler,
                      scorer, ref_profiles, device, art_attack=None,
                      label='clean', n_max=None, use_grad_norm=False,
                      use_softmax_entropy=False):
    """
    Extract feature vectors AND base Wasserstein scores.

    Feature dimension:
      37 = 36 persistence + 1 DCT energy (use_grad_norm=False, use_softmax_entropy=False)
      38 = 36 persistence + 1 DCT + 1 softmax_entropy (use_softmax_entropy=True)
      39 = 36 persistence + 1 DCT + 1 softmax_entropy + 1 grad_norm (both True)

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

        # Compute model logits for softmax-entropy feature (CW-L2 detection).
        logits_np = None
        if use_softmax_entropy:
            with torch.no_grad():
                logits_out = model(x)
            logits_np = logits_out.squeeze(0).cpu().numpy()

        feat = extract_feature_vector(
            dgms, ref_profiles, LAYER_NAMES, list(DIMS),
            image=img_for_dct.numpy() if hasattr(img_for_dct, 'numpy') else img_for_dct,
            grad_norm=gn,
            logits=logits_np,
        )
        w_score = scorer.score(dgms)

        features.append(feat)
        w_scores.append(w_score)

    if features:
        return np.stack(features, axis=0), np.array(w_scores, dtype=np.float32)
    # Dynamic feature count based on active flags
    n_feat = 36 + 1  # base TDA + DCT (always on)
    if use_softmax_entropy:
        n_feat += 1
    if use_grad_norm:
        n_feat += 1
    return np.zeros((0, n_feat), dtype=np.float32), np.zeros(0, dtype=np.float32)


def train_ensemble_scorer(
    n_train: int = 2100,
    fgsm_eps: float = EPS_LINF_STANDARD,
    fgsm_oversample: float = 1.0,
    use_grad_norm: bool = False,
    use_softmax_entropy: bool = True,
    data_root: str = './data',
    output_path: str = 'models/ensemble_scorer.pkl',
    calibrator_path: str = 'models/calibrator.pkl',
    profile_path: str = 'models/reference_profiles.pkl',
    device: str = None,
    include_cw: bool = False,
    include_autoattack: bool = False,
    cw_max_iter: int = 30,
    cw_bss: int = 3,
    no_tda_features: bool = False,
):
    """
    Args:
        fgsm_oversample: Relative weight of FGSM in the adversarial training mix.
            Default 1.0 → equal tri-split (~1/3 each).
            1.5 → FGSM gets 1.5/(1.5+1+1)=37.5% of budget; helps FGSM TPR.
        use_grad_norm: If True, append input-gradient L2 norm as a feature.
            Adds ~5ms latency per inference; requires retraining after change.
        use_softmax_entropy: If True (default), append softmax entropy of model
            logits as a feature. Captures CW-L2 decision-boundary proximity.
            Adds <1ms latency; enabled by default to fix CW detection gap.
    """
    if not ART_AVAILABLE:
        print("ERROR: adversarial-robustness-toolbox not installed.")
        print("  pip install adversarial-robustness-toolbox")
        sys.exit(1)
    if include_autoattack and not AUTOATTACK_AVAILABLE:
        print("ERROR: --include-autoattack given but autoattack not installed.")
        print("  pip install autoattack")
        sys.exit(1)

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # Dynamic attack mix. FGSM uses fgsm_oversample weight; others weight=1.
    weights = {'FGSM': fgsm_oversample, 'PGD': 1.0, 'Square': 1.0}
    if include_cw:          weights['CW']         = 1.0
    if include_autoattack:  weights['AutoAttack'] = 1.0
    total_weight = sum(weights.values())

    print(f"Device: {device}")
    print(f"Training ensemble scorer: n_train={n_train}, eps={fgsm_eps:.6f} (= 8/255 = {8/255:.6f})")
    share = {a: w / total_weight for a, w in weights.items()}
    print("Adversarial mix: " + "  ".join(
        f"{a}:{share[a]:.1%}" for a in weights))

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
    # Training data uses the active dataset's TRAINING set (cifar10 / cifar100).
    # Using train split avoids any test-set leakage for the eval phase.
    from src.data_loader import load_train_dataset
    pixel_dataset = load_train_dataset(
        root=data_root, download=True, transform=_PIXEL_TRANSFORM
    )
    rng = np.random.RandomState(42)
    all_idx = rng.permutation(len(pixel_dataset))
    clean_idx = all_idx[:n_train]
    # Pre-allocate a 3x budget pool; actual slices are taken per attack below
    adv_pool  = all_idx[n_train: n_train + 3 * n_train]

    # ART classifier for attack generation (normalisation wrapped inside model)
    norm_model = _NormalizedResNet(
        torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
    ).to(device)
    art_clf = PyTorchClassifier(
        model=norm_model,
        loss=torch.nn.CrossEntropyLoss(),
        input_shape=(3, 224, 224),
        nb_classes=1000,
        clip_values=(0.0, 1.0),
        device_type='gpu' if device == 'cuda' else 'cpu',
    )

    # Attacks at the SAME epsilon as evaluation
    attacks = {
        'FGSM': FastGradientMethod(art_clf, eps=fgsm_eps),
        'PGD': ProjectedGradientDescent(
            art_clf, eps=fgsm_eps, eps_step=fgsm_eps / 4, max_iter=20),
        # Square: gradient-free black-box; 1000 queries is enough for training
        # signal (eval uses 5000). Lower budget keeps training time reasonable.
        'Square': SquareAttack(
            art_clf, eps=fgsm_eps, max_iter=1000, batch_size=1, verbose=False),
    }
    if include_cw:
        # CW max_iter / binary_search_steps reduced vs eval for training-time
        # budget; still produces in-distribution L2 adversarials for logistic fit.
        attacks['CW'] = CarliniL2Method(
            art_clf, max_iter=cw_max_iter, confidence=0.0, learning_rate=0.01,
            binary_search_steps=cw_bss, batch_size=1,
        )
    if include_autoattack:
        # APGD-CE slice only (fastest component); full AutoAttack ensemble is
        # 4x slower and APGD alone provides the gradient-L∞ signature we want.
        attacks['AutoAttack'] = _APGDGenerator(
            norm_model, eps=fgsm_eps, device=device, n_iter=40,
        )

    # -- Extract features -------------------------------------------------------
    print(f"\nExtracting {n_train} clean feature vectors...")
    X_clean, W_clean = _extract_features(
        clean_idx, pixel_dataset, model, extractor, profiler,
        base_scorer, ref_profiles, device, art_attack=None, label='clean',
        use_grad_norm=use_grad_norm,
        use_softmax_entropy=use_softmax_entropy,
    )

    # Split adversarial budget across all active attacks by their weights.
    # Rounding remainder goes to the last attack in insertion order.
    attack_names = list(weights.keys())
    counts = {}
    running = 0
    for i, a in enumerate(attack_names):
        if i == len(attack_names) - 1:
            counts[a] = n_train - running
        else:
            counts[a] = int(n_train * weights[a] / total_weight)
            running += counts[a]

    # Widen adv_pool if the default 3x header is smaller than the total budget
    # (happens when include_cw/include_autoattack push the mix above 3 attacks).
    adv_pool_needed = sum(counts.values())
    if adv_pool_needed > len(adv_pool):
        adv_pool = all_idx[n_train: n_train + adv_pool_needed + 100]

    off = 0
    X_adv_parts = []
    W_adv_parts = []
    per_attack_shapes = {}
    for atk_name in attack_names:
        n_atk = counts[atk_name]
        print(f"\nExtracting {n_atk} {atk_name} adversarial features "
              f"(eps={fgsm_eps:.6f})...", flush=True)
        X_atk, W_atk = _extract_features(
            adv_pool[off: off + n_atk], pixel_dataset, model, extractor, profiler,
            base_scorer, ref_profiles, device,
            art_attack=attacks[atk_name], label=f'{atk_name} adv',
            use_grad_norm=use_grad_norm,
            use_softmax_entropy=use_softmax_entropy,
        )
        off += n_atk
        X_adv_parts.append(X_atk)
        W_adv_parts.append(W_atk)
        per_attack_shapes[atk_name] = X_atk.shape[0]

    X_adv = np.vstack(X_adv_parts)
    W_adv = np.concatenate(W_adv_parts)
    # Shuffle so the 80/20 train/val split sees all attack families -- without
    # shuffle, X_adv[:80%] would contain almost no samples from the last attack.
    perm  = np.random.RandomState(42).permutation(len(X_adv))
    X_adv = X_adv[perm]
    W_adv = W_adv[perm]

    print(f"\nFeature matrix shapes: clean={X_clean.shape}, adv={X_adv.shape}")
    print("  (" + ", ".join(f"{a}: {n}" for a, n in per_attack_shapes.items()) + ")")

    # P0.6 ablation: drop the 36-dim persistence-statistics block. The 36 TDA
    # columns sit at indices 0..35; DCT (optional) and grad-norm (optional)
    # follow. We keep only the trailing non-TDA columns and set alpha=0 on the
    # scorer so Wasserstein contributes nothing at inference. The ensemble
    # scorer records use_tda=False in its provenance pkl for downstream
    # identification during evaluation.
    ensemble_alpha = 0.4
    if no_tda_features:
        if X_clean.shape[1] <= 36:
            print("ERROR: --no-tda-features requires at least one non-TDA feature "
                  "(enable --use-grad-norm and/or keep use_dct=True).")
            sys.exit(1)
        print(f"[P0.6] Stripping 36 TDA columns from features. "
              f"Before: {X_clean.shape[1]}-dim; after: {X_clean.shape[1] - 36}-dim.")
        X_clean = X_clean[:, 36:]
        X_adv   = X_adv[:, 36:]
        ensemble_alpha = 0.0  # pure logistic; Wasserstein disabled

    # -- Train ensemble scorer --------------------------------------------------
    n_feat_desc = f"{X_clean.shape[1]}-dim (no_tda={no_tda_features})"
    print(f"\nFitting logistic regression ensemble component ({n_feat_desc})...")
    ensemble = PersistenceEnsembleScorer(
        base_scorer=base_scorer,
        layer_names=LAYER_NAMES,
        dims=DIMS,
        alpha=ensemble_alpha,
        use_dct=True,
        use_softmax_entropy=use_softmax_entropy,
        use_grad_norm=use_grad_norm,
        use_tda=not no_tda_features,
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

    # -- Tune α on the held-out 20% (P0.6 lever) -------------------------------
    # The α=0.4/0.5 default is a guess; grid-search on the same held-out slice
    # used above. For the P0.6 ablation (use_tda=False) the call short-circuits
    # and keeps alpha=0 since the Wasserstein head is disabled.
    if len(X_clean) > n_clean_train and len(X_adv) > n_adv_train:
        tune_summary = ensemble.tune_alpha(
            clean_features=X_clean[n_clean_train:],
            adv_features=X_adv[n_adv_train:],
            clean_w_scores=W_clean[n_clean_train:],
            adv_w_scores=W_adv[n_adv_train:],
        )
        ensemble.alpha_tune_summary = tune_summary
        print(f"Selected α = {ensemble.alpha:.3f} (held-out AUC grid search)")
    else:
        print("Skipping α tuning — validation slice is empty.")

    # -- Set provenance metadata ------------------------------------------------
    ensemble.training_eps     = fgsm_eps
    ensemble.training_attacks = attack_names          # dynamic list
    ensemble.training_n       = len(X_adv)
    ensemble.fgsm_oversample  = fgsm_oversample
    ensemble.no_tda_features  = no_tda_features
    for atk_name, cnt in per_attack_shapes.items():
        setattr(ensemble, f'training_{atk_name.lower()}_n', cnt)

    # -- Save -------------------------------------------------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ensemble.save(output_path)

    extractor.cleanup()

    mix_str = " + ".join(f"{a}({per_attack_shapes[a]})" for a in attack_names)
    print("\n[OK] Ensemble scorer trained and saved.")
    print(f"   Training eps = {fgsm_eps:.6f} (= 8/255, matches evaluation)")
    print(f"   Adversarial mix: {mix_str}")
    print(f"   training_attacks={attack_names}")
    print(f"   fgsm_oversample={fgsm_oversample:.2f}, n_features={ensemble.n_features}")
    print(f"   use_grad_norm={use_grad_norm}")
    print(f"   Trained on CIFAR-10 TRAINING set (no test-set leakage)")
    print(f"   logit_shift={ensemble.logit_shift:.4f}, w_score_mean={ensemble.w_score_mean:.4f}")
    print(f"\nNext step: python scripts/calibrate_ensemble.py")
    return ensemble


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None,
                        help='YAML config path (routes via PRISM_CONFIG env var). '
                             'Default: configs/default.yaml (CIFAR-10).')
    parser.add_argument('--n-train',   type=int,   default=3000,
                        help='Total adversarial samples. Default 3000.')
    parser.add_argument('--fgsm-eps',  type=float, default=8/255,
                        help='L-inf epsilon. Must match EPS_LINF_STANDARD=8/255.')
    parser.add_argument('--fgsm-oversample', type=float, default=2.5,
                        help='Relative FGSM weight in the training mix. '
                             'Default 2.5 per P0.3 of the publishability plan: '
                             'closes the 5-seed pooled FGSM TPR gap '
                             '(0.806 -> target >=0.85). Lower values (1.5-2.0) '
                             'regressed FGSM below gate in commits cf854f0/eabcba8.')
    parser.add_argument('--no-tda-features', action='store_true',
                        help='Ablation arm (P0.6): strip the 36-dim persistence-'
                             'statistics feature block and set alpha=0 so the '
                             'Wasserstein component is not used. Only DCT + '
                             'grad-norm features remain. Used to quantify the '
                             'marginal contribution of TAMM (C1). Saves to a '
                             'separate output pkl so the main ensemble is unaffected.')
    parser.add_argument('--use-grad-norm', action='store_true',
                        help='Append input-gradient L2 norm as a feature. '
                             'Improves FGSM discrimination; adds ~5ms latency.')
    parser.add_argument('--no-softmax-entropy', action='store_true',
                        help='Disable softmax-entropy feature (default: enabled). '
                             'Softmax entropy captures CW-L2 decision-boundary '
                             'proximity; disabling it regresses CW detection.')
    parser.add_argument('--data-root', default='./data')
    parser.add_argument('--output',    default=PATHS['ensemble_scorer'])
    parser.add_argument('--profile',   default=PATHS['reference_profiles'])
    parser.add_argument('--include-cw', action='store_true',
                        help='Add CW-L2 to training attack mix. Improves CW TPR at eval.')
    parser.add_argument('--include-autoattack', action='store_true',
                        help='Add AutoAttack-APGD-CE slice to training mix.')
    parser.add_argument('--cw-max-iter', type=int, default=40,
                        help='CW max_iter for training (default 40, matches eval).')
    parser.add_argument('--cw-bss', type=int, default=5,
                        help='CW binary_search_steps for training (default 5, matches eval).')
    parser.add_argument('--fast',      action='store_true',
                        help='Quick smoke-test: n_train=150')
    args = parser.parse_args()

    if args.fast:
        args.n_train = 150
        print("[FAST MODE] n_train=150")

    # P0.6: route no-tda-features to a separate default output file so the
    # main ensemble pkl is never accidentally overwritten by an ablation run.
    if args.no_tda_features and args.output == PATHS['ensemble_scorer']:
        _out_dir = os.path.dirname(PATHS['ensemble_scorer']) or 'models'
        args.output = os.path.join(_out_dir, 'ensemble_scorer_no_tda.pkl')
        print(f"[P0.6 ablation] --no-tda-features active: output -> {args.output}")

    train_ensemble_scorer(
        n_train=args.n_train,
        fgsm_eps=args.fgsm_eps,
        fgsm_oversample=args.fgsm_oversample,
        use_grad_norm=args.use_grad_norm,
        use_softmax_entropy=not args.no_softmax_entropy,
        data_root=args.data_root,
        output_path=args.output,
        profile_path=args.profile,
        include_cw=args.include_cw,
        include_autoattack=args.include_autoattack,
        cw_max_iter=args.cw_max_iter,
        cw_bss=args.cw_bss,
        no_tda_features=args.no_tda_features,
    )
