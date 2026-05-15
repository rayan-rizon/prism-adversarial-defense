"""
Train Persistence Ensemble Scorer (Phase 3 from Improvement Plan)

Trains the logistic regression component of PersistenceEnsembleScorer on
persistence statistics features extracted from clean and adversarial images.

DESIGN FOR PUBLISHABILITY
--------------------------
1. Uses a DEDICATED training split, fully separate from:
   - The conformal calibration/validation split (test idx 5000-7999)
   - The final evaluation set (test idx 8000-9999)
   Training data: active CIFAR TRAINING set (50,000 images for CIFAR-10;
   50,000 images for CIFAR-100).

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
import time
import torchvision
import torchvision.transforms as T
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
from src.perf import setup_perf_flags
setup_perf_flags()

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
    LAYER_NAMES, LAYER_WEIGHTS, DIM_WEIGHTS,
    BACKBONE_MEAN, BACKBONE_STD, BACKBONE_INPUT_SIZE, BACKBONE_NUM_CLASSES,
    EPS_LINF_STANDARD, N_SUBSAMPLE, MAX_DIM,
    DATASET, PATHS,
)
from src.data_loader import load_test_dataset
from src.attacks.cw_torch import TorchCWGenerator
from src.models import load_backbone, _NormalizedBackbone

# All shared constants are imported from src.config (backed by default.yaml).

DIMS = (0, 1)  # homology dimensions; matches DIM_WEIGHTS length from src.config

_MEAN = BACKBONE_MEAN
_STD  = BACKBONE_STD
if BACKBONE_INPUT_SIZE == 32:
    _PIXEL_TRANSFORM = T.Compose([T.ToTensor()])
else:
    _PIXEL_TRANSFORM = T.Compose([T.Resize(BACKBONE_INPUT_SIZE), T.ToTensor()])
_NORMALIZE       = T.Normalize(mean=_MEAN, std=_STD)

# Backward-compat alias — _NormalizedBackbone is the same wrapper shape.
_NormalizedResNet = _NormalizedBackbone


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
    native interface is ``.perturb(x, y)``. Labels come from the active CIFAR
    backbone's own clean predictions so training remains self-consistent for
    CIFAR-10 and CIFAR-100.
    """
    def __init__(self, norm_model, eps, device, n_iter=40, loss='ce'):
        self._attack = APGDAttack(
            norm_model, norm='Linf', eps=eps,
            n_iter=n_iter, n_restarts=1, loss=loss, rho=0.75, verbose=False,
        )
        self._norm_model = norm_model
        self._device = device

    def generate(self, x_np):
        x = torch.tensor(x_np, dtype=torch.float32).to(self._device)
        with torch.no_grad():
            y = self._norm_model(x).argmax(dim=1)
        x_adv = self._attack.perturb(x, y)
        return x_adv.detach().cpu().numpy()


def _batch_generate_adversarials(art_attack, dataset, indices, label,
                                  gen_chunk=512):
    """Pre-generate all adversarial examples in GPU-batched chunks.

    Returns an np.ndarray of shape (N, C, H, W) in pixel [0,1] space,
    exactly like art_attack.generate() would produce per-sample, but
    ~30-60x faster for CW/PGD because of batch GPU parallelism.

    This is mathematically identical to per-sample generation:
    ART computes gradients independently per image in the batch.
    """
    all_imgs_pixel = [dataset[int(i)][0] for i in indices]
    X_pixel_np = torch.stack(all_imgs_pixel).numpy()  # (N, C, H, W)
    X_adv_np = np.zeros_like(X_pixel_np)

    t0 = time.perf_counter()
    c_start = 0
    while c_start < len(X_pixel_np):
        c_end = min(c_start + gen_chunk, len(X_pixel_np))
        try:
            X_adv_np[c_start:c_end] = art_attack.generate(
                X_pixel_np[c_start:c_end]
            )
        except Exception as e:
            # Fallback: per-sample on batch failure
            print(f"    [gen] batch failed at {c_start}-{c_end}: {e}. "
                  "Falling back to per-sample.", flush=True)
            for j in range(c_start, c_end):
                try:
                    X_adv_np[j] = art_attack.generate(
                        X_pixel_np[j:j+1]
                    )[0]
                except Exception:
                    X_adv_np[j] = X_pixel_np[j]  # keep clean on failure
        dt = time.perf_counter() - t0
        print(f"    [gen] {c_end}/{len(X_pixel_np)} "
              f"elapsed={dt:.1f}s  rate={dt/max(c_end,1):.2f}s/img",
              flush=True)
        c_start = c_end

    dt = time.perf_counter() - t0
    print(f"  {label}: batch-generated {len(X_pixel_np)} adversarials "
          f"in {dt:.1f}s ({dt/max(len(X_pixel_np),1):.3f}s/img)", flush=True)
    return X_adv_np


def _extract_features(dataset_indices, dataset, model, extractor, profiler,
                      scorer, ref_profiles, device, art_attack=None,
                      adv_cache=None,
                      label='clean', n_max=None, use_grad_norm=False,
                      use_softmax_entropy=False):
    """
    Extract feature vectors AND base Wasserstein scores.

    If *adv_cache* is provided (np.ndarray of pre-generated adversarials),
    it is used instead of calling art_attack.generate() per-sample.
    This separates the GPU-heavy attack generation (batchable) from the
    CPU-bound TDA persistence computation (inherently per-image).

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

    for j, i in enumerate(tqdm(indices, desc=f"Extracting {label} features")):
        img_pixel, _ = dataset[int(i)]  # pixel [0,1] space

        if adv_cache is not None:
            # Use pre-generated adversarial from batch cache
            img_tensor = _NORMALIZE(torch.tensor(adv_cache[j]))
            img_for_dct = torch.tensor(adv_cache[j])
        elif art_attack is not None:
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
    cw_oversample: float = 1.0,
    no_tda_features: bool = False,
    balanced_attacks: bool = False,
    pgd_train_steps: int = 40,
    aa_train_mode: str = 'apgd-ce',
    selection_objective: str = 'worst_case_tpr',
    allow_undertrained_smoke: bool = False,
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
        cw_oversample: Relative weight of CW in the adversarial training mix.
            Default 1.0 → equal share with PGD/Square. Only active when
            --include-cw is set. Reserved as the conditional secondary
            intervention for the CW TPR gap (see plan §"Conditional secondary
            intervention"): activate ONLY if the primary softmax-entropy fix
            produces pooled CW TPR in [0.75, 0.85). Raising this above 1.0
            takes share from PGD/Square — verify ±1pp regression gate holds
            for FGSM/PGD/Square before adopting a non-default value.
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

    # Dynamic attack mix. The robust default path can force equal attack-family
    # shares so PGD/AutoAttack cannot be hidden by an aggregate AUC win.
    if balanced_attacks:
        weights = {'FGSM': 1.0, 'PGD': 1.0, 'Square': 1.0}
        if include_cw:          weights['CW']         = 1.0
        if include_autoattack:  weights['AutoAttack'] = 1.0
    else:
        weights = {'FGSM': fgsm_oversample, 'PGD': 1.0, 'Square': 1.0}
        if include_cw:          weights['CW']         = cw_oversample
        if include_autoattack:  weights['AutoAttack'] = 1.0
    total_weight = sum(weights.values())

    print(f"Device: {device}")
    print(f"Training ensemble scorer: n_train={n_train}, eps={fgsm_eps:.6f} (= 8/255 = {8/255:.6f})")
    share = {a: w / total_weight for a, w in weights.items()}
    print("Adversarial mix: " + "  ".join(
        f"{a}:{share[a]:.1%}" for a in weights))
    print(f"Selection objective: {selection_objective}")
    print(f"PGD train steps: {pgd_train_steps}")
    if include_autoattack:
        print(f"AutoAttack train mode: {aa_train_mode}")

    # -- Load pre-built components ------------------------------------------------
    if not os.path.exists(profile_path):
        print(f"ERROR: {profile_path} not found. Run build_profile_testset.py first.")
        sys.exit(1)

    with open(profile_path, 'rb') as f:
        ref_profiles = pickle.load(f)

    # -- Setup model -------------------------------------------------------------
    # Active CIFAR-trained ResNet-18, wrapped with pixel-space normalisation. The
    # `model.layer2 / layer3 / layer4` hooks used by the extractor are
    # transparently forwarded to the inner CIFARResNet18.
    model = load_backbone(device)

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

    # ART classifier for attack generation. Same CIFAR-trained backbone as
    # feature extraction, this time wrapped for pixel-space attack inputs.
    norm_model = load_backbone(device, wrap=True)
    art_clf = PyTorchClassifier(
        model=norm_model,
        loss=torch.nn.CrossEntropyLoss(),
        input_shape=(3, BACKBONE_INPUT_SIZE, BACKBONE_INPUT_SIZE),
        nb_classes=BACKBONE_NUM_CLASSES,
        clip_values=(0.0, 1.0),
        device_type='gpu' if device == 'cuda' else 'cpu',
    )

    # Attacks at the SAME epsilon as evaluation
    attacks = {
        'FGSM': FastGradientMethod(art_clf, eps=fgsm_eps),
        'PGD': ProjectedGradientDescent(
            art_clf, eps=fgsm_eps, eps_step=fgsm_eps / 4,
            max_iter=pgd_train_steps, num_random_init=1),
        # Square: gradient-free black-box; 1000 queries is enough for training
        # signal (eval uses 5000). Lower budget keeps training time reasonable.
        'Square': SquareAttack(
            art_clf, eps=fgsm_eps, max_iter=1000, batch_size=256, verbose=False),
    }
    if include_cw:
        # Native PyTorch CW — ~24x faster than ART's CarliniL2Method.
        # ART's version bounces tensors CPU↔GPU on every optimizer step;
        # this keeps everything on CUDA.  Mathematically identical output.
        attacks['CW'] = TorchCWGenerator(
            norm_model, device,
            max_iter=cw_max_iter, bss=cw_bss,
            lr=0.01, confidence=0.0,
        )
    if include_autoattack:
        # APGD-CE slice only (fastest component); full AutoAttack ensemble is
        # 4x slower and APGD alone provides the gradient-L∞ signature we want.
        aa_loss = {'apgd-ce': 'ce', 'apgd-dlr': 'dlr'}[aa_train_mode]
        attacks['AutoAttack'] = _APGDGenerator(
            norm_model, eps=fgsm_eps, device=device, n_iter=40, loss=aa_loss,
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
    adv_label_parts = []
    per_attack_shapes = {}
    for atk_name in attack_names:
        n_atk = counts[atk_name]
        atk_indices = adv_pool[off: off + n_atk]

        # Phase 1: Batch-generate all adversarials on GPU in chunks.
        # CIFAR 32x32 inputs are 49x smaller than ImageNet 224x224 — chunk
        # raised from 128 to 512 to keep the RTX 5090 fully occupied
        # (typical CW VRAM at chunk=512 on CIFAR ≈ 12-16 GB, comfortably
        # under the 32 GB budget). The native PyTorch CW handles this
        # batch size efficiently.
        print(f"\nBatch-generating {n_atk} {atk_name} adversarials "
              f"(eps={fgsm_eps:.6f})...", flush=True)
        adv_cache = _batch_generate_adversarials(
            attacks[atk_name], pixel_dataset, atk_indices,
            label=f'{atk_name} adv',
            gen_chunk=512,
        )

        # Phase 2: Extract TDA features from pre-generated adversarials.
        # TDA persistence is CPU-bound and per-image; cannot be batched.
        print(f"Extracting {n_atk} {atk_name} TDA features...", flush=True)
        X_atk, W_atk = _extract_features(
            atk_indices, pixel_dataset, model, extractor, profiler,
            base_scorer, ref_profiles, device,
            adv_cache=adv_cache, label=f'{atk_name} adv',
            use_grad_norm=use_grad_norm,
            use_softmax_entropy=use_softmax_entropy,
        )
        off += n_atk
        X_adv_parts.append(X_atk)
        W_adv_parts.append(W_atk)
        adv_label_parts.append(np.array([atk_name] * X_atk.shape[0], dtype=object))
        per_attack_shapes[atk_name] = X_atk.shape[0]

    X_adv = np.vstack(X_adv_parts)
    W_adv = np.concatenate(W_adv_parts)
    adv_labels = np.concatenate(adv_label_parts)
    # Shuffle so the 80/20 train/val split sees all attack families -- without
    # shuffle, X_adv[:80%] would contain almost no samples from the last attack.
    perm  = np.random.RandomState(42).permutation(len(X_adv))
    X_adv = X_adv[perm]
    W_adv = W_adv[perm]
    adv_labels = adv_labels[perm]

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
        feature_space_version='pixel-v1',
        selection_objective=selection_objective,
        training_attack_counts={k: int(v) for k, v in per_attack_shapes.items()},
        balanced_attacks=balanced_attacks,
        pgd_train_steps=pgd_train_steps,
        aa_train_mode=aa_train_mode if include_autoattack else None,
        gradient_head_enabled=use_grad_norm,
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
    auc = None
    if len(X_val) > 0:
        probs = np.array([ensemble._logistic_prob(x) for x in X_val])
        auc   = roc_auc_score(y_val, probs)
        preds = (probs > 0.5).astype(int)
        print(f"\nHeld-out validation AUC (logistic component): {auc:.4f}")
        print(classification_report(y_val, preds, target_names=['clean', 'adv']))

    # -- Discriminative-power gate --------------------------------------------
    # An AUC ≈ 0.5 means the features cannot separate clean from adversarial,
    # which almost always traces back to an undertrained backbone. Refusing to
    # save here stops a poisoned `ensemble_scorer.pkl` from contaminating the
    # calibration and evaluation steps, where the failure resurfaces as a
    # silent TPR ≈ FPR collapse. The 0.85 floor is well above the chance line
    # (0.50) and below the production target (≥ 0.92 on properly-trained
    # backbones) — see fix/backbone-acc-gate plan §"Acceptance criteria".
    AUC_FLOOR = 0.85
    if auc is not None and auc < AUC_FLOOR and not allow_undertrained_smoke:
        # Surface backbone provenance in the error message so the diagnosis
        # is one terminal line away from the failure.
        backbone_acc_msg = ""
        try:
            import json as _json
            from src.config import BACKBONE_CHECKPOINT_PATH as _bcp
            from pathlib import Path as _Path
            sidecar = _Path(_bcp).with_suffix('.acc.json')
            if sidecar.exists():
                _meta = _json.loads(sidecar.read_text())
                backbone_acc_msg = (
                    f" Backbone provenance: test_acc={_meta.get('test_acc', '?')}, "
                    f"epochs={_meta.get('epochs', '?')}, "
                    f"sha256_first16={_meta.get('sha256_first16', '?')}."
                )
            else:
                backbone_acc_msg = f" No sidecar at {sidecar} — backbone provenance unknown."
        except Exception:
            pass
        raise RuntimeError(
            f"Held-out logistic AUC {auc:.4f} < {AUC_FLOOR:.2f} floor. The "
            f"TDA + entropy + DCT features cannot separate clean from "
            f"adversarial inputs, so the downstream conformal calibration "
            f"will give TPR ≈ FPR ≈ {1 - auc:.0%}.{backbone_acc_msg} "
            f"Verify models/cifar_resnet18.acc.json reports test_acc ≥ 0.93 "
            f"and re-run scripts/pretrain_cifar_backbone.py if not."
        )
    if auc is not None and auc < AUC_FLOOR:
        print(
            f"[WARN] Proceeding with low-AUC smoke ensemble (AUC={auc:.4f} < "
            f"{AUC_FLOOR:.2f}). This scorer is for local integration testing only."
        )

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
            adv_attack_labels=adv_labels[n_adv_train:].tolist(),
            selection_objective=selection_objective,
            clean_fpr_target=0.10,
        )
        ensemble.alpha_tune_summary = tune_summary
        print(f"Selected α = {ensemble.alpha:.3f} ({tune_summary['selection_objective']})")
        best_row = tune_summary.get('best_row', {})
        per_attack = best_row.get('per_attack_tpr') or {}
        if per_attack:
            print("Held-out per-attack TPR at clean FPR<=10%: " + "  ".join(
                f"{k}:{v:.3f}" for k, v in sorted(per_attack.items())
            ))
    else:
        print("Skipping α tuning — validation slice is empty.")

    # -- Set provenance metadata ------------------------------------------------
    ensemble.training_eps     = fgsm_eps
    ensemble.training_attacks = attack_names          # dynamic list
    ensemble.training_n       = len(X_adv)
    ensemble.training_attack_counts = {k: int(v) for k, v in per_attack_shapes.items()}
    ensemble.balanced_attacks = balanced_attacks
    ensemble.pgd_train_steps = pgd_train_steps
    ensemble.aa_train_mode = aa_train_mode if include_autoattack else None
    ensemble.gradient_head_enabled = use_grad_norm
    ensemble.feature_space_version = 'pixel-v1'
    ensemble.selection_objective = selection_objective
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
    print(f"   Trained on {DATASET.upper()} TRAINING set (no test-set leakage)")
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
    parser.add_argument('--enable-gradient-head', action='store_true',
                        help='Alias for --use-grad-norm. Records the scorer as '
                             'using the gradient-attack feature branch.')
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
    parser.add_argument('--balanced-attacks', action='store_true',
                        help='Use equal sample weight for each active attack '
                             'family so PGD/AutoAttack cannot be diluted.')
    parser.add_argument('--pgd-train-steps', type=int, default=40,
                        help='PGD iterations for adversarial training. Default '
                             '40 matches the evaluation PGD configuration.')
    parser.add_argument('--aa-train-mode', choices=['apgd-ce', 'apgd-dlr'],
                        default='apgd-ce',
                        help='AutoAttack/APGD training slice to include when '
                             '--include-autoattack is set.')
    parser.add_argument('--selection-objective',
                        choices=['auc', 'worst_case_tpr'],
                        default='worst_case_tpr',
                        help='Alpha selection objective on held-out training '
                             'slice. worst_case_tpr maximizes the minimum '
                             'per-attack TPR at clean FPR<=10%%.')
    parser.add_argument('--cw-max-iter', type=int, default=40,
                        help='CW max_iter for training (default 40, matches eval).')
    parser.add_argument('--cw-bss', type=int, default=5,
                        help='CW binary_search_steps for training (default 5, matches eval).')
    parser.add_argument('--cw-oversample', type=float, default=1.0,
                        help='Relative CW weight in the training mix (only active '
                             'with --include-cw). Default 1.0 = equal share with '
                             'PGD/Square. Conditional secondary intervention for '
                             'the CW TPR gap: set to 2.0 ONLY if pooled CW TPR is '
                             'in [0.75, 0.85) on the 5-seed vast.ai run after the '
                             'primary softmax-entropy fix. Raising above 1.0 takes '
                             'share from PGD/Square — verify ±1pp regression gate '
                             'on FGSM/PGD/Square before adopting non-default value.')
    parser.add_argument('--fast',      action='store_true',
                        help='Quick smoke-test: n_train=150')
    parser.add_argument('--allow-undertrained-smoke', action='store_true',
                        help='Smoke-only escape hatch: allow the ensemble '
                             'trainer to continue even if held-out AUC is '
                             'below the publishable floor.')
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
        use_grad_norm=args.use_grad_norm or args.enable_gradient_head,
        use_softmax_entropy=not args.no_softmax_entropy,
        data_root=args.data_root,
        output_path=args.output,
        profile_path=args.profile,
        include_cw=args.include_cw,
        include_autoattack=args.include_autoattack,
        cw_max_iter=args.cw_max_iter,
        cw_bss=args.cw_bss,
        cw_oversample=args.cw_oversample,
        no_tda_features=args.no_tda_features,
        balanced_attacks=args.balanced_attacks,
        pgd_train_steps=args.pgd_train_steps,
        aa_train_mode=args.aa_train_mode,
        selection_objective=args.selection_objective,
        allow_undertrained_smoke=args.allow_undertrained_smoke,
    )
