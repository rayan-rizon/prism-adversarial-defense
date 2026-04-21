"""
Full Attack Evaluation — Paper-Quality (n=1000, proper ε, CI, all attacks)

Improvements over run_evaluation.py:
  1. FGSM uses standard ε=8/255≈0.0314 (NOT 0.30)
  2. n=1000 per attack (down from 300)
  3. Uses held-out test-set eval split (images 8000-9999) — zero data leakage
  4. Carlini-Wagner L2 added (excluded only when n>100 in original)
  5. AutoAttack added (gold standard since Croce et al. 2020)
  6. 95% confidence intervals via Wilson score interval
  7. Per-tier FPR breakdown reported for each attack
  8. Latency measurement (mean ± std over 200 test images)
  9. Results saved as results_paper.json with all metadata

EVAL SPLIT: CIFAR-10 test indices 8000-9999 (2000 images → n=1000 for attack eval)
  - Profiles built on : test 0-4999
  - Cal built on      : test 5000-6999
  - Val built on      : test 7000-7999
  - THIS EVAL uses    : test 8000-9999

Standard attack parameters (matching RobustBench convention):
  FGSM   : L∞, ε=8/255≈0.0314
  PGD    : L∞, ε=8/255, 40 steps, step=2/255
  CW     : L2, binary-search confidence=0
  Square : L∞, ε=8/255, 5000 queries
  AutoAttack: L∞, ε=8/255 (apgd-ce + apgd-t + fab + square)

PIPELINE GATE
-------------
After any change to PersistenceEnsembleScorer, the feature vector, or the
scorer code, you MUST re-run in order:
  1. python scripts/train_ensemble_scorer.py
  2. python scripts/calibrate_ensemble.py
  3. python scripts/compute_ensemble_val_fpr.py
  4. python experiments/evaluation/run_evaluation_full.py  (this script)
Skipping steps 1-3 causes the silent regressions documented in
results_n500_20260419.json (FGSM TPR 0.832->0.622) and
results_n500_retrained_20260419.json (L2/L3 FPR targets violated).

Local attacks only: FGSM, PGD, Square.
CW and AutoAttack: remote-only (Thundercompute), after local targets pass.

USAGE
-----
  cd prism/
  python experiments/evaluation/run_evaluation_full.py [--attacks FGSM PGD Square]
"""
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models import ResNet18_Weights
import numpy as np
import json, os, sys, ssl, certifi, time
from tqdm import tqdm

os.environ.setdefault('SSL_CERT_FILE', certifi.where())
os.environ.setdefault('REQUESTS_CA_BUNDLE', certifi.where())
ssl._create_default_https_context = ssl.create_default_context

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from art.attacks.evasion import (
        FastGradientMethod,
        ProjectedGradientDescent,
        CarliniL2Method,
        SquareAttack,
    )
    from art.estimators.classification import PyTorchClassifier
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False
    print("WARNING: ART not installed. pip install adversarial-robustness-toolbox")

try:
    from autoattack import AutoAttack as _AA
    AA_AVAILABLE = True
except ImportError:
    AA_AVAILABLE = False
    print("WARNING: AutoAttack not installed. pip install autoattack")

from src.prism import PRISM
from src.sacd.monitor import NoOpCampaignMonitor
from src.config import (
    LAYER_NAMES, LAYER_WEIGHTS, DIM_WEIGHTS,
    IMAGENET_MEAN, IMAGENET_STD, EPS_LINF_STANDARD,
    EVAL_IDX,   # single source of truth -- do not redeclare below
)

# ── Constants ─────────────────────────────────────────────────────────────────
_MEAN = IMAGENET_MEAN
_STD  = IMAGENET_STD
_PIXEL_TRANSFORM = T.Compose([T.Resize(224), T.ToTensor()])
_NORMALIZE       = T.Normalize(mean=_MEAN, std=_STD)
# EVAL_IDX is imported from src.config (configs/default.yaml splits.eval_idx).


class _NormalizedResNet(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self._model = model
        self.register_buffer('_mean', torch.tensor(_MEAN).view(3, 1, 1))
        self.register_buffer('_std',  torch.tensor(_STD).view(3, 1, 1))

    def forward(self, x):
        return self._model((x - self._mean) / self._std)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    """
    Wilson score 95% confidence interval for a proportion k/n.
    Returns (lower, upper) as fractions.
    """
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def per_tier_fpr(clean_levels: dict, n_clean: int) -> dict:
    """Compute per-tier false positive rates from clean level distribution."""
    fp_l1 = n_clean - clean_levels.get('PASS', 0)
    fp_l2 = clean_levels.get('L2', 0) + clean_levels.get('L3', 0) + clean_levels.get('L3_REJECT', 0)
    fp_l3 = clean_levels.get('L3', 0) + clean_levels.get('L3_REJECT', 0)
    return {
        'FPR_L1_plus': round(fp_l1 / max(n_clean, 1), 4),
        'FPR_L2_plus': round(fp_l2 / max(n_clean, 1), 4),
        'FPR_L3_plus': round(fp_l3 / max(n_clean, 1), 4),
        'target_L1': 0.10,
        'target_L2': 0.03,
        'target_L3': 0.005,
        'pass_L1': fp_l1 / max(n_clean, 1) <= 0.10,
        'pass_L2': fp_l2 / max(n_clean, 1) <= 0.03,
        'pass_L3': fp_l3 / max(n_clean, 1) <= 0.005,
    }


def run_latency_benchmark(prism: PRISM, dataset, device, n: int = 200) -> dict:
    """Measure PRISM.defend() wall-clock latency on n clean images."""
    rng = np.random.RandomState(999)
    idx = rng.choice(range(*EVAL_IDX), n, replace=False)
    times = []
    for i in idx:
        img, _ = dataset[int(i)]
        x = _NORMALIZE(img).unsqueeze(0).to(device)
        t0 = time.perf_counter()
        prism.defend(x)
        times.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(times)
    result = {
        'mean_ms':   round(float(arr.mean()), 2),
        'std_ms':    round(float(arr.std()), 2),
        'min_ms':    round(float(arr.min()), 2),
        'max_ms':    round(float(arr.max()), 2),
        'p50_ms':    round(float(np.percentile(arr, 50)), 2),
        'p95_ms':    round(float(np.percentile(arr, 95)), 2),
        'n':         n,
        'target_ms': 100,
        'pass':      float(arr.mean()) < 100,
    }
    print(f"\nLatency ({n} images): "
          f"mean={result['mean_ms']:.1f}ms  std={result['std_ms']:.1f}ms  "
          f"p95={result['p95_ms']:.1f}ms  "
          f"{'✅ < 100ms target' if result['pass'] else '❌ > 100ms target'}")
    return result


def run_evaluation_full(
    n_test: int = 1000,
    data_root: str = './data',
    output_path: str = 'experiments/evaluation/results_paper.json',
    attacks_to_run: list = None,
    seed: int = 42,
    checkpoint_interval: int = 100,
    device_str: str = None,
    square_max_iter: int = 5000,
):
    if not ART_AVAILABLE:
        print("ERROR: ART not installed."); sys.exit(1)

    if device_str is not None:
        device = torch.device(device_str)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Evaluation: n_test={n_test}, "
          f"eval_split=test[{EVAL_IDX[0]}-{EVAL_IDX[1]-1}]")
    print(f"Standard L∞ budget: ε = {EPS_LINF_STANDARD:.4f} ({EPS_LINF_STANDARD*255:.1f}/255)\n")

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    # ── Model ──────────────────────────────────────────────────────────────────
    model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model = model.to(device).eval()

    layer_names   = LAYER_NAMES
    layer_weights = LAYER_WEIGHTS
    dim_weights   = DIM_WEIGHTS

    # ── PRISM ──────────────────────────────────────────────────────────────────
    cal_path  = 'models/calibrator.pkl'
    prof_path = 'models/reference_profiles.pkl'
    ens_path  = 'models/ensemble_scorer.pkl'
    for p in [cal_path, prof_path]:
        if not os.path.exists(p):
            print(f"ERROR: {p} not found. Run build_profile_testset.py + "
                  f"calibrate_testset.py first.")
            sys.exit(1)

    prism_base = PRISM.from_saved(
        model=model,
        layer_names=layer_names,
        calibrator_path=cal_path,
        profile_path=prof_path,
        ensemble_path=ens_path,
        layer_weights=layer_weights,
        dim_weights=dim_weights,
        campaign_monitor=NoOpCampaignMonitor(),
    )

    # ── Capture ensemble provenance for _meta (audit trail) ───────────────────
    _ens_scorer = getattr(prism_base, 'scorer', None)
    _ens_meta = {
        'use_dct':          getattr(_ens_scorer, 'use_dct', None),
        'training_attacks': getattr(_ens_scorer, 'training_attacks', None),
        'training_n':       getattr(_ens_scorer, 'training_n', None),
        'training_eps':     getattr(_ens_scorer, 'training_eps', None),
        'n_features':       getattr(_ens_scorer, 'n_features', None),
    }
    print(f"Ensemble provenance: use_dct={_ens_meta['use_dct']}, "
          f"training_attacks={_ens_meta['training_attacks']}, "
          f"training_n={_ens_meta['training_n']}, "
          f"n_features={_ens_meta['n_features']}")

    # ── ART classifier ─────────────────────────────────────────────────────────
    # NOTE: must be on the same device as the main model so CW / PGD
    # gradient computations run on GPU (not CPU) — this is the critical fix
    # that reduces CW from ~285s/sample (CPU) to ~3s/sample (A100 GPU).
    art_model  = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device).eval()
    norm_model = _NormalizedResNet(art_model).to(device).eval()
    device_type = 'gpu' if device.type == 'cuda' else 'cpu'
    classifier = PyTorchClassifier(
        model=norm_model,
        loss=torch.nn.CrossEntropyLoss(),
        input_shape=(3, 224, 224),
        nb_classes=1000,        # ImageNet classes for ResNet-18 backbone
        clip_values=(0.0, 1.0),
        device_type=device_type,
    )

    # ── Define attacks (standard parameters) ──────────────────────────────────
    all_attacks = {
        'FGSM': lambda: FastGradientMethod(
            classifier, eps=EPS_LINF_STANDARD),
        'PGD': lambda: ProjectedGradientDescent(
            classifier,
            eps=EPS_LINF_STANDARD,
            eps_step=EPS_LINF_STANDARD / 4,   # step = 2/255
            max_iter=40,
            num_random_init=1,
        ),
        # CW-L2: batch_size=64 for GPU parallelism; max_iter=50 + binary_search_steps=5
        # is the practical "fast CW" configuration used in many robustness papers.
        # Full max_iter=100/bss=9 takes ~5h for n=1000; 50/5/bs=64 takes ~90 min.
        'CW': lambda: CarliniL2Method(
            classifier, max_iter=50, confidence=0.0, learning_rate=0.01,
            binary_search_steps=5, batch_size=64,
        ),
        'Square': lambda: SquareAttack(
            classifier, eps=EPS_LINF_STANDARD, max_iter=square_max_iter, nb_restarts=1,
        ),
    }

    attacks_to_run = attacks_to_run or ['FGSM', 'PGD', 'CW', 'Square']
    # CW warning: slow on CPU, fast on GPU (ART classifier now follows device)
    if 'CW' in attacks_to_run and device.type == 'cpu' and n_test > 50:
        print("⚠  CW on CPU with n>50 will be very slow (~285s/sample). "
              "Use a GPU instance or pass --attacks FGSM PGD Square.")
    elif 'CW' in attacks_to_run and device.type == 'cuda':
        print(f"✅ CW running on {device_type.upper()} (ART classifier on GPU) — ~3s/sample")

    # ── Dataset (held-out eval split) ──────────────────────────────────────────
    pixel_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=_PIXEL_TRANSFORM
    )
    eval_indices = list(range(*EVAL_IDX))
    n_eval = min(n_test, len(eval_indices))
    sample_idx = rng.choice(eval_indices, n_eval, replace=False)

    # ── Latency benchmark ──────────────────────────────────────────────────────
    norm_dataset_for_bench = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=False, transform=_PIXEL_TRANSFORM
    )
    latency = run_latency_benchmark(prism_base, norm_dataset_for_bench, device)

    # ── Per-attack evaluation ──────────────────────────────────────────────────
    results = {}

    # Pre-load all pixel images once (shared across attacks)
    print(f"\nPre-loading {len(sample_idx)} images...")
    all_imgs_pixel = []
    for i in sample_idx:
        img_pixel, _ = pixel_dataset[int(i)]
        all_imgs_pixel.append(img_pixel)
    # Stack: (N, 3, 224, 224) numpy array for ART batch generation
    X_pixel_np = torch.stack(all_imgs_pixel).numpy()   # shape (N,3,224,224)
    print(f"Pre-loaded {len(all_imgs_pixel)} images ✓")

    for attack_name in attacks_to_run:
        if attack_name not in all_attacks:
            print(f"Unknown attack: {attack_name}. Skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"Attack: {attack_name}")
        print(f"{'='*60}")

        attack = all_attacks[attack_name]()

        prism_attack = PRISM.from_saved(
            model=model,
            layer_names=layer_names,
            calibrator_path=cal_path,
            profile_path=prof_path,
            ensemble_path=ens_path,
            layer_weights=layer_weights,
            dim_weights=dim_weights,
            campaign_monitor=NoOpCampaignMonitor(),
        )

        tp, fp, fn, tn = 0, 0, 0, 0
        level_clean = {}
        level_adv   = {}

        # ── Batch-generate ALL adversarials at once ──────────────────────────
        # Passing the full X_pixel_np to attack.generate() lets ART split it
        # into batch_size=32 sub-batches internally, keeping the GPU pipeline
        # saturated. This is the critical speedup for CW (and PGD/Square).
        print(f"  Generating {len(sample_idx)} adversarial examples (batch={attack._batch_size if hasattr(attack, '_batch_size') else '?'})...")
        try:
            X_adv_np = attack.generate(X_pixel_np)  # shape (N,3,224,224)
        except Exception as e:
            print(f"  Batch generation failed ({e}), falling back to per-sample...")
            X_adv_np = np.zeros_like(X_pixel_np)
            for idx_i, x_np_i in enumerate(tqdm(X_pixel_np, desc="  fallback")):
                try:
                    X_adv_np[idx_i] = attack.generate(x_np_i[np.newaxis])[0]
                except Exception as e2:
                    X_adv_np[idx_i] = x_np_i
                    print(f"    Sample {idx_i} failed: {e2}")

        # ── Evaluate clean + adversarial through PRISM ────────────────────────
        for j, img_pixel in enumerate(tqdm(all_imgs_pixel)):
            x = _NORMALIZE(img_pixel).unsqueeze(0).to(device)

            # Clean
            _, lv_clean, _ = prism_attack.defend(x)
            level_clean[lv_clean] = level_clean.get(lv_clean, 0) + 1
            if lv_clean == 'PASS':
                tn += 1
            else:
                fp += 1

            # Adversarial
            x_adv = _NORMALIZE(torch.tensor(X_adv_np[j])).unsqueeze(0).to(device)
            _, lv_adv, _ = prism_attack.defend(x_adv)
            level_adv[lv_adv] = level_adv.get(lv_adv, 0) + 1
            if lv_adv != 'PASS':
                tp += 1
            else:
                fn += 1

            if (j + 1) % checkpoint_interval == 0:
                _n_adv = tp + fn
                _n_clean = fp + tn
                _tpr = tp / max(_n_adv, 1)
                _fpr = fp / max(_n_clean, 1)
                _prec = tp / max(tp + fp, 1)
                _f1 = 2 * _prec * _tpr / max(_prec + _tpr, 1e-8)
                print(f"\n  [Checkpoint {j+1}/{len(all_imgs_pixel)}] TPR={_tpr:.4f} "
                      f"({'✅' if _tpr >= 0.88 else '❌' if _tpr < 0.70 else '⚠'}) | "
                      f"FPR={_fpr:.4f} ({'✅' if _fpr <= 0.10 else '❌'}) | F1={_f1:.4f}")

        # Metrics
        n_adv   = tp + fn
        n_clean = fp + tn
        tpr = tp / max(n_adv, 1)
        fpr = fp / max(n_clean, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 2 * precision * tpr / max(precision + tpr, 1e-8)

        # 95% Wilson confidence intervals
        tpr_ci = wilson_ci(tp, n_adv)
        fpr_ci = wilson_ci(fp, n_clean)

        # Per-tier FPR
        tier_fpr = per_tier_fpr(level_clean, n_clean)

        results[attack_name] = {
            'TPR':       round(tpr, 4),
            'TPR_CI_95': [round(tpr_ci[0], 4), round(tpr_ci[1], 4)],
            'FPR':       round(fpr, 4),
            'FPR_CI_95': [round(fpr_ci[0], 4), round(fpr_ci[1], 4)],
            'Precision': round(precision, 4),
            'F1':        round(f1, 4),
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'n_adv': n_adv, 'n_clean': n_clean,
            'per_tier_fpr': tier_fpr,
            'clean_level_distribution': level_clean,
            'adversarial_level_distribution': level_adv,
        }

        print(f"\n  TPR  = {tpr:.4f}  95% CI [{tpr_ci[0]:.4f}, {tpr_ci[1]:.4f}]"
              f"  {'✅' if tpr >= 0.90 else '❌' if tpr < 0.70 else '⚠'}")
        print(f"  FPR  = {fpr:.4f}  95% CI [{fpr_ci[0]:.4f}, {fpr_ci[1]:.4f}]"
              f"  {'✅' if fpr <= 0.10 else '❌'}")
        print(f"  F1   = {f1:.4f}")
        print(f"  Per-tier FPR: L1+={tier_fpr['FPR_L1_plus']:.4f} "
              f"L2+={tier_fpr['FPR_L2_plus']:.4f} "
              f"L3+={tier_fpr['FPR_L3_plus']:.4f}")

    # ── AutoAttack ─────────────────────────────────────────────────────────────
    if 'AutoAttack' in (attacks_to_run or []) and AA_AVAILABLE:
        print(f"\n{'='*60}")
        print("Attack: AutoAttack (L∞, ε=8/255, standard)")
        print(f"{'='*60}")
        results['AutoAttack'] = _run_autoattack(
            prism_base, pixel_dataset, sample_idx, device, EPS_LINF_STANDARD
        )

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"{'Attack':>10} {'TPR':>8} {'FPR':>8} {'F1':>8} {'TPR≥90%':>9}")
    print(f"{'-'*70}")
    for name, r in results.items():
        if name == '_meta':
            continue
        status = '✅' if r['TPR'] >= 0.90 else ('⚠' if r['TPR'] >= 0.70 else '❌')
        print(f"{name:>10} {r['TPR']:>8.4f} {r['FPR']:>8.4f} "
              f"{r['F1']:>8.4f} {status:>9}")

    # ── Save ───────────────────────────────────────────────────────────────────
    results['_meta'] = {
        'n_test':       n_test,
        'n_actual':     int(len(sample_idx)),
        'eval_split':   f'CIFAR-10 test idx {EVAL_IDX[0]}-{EVAL_IDX[1]-1}',
        'seed':         seed,
        'eps_linf':     EPS_LINF_STANDARD,
        'eps_note':     '8/255 = standard RobustBench/AutoAttack convention',
        'attacks':      attacks_to_run,
        'latency':      latency,
        'device':       str(device),
        'ensemble':     _ens_meta,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {output_path}")

    return results


def _run_autoattack(prism, pixel_dataset, sample_idx, device, eps):
    """Run AutoAttack and evaluate each example through PRISM."""
    if not AA_AVAILABLE:
        return {'error': 'autoattack not installed'}

    # Build a wrapped model for AutoAttack on the same device (GPU if available)
    from torchvision.models import ResNet18_Weights
    import torchvision

    model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device).eval()
    norm_model = _NormalizedResNet(model).to(device).eval()

    adversary = _AA(norm_model, norm='Linf', eps=eps, version='standard', device=device)

    tp, fp, fn, tn = 0, 0, 0, 0
    level_clean, level_adv = {}, {}

    # Build batched pixel tensors for AutoAttack
    imgs_pixel   = []
    labels_list  = []
    for i in sample_idx:
        img, lbl = pixel_dataset[int(i)]
        imgs_pixel.append(img)
        labels_list.append(lbl)

    X = torch.stack(imgs_pixel).to(device)     # (N, 3, 224, 224)
    y = torch.tensor(labels_list).to(device)   # (N,) — used by AutoAttack

    X_adv = adversary.run_standard_evaluation(X, y, bs=32)

    for j, (pix, adv_pix) in enumerate(zip(X, X_adv)):
        x_clean = _NORMALIZE(pix.cpu()).unsqueeze(0).to(device)
        x_adv   = _NORMALIZE(adv_pix.cpu()).unsqueeze(0).to(device)

        _, lv_c, _ = prism.defend(x_clean)
        level_clean[lv_c] = level_clean.get(lv_c, 0) + 1
        if lv_c == 'PASS': tn += 1
        else:               fp += 1

        _, lv_a, _ = prism.defend(x_adv)
        level_adv[lv_a] = level_adv.get(lv_a, 0) + 1
        if lv_a != 'PASS': tp += 1
        else:               fn += 1

    n_adv, n_clean = tp + fn, fp + tn
    tpr = tp / max(n_adv, 1)
    fpr = fp / max(n_clean, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * tpr / max(prec + tpr, 1e-8)

    print(f"  AutoAttack: TPR={tpr:.4f}  FPR={fpr:.4f}  F1={f1:.4f}")
    return {
        'TPR': round(tpr, 4), 'FPR': round(fpr, 4),
        'Precision': round(prec, 4), 'F1': round(f1, 4),
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
        'n_adv': n_adv, 'n_clean': n_clean,
        'per_tier_fpr': per_tier_fpr(level_clean, n_clean),
        'clean_level_distribution': level_clean,
        'adversarial_level_distribution': level_adv,
        'TPR_CI_95': [round(v, 4) for v in wilson_ci(tp, n_adv)],
        'FPR_CI_95': [round(v, 4) for v in wilson_ci(fp, n_clean)],
    }


def run_evaluation_multiseed(
    seeds: list = None,
    n_test: int = 1000,
    data_root: str = './data',
    output_path: str = 'experiments/evaluation/results_paper_multiseed.json',
    attacks_to_run: list = None,
    checkpoint_interval: int = 100,
):
    """
    Run run_evaluation_full() over multiple seeds and aggregate statistics.

    Different seeds sample different subsets of EVAL_IDX=(8000,10000) —
    providing genuine variance estimates for paper reporting.

    Each per-seed result is stored as-is; the aggregate reports mean±std
    over seeds for TPR, FPR, and F1 per attack.  Wilson CI is also
    pooled across seeds via the sum of TP/FP/FN/TN counts.

    Output format
    -------------
    {
      "seeds": [42, ...],
      "per_seed": { "42": {...full single-seed result...}, ... },
      "aggregate": {
        "FGSM": {
          "TPR_mean": ..., "TPR_std": ...,
          "FPR_mean": ..., "FPR_std": ...,
          "F1_mean":  ..., "F1_std":  ...,
          "TPR_CI_95_pooled": [...],   # from pooled TP / n
          "FPR_CI_95_pooled": [...],
        },
        ...
      },
      "metadata": { "n_test": ..., "eps": ..., "eval_split": ... }
    }
    """
    if seeds is None:
        seeds = [42, 123, 456, 789, 999]

    attacks_to_run = attacks_to_run or ['FGSM', 'PGD', 'Square']

    print(f"\n{'='*65}")
    print(f"Multi-seed evaluation: seeds={seeds}")
    print(f"n_test={n_test}, attacks={attacks_to_run}")
    print(f"{'='*65}\n")

    per_seed_results = {}
    for seed in seeds:
        print(f"\n{'─'*65}")
        print(f"Running seed={seed}")
        print(f"{'─'*65}")
        # Each seed writes its own checkpoint file; we collect the return value.
        seed_out = os.path.join(
            os.path.dirname(output_path),
            f"results_paper_seed{seed}.json",
        )
        result = run_evaluation_full(
            n_test=n_test,
            data_root=data_root,
            output_path=seed_out,
            attacks_to_run=attacks_to_run,
            seed=seed,
            checkpoint_interval=checkpoint_interval,
        )
        per_seed_results[str(seed)] = result

    # ── Aggregate across seeds ────────────────────────────────────────────────
    aggregate = {}
    for atk in attacks_to_run:
        tprs, fprs, f1s = [], [], []
        pool_tp = pool_fp = pool_fn = pool_tn = 0

        for seed_str, seed_res in per_seed_results.items():
            # run_evaluation_full() returns attack results at top level keyed by attack name
            atk_res = seed_res.get(atk)
            if atk_res is None:
                continue
            tprs.append(atk_res['TPR'])
            fprs.append(atk_res['FPR'])
            f1s.append(atk_res['F1'])
            pool_tp += atk_res.get('TP', 0)
            pool_fp += atk_res.get('FP', 0)
            pool_fn += atk_res.get('FN', 0)
            pool_tn += atk_res.get('TN', 0)

        if not tprs:
            continue

        pool_n_adv   = pool_tp + pool_fn
        pool_n_clean = pool_fp + pool_tn
        aggregate[atk] = {
            'TPR_mean':          round(float(np.mean(tprs)), 4),
            'TPR_std':           round(float(np.std(tprs, ddof=1) if len(tprs) > 1 else 0.0), 4),
            'FPR_mean':          round(float(np.mean(fprs)), 4),
            'FPR_std':           round(float(np.std(fprs, ddof=1) if len(fprs) > 1 else 0.0), 4),
            'F1_mean':           round(float(np.mean(f1s)),  4),
            'F1_std':            round(float(np.std(f1s, ddof=1) if len(f1s) > 1 else 0.0),  4),
            'TPR_CI_95_pooled':  [round(v, 4) for v in wilson_ci(pool_tp, pool_n_adv)],
            'FPR_CI_95_pooled':  [round(v, 4) for v in wilson_ci(pool_fp, pool_n_clean)],
            'n_seeds':           len(tprs),
            'pool_TP': pool_tp, 'pool_FP': pool_fp,
            'pool_FN': pool_fn, 'pool_TN': pool_tn,
        }

    multi_seed_output = {
        'seeds':     seeds,
        'per_seed':  per_seed_results,
        'aggregate': aggregate,
        'metadata': {
            'n_test':      n_test,
            'eps':         round(EPS_LINF_STANDARD, 6),
            'eps_255':     round(EPS_LINF_STANDARD * 255, 2),
            'eval_split':  list(EVAL_IDX),
            'attacks':     attacks_to_run,
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(multi_seed_output, f, indent=2)
    print(f"\nMulti-seed results saved → {output_path}")

    # ── Print summary table ────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"{'Attack':10} {'TPR mean±std':>20} {'FPR mean±std':>20} {'F1 mean±std':>20}")
    print(f"{'─'*65}")
    for atk, ag in aggregate.items():
        print(f"{atk:10} {ag['TPR_mean']:.4f}±{ag['TPR_std']:.4f}"
              f"  {ag['FPR_mean']:.4f}±{ag['FPR_std']:.4f}"
              f"  {ag['F1_mean']:.4f}±{ag['F1_std']:.4f}")
    print(f"{'='*65}")

    return multi_seed_output


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="PRISM paper-quality evaluation")
    parser.add_argument('--n-test',    type=int, default=1000)
    parser.add_argument('--data-root', default='./data')
    parser.add_argument('--output',    default='experiments/evaluation/results_paper.json')
    parser.add_argument('--attacks',   nargs='+',
                        default=['FGSM', 'PGD', 'Square'],
                        help='FGSM PGD CW Square AutoAttack')
    parser.add_argument('--seed',      type=int, default=42)
    parser.add_argument('--checkpoint-interval', type=int, default=100,
                        help='Interval for printing live metrics')
    parser.add_argument('--device', default=None,
                        help='Force device: cpu or cuda (default: auto-detect)')
    parser.add_argument('--square-max-iter', type=int, default=5000,
                        help='Max iterations for Square attack (default: 5000)')
    parser.add_argument('--multi-seed', action='store_true',
                        help='Run over 5 seeds and report mean±std (paper mode)')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456, 789, 999],
                        help='Seeds to use with --multi-seed')
    args = parser.parse_args()

    if args.multi_seed:
        run_evaluation_multiseed(
            seeds=args.seeds,
            n_test=args.n_test,
            data_root=args.data_root,
            output_path=args.output,
            attacks_to_run=args.attacks,
            checkpoint_interval=args.checkpoint_interval,
        )
    else:
        run_evaluation_full(
            n_test=args.n_test,
            data_root=args.data_root,
            output_path=args.output,
            attacks_to_run=args.attacks,
            seed=args.seed,
            checkpoint_interval=args.checkpoint_interval,
            device_str=args.device,
            square_max_iter=args.square_max_iter,
        )
