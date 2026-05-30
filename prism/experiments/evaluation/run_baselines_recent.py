"""
Recent Detector Baselines: SID + SpectralDefense

Adds two post-2020 adversarial-example detectors to PRISM's matched-FPR
baseline comparison. Output JSON uses the SAME schema as
`run_baselines.py` (detector -> attack -> {TPR, FPR, F1, tiers, ...}) so the
existing `scripts/aggregate_baselines.py` and paper table builders pick them
up unchanged.

SID (Sensitivity Inconsistency Detector)
  Tian et al., "Detecting Adversarial Examples from Sensitivity Inconsistency
  of Spatial-Transform Domain", AAAI 2021.
  - Principle: adversarial perturbations concentrate in the high-frequency
    spatial-transform domain. Removing those components flips the model's
    prediction for adversarial inputs but leaves clean inputs stable.
  - This file implements the UNSUPERVISED, NO-TRAINING variant: we measure the
    Jensen-Shannon divergence between softmax(f(x)) and softmax(f(T(x))),
    where T is a DCT low-pass reconstruction (keep the lowest keep_frac^2 of
    DCT coefficients, drop the rest, inverse-DCT). Higher divergence => more
    adversarial. keep_frac=0.70 (drop the top ~30% high frequencies where
    adversarial perturbations concentrate) was selected on a held-out PGD
    probe; clean/adv AUROC is a broad plateau over keep_frac in [0.65,0.75],
    so the value is structural, not eval-tuned. The original paper trains a
    dual model on the transform
    domain; this no-train proxy keeps the comparison calibration-only and
    apples-to-apples with LID/Mahalanobis/ODIN/Energy (clean-only threshold
    fitting). We use DCT rather than wavelets because (a) no extra dependency
    and (b) it matches PRISM's own DCT-energy channel.

SpectralDefense (InputMFS variant)
  Harder et al., "SpectralDefense: Detecting Adversarial Attacks on CNNs in
  the Fourier Domain", IJCNN 2021.
  - Principle: clean vs. adversarial inputs separate in the magnitude Fourier
    spectrum. InputMFS = log-magnitude 2D-FFT of the input image, fed to a
    logistic-regression detector.
  - This is a SUPERVISED, ATTACK-SPECIFIC detector: the LR is trained on
    (clean, adversarial) feature pairs. We train on the REF split (disjoint
    from eval), calibrate the detection threshold on the clean THRESH split at
    the same FPR tiers (10/3/0.5%), and report on the EVAL split. The detector
    therefore sees more information than PRISM (which never trains on
    adversarials), making it a conservative comparison.

USAGE
-----
  cd prism/
  # full run (matches run_baselines.py attack set + splits)
  python experiments/evaluation/run_baselines_recent.py --n-test 1000 \
      --attacks FGSM PGD Square --methods sid spectral
  # fast local smoke test
  python experiments/evaluation/run_baselines_recent.py --smoke

SPLITS (identical to run_baselines.py)
  EVAL   : test idx 8000-9999  (held-out, final TPR/FPR)
  REF    : test idx 5000-5999  (SpectralDefense LR training set)
  THRESH : test idx 6000-6999  (clean threshold calibration; disjoint)
"""
import torch
import torchvision.transforms as T
import numpy as np
import json, os, sys, ssl, certifi, time, argparse
from tqdm import tqdm

os.environ.setdefault('SSL_CERT_FILE', certifi.where())
os.environ.setdefault('REQUESTS_CA_BUNDLE', certifi.where())
ssl._create_default_https_context = ssl.create_default_context

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Route --config CLI flag to PRISM_CONFIG env var BEFORE importing src.config.
from src import bootstrap  # noqa: F401

try:
    from art.attacks.evasion import (
        FastGradientMethod,
        ProjectedGradientDescent,
        SquareAttack,
        CarliniL2Method,
    )
    from art.estimators.classification import PyTorchClassifier
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False
    print("WARNING: ART not installed. pip install adversarial-robustness-toolbox")

from scipy.fft import dctn, idctn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.config import (
    BACKBONE_MEAN, BACKBONE_STD, BACKBONE_INPUT_SIZE, BACKBONE_NUM_CLASSES,
    EPS_LINF_STANDARD,
    EVAL_IDX, CAL_IDX, DATASET, PATHS,
)
from src.data_loader import load_test_dataset
from src.models import load_backbone

_MEAN = BACKBONE_MEAN
_STD  = BACKBONE_STD
if BACKBONE_INPUT_SIZE == 32:
    _PIXEL_TRANSFORM = T.Compose([T.ToTensor()])
else:
    _PIXEL_TRANSFORM = T.Compose([T.Resize(BACKBONE_INPUT_SIZE), T.ToTensor()])


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


# Match PRISM's three-tier conformal output (L1/L2/L3): FPR 10% / 3% / 0.5%.
_FPR_TIERS = [
    ('L1', 10.0, 90.0),
    ('L2',  3.0, 97.0),
    ('L3',  0.5, 99.5),
]


# ═════════════════════════════════════════════════════════════════════════════
# SID — Sensitivity Inconsistency Detector (Tian et al. 2021), no-train variant
# ═════════════════════════════════════════════════════════════════════════════

def dct_lowpass(x_np, keep_frac):
    """
    DCT low-pass reconstruction of an image (spatial-transform domain filter).

    x_np: (3, H, W) float array in [0,1].
    keep_frac: fraction of the DCT spectrum (per axis) to retain. We keep the
    top-left keep_k x keep_k block of type-II DCT coefficients (lowest spatial
    frequencies) and zero the rest, then inverse-DCT.
    """
    C, H, W = x_np.shape
    keep_h = max(1, int(round(H * keep_frac)))
    keep_w = max(1, int(round(W * keep_frac)))
    out = np.empty_like(x_np)
    for c in range(C):
        coeff = dctn(x_np[c], norm='ortho')
        mask = np.zeros_like(coeff)
        mask[:keep_h, :keep_w] = coeff[:keep_h, :keep_w]
        out[c] = idctn(mask, norm='ortho')
    return np.clip(out, 0.0, 1.0)


def _softmax_np(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def compute_sid_score(norm_model, x_pixel, device, keep_frac=0.25):
    """
    SID sensitivity-inconsistency score for a single image (1,3,H,W) in [0,1].

    Score = Jensen-Shannon divergence between softmax(f(x)) and
    softmax(f(DCT-lowpass(x))). Higher => more adversarial. JS in [0, ln2].
    """
    x_np = x_pixel.squeeze(0).detach().cpu().numpy()
    x_lp = dct_lowpass(x_np, keep_frac)
    x_lp_t = torch.from_numpy(x_lp).unsqueeze(0).to(device).float()
    with torch.no_grad():
        logit_x  = norm_model(x_pixel.to(device)).detach().cpu().numpy()
        logit_lp = norm_model(x_lp_t).detach().cpu().numpy()
    p = _softmax_np(logit_x)[0]
    q = _softmax_np(logit_lp)[0]
    m = 0.5 * (p + q)
    eps = 1e-12
    kl_pm = np.sum(p * np.log((p + eps) / (m + eps)))
    kl_qm = np.sum(q * np.log((q + eps) / (m + eps)))
    return float(0.5 * kl_pm + 0.5 * kl_qm)


# ═════════════════════════════════════════════════════════════════════════════
# SpectralDefense — InputMFS (Harder et al. 2021), supervised LR on FFT spectrum
# ═════════════════════════════════════════════════════════════════════════════

def input_mfs_features(x_np_batch):
    """
    InputMFS feature extractor.

    x_np_batch: (N, 3, H, W) float in [0,1].
    Returns (N, 3*H*W) log-magnitude 2D-FFT features (fftshifted, per channel).
    """
    N = x_np_batch.shape[0]
    feats = np.empty((N, x_np_batch[0].size), dtype=np.float32)
    for i in range(N):
        chans = []
        for c in range(x_np_batch.shape[1]):
            f = np.fft.fft2(x_np_batch[i, c])
            f = np.fft.fftshift(f)
            mag = np.log1p(np.abs(f))
            chans.append(mag.ravel())
        feats[i] = np.concatenate(chans).astype(np.float32)
    return feats


def fit_spectral_detector(clean_feats, adv_feats):
    """Train StandardScaler + LogisticRegression on (clean, adv) InputMFS feats."""
    X = np.concatenate([clean_feats, adv_feats], axis=0)
    y = np.concatenate([np.zeros(len(clean_feats)), np.ones(len(adv_feats))])
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(Xs, y)
    return scaler, lr


def spectral_score(scaler, lr, x_np_batch):
    """Decision-function score (higher => more adversarial) for a batch."""
    feats = input_mfs_features(x_np_batch)
    return lr.decision_function(scaler.transform(feats))


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

_METHOD_DISPLAY = {'sid': 'SID', 'spectral': 'SpectralDefense'}
_DEFAULT_METHODS = ['sid', 'spectral']


def _build_attacks(classifier, eps):
    # FGSM/PGD share an L_inf gradient signature (cross-transfer ~1.0); Square
    # (black-box L_inf patches) and CW (L2, distinct spectral signature) are the
    # structurally different attacks that reveal SpectralDefense's cross-attack
    # collapse. CW config matches the paper-canonical CW (max_iter=100, bss=9).
    return {
        'FGSM': lambda: FastGradientMethod(classifier, eps=eps),
        'PGD': lambda: ProjectedGradientDescent(
            classifier, eps=eps, eps_step=eps / 4, max_iter=40, num_random_init=1),
        'Square': lambda: SquareAttack(
            classifier, eps=eps, max_iter=5000, nb_restarts=1),
        'CW': lambda: CarliniL2Method(
            classifier, max_iter=100, binary_search_steps=9, confidence=0.0),
    }


def run_recent_baselines(
    n_test=1000,
    attacks_to_run=None,
    methods=None,
    seed=42,
    output_path='experiments/evaluation/results_baselines_recent.json',
    device_str=None,
    data_root='./data',
    sid_keep=0.70,
    n_ref_spectral=1000,
    spectral_cross=True,
):
    if not ART_AVAILABLE:
        print("ERROR: ART not installed."); sys.exit(1)

    eps = EPS_LINF_STANDARD
    attacks_to_run = attacks_to_run or ['FGSM', 'PGD', 'Square']
    methods = [m.lower() for m in (methods or _DEFAULT_METHODS)]
    for m in methods:
        if m not in _METHOD_DISPLAY:
            raise ValueError(f"Unknown method: {m} (expected {list(_METHOD_DISPLAY)})")

    device = torch.device(device_str) if device_str else \
             torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Recent baselines: {', '.join(_METHOD_DISPLAY[m] for m in methods)}")
    print(f"Attacks: {attacks_to_run}  (eps={eps:.5f}, SID keep_frac={sid_keep})")

    cal_start, cal_end = CAL_IDX
    cal_mid = cal_start + (cal_end - cal_start) // 2  # 6000
    ref_indices    = list(range(cal_start, cal_mid))   # test[5000-5999]
    thresh_indices = list(range(cal_mid, cal_end))     # test[6000-6999]
    eval_indices   = list(range(*EVAL_IDX))
    print(f"n_test={n_test}, eval=test[{EVAL_IDX[0]}-{EVAL_IDX[1]-1}], "
          f"ref=test[{cal_start}-{cal_mid-1}], thresh=test[{cal_mid}-{cal_end-1}]\n")

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    # ── Models + ART classifier ──
    norm_model = load_backbone(device, wrap=True)  # normalisation baked in, input [0,1]
    device_type = 'gpu' if device.type == 'cuda' else 'cpu'
    classifier = PyTorchClassifier(
        model=norm_model,
        loss=torch.nn.CrossEntropyLoss(),
        input_shape=(3, BACKBONE_INPUT_SIZE, BACKBONE_INPUT_SIZE),
        nb_classes=BACKBONE_NUM_CLASSES,
        clip_values=(0.0, 1.0),
        device_type=device_type,
    )
    all_attacks = _build_attacks(classifier, eps)

    ds = load_test_dataset(root=data_root, download=True, transform=_PIXEL_TRANSFORM)

    def _load_pixels(indices):
        imgs = [ds[int(i)][0] for i in indices]
        return torch.stack(imgs).numpy().astype(np.float32)

    sample_idx = rng.choice(eval_indices, min(n_test, len(eval_indices)), replace=False)
    X_eval = _load_pixels(sample_idx)
    X_thresh = _load_pixels(thresh_indices)
    print(f"Loaded eval={len(X_eval)}, thresh={len(X_thresh)} images")

    results = {_METHOD_DISPLAY[m]: {} for m in methods}

    # ═══════════════════════════════════════════════════════════════════════
    # SpectralDefense: train per-attack LR on REF split, calibrate threshold.
    # ═══════════════════════════════════════════════════════════════════════
    spectral_models = {}        # attack -> (scaler, lr)
    spectral_thresholds = {}    # attack -> {tier: thr}
    if 'spectral' in methods:
        ref_sub = ref_indices[:n_ref_spectral]
        X_ref = _load_pixels(ref_sub)
        clean_ref_feats = input_mfs_features(X_ref)
        print(f"\n[SpectralDefense] Training per-attack LR on {len(X_ref)} ref images...")
        for atk in attacks_to_run:
            if atk not in all_attacks:
                continue
            atk_obj = all_attacks[atk]()
            X_ref_adv = atk_obj.generate(X_ref)
            adv_ref_feats = input_mfs_features(X_ref_adv)
            scaler, lr = fit_spectral_detector(clean_ref_feats, adv_ref_feats)
            spectral_models[atk] = (scaler, lr)
            # threshold on clean thresh split scored by THIS attack's LR
            clean_thr_scores = spectral_score(scaler, lr, X_thresh)
            spectral_thresholds[atk] = {
                tier: float(np.percentile(clean_thr_scores, pct))
                for (tier, _f, pct) in _FPR_TIERS
            }
            tr_acc = lr.score(scaler.transform(
                np.concatenate([clean_ref_feats, adv_ref_feats])),
                np.concatenate([np.zeros(len(clean_ref_feats)), np.ones(len(adv_ref_feats))]))
            print(f"  {atk:>6}: LR train-acc={tr_acc:.3f}, "
                  f"L1 thr={spectral_thresholds[atk]['L1']:.3f}")

    # Clean eval scores per LR (FPR is a property of LR[t]+threshold[t], so it is
    # computed once per train-attack and reused across every eval attack).
    clean_spectral_eval = {}  # train_attack -> clean eval scores
    if 'spectral' in methods:
        for t, (scaler, lr) in spectral_models.items():
            clean_spectral_eval[t] = spectral_score(scaler, lr, X_eval)

    # ═══════════════════════════════════════════════════════════════════════
    # SID: attack-agnostic threshold on clean thresh split.
    # ═══════════════════════════════════════════════════════════════════════
    sid_thresholds = {}
    if 'sid' in methods:
        print(f"\n[SID] Calibrating threshold on {len(X_thresh)} clean images...")
        clean_sid = np.array([
            compute_sid_score(norm_model, torch.from_numpy(X_thresh[j]).unsqueeze(0),
                              device, keep_frac=sid_keep)
            for j in tqdm(range(len(X_thresh)), desc="  SID thresh")
        ])
        sid_thresholds = {
            tier: float(np.percentile(clean_sid, pct)) for (tier, _f, pct) in _FPR_TIERS
        }
        print(f"  SID thresholds: " + ", ".join(
            f"{t}={sid_thresholds[t]:.4f}" for (t, _f, _p) in _FPR_TIERS))

    # ═══════════════════════════════════════════════════════════════════════
    # Eval per attack
    # ═══════════════════════════════════════════════════════════════════════
    def _tiers_from_scores(scores_adv, scores_clean, thr_map):
        per_tier = {}
        for (tier, fpr_target, _pct) in _FPR_TIERS:
            thr = thr_map[tier]
            det_adv = scores_adv > thr
            det_clean = scores_clean > thr
            tp = int(det_adv.sum()); fn = int((~det_adv).sum())
            fp = int(det_clean.sum()); tn = int((~det_clean).sum())
            n_adv = tp + fn; n_clean = fp + tn
            tpr = tp / max(n_adv, 1); fpr_emp = fp / max(n_clean, 1)
            prec = tp / max(tp + fp, 1)
            f1 = 2 * prec * tpr / max(prec + tpr, 1e-8)
            tpr_ci = wilson_ci(tp, n_adv); fpr_ci = wilson_ci(fp, n_clean)
            per_tier[tier] = {
                'TPR': round(tpr, 4), 'TPR_CI_95': [round(tpr_ci[0], 4), round(tpr_ci[1], 4)],
                'FPR': round(fpr_emp, 4), 'FPR_CI_95': [round(fpr_ci[0], 4), round(fpr_ci[1], 4)],
                'FPR_target': fpr_target / 100.0,
                'Precision': round(prec, 4), 'F1': round(f1, 4),
                'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
                'n_adv': n_adv, 'n_clean': n_clean, 'threshold': round(thr, 6),
            }
        return per_tier

    # spectral_cross_matrix[train_attack][eval_attack] = per_tier dict.
    # Diagonal (train==eval) is the in-distribution result; off-diagonal is
    # cross-attack transfer (the headline collapse showing SpectralDefense is
    # attack-specific while PRISM is attack-agnostic).
    spectral_cross_matrix = {t: {} for t in spectral_models}

    t_start = time.time()
    for atk in attacks_to_run:
        if atk not in all_attacks:
            print(f"Unknown attack {atk}, skipping."); continue
        print(f"\n{'='*60}\nEval attack: {atk}\n{'='*60}")
        atk_obj = all_attacks[atk]()
        print(f"  Generating {len(X_eval)} adversarials...")
        X_eval_adv = atk_obj.generate(X_eval)

        if 'sid' in methods:
            adv_scores = np.array([
                compute_sid_score(norm_model, torch.from_numpy(X_eval_adv[j]).unsqueeze(0),
                                  device, keep_frac=sid_keep)
                for j in tqdm(range(len(X_eval_adv)), desc="  SID adv")
            ])
            clean_scores = np.array([
                compute_sid_score(norm_model, torch.from_numpy(X_eval[j]).unsqueeze(0),
                                  device, keep_frac=sid_keep)
                for j in tqdm(range(len(X_eval)), desc="  SID clean")
            ])
            per_tier = _tiers_from_scores(adv_scores, clean_scores, sid_thresholds)
            results['SID'][atk] = {**per_tier['L1'], 'tiers': per_tier}
            l1 = per_tier['L1']
            print(f"  SID: L1 TPR={l1['TPR']:.4f} FPR={l1['FPR']:.4f} | "
                  f"L2 TPR={per_tier['L2']['TPR']:.4f} | L3 TPR={per_tier['L3']['TPR']:.4f}")

        if 'spectral' in methods:
            # Score this eval attack's adversarials with EVERY train-attack LR.
            train_attacks = list(spectral_models) if spectral_cross else (
                [atk] if atk in spectral_models else [])
            for t in train_attacks:
                scaler, lr = spectral_models[t]
                adv_scores = spectral_score(scaler, lr, X_eval_adv)
                per_tier = _tiers_from_scores(
                    adv_scores, clean_spectral_eval[t], spectral_thresholds[t])
                spectral_cross_matrix[t][atk] = per_tier
                if t == atk:  # diagonal = in-distribution, table-facing detector entry
                    results['SpectralDefense'][atk] = {**per_tier['L1'], 'tiers': per_tier}
                    l1 = per_tier['L1']
                    print(f"  SpectralDefense (train={t}): L1 TPR={l1['TPR']:.4f} "
                          f"FPR={l1['FPR']:.4f} [in-dist]")
                else:
                    print(f"  SpectralDefense (train={t}->eval={atk}): "
                          f"L1 TPR={spectral_cross_matrix[t][atk]['L1']['TPR']:.4f} [cross]")

    elapsed = time.time() - t_start

    # ── Cross-attack summary: in-dist (diagonal) vs cross-attack (off-diagonal mean) ──
    spectral_cross_summary = {}
    if 'spectral' in methods and spectral_cross:
        print(f"\n{'='*70}\nSpectralDefense cross-attack matrix (L1 TPR; rows=train, cols=eval)\n{'-'*70}")
        evals = [a for a in attacks_to_run if a in all_attacks]
        hdr = "train\\eval".ljust(12) + "".join(f"{e:>10}" for e in evals)
        print(hdr)
        for t in spectral_models:
            row = t.ljust(12)
            for e in evals:
                cell = spectral_cross_matrix[t].get(e)
                row += f"{cell['L1']['TPR']:>10.3f}" if cell else f"{'-':>10}"
            print(row)
        for t in spectral_models:
            diag = spectral_cross_matrix[t].get(t, {}).get('L1', {}).get('TPR')
            off = [spectral_cross_matrix[t][e]['L1']['TPR']
                   for e in evals if e != t and e in spectral_cross_matrix[t]]
            spectral_cross_summary[t] = {
                'in_dist_TPR': diag,
                'cross_attack_mean_TPR': round(float(np.mean(off)), 4) if off else None,
                'cross_attack_min_TPR': round(float(np.min(off)), 4) if off else None,
                'n_cross': len(off),
            }
        print(f"{'-'*70}")
        for t, s in spectral_cross_summary.items():
            print(f"  train={t:>6}: in-dist TPR={s['in_dist_TPR']}, "
                  f"cross-attack mean={s['cross_attack_mean_TPR']}, "
                  f"min={s['cross_attack_min_TPR']}")

    # ── Summary ──
    print(f"\n{'='*70}\n{'Detector':>16} {'Attack':>8} {'TPR':>8} {'FPR':>8} {'F1':>8}\n{'-'*70}")
    for m in methods:
        d = _METHOD_DISPLAY[m]
        for atk in attacks_to_run:
            if atk in results[d]:
                r = results[d][atk]
                print(f"{d:>16} {atk:>8} {r['TPR']:>8.4f} {r['FPR']:>8.4f} {r['F1']:>8.4f}")

    refs = {
        'SID': 'Tian et al., 2021. Detecting Adversarial Examples from Sensitivity '
               'Inconsistency of Spatial-Transform Domain. AAAI 2021. '
               '(no-training DCT sensitivity-inconsistency variant)',
        'SpectralDefense': 'Harder et al., 2021. SpectralDefense: Detecting Adversarial '
                           'Attacks on CNNs in the Fourier Domain. IJCNN 2021. '
                           '(InputMFS: log-magnitude FFT + logistic regression, '
                           'supervised per-attack)',
    }
    results['_meta'] = {
        'n_test': n_test, 'n_actual': int(len(sample_idx)),
        'dataset': DATASET,
        'eval_split': f'{DATASET.upper()} test idx {EVAL_IDX[0]}-{EVAL_IDX[1]-1}',
        'ref_split': f'{DATASET.upper()} test idx {cal_start}-{cal_mid-1}',
        'thresh_split': f'{DATASET.upper()} test idx {cal_mid}-{cal_end-1}',
        'seed': seed, 'device': str(device), 'attacks': attacks_to_run,
        'methods': methods, 'eps': round(eps, 6),
        'sid_keep_frac': sid_keep, 'n_ref_spectral': n_ref_spectral,
        'fpr_tiers': [{'name': n, 'target_fpr': f / 100.0, 'percentile': p}
                      for (n, f, p) in _FPR_TIERS],
        'sid_thresholds': {t: round(v, 6) for t, v in sid_thresholds.items()},
        'spectral_thresholds': {a: {t: round(v, 6) for t, v in d.items()}
                                for a, d in spectral_thresholds.items()},
        'protocol_notes': {
            'SID': 'unsupervised, clean-only threshold calibration (like LID/Maha/ODIN/Energy)',
            'SpectralDefense': 'SUPERVISED, attack-specific LR trained on ref-split '
                               '(clean,adv) pairs; sees more info than PRISM (conservative)',
        },
        'spectral_cross_attack_enabled': bool(spectral_cross and 'spectral' in methods),
        'spectral_cross_summary': spectral_cross_summary,
        'elapsed_s': round(elapsed, 1),
        'references': {_METHOD_DISPLAY[m]: refs[_METHOD_DISPLAY[m]] for m in methods},
    }

    # Full cross-attack matrix (underscore-prefixed so aggregate_baselines.py skips it).
    if 'spectral' in methods and spectral_cross:
        results['_spectral_cross_attack'] = spectral_cross_matrix

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved -> {output_path}  ({elapsed:.1f}s)")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Recent detector baselines: SID + SpectralDefense")
    parser.add_argument('--config', default=None,
                        help='YAML config path (routes via PRISM_CONFIG env var).')
    parser.add_argument('--n-test', type=int, default=1000)
    parser.add_argument('--attacks', nargs='+', default=['FGSM', 'PGD', 'Square'])
    parser.add_argument('--methods', nargs='+', default=_DEFAULT_METHODS,
                        choices=list(_METHOD_DISPLAY))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sid-keep', type=float, default=0.70,
                        help='DCT low-pass keep fraction per axis (default 0.70; '
                             'selected on a held-out PGD probe, AUROC plateau 0.65-0.75).')
    parser.add_argument('--n-ref-spectral', type=int, default=1000,
                        help='Ref images for SpectralDefense LR training.')
    parser.add_argument('--no-cross', action='store_true',
                        help='Disable SpectralDefense cross-attack matrix (diagonal only).')
    parser.add_argument('--smoke', action='store_true',
                        help='Fast local sanity run: n_test=64, FGSM+PGD, ref=128.')
    parser.add_argument('--output', default='experiments/evaluation/results_baselines_recent.json')
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    if args.smoke:
        args.n_test = min(args.n_test, 64)
        args.attacks = ['FGSM', 'PGD']
        args.n_ref_spectral = min(args.n_ref_spectral, 128)
        args.output = 'experiments/evaluation/results_baselines_recent_smoke.json'
        print("=== SMOKE TEST MODE: n_test=64, FGSM+PGD, ref=128 ===\n")

    run_recent_baselines(
        n_test=args.n_test,
        attacks_to_run=args.attacks,
        methods=args.methods,
        seed=args.seed,
        output_path=args.output,
        device_str=args.device,
        sid_keep=args.sid_keep,
        n_ref_spectral=args.n_ref_spectral,
        spectral_cross=not args.no_cross,
    )
