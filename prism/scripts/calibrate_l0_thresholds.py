"""
P0.4 L0 Threshold Calibration

`CampaignMonitor` defaults were tuned on synthetic prior-predictive streams,
not on the trained scorer's actual output. This script grid-searches a
centered-score CUSUM detector on score streams produced by the real ensemble
scorer, then writes the best cell to `models/l0_thresholds.pkl`.

Selection rule:
  argmax  ASR_gap_pp   subject to   clean_only_L0_active_frac <= 0.01

If no grid cell is feasible, the script aborts with a non-zero exit code
so `run_vastai_full.sh` Step 6b fails fast instead of launching Step 7a
with uncalibrated thresholds.

The loaded CampaignMonitor is given each cell via `thresholds_path`
(see src/sacd/monitor.py::_load_thresholds), so no other call-sites need to
change once this pkl exists.

USAGE
-----
  cd prism/
  python scripts/calibrate_l0_thresholds.py \
         [--n-clean 500] [--n-adv 500] [--output models/l0_thresholds.pkl]

OUTPUT
------
  models/l0_thresholds.pkl:
    {'detection_mode': 'cusum', 'score_center': float, 'score_scale': float,
     'cusum_drift': float, 'cusum_threshold': float, 'warmup_steps': int,
     'l0_factor': float,
     'calibration_metrics': {...grid details, selected cell...}}
"""
import os
import sys
import pickle
import argparse
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Route --config CLI flag to PRISM_CONFIG env var BEFORE importing src.config.
from src import bootstrap  # noqa: F401

from src.tamm.extractor import ActivationExtractor
from src.tamm.tda import TopologicalProfiler
from src.tamm.scorer import TopologicalScorer
from src.cadg.ensemble_scorer import PersistenceEnsembleScorer
from src.cadg.calibrate import ConformalCalibrator
from src.sacd.monitor import CampaignMonitor
from src.config import (
    LAYER_NAMES, LAYER_WEIGHTS, DIM_WEIGHTS,
    BACKBONE_MEAN, BACKBONE_STD, BACKBONE_INPUT_SIZE, BACKBONE_NUM_CLASSES,
    EPS_LINF_STANDARD,
    CAL_IDX, VAL_IDX, EVAL_IDX,
    N_SUBSAMPLE, MAX_DIM, DATASET, PATHS,
)
from src.data_loader import load_test_dataset
from src.models import load_backbone

try:
    from art.attacks.evasion import ProjectedGradientDescent
    from art.estimators.classification import PyTorchClassifier
except ImportError:
    print("ERROR: ART not installed. pip install adversarial-robustness-toolbox")
    sys.exit(1)


if BACKBONE_INPUT_SIZE == 32:
    _PIXEL_TRANSFORM = T.Compose([T.ToTensor()])
else:
    _PIXEL_TRANSFORM = T.Compose([T.Resize(BACKBONE_INPUT_SIZE), T.ToTensor()])
_NORMALIZE       = T.Normalize(mean=BACKBONE_MEAN, std=BACKBONE_STD)


# Backward-compat alias — _NormalizedBackbone in src.models is the same wrapper.
from src.models import _NormalizedBackbone as _NormalizedResNet


def _compute_score_stream(
    imgs_pixel, profiler, extractor, ensemble, model, device, label,
):
    """Return (N,) scorer outputs matching PRISM.defend's runtime inputs."""
    scores = np.empty(len(imgs_pixel), dtype=np.float32)
    use_dct = getattr(ensemble, 'use_dct', False)
    use_grad_norm = getattr(ensemble, 'use_grad_norm', False)
    use_softmax_entropy = getattr(ensemble, 'use_softmax_entropy', False)
    for i, img_pixel in enumerate(tqdm(imgs_pixel, desc=f"  score[{label}]")):
        x_norm = _NORMALIZE(img_pixel).unsqueeze(0).to(device)
        acts = extractor.extract(x_norm)
        dgms = {
                L: profiler.compute_diagram(acts[L].squeeze(0).cpu().numpy())
                for L in LAYER_NAMES
        }
        img_np = img_pixel.detach().cpu().numpy() if use_dct else None

        grad_norm = None
        if use_grad_norm:
            x_g = x_norm.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                logits_g = model(x_g)
                pred_idx = int(logits_g.argmax(1).item())
                (grad_x,) = torch.autograd.grad(logits_g[0, pred_idx], x_g)
            grad_norm = float(grad_x.norm().item())

        logits_np = None
        if use_softmax_entropy:
            with torch.no_grad():
                logits = model(x_norm)
            logits_np = logits.squeeze(0).detach().cpu().numpy()

        scores[i] = ensemble.score(
            dgms,
            image=img_np,
            grad_norm=grad_norm,
            logits=logits_np,
        )
    return scores


def _simulate_stream(scores, tier_thresh_L3, cell, l0_factor=0.8):
    """
    Run a single stream of scores through a fresh CampaignMonitor at this cell.
    Return (n_pass_l0_on, n_pass_l0_off, l0_active_frac).

    A score "passes" (adversary wins) iff score < tier_thresh_L3; when L0 is
    active on that step, the threshold tightens to tier_thresh_L3 * l0_factor.
    """
    monitor = CampaignMonitor(
        hazard_rate=cell.get('hazard_rate', 1 / 30),
        alert_run_prob=cell.get('alert_run_prob', 0.60),
        warmup_steps=cell['warmup_steps'],
        l0_factor=l0_factor,
        detection_mode=cell.get('detection_mode', 'bocpd'),
        score_center=cell.get('score_center', 0.0),
        score_scale=cell.get('score_scale', 1.0),
        cusum_drift=cell.get('cusum_drift', 0.25),
        cusum_threshold=cell.get('cusum_threshold'),
    )
    n_pass_on = 0
    n_pass_off = 0
    n_l0_active = 0
    for s in scores:
        state = monitor.process_score(float(s))
        l0_active = bool(state['l0_active'])
        n_l0_active += int(l0_active)
        thresh_eff = tier_thresh_L3 * (l0_factor if l0_active else 1.0)
        if s < tier_thresh_L3:
            n_pass_off += 1
        if s < thresh_eff:
            n_pass_on += 1
    return n_pass_on, n_pass_off, n_l0_active / max(len(scores), 1)


def calibrate_l0_thresholds(
    n_clean: int = 500,
    n_adv: int = 500,
    data_root: str = './data',
    output_path: str = None,
    dry_run: bool = False,
    seed: int = 42,
):
    output_path = output_path or os.path.join(
        os.path.dirname(PATHS['calibrator']), 'l0_thresholds.pkl'
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  Dataset: {DATASET}")
    print(f"Output: {output_path}")

    # ── Load frozen artefacts ────────────────────────────────────────────────
    with open(PATHS['reference_profiles'], 'rb') as f:
        ref_profiles = pickle.load(f)
    with open(PATHS['calibrator'], 'rb') as f:
        calibrator: ConformalCalibrator = pickle.load(f)
    tier_thresh_L3 = float(calibrator.thresholds['L3'])
    print(f"Loaded calibrator. L3 threshold = {tier_thresh_L3:.6f}")

    base_scorer = TopologicalScorer(
        ref_profiles=ref_profiles,
        layer_names=LAYER_NAMES,
        layer_weights=LAYER_WEIGHTS,
        dim_weights=DIM_WEIGHTS,
    )
    ensemble = PersistenceEnsembleScorer.load(
        PATHS['ensemble_scorer'], base_scorer, LAYER_NAMES,
    )
    print(f"Loaded ensemble scorer (α={ensemble.alpha:.3f}, "
          f"use_tda={ensemble.use_tda}).")

    # ── Model + attack ───────────────────────────────────────────────────────
    # Same active CIFAR-trained backbone used by the rest of the pipeline.
    model = load_backbone(device)
    extractor = ActivationExtractor(model, LAYER_NAMES)
    profiler  = TopologicalProfiler(n_subsample=N_SUBSAMPLE, max_dim=MAX_DIM)
    norm_model = load_backbone(device, wrap=True)

    ds = load_test_dataset(root=data_root, download=True, transform=_PIXEL_TRANSFORM)
    rng = np.random.RandomState(seed)
    val_pool = list(range(*VAL_IDX))
    # Use VAL split so we never touch the EVAL split used by campaign/recovery.
    # VAL is also disjoint from CAL, which the calibrator was fitted on.
    chosen = rng.choice(val_pool, min(n_clean + n_adv, len(val_pool)), replace=False)
    clean_idx = chosen[:n_clean]
    adv_src_idx = chosen[n_clean: n_clean + n_adv]

    print(f"Collecting {n_clean} clean images + {n_adv} PGD-40 adversarials "
          f"from VAL split ({VAL_IDX[0]}..{VAL_IDX[1]-1})...")
    clean_pixel = [ds[int(i)][0] for i in clean_idx]
    adv_src_pixel = [ds[int(i)][0] for i in adv_src_idx]

    # PGD-40 on the adv source pool (matches run_campaign_eval.py).
    device_type = 'gpu' if device.type == 'cuda' else 'cpu'
    classifier = PyTorchClassifier(
        model=norm_model, loss=torch.nn.CrossEntropyLoss(),
        input_shape=(3, BACKBONE_INPUT_SIZE, BACKBONE_INPUT_SIZE), nb_classes=BACKBONE_NUM_CLASSES,
        clip_values=(0.0, 1.0), device_type=device_type,
    )
    attack = ProjectedGradientDescent(
        classifier, eps=EPS_LINF_STANDARD,
        eps_step=EPS_LINF_STANDARD / 4, max_iter=40, num_random_init=1,
    )
    X_adv_np = attack.generate(torch.stack(adv_src_pixel).numpy())
    adv_pixel = [torch.tensor(x) for x in X_adv_np]

    if dry_run:
        print("Dry-run: skipping score extraction and grid search.")
        return

    clean_scores = _compute_score_stream(
        clean_pixel, profiler, extractor, ensemble, model, device, label='clean')
    adv_scores = _compute_score_stream(
        adv_pixel, profiler, extractor, ensemble, model, device, label='adv')
    extractor.cleanup()

    print(f"clean_scores: mean={clean_scores.mean():.4f} std={clean_scores.std():.4f} "
          f"max={clean_scores.max():.4f}")
    print(f"adv_scores:   mean={adv_scores.mean():.4f} std={adv_scores.std():.4f} "
          f"max={adv_scores.max():.4f}")
    score_center = float(np.median(clean_scores))
    mad = float(np.median(np.abs(clean_scores - score_center)))
    score_scale = max(1.4826 * mad, float(clean_scores.std()) * 0.5, 1e-6)
    print(f"CUSUM clean baseline: center={score_center:.4f} scale={score_scale:.4f}")

    # ── Grid search ──────────────────────────────────────────────────────────
    # Sustained stream layout: n_clean_prefix clean scores (to let BOCPD
    # accumulate a long run) followed by n_adv adversarial scores. Matches
    # the `sustained_rho100` scenario in run_campaign_eval.py most closely.
    n_clean_prefix = min(60, len(clean_scores) // 3)  # ≥ warmup_steps
    sustained_stream = np.concatenate(
        [clean_scores[:n_clean_prefix], adv_scores]
    )

    # Recovery calibration: use a CUSUM over centered scorer output rather than
    # an L3-only score stream. This lets L0 react to sustained low-grade score
    # drift even when single PGD/AA examples rarely exceed the L3 threshold.
    grid = []
    for drift in (0.00, 0.10, 0.25, 0.50, 0.75):
        for threshold in (5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0):
            for ws in (20, 35, 50, 75, 100):
                grid.append({
                    'detection_mode': 'cusum',
                    'score_center': score_center,
                    'score_scale': score_scale,
                    'cusum_drift': drift,
                    'cusum_threshold': threshold,
                    'warmup_steps': ws,
                })

    FPR_BUDGET = 0.01
    results = []
    best = None
    print(f"\nGrid search: {len(grid)} cells  FPR budget <= {FPR_BUDGET:.2%}")
    for cell in grid:
        # Clean stream → false-alarm probe
        _, _, clean_l0_frac = _simulate_stream(
            clean_scores, tier_thresh_L3, cell,
        )
        # Sustained stream → ASR gap (l0_off vs l0_on, adv portion only)
        n_pass_on, n_pass_off, _ = _simulate_stream(
            sustained_stream, tier_thresh_L3, cell,
        )
        # ASR is measured over the adversarial tail of the sustained stream.
        # Clean prefix passes both conditions identically so it cancels.
        n_adv_in_stream = len(adv_scores)
        # Restrict to the adversarial suffix:
        adv_off_pass, adv_on_pass = 0, 0
        monitor = CampaignMonitor(
            hazard_rate=cell.get('hazard_rate', 1 / 30),
            alert_run_prob=cell.get('alert_run_prob', 0.60),
            warmup_steps=cell['warmup_steps'],
            detection_mode=cell.get('detection_mode', 'bocpd'),
            score_center=cell.get('score_center', 0.0),
            score_scale=cell.get('score_scale', 1.0),
            cusum_drift=cell.get('cusum_drift', 0.25),
            cusum_threshold=cell.get('cusum_threshold'),
        )
        for i, s in enumerate(sustained_stream):
            state = monitor.process_score(float(s))
            if i < n_clean_prefix:
                continue
            l0_active = bool(state['l0_active'])
            thresh_eff = tier_thresh_L3 * (0.8 if l0_active else 1.0)
            if s < tier_thresh_L3:
                adv_off_pass += 1
            if s < thresh_eff:
                adv_on_pass += 1

        asr_off = adv_off_pass / max(n_adv_in_stream, 1)
        asr_on  = adv_on_pass  / max(n_adv_in_stream, 1)
        asr_gap_pp = 100.0 * (asr_off - asr_on)

        row = {
            **cell,
            'clean_l0_active_frac': round(float(clean_l0_frac), 4),
            'asr_off': round(float(asr_off), 4),
            'asr_on':  round(float(asr_on),  4),
            'asr_gap_pp': round(float(asr_gap_pp), 2),
            'feasible': clean_l0_frac <= FPR_BUDGET,
        }
        results.append(row)
        if row['feasible'] and (best is None or row['asr_gap_pp'] > best['asr_gap_pp']):
            best = row

    results.sort(key=lambda r: (-r['feasible'], -r['asr_gap_pp']))
    print("\nTop 8 cells:")
    print(f"  {'feas':>5} {'mode':>6} {'drift':>6} {'thr':>6} {'ws':>4} {'clean_fp':>9} "
          f"{'asr_off':>8} {'asr_on':>8} {'gap_pp':>7}")
    for r in results[:8]:
        print(f"  {str(r['feasible']):>5} {r.get('detection_mode', 'bocpd'):>6} "
              f"{r.get('cusum_drift', 0.0):>6.2f} "
              f"{r.get('cusum_threshold', 0.0):>6.1f} {r['warmup_steps']:>4d} "
              f"{r['clean_l0_active_frac']:>9.4f} "
              f"{r['asr_off']:>8.4f} {r['asr_on']:>8.4f} {r['asr_gap_pp']:>7.2f}")

    if best is None:
        print(f"\n[FAIL] No grid cell meets the {FPR_BUDGET:.2%} clean-FPR budget.")
        print("P0.4 CALIBRATION FAILED. Aborting without writing pkl.")
        sys.exit(1)

    print(f"\n[OK] Selected cell: mode={best.get('detection_mode', 'bocpd')}, "
          f"cusum_drift={best.get('cusum_drift', 0.0):.2f}, "
          f"cusum_threshold={best.get('cusum_threshold', 0.0):.1f}, "
          f"warmup_steps={best['warmup_steps']}  "
          f"(clean FPR={best['clean_l0_active_frac']:.4f}, "
          f"ASR gap={best['asr_gap_pp']:.2f}pp).")

    payload = {
        'detection_mode':   best.get('detection_mode', 'cusum'),
        'score_center':     best.get('score_center', score_center),
        'score_scale':      best.get('score_scale', score_scale),
        'cusum_drift':      best.get('cusum_drift', 0.25),
        'cusum_threshold':  best.get('cusum_threshold'),
        'hazard_rate':      best.get('hazard_rate', 1 / 30),
        'alert_run_prob':   best.get('alert_run_prob', 0.60),
        'warmup_steps':     best['warmup_steps'],
        'alert_run_length': 10,
        'l0_factor':        0.8,
        'calibration_metrics': {
            'dataset':        DATASET,
            'n_clean':        int(n_clean),
            'n_adv':          int(n_adv),
            'n_clean_prefix': int(n_clean_prefix),
            'tier_thresh_L3': tier_thresh_L3,
            'fpr_budget':     FPR_BUDGET,
            'score_center':   score_center,
            'score_scale':    score_scale,
            'selected_cell':  best,
            'top_rows':       results[:16],
        },
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(payload, f)
    print(f"Wrote {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calibrate L0 thresholds (P0.4)")
    parser.add_argument('--config', default=None,
                        help='YAML config path (routes via PRISM_CONFIG env var).')
    parser.add_argument('--n-clean', type=int, default=500)
    parser.add_argument('--n-adv',   type=int, default=500)
    parser.add_argument('--data-root', default='./data')
    parser.add_argument('--output', default=None,
                        help='Output pkl path. Default: next to calibrator.pkl.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only check argparse + data loading; skip attack + grid.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    calibrate_l0_thresholds(
        n_clean=args.n_clean,
        n_adv=args.n_adv,
        data_root=args.data_root,
        output_path=args.output,
        dry_run=args.dry_run,
        seed=args.seed,
    )
