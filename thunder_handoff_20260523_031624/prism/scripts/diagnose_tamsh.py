"""
TAMSH Recovery Failure Diagnostic (P0.5)

Recovery gate fails on all 5 seeds: tamsh − passthrough = 0.5–0.9 pp (target ≥ 15 pp).
Expert selection collapses to expert 0 on ~94% of L3-rejected inputs.

This script isolates the cause without changing any code. It reports:

  1. experts.pkl structure  : per-expert medoid diagram H0/H1 sizes.
  2. Router behavior        : on a small PGD pool, for each input it records
                              H0/H1 sizes, per-expert Wasserstein distances,
                              the argmin, and a 'tied' flag.
  3. Routing statistics     : expert use histogram, routing entropy, tie rate.
  4. Per-expert recovery    : forces each input through each expert and reports
                              per-expert recovery_acc — isolates router vs
                              expert-capacity contributions.

Read-only: does not write to models/ or experiments/. Only writes a JSON
diagnostic to logs/local/tamsh_diagnostic.json.

USAGE
  cd prism/
  python scripts/diagnose_tamsh.py --n-test 200 --seed 42
"""
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import json, os, sys, ssl, certifi, argparse, pickle
from tqdm import tqdm

os.environ.setdefault('SSL_CERT_FILE', certifi.where())
os.environ.setdefault('REQUESTS_CA_BUNDLE', certifi.where())
ssl._create_default_https_context = ssl.create_default_context

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src import bootstrap  # noqa: F401

from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier

from src.prism import PRISM
from src.sacd.monitor import NoOpCampaignMonitor
from src.tamsh.experts import TopologyAwareMoE, ExpertSubNetwork
from src.config import (
    LAYER_NAMES, LAYER_WEIGHTS, DIM_WEIGHTS,
    BACKBONE_MEAN, BACKBONE_STD, BACKBONE_INPUT_SIZE, BACKBONE_NUM_CLASSES,
    EPS_LINF_STANDARD, EVAL_IDX, DATASET, PATHS,
)
from src.data_loader import load_test_dataset
from src.models import load_backbone

_MEAN = BACKBONE_MEAN
_STD = BACKBONE_STD
_PIXEL = T.Compose([T.ToTensor()]) if BACKBONE_INPUT_SIZE == 32 else T.Compose(
    [T.Resize(BACKBONE_INPUT_SIZE), T.ToTensor()])
_NORM = T.Normalize(mean=_MEAN, std=_STD)

try:
    from gudhi.wasserstein import wasserstein_distance as _wass
except ImportError:
    _wass = None


def _safe_wasserstein(a, b):
    if _wass is None:
        return float('nan')
    if len(a) == 0 and len(b) == 0:
        return 0.0
    if len(a) == 0 or len(b) == 0:
        non_empty = a if len(a) > 0 else b
        return float(np.sum(np.abs(non_empty[:, 1] - non_empty[:, 0])))
    return float(_wass(a, b, order=2))


def _diag_shape(dgm):
    """Return [n_points, n_finite, n_infinite] for a single diagram array."""
    if len(dgm) == 0:
        return [0, 0, 0]
    arr = np.asarray(dgm, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return [int(len(arr)), 0, 0]
    lifetimes = arr[:, 1] - arr[:, 0]
    n_finite = int(np.sum(np.isfinite(lifetimes)))
    n_infinite = int(np.sum(~np.isfinite(lifetimes)))
    return [int(arr.shape[0]), n_finite, n_infinite]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-test', type=int, default=200)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default=None)
    ap.add_argument('--output', default='logs/local/tamsh_diagnostic.json')
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    # ── (1) Inspect experts.pkl structure ──
    print("\n=== (1) experts.pkl structure ===")
    with open(PATHS['experts'], 'rb') as f:
        data = pickle.load(f)
    n_experts = len(data['experts'])
    medoids = data['medoid_diagrams']
    expert_names = data.get('expert_names', [f"expert{i}" for i in range(n_experts)])
    attack_mixes = data.get('attack_mixes', [['?']] * n_experts)
    print(f"n_experts            = {n_experts}")
    print(f"input_dim            = {data['input_dim']}")
    print(f"output_dim           = {data['output_dim']}")
    print(f"hidden_dim           = {data.get('hidden_dim', '?')}")
    print(f"activation_layer     = {data.get('activation_layer', '?')}")
    print(f"target_source        = {data.get('target_source', '?')}")

    medoid_summary = []
    for i, (name, mix, dgm_set) in enumerate(zip(expert_names, attack_mixes, medoids)):
        h0 = _diag_shape(dgm_set[0]) if len(dgm_set) > 0 else [0, 0, 0]
        h1 = _diag_shape(dgm_set[1]) if len(dgm_set) > 1 else [0, 0, 0]
        print(f"  expert {i} [{name}]  mix={mix}")
        print(f"    H0 = {h0[0]} pts ({h0[1]} finite, {h0[2]} inf)")
        print(f"    H1 = {h1[0]} pts ({h1[1]} finite, {h1[2]} inf)")
        medoid_summary.append({
            'idx': i, 'name': name, 'attack_mix': mix,
            'H0_shape': h0, 'H1_shape': h1,
        })

    # ── Pairwise distance between medoids (router can only distinguish what's distinct) ──
    print("\n  Pairwise H1 Wasserstein between medoids:")
    pairwise_h1 = np.zeros((n_experts, n_experts))
    for i in range(n_experts):
        for j in range(i + 1, n_experts):
            d = _safe_wasserstein(
                medoids[i][1] if len(medoids[i]) > 1 else np.zeros((0, 2)),
                medoids[j][1] if len(medoids[j]) > 1 else np.zeros((0, 2)),
            )
            pairwise_h1[i, j] = d
            pairwise_h1[j, i] = d
            print(f"    d(e{i}, e{j}) H1 = {d:.6f}")

    print("\n  Pairwise H0 Wasserstein between medoids:")
    pairwise_h0 = np.zeros((n_experts, n_experts))
    for i in range(n_experts):
        for j in range(i + 1, n_experts):
            d = _safe_wasserstein(
                medoids[i][0] if len(medoids[i]) > 0 else np.zeros((0, 2)),
                medoids[j][0] if len(medoids[j]) > 0 else np.zeros((0, 2)),
            )
            pairwise_h0[i, j] = d
            pairwise_h0[j, i] = d
            print(f"    d(e{i}, e{j}) H0 = {d:.6f}")

    # ── (2) Run a small PGD recovery cycle, capture routing trace ──
    print(f"\n=== (2) Routing trace on {args.n_test} PGD-{EPS_LINF_STANDARD:.4f} samples ===")
    model = load_backbone(device)
    norm_model = load_backbone(device, wrap=True)

    # Rebuild MoE
    rebuilt = []
    for sd in data['experts']:
        net = ExpertSubNetwork(
            input_dim=data['input_dim'],
            output_dim=data['output_dim'],
            hidden_dim=data.get('hidden_dim', 256),
        )
        net.load_state_dict(sd)
        net.eval().to(device)
        rebuilt.append(net)
    moe = TopologyAwareMoE(experts=rebuilt, expert_ref_diagrams=medoids)

    # Build PGD adversarials on EVAL split
    classifier = PyTorchClassifier(
        model=norm_model, loss=torch.nn.CrossEntropyLoss(),
        input_shape=(3, BACKBONE_INPUT_SIZE, BACKBONE_INPUT_SIZE),
        nb_classes=BACKBONE_NUM_CLASSES,
        clip_values=(0.0, 1.0),
        device_type='gpu' if device.type == 'cuda' else 'cpu',
    )
    attack = ProjectedGradientDescent(
        classifier, eps=EPS_LINF_STANDARD, eps_step=EPS_LINF_STANDARD / 4,
        max_iter=40, num_random_init=1,
    )

    ds = load_test_dataset(root='./data', download=True, transform=_PIXEL)
    eval_indices = list(range(*EVAL_IDX))
    sample_idx = rng.choice(eval_indices, min(args.n_test, len(eval_indices)),
                            replace=False)
    pixel_imgs, labels = [], []
    for i in sample_idx:
        img, y = ds[int(i)]
        pixel_imgs.append(img)
        labels.append(int(y))
    X_pixel_np = torch.stack(pixel_imgs).numpy()
    print(f"  generating PGD adversarials...")
    X_adv_np = attack.generate(X_pixel_np)

    prism = PRISM.from_saved(
        model=model,
        layer_names=LAYER_NAMES,
        calibrator_path=PATHS['calibrator'],
        profile_path=PATHS['reference_profiles'],
        ensemble_path=PATHS['ensemble_scorer'],
        layer_weights=LAYER_WEIGHTS,
        dim_weights=DIM_WEIGHTS,
        campaign_monitor=NoOpCampaignMonitor(),
        moe=moe,
    )

    # Identify L3-rejected, capture routing trace
    last_layer = LAYER_NAMES[-1]
    routing_trace = []
    l3_indices = []
    print(f"  triage + routing trace...")
    for j in tqdm(range(len(pixel_imgs))):
        x_clean = pixel_imgs[j].unsqueeze(0).to(device)
        x_adv_pixel = torch.tensor(X_adv_np[j]).unsqueeze(0).to(device)
        x_adv_norm = _NORM(x_adv_pixel.squeeze(0).cpu()).unsqueeze(0).to(device)

        _, level, _ = prism.defend(x_adv_norm)
        if level not in ('L3', 'L3_REJECT'):
            continue

        # Re-extract for routing inspection
        acts = prism.extractor.extract(x_adv_norm)
        diagrams = {
            L: prism.profiler.compute_diagram(acts[L].squeeze(0).cpu().numpy())
            for L in LAYER_NAMES
        }
        in_dgm = diagrams[last_layer]
        in_h0_shape = _diag_shape(in_dgm[0]) if len(in_dgm) > 0 else [0, 0, 0]
        in_h1_shape = _diag_shape(in_dgm[1]) if len(in_dgm) > 1 else [0, 0, 0]

        # Per-expert distances at H1 (router's dim) and H0 (alternate)
        d_h1 = []
        d_h0 = []
        for ref_set in medoids:
            ref_h1 = ref_set[1] if len(ref_set) > 1 else np.zeros((0, 2))
            ref_h0 = ref_set[0] if len(ref_set) > 0 else np.zeros((0, 2))
            in_h1 = in_dgm[1] if len(in_dgm) > 1 else np.zeros((0, 2))
            in_h0 = in_dgm[0] if len(in_dgm) > 0 else np.zeros((0, 2))
            d_h1.append(_safe_wasserstein(in_h1, ref_h1))
            d_h0.append(_safe_wasserstein(in_h0, ref_h0))

        argmin_h1 = int(np.argmin(d_h1))
        argmin_h0 = int(np.argmin(d_h0))
        tied_h1 = sum(1 for x in d_h1 if abs(x - d_h1[argmin_h1]) < 1e-9) > 1
        tied_h0 = sum(1 for x in d_h0 if abs(x - d_h0[argmin_h0]) < 1e-9) > 1

        # Force-run every expert; record per-expert prediction correctness
        a = acts[last_layer]
        if a.dim() > 2:
            a_flat = F.adaptive_avg_pool2d(a, 1).view(a.size(0), -1)
        else:
            a_flat = a
        a_flat = a_flat.to(device)
        per_expert_pred = []
        for k in range(n_experts):
            with torch.no_grad():
                logits = rebuilt[k](a_flat)
            per_expert_pred.append(int(logits.argmax(1).item()))

        # Clean argmax through backbone
        with torch.no_grad():
            clean_logits = norm_model(x_clean)
        clean_argmax = int(clean_logits.argmax(1).item())
        with torch.no_grad():
            adv_logits = norm_model(x_adv_pixel)
        adv_argmax = int(adv_logits.argmax(1).item())

        true_label = int(labels[j])
        routing_trace.append({
            'j': int(j),
            'true_label': true_label,
            'clean_argmax': clean_argmax,
            'adv_argmax': adv_argmax,
            'clean_correct': clean_argmax == true_label,
            'in_H0_shape': in_h0_shape,
            'in_H1_shape': in_h1_shape,
            'd_H1': [round(x, 6) for x in d_h1],
            'd_H0': [round(x, 6) for x in d_h0],
            'router_pick_H1': argmin_h1,
            'router_pick_H0': argmin_h0,
            'tied_H1': bool(tied_h1),
            'tied_H0': bool(tied_h0),
            'per_expert_pred': per_expert_pred,
            'per_expert_correct': [int(p == true_label) for p in per_expert_pred],
        })
        l3_indices.append(int(j))

    print(f"  L3-triggered: {len(routing_trace)}/{args.n_test}")

    # ── (3) Routing statistics ──
    print("\n=== (3) Routing statistics ===")
    if len(routing_trace) > 0:
        picks_h1 = [t['router_pick_H1'] for t in routing_trace]
        picks_h0 = [t['router_pick_H0'] for t in routing_trace]
        ties_h1 = sum(t['tied_H1'] for t in routing_trace)
        ties_h0 = sum(t['tied_H0'] for t in routing_trace)
        empty_h1 = sum(t['in_H1_shape'][0] == 0 for t in routing_trace)
        empty_h0 = sum(t['in_H0_shape'][0] == 0 for t in routing_trace)

        def _entropy(counts, k):
            total = sum(counts)
            if total == 0:
                return 0.0
            p = np.array([c / total for c in counts])
            p = p[p > 0]
            return float(-np.sum(p * np.log2(p)))

        hist_h1 = [picks_h1.count(k) for k in range(n_experts)]
        hist_h0 = [picks_h0.count(k) for k in range(n_experts)]
        ent_h1 = _entropy(hist_h1, n_experts)
        ent_h0 = _entropy(hist_h0, n_experts)
        max_ent = np.log2(n_experts)

        print(f"  routing picks (H1, current default):  {hist_h1}")
        print(f"  routing picks (H0, alternate):        {hist_h0}")
        print(f"  routing entropy H1: {ent_h1:.3f} / max {max_ent:.3f} bits")
        print(f"  routing entropy H0: {ent_h0:.3f} / max {max_ent:.3f} bits")
        print(f"  inputs with empty H1: {empty_h1}/{len(routing_trace)} ({100*empty_h1/len(routing_trace):.1f}%)")
        print(f"  inputs with empty H0: {empty_h0}/{len(routing_trace)} ({100*empty_h0/len(routing_trace):.1f}%)")
        print(f"  tied routing (H1): {ties_h1}/{len(routing_trace)} ({100*ties_h1/len(routing_trace):.1f}%)")
        print(f"  tied routing (H0): {ties_h0}/{len(routing_trace)} ({100*ties_h0/len(routing_trace):.1f}%)")

        # Per-expert capacity: if oracle routed to best expert, what's the cap?
        per_expert_acc = np.zeros(n_experts)
        for k in range(n_experts):
            per_expert_acc[k] = np.mean([t['per_expert_correct'][k] for t in routing_trace])
        oracle_acc = np.mean([max(t['per_expert_correct']) for t in routing_trace])
        current_router_acc = np.mean(
            [t['per_expert_correct'][t['router_pick_H1']] for t in routing_trace]
        )
        h0_router_acc = np.mean(
            [t['per_expert_correct'][t['router_pick_H0']] for t in routing_trace]
        )
        avg_acc = float(np.mean(per_expert_acc))

        print(f"\n  per-expert recovery_acc (force-route to each):")
        for k in range(n_experts):
            print(f"    expert {k}: {per_expert_acc[k]:.4f}")
        print(f"  AVERAGE expert recovery_acc:           {avg_acc:.4f}")
        print(f"  ORACLE (best-of-K per-input):          {oracle_acc:.4f}  <- ceiling")
        print(f"  current H1 router recovery_acc:        {current_router_acc:.4f}")
        print(f"  H0 router recovery_acc (alternate):    {h0_router_acc:.4f}")

        # Diagnosis
        print("\n=== DIAGNOSIS ===")
        if empty_h1 / len(routing_trace) > 0.5:
            print(f"  PRIMARY: input H1 diagrams empty in {100*empty_h1/len(routing_trace):.0f}% of cases → router falls through to argmin of ties → expert 0 selected.")
        if ties_h1 / len(routing_trace) > 0.5:
            print(f"  PRIMARY: H1 distances tie in {100*ties_h1/len(routing_trace):.0f}% of cases → router non-discriminative.")
        if oracle_acc <= 0.15:
            print(f"  PRIMARY: ORACLE recovery_acc only {oracle_acc:.3f} — no expert exists that recovers the label. Expert capacity is the ceiling. Router fixes won't help.")
        elif oracle_acc > 0.20 and current_router_acc < oracle_acc * 0.5:
            print(f"  PRIMARY: oracle {oracle_acc:.3f} vs current {current_router_acc:.3f} — router is the bottleneck. Fixable.")
        else:
            print(f"  PARTIAL: oracle {oracle_acc:.3f}, current {current_router_acc:.3f} — limited headroom even with perfect routing.")
    else:
        hist_h1 = hist_h0 = []
        ent_h1 = ent_h0 = 0.0
        per_expert_acc = np.zeros(n_experts)
        oracle_acc = current_router_acc = h0_router_acc = avg_acc = 0.0
        ties_h1 = ties_h0 = empty_h1 = empty_h0 = 0
        print("  No L3-triggered samples in this pool.")

    # ── Save diagnostic JSON ──
    out = {
        'n_test': int(args.n_test),
        'seed': int(args.seed),
        'n_l3_triggered': int(len(routing_trace)),
        'experts': medoid_summary,
        'pairwise_H1_distance': pairwise_h1.tolist(),
        'pairwise_H0_distance': pairwise_h0.tolist(),
        'routing_stats': {
            'hist_H1': hist_h1,
            'hist_H0': hist_h0,
            'entropy_H1_bits': float(ent_h1),
            'entropy_H0_bits': float(ent_h0),
            'max_entropy_bits': float(np.log2(max(n_experts, 1))),
            'empty_input_H1_count': int(empty_h1),
            'empty_input_H0_count': int(empty_h0),
            'tied_H1_count': int(ties_h1),
            'tied_H0_count': int(ties_h0),
        },
        'per_expert_acc': per_expert_acc.tolist() if len(routing_trace) > 0 else [],
        'oracle_acc': float(oracle_acc) if len(routing_trace) > 0 else 0.0,
        'current_router_acc_H1': float(current_router_acc) if len(routing_trace) > 0 else 0.0,
        'alternate_router_acc_H0': float(h0_router_acc) if len(routing_trace) > 0 else 0.0,
        'avg_per_expert_acc': float(avg_acc) if len(routing_trace) > 0 else 0.0,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"\nDiagnostic JSON → {args.output}")


if __name__ == '__main__':
    main()
