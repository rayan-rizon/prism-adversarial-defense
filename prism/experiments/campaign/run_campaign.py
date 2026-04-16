"""
Real PRISM Campaign Detection Experiment
Uses actual PRISM inference to generate anomaly scores.
Scenario: 50 clean CIFAR-10 images, then 100 FGSM adversarials.
Measures: step at which SACD/L0 fires (campaign detection latency).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import json

from torchvision.models import ResNet18_Weights
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier

from src.prism import PRISM
from src.tamsh.experts import TopologyAwareMoE
from src.sacd.monitor import CampaignMonitor

# ── Transforms ───────────────────────────────────────────────────────────────
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

class _NormalizedResNet(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.register_buffer('mean', torch.tensor(_MEAN).view(1,3,1,1))
        self.register_buffer('std',  torch.tensor(_STD).view(1,3,1,1))
    def forward(self, x):
        return self.backbone((x - self.mean) / self.std)

_PIXEL = T.Compose([T.Resize(224), T.ToTensor()])
_NORM  = T.Normalize(_MEAN, _STD)


def run_campaign_experiment(n_clean=50, n_adv=100, eps=0.05, seed=42):
    print(f"=== Real PRISM Campaign Detection ===")
    print(f"  n_clean={n_clean}, n_adv={n_adv}, eps={eps}\n")

    rng = np.random.RandomState(seed)

    # ── Load PRISM ────────────────────────────────────────────────────────────
    backbone = torchvision.models.resnet18(
        weights=ResNet18_Weights.IMAGENET1K_V1
    ).eval()
    wrapped = _NormalizedResNet(backbone).eval()

    prism = PRISM.from_saved(
        model=backbone,
        layer_names=['layer2', 'layer3', 'layer4'],
        layer_weights={'layer2': 0.15, 'layer3': 0.30, 'layer4': 0.55},
        dim_weights=[0.5, 0.5],
        calibrator_path='models/calibrator.pkl',
        profile_path='models/reference_profiles.pkl',
        # Calibrated BOCPD prior for CIFAR-10 clean scores (mean~4.6, std~1.1)
        # alert_run_prob=0.3 → fire when P(run_length≤alert_run_length)>30%
        # hazard_rate=1/30 → prior expects changepoint every ~30 steps
        campaign_monitor=CampaignMonitor(
            mu0=7.0, kappa0=5.0, alpha0=3.0, beta0=15.0,
            hazard_rate=1/30, alert_run_length=10, alert_run_prob=0.60,
            warmup_steps=35, l0_factor=0.8, cooldown_steps=30,
        ),
    )

    # ── Load dataset ──────────────────────────────────────────────────────────
    ds = torchvision.datasets.CIFAR10(
        './data', train=False, download=False, transform=_PIXEL
    )

    # ── ART classifier ────────────────────────────────────────────────────────
    art_clf = PyTorchClassifier(
        model=wrapped,
        loss=torch.nn.CrossEntropyLoss(),
        input_shape=(3, 224, 224),
        nb_classes=10,
        clip_values=(0.0, 1.0),
    )
    fgsm = FastGradientMethod(art_clf, eps=eps)

    # ── Build query stream ────────────────────────────────────────────────────
    # Fresh PRISM monitor (reset L0 state)
    prism.monitor.reset()

    indices = rng.choice(len(ds), n_clean + n_adv, replace=False)
    clean_idx = indices[:n_clean]
    adv_idx   = indices[n_clean:]

    scores = []
    levels = []
    l0_fired_at = None

    print("Phase 1: Clean queries")
    for t, idx in enumerate(clean_idx):
        pixel_img, _ = ds[int(idx)]
        norm_img = _NORM(pixel_img).unsqueeze(0)
        _, level, meta = prism.defend(norm_img)
        s = meta.get('anomaly_score', 0.0)
        scores.append(s)
        levels.append(level)
        l0 = meta.get('l0_state', {}).get('l0_active', False)
        if l0 and l0_fired_at is None:
            l0_fired_at = t
            print(f"  !! False alarm at clean step {t} (score={s:.3f})")

    print(f"  Done. Mean score={np.mean(scores[:n_clean]):.3f}  "
          f"Std={np.std(scores[:n_clean]):.3f}")
    print(f"  L0 so far: {'FIRED (false alarm)'  if l0_fired_at is not None else 'quiet'}\n")

    print("Phase 2: Adversarial queries (FGSM)")
    adv_l0_at = None
    for t, idx in enumerate(adv_idx):
        pixel_img, _ = ds[int(idx)]
        x_np = pixel_img.unsqueeze(0).numpy()
        x_adv_np = fgsm.generate(x_np)
        x_adv_norm = _NORM(torch.tensor(x_adv_np[0])).unsqueeze(0)

        _, level, meta = prism.defend(x_adv_norm)
        s = meta.get('anomaly_score', 0.0)
        scores.append(s)
        levels.append(level)

        l0 = meta.get('l0_state', {}).get('l0_active', False)
        if l0 and adv_l0_at is None:
            adv_l0_at = t
            step_global = n_clean + t
            print(f"  *** L0 fired at adversarial step {t} "
                  f"(global step {step_global}, score={s:.3f})")

        if (t + 1) % 20 == 0:
            print(f"  Processed {t+1}/{n_adv} adversarials...")

    if adv_l0_at is None:
        print("  L0 never fired during adversarial phase.")

    # ── Results ───────────────────────────────────────────────────────────────
    adv_scores = np.array(scores[n_clean:])
    clean_scores_arr = np.array(scores[:n_clean])

    print(f"\n=== Campaign Detection Results ===")
    print(f"  Clean scores:  mean={clean_scores_arr.mean():.3f} std={clean_scores_arr.std():.3f}")
    print(f"  Adv scores:    mean={adv_scores.mean():.3f}  std={adv_scores.std():.3f}")
    print(f"  L0 detection:  {'step ' + str(adv_l0_at) + ' after campaign onset' if adv_l0_at is not None else 'NOT DETECTED'}")

    if adv_l0_at is not None:
        print(f"  Lead time:     {adv_l0_at} adversarial queries before L0 fired")
        if adv_l0_at < 20:
            print(f"  ✓ Within target (<20 queries)")
        else:
            print(f"  ⚠ Slower than target (>20 queries)")

    # ── Save ──────────────────────────────────────────────────────────────────
    result = {
        'n_clean': n_clean,
        'n_adv': n_adv,
        'eps': eps,
        'l0_detected_at_adv_step': int(adv_l0_at) if adv_l0_at is not None else None,
        'l0_global_step': int(n_clean + adv_l0_at) if adv_l0_at is not None else None,
        'false_alarm': l0_fired_at is not None,
        'clean_score_mean': float(clean_scores_arr.mean()),
        'clean_score_std': float(clean_scores_arr.std()),
        'adv_score_mean': float(adv_scores.mean()),
        'adv_score_std': float(adv_scores.std()),
        'all_levels': levels,
    }
    os.makedirs('experiments/campaign', exist_ok=True)
    with open('experiments/campaign/results.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved → experiments/campaign/results.json")
    return result


if __name__ == '__main__':
    run_campaign_experiment()
