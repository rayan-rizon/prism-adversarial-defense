"""
CIFAR-100 Generalization Test (Appendix B)
Tests PRISM (calibrated on CIFAR-10) on CIFAR-100 clean + FGSM adversarials.
Measures: score distribution, FPR on OOD clean data, TPR on OOD adversarials.
Key question: do topological features generalize across datasets?
"""
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import ssl
import certifi
import torch
import torchvision
import torchvision.transforms as T
import numpy as np
from torchvision.models import ResNet18_Weights

os.environ.setdefault('SSL_CERT_FILE', certifi.where())
os.environ.setdefault('REQUESTS_CA_BUNDLE', certifi.where())
ssl._create_default_https_context = ssl.create_default_context

from src.prism import PRISM

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

_PIXEL = T.Compose([T.Resize(224), T.ToTensor()])
_NORM  = T.Normalize(_MEAN, _STD)


class _NormalizedResNet(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.register_buffer('mean', torch.tensor(_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor(_STD).view(1, 3, 1, 1))

    def forward(self, x):
        return self.backbone((x - self.mean) / self.std)


def run_cifar100_test(n_clean=200, n_adv=200, eps=0.05, seed=42,
                      data_root='./data',
                      output_path='experiments/generalization/cifar100_results.json'):

    print(f"=== CIFAR-100 Generalization Test ===")
    print(f"  n_clean={n_clean}, n_adv={n_adv}, eps={eps}\n")

    rng = np.random.RandomState(seed)

    # ── Load PRISM (calibrated on CIFAR-10) ───────────────────────────────────
    backbone = torchvision.models.resnet18(
        weights=ResNet18_Weights.IMAGENET1K_V1
    ).eval()

    prism = PRISM.from_saved(
        model=backbone,
        layer_names=['layer2', 'layer3', 'layer4'],
        layer_weights={'layer2': 0.15, 'layer3': 0.30, 'layer4': 0.55},
        calibrator_path='models/calibrator.pkl',
        profile_path='models/reference_profiles.pkl',
    )
    print(f"  Thresholds: L1={prism.calibrator.thresholds.get('L1', '?'):.4f}  "
          f"L2={prism.calibrator.thresholds.get('L2', '?'):.4f}  "
          f"L3={prism.calibrator.thresholds.get('L3', '?'):.4f}\n")

    # ── Load CIFAR-100 ────────────────────────────────────────────────────────
    print("  Loading CIFAR-100 test set...")
    ds = torchvision.datasets.CIFAR100(
        data_root, train=False, download=True, transform=_PIXEL
    )
    idx = rng.choice(len(ds), n_clean + n_adv, replace=False)
    clean_idx = idx[:n_clean]
    adv_idx   = idx[n_clean:]
    print(f"  Dataset size: {len(ds)}\n")

    # ART adversarials on CIFAR-100
    try:
        from art.attacks.evasion import FastGradientMethod
        from art.estimators.classification import PyTorchClassifier
        wrapped = _NormalizedResNet(backbone).eval()
        art_clf = PyTorchClassifier(
            model=wrapped,
            loss=torch.nn.CrossEntropyLoss(),
            input_shape=(3, 224, 224),
            nb_classes=1000,  # ImageNet backbone (ResNet-18) has 1000 output classes
            clip_values=(0.0, 1.0),
        )
        fgsm = FastGradientMethod(art_clf, eps=eps)
        art_available = True
    except ImportError:
        art_available = False
        print("  WARNING: ART not available, skipping adversarial phase")

    # ── Phase 1: Clean CIFAR-100 ──────────────────────────────────────────────
    print("Phase 1: Clean CIFAR-100 queries")
    clean_scores = []
    clean_levels = []

    for i, idx_item in enumerate(clean_idx):
        pixel_img, _ = ds[int(idx_item)]
        norm_img = _NORM(pixel_img).unsqueeze(0)
        _, level, meta = prism.defend(norm_img)
        s = meta.get('anomaly_score', 0.0)
        clean_scores.append(s)
        clean_levels.append(level)
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{n_clean} clean images...")

    clean_detected = sum(1 for l in clean_levels if l != 'PASS')
    clean_fpr = clean_detected / n_clean
    print(f"  Done. Mean={np.mean(clean_scores):.3f}  Std={np.std(clean_scores):.3f}")
    print(f"  FPR on CIFAR-100 clean: {clean_fpr:.3f} ({clean_detected}/{n_clean} flagged)\n")

    # ── Phase 2: FGSM on CIFAR-100 ────────────────────────────────────────────
    adv_scores = []
    adv_levels = []
    tpr = None

    if art_available:
        print("Phase 2: FGSM adversarials on CIFAR-100")
        for i, idx_item in enumerate(adv_idx):
            pixel_img, _ = ds[int(idx_item)]
            x_np = pixel_img.unsqueeze(0).numpy()
            x_adv_np = fgsm.generate(x_np)
            x_adv_norm = _NORM(torch.tensor(x_adv_np[0])).unsqueeze(0)
            _, level, meta = prism.defend(x_adv_norm)
            s = meta.get('anomaly_score', 0.0)
            adv_scores.append(s)
            adv_levels.append(level)
            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{n_adv} adversarial images...")

        adv_detected = sum(1 for l in adv_levels if l != 'PASS')
        tpr = adv_detected / n_adv
        print(f"  Done. Mean={np.mean(adv_scores):.3f}  Std={np.std(adv_scores):.3f}")
        print(f"  TPR on CIFAR-100 FGSM: {tpr:.3f} ({adv_detected}/{n_adv} detected)\n")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "dataset": "CIFAR-100",
        "calibrated_on": "CIFAR-10",
        "n_clean": n_clean,
        "n_adv": n_adv,
        "eps": eps,
        "clean_score_mean": float(np.mean(clean_scores)),
        "clean_score_std":  float(np.std(clean_scores)),
        "clean_fpr": clean_fpr,
        "clean_level_distribution": {l: clean_levels.count(l) for l in set(clean_levels)},
        "adv_score_mean": float(np.mean(adv_scores)) if adv_scores else None,
        "adv_score_std":  float(np.std(adv_scores)) if adv_scores else None,
        "tpr_cifar100_fgsm": tpr,
        "adv_level_distribution": ({l: adv_levels.count(l) for l in set(adv_levels)}
                                    if adv_levels else {}),
        "note": "PRISM calibrated on CIFAR-10; generalization test on CIFAR-100",
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → {output_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== Generalization Summary ===")
    print(f"  CIFAR-10  clean FPR: 0.05  (calibrated, alpha=0.05)")
    print(f"  CIFAR-100 clean FPR: {clean_fpr:.3f}")
    c10_clean_mean = 4.634
    print(f"  Score shift: CIFAR-10 clean={c10_clean_mean:.3f} → "
          f"CIFAR-100 clean={np.mean(clean_scores):.3f}")
    if tpr is not None:
        print(f"  CIFAR-100 FGSM TPR: {tpr:.3f}")

    return results


if __name__ == '__main__':
    run_cifar100_test(n_clean=200, n_adv=200, eps=0.05)
