# Exp 2 — ImageNet-scale standard-attack run (RUNBOOK, not a drop-in script)

## Honest status

Exp 1 and Exp 3 are **drop-in** (one script each, consume frozen artifacts).
Exp 2 is **not**. The codebase deliberately dropped ImageNet
(`run_vastai_full.sh:226`: *"Replaces the prior ImageNet-pretrained ResNet-18"*).
Detection on a new dataset requires the **full pipeline re-instantiated**:
reference profiles -> ensemble scorer -> conformal calibration -> eval. There is
no ImageNet dataset loader, no ResNet-50 arch dispatch, and the split logic
assumes a 10k CIFAR test file. So Exp 2 is a 1-2 day port with real debugging,
not a script I can hand you "ready" without lying about it.

I did **not** fabricate a ready launcher for it. Below is the exact work it
needs, plus a cheaper variant that still answers the reviewer.

## What the reviewer actually wants

"Does PRISM work beyond 32x32 / CIFAR-derived?" One scaled standard-attack run
(no adaptive, no latency claim) deflects "CIFAR-only." It does **not** need to
be full ImageNet-1k.

## Recommended scope: ImageNet-100 (cheaper, still convincing)

Full ImageNet-1k adds cost and a real scientific risk: TDA subsamples to 150
points over much larger ResNet-50 activation maps — persistence separation may
degrade, and a weak headline number hurts more than no number. ImageNet-100
(100-class subset, torchvision ResNet-50 features, 224x224) is enough to claim
"scales to 224x224 / ImageNet-grade inputs" with bounded cost and risk. Decide
1k vs 100 before spending GPU.

## Required code changes (each small, but must all land + be tested)

1. **`src/data_loader.py`** — add an `imagenet` branch to `_resolve_class`
   (use `torchvision.datasets.ImageFolder` over a val directory; CIFAR loaders
   won't apply). Split indices (`PROFILE/CAL/VAL/EVAL_IDX`) assume one flat 10k
   test file — replace with a deterministic per-class index partition over the
   ImageFolder, or add an `imagenet`-specific splitter.
2. **`src/models/backbone.py`** — add a `resnet50` arch branch:
   `torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)`; it
   already exposes `layer2/layer3/layer4`, so the extractor needs no change.
3. **`configs/imagenet.yaml`** — `data.dataset: imagenet`, `data.image_size: 224`,
   ImageNet `mean=[0.485,0.456,0.406]`/`std=[0.229,0.224,0.225]`,
   `model.arch: resnet50`, `model.num_classes: 100` (or 1000),
   per-dataset `paths:` so CIFAR artifacts aren't clobbered
   (`models/imagenet/...`).
4. **Attacks** — FGSM/PGD/Square already run through ART/native on pixel space;
   confirm `eps=8/255` is the intended ImageNet budget (4/255 is also common —
   pick and state it). No adaptive, no CW, no AutoAttack for this row.

## Pipeline stages (reuse existing scripts via `--config configs/imagenet.yaml`)

All of these already dispatch on `load_backbone` + `load_test_dataset` + config,
so once the three code changes above land they should run unmodified:

```
PRISM_CONFIG=configs/imagenet.yaml
python scripts/build_profile_testset.py          # reference_profiles.pkl  (TDA profiling — SLOW at 224x224)
python scripts/train_ensemble_scorer.py --config configs/imagenet.yaml \
    --balanced-attacks --use-stability-features --use-logit-profile-features \
    --use-side-quadratic-features --use-grad-norm --output models/imagenet/ensemble_scorer.pkl
python scripts/calibrate_ensemble.py             # calibrator.pkl
python scripts/compute_ensemble_val_fpr.py       # FPR gate (must pass alpha targets)
python experiments/evaluation/run_evaluation_full.py \
    --n-test 1000 --attacks FGSM PGD Square --multi-seed --seeds 42 123 456 \
    --skip-latency --output experiments/evaluation/results_imagenet.json
```

## Cost / risk

- **Profiling is the bottleneck**: persistent homology over 224x224 ResNet-50
  activations is far heavier than CIFAR. Budget several GPU-hours just for
  `build_profile_testset.py`; consider monitoring only `layer3/layer4` if
  `layer2` profiling is too slow.
- **Risk**: TDA channel may add little at scale (the paper already shows TDA is
  +13.1pp on CIFAR; at 224x224 it could shrink). The ensemble's DCT/entropy/
  logit/stability channels should still separate FGSM/PGD/Square, so the row is
  likely safe for *standard* attacks — which is all this row claims.

## Decision needed

Confirm **(a) ImageNet-100 vs ImageNet-1k** and **(b) eps = 4/255 vs 8/255**.
Once chosen I can implement the three code changes + `configs/imagenet.yaml` +
`run_vastai_imagenet.sh`. Until then this stays a runbook, not a "ready" script,
because shipping it as ready would be a mistake.
