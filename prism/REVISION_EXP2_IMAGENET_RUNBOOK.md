# Exp 2 — ImageNet-100 / ResNet-50 standard-attack run (READY)

## Status: wired + offline-validated (2026-06-13)

Decisions baked in: **ImageNet-100** (100-class subset) and **eps = 8/255**
(matches every other PRISM setting). The port is done — all code changes
landed and the ResNet-50 @ 224x224 detection path is validated offline. What
remains is staging real data and running on a GPU box.

### What was implemented

1. **`src/models/backbone.py`** — `resnet50` arch branch. Stock torchvision
   ResNet-50 with a num_classes head; loads the fine-tuned checkpoint. Native
   `layer2/layer3/layer4` are unchanged, so the activation extractor needs no
   edit.
2. **`src/data_loader.py`** — `imagenet` branch. ImageNet has no flat indexable
   test file, so we build ONE deterministic fixed-seed permuted ImageFolder
   pool (`_PermutedImageFolder` / `_imagenet_pool`). The flat split ranges
   (profile/cal/val/eval) carve disjoint, class-spanning slices out of it; eval
   is a held-out slice of the same pool — identical in spirit to the disclosed
   CIFAR protocol. The central transform builder now emits a proper
   Resize(256)->CenterCrop(224) for `dataset: imagenet` (a bare Resize would
   leave non-square, unbatchable tensors).
3. **`experiments/evaluation/run_evaluation_full.py`** and
   **`scripts/compute_ensemble_val_fpr.py`** — their locally-built pixel
   transforms now square-crop for ImageNet (same Resize+CenterCrop), fixing the
   unbatchable-tensor bug. CIFAR/ViT behaviour byte-unchanged.
4. **`configs/imagenet.yaml`** — `dataset: imagenet`, `image_size: 224`,
   ImageNet mean/std, `arch: resnet50`, `num_classes: 100`, `imagenet_dir`,
   `imagenet_pool_seed`, per-dataset `paths:` under `models/imagenet/`, and the
   flat splits over the pool.
5. **`scripts/pretrain_imagenet100_backbone.py`** — fine-tunes ResNet-50
   (ImageNet-1k init) to a 100-way head on the staged ImageFolder. Labels 0..99
   come from the ImageFolder's sorted class dirs, matching the data_loader pool
   exactly — no wnid remapping anywhere.
6. **`run_vastai_imagenet.sh`** — full pipeline launcher (fine-tune ->
   TDA-compat smoke -> profile -> ensemble -> calibrate -> FPR gate -> eval),
   with a `SMOKE_ONLY=1` wiring check.

### Offline validation done (no data / no GPU needed)

- `py_compile` clean on all edited + new files.
- `configs/imagenet.yaml` resolves: DATASET=imagenet, arch=resnet50,
  num_classes=100, input_size=224, splits (0,5000)/(5000,7000)/(7000,8000)/
  (8000,10000), paths under models/imagenet/.
- imagenet-aware transforms build (Resize256 + CenterCrop224).
- backbone dispatch reaches the resnet50 branch with the correct
  missing-checkpoint hint.
- **End-to-end synthetic smoke (the main scientific risk):** built a fake
  4-class ImageFolder + stub ResNet-50 checkpoint, then ran
  pool -> load_backbone -> ActivationExtractor -> TopologicalProfiler. Result:
  `layer2 (512,28,28)`, `layer3 (1024,14,14)`, `layer4 (2048,7,7)`, all with
  non-empty H0/H1 persistence diagrams. **The ResNet-50 @ 224x224 extraction +
  TDA path is mechanically sound.** (Fixtures were deleted after the check.)

The one thing offline validation cannot answer is data-quality: whether the
ensemble actually separates clean vs adversarial on real ImageNet-100. That is
exactly what the run produces.

## To run (on the GPU box)

1. **Stage ImageNet-100** as an ImageFolder at `data/imagenet100/` (one subdir
   per class, the standard ~130k-image ImageNet-100 train split; >=10k images
   required so the flat splits fit). Override with `IMAGENET_DIR=...` if staged
   elsewhere.
2. ```
   SEEDS="42 123 456" N_TEST=1000 bash run_vastai_imagenet.sh
   ```
   Optional wiring check first: `SMOKE_ONLY=1 bash run_vastai_imagenet.sh`.
3. Output: `experiments/imagenet/evaluation/results_imagenet_multiseed.json`,
   matching the JSON shape the paper-table builder already consumes.

## Disk / dataset size (vast.ai)

| Item | Size |
|---|---|
| **ImageNet-100** train (~130k JPEGs, 100 classes) | **~13-16 GB** |
| ImageNet-100 val (5k JPEGs) | ~0.5 GB |
| Full ImageNet-1k (for reference — NOT used here) | ~150 GB (1.28M imgs) |
| venv + torch(cu)+torchvision+deps | ~8-10 GB |
| ResNet-50 IMAGENET1K_V2 weights (auto-download) | ~100 MB |
| profiles/calibrator/scorer pkls + result JSON | < 1 GB |

**Provision >= 40 GB** on the vast.ai instance (15 GB data + 10 GB env +
headroom for the profiling pass / checkpoints). 50 GB comfortable. Full
ImageNet-1k would need 200 GB+ — another reason IN-100 is the chosen scope.

## Cost / risk

- **Profiling is the bottleneck**: persistent homology over 224x224 ResNet-50
  activations is heavier than CIFAR. Budget several GPU-hours for
  `build_profile_testset.py`. If `layer2` profiling is too slow, drop to
  monitoring `layer3/layer4` only (edit `model.layer_names`/`layer_weights` in
  `configs/imagenet.yaml`).
- **Residual risk**: TDA may add less at scale (CIFAR shows TDA = +13.1pp; at
  224x224 it could shrink). The DCT/entropy/logit/stability channels should
  still separate FGSM/PGD/Square, so the standard-attack row — all this row
  claims — is likely safe.

## After the run lands

Add one detection table + a one-paragraph scope note to all 4 papers (mirror
the ViT-B/16 transfer row: standard attacks only, no latency / CW / AA /
adaptive claim), rebuild PDFs, repackage source zips.
