#!/bin/bash
# =============================================================================
# PRISM — Vast.ai ENSEMBLE-COMPLETE adaptive PGD on CIFAR-100 (paper revision R1)
# =============================================================================
# Closes the reviewer-blocking gap: the ensemble-complete BPDA-style adaptive
# attack (which the paper reports on CIFAR-10/ResNet-18, TPR 0.92 -> 0.479) has
# NOT been run on CIFAR-100. The stock run_vastai_cifar100.sh runs adaptive PGD
# with --through-scorer only (two-channel), which the appendix itself shows
# misses StabilityV2 — CIFAR-100's dominant detection channel. This script runs
# the FULL ensemble-complete attack (activation-matching for TDA + DCT + entropy
# + logit-profile + stability-v2 + grad-norm + the fitted side-quadratic logistic
# surrogate) so the CIFAR-100 adaptive-robustness claim is complete, not scoped
# by an incomplete attack.
#
# What it does:
#   1. Builds the CIFAR-100 detector if artifacts are missing (backbone ->
#      reference profiles -> 55-feature ensemble scorer -> conformal calibration
#      -> val-FPR gate). Idempotent: existing artifacts are reused, so re-runs
#      resume rather than retrain.
#   2. Runs the two-stage ensemble-complete adaptive PGD:
#        Stage A (scan):    n=50,  lambda in {0,0.5,1,2,5,10}, 100 steps x 10
#                           restarts, 1 seed (SCAN_SEEDS) -- picks a lambda,
#                           is not itself a reported number, so it runs cheap.
#        Stage B (confirm): n=200, worst-case lambda from Stage A, full SEEDS
#                           (default 5) -- this is the number that gets
#                           compared against the CIFAR-10 result in the paper.
#      "Worst" = the lambda maximising undetected-success-rate (successful model
#      fooling that also evades PRISM) — the true deployment-risk operating point.
#
# Usage (on the Vast.ai instance, repo cloned, GPU present):
#   bash run_vastai_ec_adaptive_cifar100.sh
#
# Quick smoke test (verifies wiring end-to-end in ~minutes, tiny n/steps):
#   SMOKE=1 bash run_vastai_ec_adaptive_cifar100.sh
#
# Run WRN-28-10 instead (slower — larger backbone; kept for completeness):
#   CONFIG=configs/wrn_cifar10.yaml TAG=wrn_ec bash run_vastai_ec_adaptive_cifar100.sh
#
# Useful overrides:
#   SCAN_N=50 CONFIRM_N=200 SEEDS="42 123 456 789 999" STEPS=100 RESTARTS=10
#   SKIP_BUILD=1   # skip detector build (artifacts already present)
#   SKIP_GRADNORM=1
#
# Exit codes: 0=success, 1=env/root error, 2=detector build failed,
#             3=attack harness missing --ensemble-complete, 4=attack failed.
# =============================================================================

set -euo pipefail

# ── Resolve PRISM root (both common Vast.ai clone layouts) ───────────────────
if [ -d /workspace/prism-repo/prism/prism/src ] && [ -f /workspace/prism-repo/prism/prism/requirements.txt ]; then
  PRISM_ROOT=/workspace/prism-repo/prism/prism
elif [ -d /workspace/prism-repo/prism/src ] && [ -f /workspace/prism-repo/prism/requirements.txt ]; then
  PRISM_ROOT=/workspace/prism-repo/prism
elif [ -d "$(pwd)/src" ] && [ -f "$(pwd)/requirements.txt" ]; then
  PRISM_ROOT="$(pwd)"
elif [ -d "$(dirname "$0")/src" ] && [ -f "$(dirname "$0")/requirements.txt" ]; then
  PRISM_ROOT="$(cd "$(dirname "$0")" && pwd)"
else
  echo "ERROR: Could not locate PRISM root (expected src/ and requirements.txt)."
  exit 1
fi
cd "$PRISM_ROOT"

# Activate the standard Vast.ai venv if present (matches other run scripts).
if [ -f /workspace/prism-venv/bin/activate ]; then
  # shellcheck disable=SC1091
  . /workspace/prism-venv/bin/activate
fi

unset PYTHONSAFEPATH || true
export PYTHONPATH="$PRISM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}"

# ── Configuration ────────────────────────────────────────────────────────────
CONFIG="${CONFIG:-configs/cifar100.yaml}"
TAG="${TAG:-cifar100_ec}"
export PRISM_CONFIG="$CONFIG"

SEEDS="${SEEDS:-42 123 456 789 999}"
STEPS="${STEPS:-100}"
RESTARTS="${RESTARTS:-10}"
SCAN_LAMBDAS="${SCAN_LAMBDAS:-0.0 0.5 1.0 2.0 5.0 10.0}"
SCAN_N="${SCAN_N:-50}"
CONFIRM_N="${CONFIRM_N:-200}"
# Stage A (lambda scan) only picks which lambda to confirm at — it is not
# itself a reported number, so by default it runs on ONE seed instead of all
# five. This is the single biggest cost lever: with 6 lambdas x n=50, 5 seeds
# vs 1 seed is a 5x cost difference on the scan, which otherwise dominates
# wall-clock alongside the confirm phase. Override with SCAN_SEEDS="..." to
# scan on more seeds if you want the lambda choice itself to be seed-robust.
first_seed=""; for s in $SEEDS; do first_seed="$s"; break; done
SCAN_SEEDS="${SCAN_SEEDS:-$first_seed}"
EOT_SAMPLES="${EOT_SAMPLES:-1}"
EOT_VERIFY_SAMPLES="${EOT_VERIFY_SAMPLES:-20}"
EC_MATCH_COEFF="${EC_MATCH_COEFF:-0.5}"
EC_SCORE_COEFF="${EC_SCORE_COEFF:-0.25}"
SKIP_GRADNORM="${SKIP_GRADNORM:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"

# CIFAR-100 detector-build parameters (mirror run_vastai_cifar100.sh exactly).
ENSEMBLE_N_TRAIN="${ENSEMBLE_N_TRAIN:-1500}"
ENSEMBLE_SOURCE_SPLIT="${ENSEMBLE_SOURCE_SPLIT:-profile}"
ENSEMBLE_GEN_CHUNK="${ENSEMBLE_GEN_CHUNK:-512}"
BACKBONE_MIN_ACC="${BACKBONE_MIN_ACC:-0.73}"

if [ "${SMOKE:-0}" = "1" ]; then
  TAG="${TAG}_smoke"
  first_seed=""; for s in $SEEDS; do first_seed="$s"; break; done
  SEEDS="$first_seed"
  SCAN_SEEDS="$first_seed"
  SCAN_N="${SMOKE_SCAN_N:-4}"
  CONFIRM_N="${SMOKE_CONFIRM_N:-4}"
  STEPS="${SMOKE_STEPS:-2}"
  RESTARTS="${SMOKE_RESTARTS:-1}"
  SCAN_LAMBDAS="${SMOKE_LAMBDAS:-0.0 10.0}"
fi

SCAN_TAG="${TAG}_lambda_scan_n${SCAN_N}"
CONFIRM_TAG="${TAG}_worst_lambda_n${CONFIRM_N}"

mkdir -p "logs/${TAG}"

echo "============================================================"
echo "PRISM ensemble-complete adaptive PGD — ${TAG}"
echo "Repo root: $PRISM_ROOT"
echo "Config:    $CONFIG"
echo "Seeds (confirm, reported number): $SEEDS"
echo "Scan:      n=$SCAN_N  lambdas=[$SCAN_LAMBDAS]  seeds=[$SCAN_SEEDS]  ${STEPS} steps x ${RESTARTS} restarts"
echo "Confirm:   n=$CONFIRM_N (worst lambda), seeds=[$SEEDS]"
echo "ec_match_coeff=$EC_MATCH_COEFF  ec_score_coeff=$EC_SCORE_COEFF  skip_gradnorm=$SKIP_GRADNORM"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# ── Preflight: dependencies ──────────────────────────────────────────────────
if ! python -c "import torch" 2>/dev/null; then
  echo "PyTorch not found; installing requirements.txt ..."
  pip install --no-cache-dir --upgrade pip setuptools wheel
  pip install --no-cache-dir -r requirements.txt
fi
python -c "
import importlib, sys
req = ['torch','torchvision','numpy','scipy','sklearn','yaml','tqdm','ripser','gudhi']
missing = []
for m in req:
    try: importlib.import_module(m)
    except Exception as e: missing.append((m, str(e).splitlines()[0]))
if missing:
    print('MISSING modules:'); [print(f'  - {m}: {e}') for m, e in missing]; sys.exit(1)
print('Preflight: dependencies OK.')
" || { echo "ERROR: dependencies missing."; exit 1; }

# Resolve artifact paths from the config (single source of truth).
read -r CKPT REF ENS CAL EXP < <(python - <<'PY'
import yaml, os
cfg = yaml.safe_load(open(os.environ["PRISM_CONFIG"]))
m = cfg.get("model", {}); p = cfg.get("paths", {})
print(m.get("backbone_checkpoint",""), p.get("reference_profiles",""),
      p.get("ensemble_scorer",""), p.get("calibrator",""), p.get("experts",""))
PY
)
echo "Artifacts: backbone=$CKPT  ref=$REF  ensemble=$ENS  calibrator=$CAL"

# ── Stage 0: build the detector if any artifact is missing ───────────────────
build_detector() {
  echo ""; echo "=== Building CIFAR-100 detector (missing artifacts) ==="

  # 0a: backbone
  if [ -f "$CKPT" ]; then
    echo "  backbone present: $CKPT (skip pretrain)"
  else
    echo "  pretraining backbone -> $CKPT"
    python scripts/pretrain_cifar_backbone.py \
      --dataset cifar100 --num-classes 100 \
      --output "$CKPT" --min-test-acc "$BACKBONE_MIN_ACC" \
      2>&1 | tee "logs/${TAG}/build_0a_backbone.log"
  fi

  # 1: reference profiles
  if [ -f "$REF" ]; then
    echo "  reference profiles present: $REF (skip)"
  else
    echo "  building reference profiles -> $REF"
    python scripts/build_profile_testset.py --config "$CONFIG" \
      2>&1 | tee "logs/${TAG}/build_1_profiles.log"
  fi

  # 2: 55-feature ensemble scorer (balanced FGSM/PGD/Square; CW/AA stay eval-only)
  if [ -f "$ENS" ]; then
    echo "  ensemble scorer present: $ENS (skip)"
  else
    echo "  training 55-feature ensemble scorer -> $ENS"
    python scripts/train_ensemble_scorer.py \
      --config "$CONFIG" \
      --n-train "$ENSEMBLE_N_TRAIN" \
      --source-split "$ENSEMBLE_SOURCE_SPLIT" \
      --balanced-attacks \
      --pgd-train-steps 40 \
      --square-train-max-iter 500 \
      --gen-chunk "$ENSEMBLE_GEN_CHUNK" \
      --selection-objective worst_case_tpr \
      --use-stability-features \
      --use-logit-profile-features \
      --use-side-quadratic-features \
      --use-grad-norm \
      --output "$ENS" \
      2>&1 | tee "logs/${TAG}/build_2_scorer.log"
  fi

  # 2d: differentiated experts (needed only if a later recovery pass is run;
  # harmless to build now and keeps the CIFAR-100 model dir complete).
  if [ -n "$EXP" ] && [ ! -f "$EXP" ] && [ -f scripts/train_experts.py ]; then
    echo "  training experts -> $EXP"
    python scripts/train_experts.py --config "$CONFIG" --output "$EXP" \
      2>&1 | tee "logs/${TAG}/build_2d_experts.log" || \
      echo "  WARN: expert training failed (non-fatal for the adaptive attack)."
  fi

  # 3: conformal calibration
  if [ -f "$CAL" ]; then
    echo "  calibrator present: $CAL (skip)"
  else
    echo "  calibrating conformal thresholds -> $CAL"
    python scripts/calibrate_ensemble.py --config "$CONFIG" \
      2>&1 | tee "logs/${TAG}/build_3_calibrate.log"
  fi

  # 4: val-FPR gate (report only — do not hard-fail the attack on a marginal gate)
  if [ -f scripts/compute_ensemble_val_fpr.py ]; then
    echo "  validation FPR gate:"
    python scripts/compute_ensemble_val_fpr.py --config "$CONFIG" \
      2>&1 | tee "logs/${TAG}/build_4_val_fpr.log" || \
      echo "  WARN: val-FPR gate reported non-zero (inspect build_4_val_fpr.log)."
  fi
}

need_build=0
for f in "$CKPT" "$REF" "$ENS" "$CAL"; do
  [ -f "$f" ] || need_build=1
done
if [ "$SKIP_BUILD" = "1" ]; then
  echo "SKIP_BUILD=1 — assuming detector artifacts are present."
elif [ "$need_build" = "1" ]; then
  build_detector || { echo "ERROR: detector build failed."; exit 2; }
else
  echo "All detector artifacts present — skipping build."
fi

# ── Confirm the attack harness supports --ensemble-complete ──────────────────
if ! python experiments/evaluation/run_adaptive_pgd.py --help 2>&1 | grep -q -- '--ensemble-complete'; then
  echo "ERROR: run_adaptive_pgd.py does not expose --ensemble-complete on this checkout."
  exit 3
fi

# ── run one (tag, n_test, lambdas, seeds) phase ───────────────────────────────
run_phase() {
  local tag="$1" n_test="$2" lambdas="$3" seeds="$4"
  local outdir="experiments/evaluation/${tag}"
  local logdir="logs/${tag}"
  mkdir -p "$outdir" "$logdir"
  echo ""; echo "=== PHASE ${tag}: n=${n_test} lambdas=[${lambdas}] seeds=[${seeds}] ==="

  local extra=(--ensemble-complete --through-scorer
               --ec-match-coeff "$EC_MATCH_COEFF" --ec-score-coeff "$EC_SCORE_COEFF")
  [ "$SKIP_GRADNORM" = "1" ] && extra+=(--skip-gradnorm-surrogate)

  for s in $seeds; do
    local out_json="$outdir/results_ensemble_complete_adaptive_pgd_seed${s}.json"
    local out_jsonl="$outdir/results_ensemble_complete_adaptive_pgd_seed${s}.jsonl"
    echo "  --- seed $s -> $out_json"
    python experiments/evaluation/run_adaptive_pgd.py \
      --config "$CONFIG" \
      --n-test "$n_test" \
      --seed "$s" \
      --lambdas $lambdas \
      --pgd-steps "$STEPS" \
      --pgd-restarts "$RESTARTS" \
      --eot-samples "$EOT_SAMPLES" \
      --eot-verify-samples "$EOT_VERIFY_SAMPLES" \
      "${extra[@]}" \
      --checkpoint-jsonl "$out_jsonl" \
      --resume \
      --output "$out_json" \
      2>&1 | tee "$logdir/ec_adaptive_seed${s}.log"
  done
}

# ── choose the worst-case (highest deployment-risk) lambda from the scan ─────
choose_worst_lambda() {
  SCAN_DIR="experiments/evaluation/${SCAN_TAG}" python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ["SCAN_DIR"])
rows = {}
for path in sorted(root.glob("results_ensemble_complete_adaptive_pgd_seed*.json")):
    data = json.loads(path.read_text())
    for key, value in data.items():
        if key.startswith("AdaptivePGD_lambda_"):
            rows.setdefault(str(value["lambda"]), []).append(value)
if not rows:
    raise SystemExit("No scan result JSON files found in %s" % root)

summary = []
for lam, vals in sorted(rows.items(), key=lambda kv: float(kv[0])):
    tp = sum(int(v.get("TP", 0)) for v in vals)
    fp = sum(int(v.get("FP", 0)) for v in vals)
    fn = sum(int(v.get("FN", 0)) for v in vals)
    tn = sum(int(v.get("TN", 0)) for v in vals)
    n_adv = max(tp + fn, 1); n_clean = max(fp + tn, 1)
    n_success = sum(int(v.get("n_successful_adv", 0)) for v in vals)
    detected = sum(int(v.get("detected_successful_adv", 0)) for v in vals)
    evaded = sum(int(v.get("evaded_successful_adv", 0)) for v in vals)
    summary.append({
        "lambda": lam, "n_seeds": len(vals),
        "TPR": tp / n_adv, "FPR": fp / n_clean,
        "model_ASR": n_success / n_adv,
        "TPR_on_successful_attacks": detected / max(n_success, 1),
        "evasion_rate_on_successful_attacks": evaded / max(n_success, 1),
        "undetected_success_rate": evaded / n_adv,
    })
worst = max(summary, key=lambda x: (
    x["undetected_success_rate"], x["model_ASR"],
    x["evasion_rate_on_successful_attacks"],
    -x["TPR_on_successful_attacks"], -x["TPR"]))
Path("/tmp").mkdir(exist_ok=True)
Path("prism_ec_worst_lambda.txt").write_text(worst["lambda"] + "\n")
print(json.dumps({"worst": worst, "summary": summary}, indent=2))
PY
}

# ── Stage A: lambda scan (cheap — SCAN_SEEDS, default 1 seed) ────────────────
run_phase "$SCAN_TAG" "$SCAN_N" "$SCAN_LAMBDAS" "$SCAN_SEEDS" || { echo "ERROR: scan phase failed."; exit 4; }

echo ""; echo "=== Selecting worst-case lambda from scan ==="
choose_worst_lambda | tee "logs/${SCAN_TAG}/worst_lambda.log"
WORST_LAMBDA="$(cat prism_ec_worst_lambda.txt)"
echo "WORST_LAMBDA=${WORST_LAMBDA}"

# ── Stage B: confirm at the worst lambda, full n, full SEEDS (reported number) ─
run_phase "$CONFIRM_TAG" "$CONFIRM_N" "$WORST_LAMBDA" "$SEEDS" || { echo "ERROR: confirm phase failed."; exit 4; }

echo ""
echo "============================================================"
echo "DONE. Ensemble-complete adaptive PGD (${TAG}) complete."
echo "  scan    -> experiments/evaluation/${SCAN_TAG}/"
echo "  confirm -> experiments/evaluation/${CONFIRM_TAG}/"
echo "  worst lambda = ${WORST_LAMBDA}"
echo "Compare the confirm-phase TPR / undetected-success against the"
echo "CIFAR-10/ResNet-18 result (all-input TPR 0.479) reported in the paper."
echo "============================================================"
