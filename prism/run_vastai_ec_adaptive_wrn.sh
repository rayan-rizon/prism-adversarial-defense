#!/bin/bash
# =============================================================================
# PRISM — Vast.ai ENSEMBLE-COMPLETE adaptive PGD on WRN-28-10/CIFAR-10 (R1)
# =============================================================================
# Closes the second half of the reviewer-blocking gap (R1): the paper's
# ensemble-complete BPDA-style adaptive attack (all-input TPR 0.92 -> 0.479 on
# CIFAR-10/ResNet-18) has so far only been extended to CIFAR-100/ResNet-18
# (run_vastai_ec_adaptive_cifar100.sh). This script runs the identical FULL
# ensemble-complete attack (activation-matching for TDA + DCT + entropy +
# logit-profile + StabilityV2 + grad-norm + the fitted side-quadratic logistic
# surrogate) against the WRN-28-10/CIFAR-10 detector, so the adaptive-
# robustness claim covers all three CNN settings the paper reports clean/
# standard-attack numbers for.
#
# This is a dedicated sibling of run_vastai_ec_adaptive_cifar100.sh rather
# than a generic multi-arch script: the WRN backbone is pretrained by a
# different script with a different CLI (scripts/pretrain_wrn_backbone.py,
# no --dataset/--num-classes — WRN-28-10/CIFAR-10 is its only target) and a
# different default min-test-acc gate (0.93 vs 0.73 for CIFAR-100/ResNet-18).
# Everything downstream of the backbone (profile build, ensemble scorer,
# calibration, experts, and the adaptive-attack harness itself) is config-
# driven and identical in structure to the CIFAR-100 script.
#
# What it does:
#   1. Builds the WRN-28-10 detector if artifacts are missing (backbone ->
#      reference profiles -> ensemble scorer -> conformal calibration ->
#      val-FPR gate). Idempotent: existing artifacts are reused, so if you
#      upload the already-trained WRN artifacts (see below) this step is
#      skipped entirely.
#   2. Runs the two-stage ensemble-complete adaptive PGD:
#        Stage A (scan):    n=50,  lambda in {0,0.5,1,2,5,10}, 100 steps x 10
#                           restarts, 1 seed (SCAN_SEEDS) -- picks a lambda,
#                           is not itself a reported number, so it runs cheap.
#        Stage B (confirm): n=200, worst-case lambda from Stage A, full SEEDS
#                           (default 5) -- this is the number that gets
#                           compared against the CIFAR-10/ResNet-18 result
#                           (all-input TPR 0.479) in the paper.
#      "Worst" = the lambda maximising undetected-success-rate (successful
#      model fooling that also evades PRISM) — the true deployment-risk
#      operating point.
#
# Artifacts needed (already built and verified locally under
# WRN/thunder_handoff_20260523_031624/prism/models/ — upload these to the new
# Vast.ai box under $PRISM_ROOT/models/cifar_wrn28_10.pt and
# $PRISM_ROOT/models/wrn/{calibrator,ensemble_scorer,reference_profiles}.pkl
# to skip the (slow) detector build entirely):
#   models/cifar_wrn28_10.pt          (146 MB, WRN-28-10 backbone, test_acc=0.9636)
#   models/wrn/calibrator.pkl
#   models/wrn/ensemble_scorer.pkl
#   models/wrn/reference_profiles.pkl
#   data/cifar-10-python.tar.gz / cifar-10-batches-py   (or let torchvision fetch it)
# experts.pkl is optional for this attack (non-fatal if missing/not built).
#
# Usage (on the Vast.ai instance, repo cloned, GPU present):
#   bash run_vastai_ec_adaptive_wrn.sh
#
# Quick smoke test (verifies wiring end-to-end in ~minutes, tiny n/steps):
#   SMOKE=1 bash run_vastai_ec_adaptive_wrn.sh
#
# Useful overrides (identical semantics to the CIFAR-100 sibling script):
#   SCAN_N=50 CONFIRM_N=200 SEEDS="42 123 456 789 999" STEPS=100 RESTARTS=10
#   PARALLEL_SCAN_LAMBDAS=6   # scheduling-only speedup for the unreported scan
#   PARALLEL_SEEDS=8          # concurrent confirm-phase seeds (scheduling only)
#   SKIP_BUILD=1              # skip detector build (artifacts already present)
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

# Activate a PyTorch venv if present. Tries the repo-convention path first
# (matches other run scripts), then the Vast.ai template's own venv
# (/venv/main -- present on this provider's stock PyTorch images), then
# falls back to whatever `python`/`pip` already resolve to on PATH.
if [ -f /workspace/prism-venv/bin/activate ]; then
  # shellcheck disable=SC1091
  . /workspace/prism-venv/bin/activate
elif [ -f /venv/main/bin/activate ]; then
  # shellcheck disable=SC1091
  . /venv/main/bin/activate
fi

unset PYTHONSAFEPATH || true
export PYTHONPATH="$PRISM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}"

# ── Configuration ────────────────────────────────────────────────────────────
CONFIG="${CONFIG:-configs/wrn_cifar10.yaml}"
TAG="${TAG:-wrn_ec}"
export PRISM_CONFIG="$CONFIG"

SEEDS="${SEEDS:-42 123 456 789 999}"
STEPS="${STEPS:-100}"
RESTARTS="${RESTARTS:-10}"
SCAN_LAMBDAS="${SCAN_LAMBDAS:-0.0 0.5 1.0 2.0 5.0 10.0}"
SCAN_N="${SCAN_N:-50}"
CONFIRM_N="${CONFIRM_N:-200}"
# The lambda scan chooses a setting for Stage B and is not a reported result.
# Set this above one to evaluate independent (seed, lambda) cells concurrently.
# Every worker retains the same fixed seed, sample selection, model, and attack
# configuration; only the order in which independent cells occupy the GPU changes.
PARALLEL_SCAN_LAMBDAS="${PARALLEL_SCAN_LAMBDAS:-1}"
# Stage A (lambda scan) only picks which lambda to confirm at — it is not
# itself a reported number, so by default it runs on ONE seed instead of all
# five. Override with SCAN_SEEDS="..." to scan on more seeds if you want the
# lambda choice itself to be seed-robust.
first_seed=""; for s in $SEEDS; do first_seed="$s"; break; done
SCAN_SEEDS="${SCAN_SEEDS:-$first_seed}"
EOT_SAMPLES="${EOT_SAMPLES:-1}"
EOT_VERIFY_SAMPLES="${EOT_VERIFY_SAMPLES:-20}"
EC_MATCH_COEFF="${EC_MATCH_COEFF:-0.5}"
EC_SCORE_COEFF="${EC_SCORE_COEFF:-0.25}"
SKIP_GRADNORM="${SKIP_GRADNORM:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"

# WRN-28-10 detector-build parameters. min-test-acc gate is 0.93 (vs 0.73 for
# CIFAR-100/ResNet-18) to match the paper's actual WRN backbone (test_acc=0.9636).
ENSEMBLE_N_TRAIN="${ENSEMBLE_N_TRAIN:-1500}"
ENSEMBLE_SOURCE_SPLIT="${ENSEMBLE_SOURCE_SPLIT:-profile}"
ENSEMBLE_GEN_CHUNK="${ENSEMBLE_GEN_CHUNK:-512}"
BACKBONE_MIN_ACC="${BACKBONE_MIN_ACC:-0.93}"

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

# ── Hardware auto-detection (CPU/GPU saturation, no numeric effect) ──────────
# Same correctness-preserving scheduling levers as the CIFAR-100 sibling
# script: concurrent independent seed processes (own model load, own RNG
# stream, own output files) and a cgroup-aware CPU-core count so the thread
# plan is sized to what the container is actually allotted, not the host.
NPROC="$(python3 -c '
import os, math
c = [os.cpu_count() or 4]
try:
    q, p = open("/sys/fs/cgroup/cpu.max").read().split()
    if q != "max": c.append(max(1, math.floor(int(q)/int(p))))
except Exception: pass
try:
    q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
    p = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
    if q > 0 and p > 0: c.append(max(1, math.floor(q/p)))
except Exception: pass
try:
    c.append(len(os.sched_getaffinity(0)))
except Exception: pass
print(max(1, min(c)))
' 2>/dev/null || echo 4)"
# `|| true` neutralizes the whole pipeline's exit status: under `set -o
# pipefail`, a missing/erroring `nvidia-smi` makes the pipe report its exit
# code even though wc/tr succeeded and produced "0" -- and a bare
# `VAR=$(...)` assignment propagates that under `set -e`, aborting the script
# before any real work starts. This must degrade to GPU_COUNT=0, never abort.
GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ' || true)"
if [ -z "$GPU_COUNT" ]; then GPU_COUNT=0; fi

# How many seeds to run concurrently. Default: all seeds in the confirm phase
# (typically 5) up to a safety cap of 8. The WRN-28-10 backbone is much larger
# than ResNet-18 (146 MB vs 45 MB) and each forward/backward pass is heavier,
# so per-seed VRAM/compute cost is higher than the CIFAR-100/ResNet-18 run --
# if VRAM pressure appears on a smaller GPU, override PARALLEL_SEEDS down.
PARALLEL_SEEDS="${PARALLEL_SEEDS:-8}"

# CPU threads handed to each concurrent seed process, sized so
# PARALLEL_SEEDS-way concurrency never oversubscribes the box
# (reserves ~2 cores for the OS/orchestration).
THREADS_PER_JOB="${THREADS_PER_JOB:-$(( (NPROC > 2 ? NPROC - 2 : NPROC) / (PARALLEL_SEEDS > 0 ? PARALLEL_SEEDS : 1) ))}"
[ "$THREADS_PER_JOB" -lt 1 ] && THREADS_PER_JOB=1

# Workers for the CPU-bound persistent-homology step (build only, Step 1).
PROFILE_WORKERS="${PROFILE_WORKERS:-$(( NPROC > 2 ? NPROC - 2 : 1 ))}"

SCAN_TAG="${TAG}_lambda_scan_n${SCAN_N}"
CONFIRM_TAG="${TAG}_worst_lambda_n${CONFIRM_N}"

mkdir -p "logs/${TAG}"

echo "============================================================"
echo "PRISM ensemble-complete adaptive PGD — ${TAG} (WRN-28-10/CIFAR-10)"
echo "Repo root: $PRISM_ROOT"
echo "Config:    $CONFIG"
echo "Seeds (confirm, reported number): $SEEDS"
echo "Scan:      n=$SCAN_N  lambdas=[$SCAN_LAMBDAS]  seeds=[$SCAN_SEEDS]  ${STEPS} steps x ${RESTARTS} restarts"
echo "Confirm:   n=$CONFIRM_N (worst lambda), seeds=[$SEEDS]"
echo "ec_match_coeff=$EC_MATCH_COEFF  ec_score_coeff=$EC_SCORE_COEFF  skip_gradnorm=$SKIP_GRADNORM"
echo "------------------------------------------------------------"
echo "Hardware:  ${NPROC} CPU cores detected, ${GPU_COUNT} GPU(s)"
echo "Concurrency: up to ${PARALLEL_SEEDS} seeds run in parallel, ${THREADS_PER_JOB} CPU threads/job"
echo "Profile-build workers (Step 1, ripser pool): ${PROFILE_WORKERS}"
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

# Also need POT (import name 'ot') for gudhi.wasserstein -- required by the
# TDA channel of the ensemble-complete attack (matches the CIFAR-100 box fix).
python -c "import ot" 2>/dev/null || { echo "Installing POT (import name 'ot', needed by gudhi.wasserstein) ..."; pip install --no-cache-dir POT; }

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
  echo ""; echo "=== Building WRN-28-10 detector (missing artifacts) ==="

  # 0a: backbone. WRN-28-10 has its own dedicated pretrain script (no
  # --dataset/--num-classes -- WRN-28-10/CIFAR-10 is its only target, unlike
  # pretrain_cifar_backbone.py which is shared across CIFAR-10/CIFAR-100).
  if [ -f "$CKPT" ]; then
    echo "  backbone present: $CKPT (skip pretrain)"
  else
    echo "  pretraining WRN-28-10 backbone -> $CKPT (this is slow: ~146MB model, 200 epochs)"
    python scripts/pretrain_wrn_backbone.py \
      --output "$CKPT" --min-test-acc "$BACKBONE_MIN_ACC" \
      2>&1 | tee "logs/${TAG}/build_0a_backbone.log"
  fi

  # 1: reference profiles (--workers fans the CPU-bound ripser calls across a
  # process pool; numerically identical to the serial path per the script's
  # own documentation -- same subsample, same ripser call, only faster).
  if [ -f "$REF" ]; then
    echo "  reference profiles present: $REF (skip)"
  else
    echo "  building reference profiles -> $REF (workers=$PROFILE_WORKERS)"
    python scripts/build_profile_testset.py --config "$CONFIG" \
      --workers "$PROFILE_WORKERS" \
      2>&1 | tee "logs/${TAG}/build_1_profiles.log"
  fi

  # 2: ensemble scorer (balanced FGSM/PGD/Square; CW/AA stay eval-only)
  if [ -f "$ENS" ]; then
    echo "  ensemble scorer present: $ENS (skip)"
  else
    echo "  training ensemble scorer -> $ENS"
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
  # harmless to build now and keeps the WRN model dir complete).
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

  # Each seed is an independent, self-seeded, deterministic process with its
  # own output/checkpoint files (results_..._seed${s}.json{,l}) -- there is no
  # shared state between them, so running up to PARALLEL_SEEDS at once changes
  # only wall-clock, never a result. Concurrency is capped (`wait -n`-style
  # pool) so we never spawn more processes than PARALLEL_SEEDS at a time.
  # `wait "$pid1" "$pid2" ...` only reports the exit status of the LAST pid
  # given -- an earlier seed's failure would otherwise be silently swallowed
  # under `set -e`. Track (seed, pid) pairs and check each exit code
  # individually so a failed seed is never missed.
  local pids=() pid_seeds=() running=0 any_failed=0

  drain_batch() {
    local i
    for i in "${!pids[@]}"; do
      if ! wait "${pids[$i]}"; then
        echo "  ERROR: seed ${pid_seeds[$i]} failed (see $logdir/ec_adaptive_seed${pid_seeds[$i]}.log)"
        any_failed=1
      fi
    done
    pids=(); pid_seeds=(); running=0
  }

  for s in $seeds; do
    local out_json="$outdir/results_ensemble_complete_adaptive_pgd_seed${s}.json"
    local out_jsonl="$outdir/results_ensemble_complete_adaptive_pgd_seed${s}.jsonl"
    local gpu_idx=0
    [ "$GPU_COUNT" -gt 0 ] && gpu_idx=$(( running % GPU_COUNT ))
    local dev_args=()
    [ "$GPU_COUNT" -gt 0 ] && dev_args=(--device "cuda:${gpu_idx}")

    echo "  --- launching seed $s (gpu=${gpu_idx}, threads=${THREADS_PER_JOB}) -> $out_json"
    (
      export OMP_NUM_THREADS="$THREADS_PER_JOB" MKL_NUM_THREADS="$THREADS_PER_JOB"
      python experiments/evaluation/run_adaptive_pgd.py \
        --config "$CONFIG" \
        --n-test "$n_test" \
        --seed "$s" \
        --lambdas $lambdas \
        --pgd-steps "$STEPS" \
        --pgd-restarts "$RESTARTS" \
        --eot-samples "$EOT_SAMPLES" \
        --eot-verify-samples "$EOT_VERIFY_SAMPLES" \
        "${extra[@]}" "${dev_args[@]+${dev_args[@]}}" \
        --checkpoint-jsonl "$out_jsonl" \
        --resume \
        --output "$out_json"
    ) > "$logdir/ec_adaptive_seed${s}.log" 2>&1 &
    pids+=($!)
    pid_seeds+=("$s")
    running=$((running + 1))

    if [ "$running" -ge "$PARALLEL_SEEDS" ]; then
      drain_batch
    fi
  done
  [ "${#pids[@]}" -gt 0 ] && drain_batch

  # Surface each seed's log tail so failures are visible without opening files.
  for s in $seeds; do
    echo "  --- tail: seed $s ---"
    tail -n 5 "$logdir/ec_adaptive_seed${s}.log" 2>/dev/null || echo "  (no log found)"
  done

  if [ "$any_failed" = "1" ]; then
    echo "  ERROR: one or more seeds in phase ${tag} failed."
    return 1
  fi
}

# ── run the unreported lambda scan with independent lambda workers ───────────
# `run_adaptive_pgd.py` evaluates every lambda independently: it reloads PRISM
# per lambda, derives the same sample list from `--seed`, and writes a complete
# result dictionary for that lambda.  Splitting these cells into separate OS
# processes does not modify the experimental inputs, attack configuration, or
# metric computation; it increases GPU throughput.  Each worker has its own
# output, checkpoint, and log so concurrent writers never share state.
run_parallel_lambda_scan() {
  local tag="$1" n_test="$2" lambdas="$3" seeds="$4"
  local outdir="experiments/evaluation/${tag}"
  local logdir="logs/${tag}"
  mkdir -p "$outdir" "$logdir"
  echo ""; echo "=== PARALLEL SCAN ${tag}: n=${n_test} lambdas=[${lambdas}] seeds=[${seeds}] workers=${PARALLEL_SCAN_LAMBDAS} ==="

  local extra=(--ensemble-complete --through-scorer
               --ec-match-coeff "$EC_MATCH_COEFF" --ec-score-coeff "$EC_SCORE_COEFF")
  [ "$SKIP_GRADNORM" = "1" ] && extra+=(--skip-gradnorm-surrogate)

  local pids=() job_labels=() running=0 any_failed=0
  drain_parallel_scan_batch() {
    local i
    for i in "${!pids[@]}"; do
      if ! wait "${pids[$i]}"; then
        echo "  ERROR: scan cell ${job_labels[$i]} failed (see $logdir/ec_adaptive_${job_labels[$i]}.log)"
        any_failed=1
      fi
    done
    pids=(); job_labels=(); running=0
  }

  local s lam safe_lam label out_json out_jsonl gpu_idx
  for s in $seeds; do
    for lam in $lambdas; do
      safe_lam="${lam//./_}"
      safe_lam="${safe_lam//-/m}"
      label="seed${s}_lambda${safe_lam}"
      out_json="$outdir/results_ensemble_complete_adaptive_pgd_${label}.json"
      out_jsonl="$outdir/results_ensemble_complete_adaptive_pgd_${label}.jsonl"
      gpu_idx=0

      echo "  --- launching scan cell seed=${s} lambda=${lam} (gpu=${gpu_idx}, threads=${THREADS_PER_JOB}) -> $out_json"
      (
        export OMP_NUM_THREADS="$THREADS_PER_JOB" MKL_NUM_THREADS="$THREADS_PER_JOB"
        python experiments/evaluation/run_adaptive_pgd.py \
          --config "$CONFIG" \
          --n-test "$n_test" \
          --seed "$s" \
          --lambdas "$lam" \
          --pgd-steps "$STEPS" \
          --pgd-restarts "$RESTARTS" \
          --eot-samples "$EOT_SAMPLES" \
          --eot-verify-samples "$EOT_VERIFY_SAMPLES" \
          "${extra[@]}" \
          --device "cuda:${gpu_idx}" \
          --checkpoint-jsonl "$out_jsonl" \
          --resume \
          --output "$out_json"
      ) > "$logdir/ec_adaptive_${label}.log" 2>&1 &
      pids+=($!)
      job_labels+=("$label")
      running=$((running + 1))

      if [ "$running" -ge "$PARALLEL_SCAN_LAMBDAS" ]; then
        drain_parallel_scan_batch
      fi
    done
  done
  [ "${#pids[@]}" -gt 0 ] && drain_parallel_scan_batch

  if [ "$any_failed" = "1" ]; then
    echo "  ERROR: one or more parallel scan cells failed."
    return 1
  fi
}

# ── choose the worst-case (highest deployment-risk) lambda from the scan ─────
choose_worst_lambda() {
  SCAN_DIR="experiments/evaluation/${SCAN_TAG}" WORST_LAMBDA_FILE="logs/${SCAN_TAG}/prism_ec_worst_lambda.txt" python - <<'PY'
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
out_file = Path(os.environ["WORST_LAMBDA_FILE"])
out_file.parent.mkdir(parents=True, exist_ok=True)
out_file.write_text(worst["lambda"] + "\n")
print(json.dumps({"worst": worst, "summary": summary}, indent=2))
PY
}

# ── Stage A: lambda scan (cheap — SCAN_SEEDS, default 1 seed) ────────────────
if [ "$PARALLEL_SCAN_LAMBDAS" -gt 1 ]; then
  run_parallel_lambda_scan "$SCAN_TAG" "$SCAN_N" "$SCAN_LAMBDAS" "$SCAN_SEEDS" || { echo "ERROR: parallel scan phase failed."; exit 4; }
else
  run_phase "$SCAN_TAG" "$SCAN_N" "$SCAN_LAMBDAS" "$SCAN_SEEDS" || { echo "ERROR: scan phase failed."; exit 4; }
fi

echo ""; echo "=== Selecting worst-case lambda from scan ==="
choose_worst_lambda | tee "logs/${SCAN_TAG}/worst_lambda.log"
WORST_LAMBDA="$(cat "logs/${SCAN_TAG}/prism_ec_worst_lambda.txt")"
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
echo "CIFAR-10/ResNet-18 result (all-input TPR 0.479) and the CIFAR-100/"
echo "ResNet-18 result reported in the paper's revision."
echo "============================================================"
