#!/usr/bin/env bash
# ============================================================
# PRISM n=1000 ThunderCompute Full Evaluation
# Excludes: CW, AutoAttack (to be run later)
# Instance: uqa02qde | A100-SXM4-80GB
# Date: 2026-04-21
# ============================================================
set -euo pipefail

REPO_ARCHIVE="$HOME/prism_repo.tar.gz"
WORKDIR="$HOME/prism-run"
PRISM_DIR="$WORKDIR/prism-adversarial-defense/prism"
LOGDIR="$PRISM_DIR/logs"
RUNDATE=$(date +%Y%m%d)

# ── Colours ────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log()  { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $*"; }
pass() { echo -e "${GREEN}[PASS]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }

# ── GPU / env check ────────────────────────────────────────
log "=== ENVIRONMENT ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python3 --version
echo "CUDA devices: $(python3 -c 'import torch; print(torch.cuda.device_count(), "x", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")')" 2>/dev/null || true

# ── Step 0: Extract repo archive ──────────────────────────
log "=== STEP 0: EXTRACT REPO ==="
if [ -d "$WORKDIR" ]; then
    warn "Work directory already exists — removing and re-extracting"
    rm -rf "$WORKDIR"
fi
mkdir -p "$WORKDIR/prism-adversarial-defense"
tar -xzf "$REPO_ARCHIVE" -C "$WORKDIR/prism-adversarial-defense"
cd "$PRISM_DIR"
log "Repo extracted to: $PRISM_DIR"

# ── Step 1: Virtual environment ────────────────────────────
log "=== STEP 1: VENV & DEPS ==="
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip wheel --quiet

# Install requirements
pip install -r requirements.txt --quiet
pip install adversarial-robustness-toolbox autoattack --quiet

# GPU-aware PyTorch (already should be CUDA on this image)
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA NOT AVAILABLE — aborting'
print(f'Torch: {torch.__version__}  CUDA: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

# Env vars
export PYTHONIOENCODING=utf-8
export SSL_CERT_FILE=$(python3 -c 'import certifi; print(certifi.where())')
export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE

mkdir -p "$LOGDIR"

# ── Phase A: Preflight verification ───────────────────────
log "=== PHASE A: PREFLIGHT ==="
FAIL_COUNT=0

check_grep() {
    local pattern="$1" file="$2" label="$3"
    if grep -qn "$pattern" "$file" 2>/dev/null; then
        pass "$label"
    else
        warn "PREFLIGHT: $label not found — may be using default params, continuing"
    fi
}

check_grep "default=3000"      scripts/train_ensemble_scorer.py  "n-train=3000 default"
check_grep "default=1.5"       scripts/train_ensemble_scorer.py  "fgsm-oversample=1.5 default"
check_grep "alpha=0.4"         scripts/train_ensemble_scorer.py  "alpha=0.4"
check_grep "n_subsample: 150"  configs/default.yaml              "TDA n_subsample=150"
check_grep "cal_alpha_factor: 0.7" configs/default.yaml          "cal_alpha_factor=0.7"

# ── Phase B: Artifact rebuild ──────────────────────────────
log "=== PHASE B1: BUILD PROFILE (test[0:5000]) ==="
python3 scripts/build_profile_testset.py \
    2>&1 | tee "$LOGDIR/B1_build_profile.log"
B1_EXIT=${PIPESTATUS[0]}
log "B1 exit: $B1_EXIT"
[ "$B1_EXIT" -eq 0 ] || fail "Phase B1 failed"
pass "B1 complete"

log "=== PHASE B2: TRAIN ENSEMBLE SCORER ==="
python3 scripts/train_ensemble_scorer.py \
    --n-train 3000 \
    --fgsm-oversample 1.5 \
    2>&1 | tee "$LOGDIR/B2_train_ensemble.log"
B2_EXIT=${PIPESTATUS[0]}
log "B2 exit: $B2_EXIT"
[ "$B2_EXIT" -eq 0 ] || fail "Phase B2 failed"
pass "B2 complete"

log "=== PHASE B3: CALIBRATE ENSEMBLE ==="
python3 scripts/calibrate_ensemble.py \
    2>&1 | tee "$LOGDIR/B3_calibrate.log"
B3_EXIT=${PIPESTATUS[0]}
log "B3 exit: $B3_EXIT"
[ "$B3_EXIT" -eq 0 ] || fail "Phase B3 failed"
pass "B3 complete"

log "=== PHASE B4: VALIDATION FPR GATE ==="
python3 scripts/compute_ensemble_val_fpr.py \
    2>&1 | tee "$LOGDIR/B4_val_fpr.log"
B4_EXIT=${PIPESTATUS[0]}
log "B4 exit: $B4_EXIT"
[ "$B4_EXIT" -eq 0 ] || fail "Phase B4 failed"

# FPR gate check — ABORT if not passing
python3 -c "
import json, sys
try:
    with open('experiments/calibration/ensemble_fpr_report.json') as f:
        r = json.load(f)
except FileNotFoundError:
    print('[WARN] ensemble_fpr_report.json not found — skipping gate check')
    sys.exit(0)

failed = False
for tier, tgt in [('L1', 0.10), ('L2', 0.03), ('L3', 0.005)]:
    fpr = r.get(f'FPR_{tier}_plus', r.get(tier, {}).get('FPR', 999))
    status = 'PASS' if fpr <= tgt else 'FAIL'
    print(f'{tier} FPR={fpr:.4f}  target={tgt}  [{status}]')
    if fpr > tgt:
        failed = True

if failed:
    print('\n[ABORT] FPR gate FAILED — do NOT proceed to evaluation.')
    print('Fix: reduce cal_alpha_factor in configs/default.yaml (e.g. 0.7->0.6), re-run B3+B4.')
    sys.exit(1)
else:
    print('\n[PASS] All FPR tiers pass — proceeding to evaluation.')
"
FPR_EXIT=$?
[ "$FPR_EXIT" -eq 0 ] || fail "FPR gate failed — aborting to save money"

log "=== PHASE B5: SANITY CHECKS ==="
python3 sanity_checks.py \
    2>&1 | tee "$LOGDIR/B5_sanity.log"
B5_EXIT=${PIPESTATUS[0]}
log "B5 exit: $B5_EXIT"
[ "$B5_EXIT" -eq 0 ] || fail "Phase B5 sanity checks failed"
pass "All Phase B checks passed"

# ── Phase C1: Multi-seed eval (FGSM/PGD/Square, 5 seeds, n=1000) ──
log "=== PHASE C1: MULTI-SEED EVAL (n=1000, FGSM/PGD/Square, 5 seeds) ==="
C1_OUTPUT="experiments/evaluation/results_n1000_multiseed_${RUNDATE}.json"
python3 experiments/evaluation/run_evaluation_full.py \
    --n-test 1000 \
    --attacks FGSM PGD Square \
    --multi-seed \
    --seeds 42 123 456 789 999 \
    --device cuda \
    --square-max-iter 5000 \
    --output "$C1_OUTPUT" \
    --checkpoint-interval 200 \
    2>&1 | tee "$LOGDIR/C1_multiseed.log"
C1_EXIT=${PIPESTATUS[0]}
log "C1 exit: $C1_EXIT"
[ "$C1_EXIT" -eq 0 ] || fail "Phase C1 failed"

# C1 acceptance check
python3 -c "
import json, sys, glob
files = sorted(glob.glob('${C1_OUTPUT}'))
if not files:
    print('[WARN] C1 output file not found')
    sys.exit(0)
try:
    with open(files[0]) as f:
        r = json.load(f)
except Exception as e:
    print(f'[WARN] Cannot check C1 results: {e}')
    sys.exit(0)

print('\n=== C1 ACCEPTANCE CHECK ===')
failed = False
attacks_data = r.get('attacks', r.get('results', {}))
for atk, target in [('FGSM', 0.85), ('PGD', 0.95), ('Square', 0.85)]:
    data = attacks_data.get(atk, {})
    tpr = data.get('tpr', data.get('TPR', data.get('detection_rate', None)))
    if tpr is None:
        print(f'  {atk}: TPR not found in result JSON — check structure')
        continue
    status = 'PASS' if tpr >= target else 'FAIL'
    ci = data.get('ci_95', data.get('wilson_ci', '?'))
    print(f'  {atk}: TPR={tpr:.4f}  target>={target}  [{status}]  CI={ci}')
    if tpr < target:
        failed = True

if failed:
    print('\n[WARNING] Some TPR targets not met. Check logs before proceeding.')
else:
    print('\n[PASS] All C1 TPR targets met.')
"
pass "C1 complete"

# ── Phase D1: Baselines ────────────────────────────────────
log "=== PHASE D1: BASELINES (3 seeds, n=1000) ==="
for SEED in 42 123 456; do
    log "D1 baselines seed=$SEED"
    python3 experiments/evaluation/run_baselines.py \
        --n-test 1000 \
        --attacks FGSM PGD Square \
        --seed $SEED \
        --output "experiments/evaluation/results_baselines_n1000_seed${SEED}_${RUNDATE}.json" \
        2>&1 | tee -a "$LOGDIR/D1_baselines.log"
    D1_EXIT=${PIPESTATUS[0]}
    log "D1 seed=$SEED exit: $D1_EXIT"
    [ "$D1_EXIT" -eq 0 ] || warn "D1 seed=$SEED failed — continuing"
done
pass "D1 complete"

# ── Phase D2: Ablation ─────────────────────────────────────
log "=== PHASE D2: ABLATION (n=1000) ==="
python3 experiments/ablation/run_ablation_paper.py \
    --n-test 1000 \
    --output "experiments/ablation/results_ablation_n1000_${RUNDATE}.json" \
    2>&1 | tee "$LOGDIR/D2_ablation.log"
D2_EXIT=${PIPESTATUS[0]}
log "D2 exit: $D2_EXIT"
[ "$D2_EXIT" -eq 0 ] || warn "D2 ablation failed — continuing"
pass "D2 complete"

# ── Phase D3: Adaptive PGD ─────────────────────────────────
log "=== PHASE D3: ADAPTIVE PGD --through-scorer (n=1000) ==="
python3 experiments/evaluation/run_adaptive_pgd.py \
    --n-test 1000 \
    --pgd-steps 40 \
    --lambdas 0.0 0.5 1.0 2.0 5.0 \
    --through-scorer \
    --output "experiments/evaluation/results_adaptive_pgd_n1000_${RUNDATE}.json" \
    2>&1 | tee "$LOGDIR/D3_adaptive.log"
D3_EXIT=${PIPESTATUS[0]}
log "D3 exit: $D3_EXIT"
[ "$D3_EXIT" -eq 0 ] || warn "D3 adaptive PGD failed — continuing"
pass "D3 complete"

# ── Phase D4: CIFAR-100 generalization ────────────────────
log "=== PHASE D4: CIFAR-100 GENERALIZATION (n=500) ==="
python3 experiments/generalization/run_cifar100.py \
    --n-clean 500 \
    --n-adv 500 \
    --seed 42 \
    --output "experiments/generalization/results_cifar100_n500_${RUNDATE}.json" \
    2>&1 | tee "$LOGDIR/D4_cifar100.log"
D4_EXIT=${PIPESTATUS[0]}
log "D4 exit: $D4_EXIT"
[ "$D4_EXIT" -eq 0 ] || warn "D4 CIFAR-100 failed — continuing"
pass "D4 complete"

# ── Phase D5: Campaign detection ──────────────────────────
log "=== PHASE D5: CAMPAIGN DETECTION (10 trials) ==="
python3 experiments/campaign/run_campaign_real.py \
    --n-trials 10 \
    --output "experiments/campaign/results_campaign_n1000_${RUNDATE}.json" \
    2>&1 | tee "$LOGDIR/D5_campaign.log"
D5_EXIT=${PIPESTATUS[0]}
log "D5 exit: $D5_EXIT"
[ "$D5_EXIT" -eq 0 ] || warn "D5 campaign failed — continuing"
pass "D5 complete"

# ── Phase E: Aggregate results ─────────────────────────────
log "=== PHASE E: AGGREGATE RESULTS ==="
REPORT="experiments/evaluation/run_report_n1000_thundercompute_${RUNDATE}.md"

python3 -c "
import json, glob, os

rundate = '${RUNDATE}'
results = {}
patterns = {
    'multiseed':    f'experiments/evaluation/results_n1000_multiseed_{rundate}.json',
    'baselines_42': f'experiments/evaluation/results_baselines_n1000_seed42_{rundate}.json',
    'baselines_123':f'experiments/evaluation/results_baselines_n1000_seed123_{rundate}.json',
    'baselines_456':f'experiments/evaluation/results_baselines_n1000_seed456_{rundate}.json',
    'ablation':     f'experiments/ablation/results_ablation_n1000_{rundate}.json',
    'adaptive_pgd': f'experiments/evaluation/results_adaptive_pgd_n1000_{rundate}.json',
    'cifar100':     f'experiments/generalization/results_cifar100_n500_{rundate}.json',
    'campaign':     f'experiments/campaign/results_campaign_n1000_{rundate}.json',
}
for name, path in patterns.items():
    if os.path.exists(path):
        try:
            with open(path) as f:
                results[name] = json.load(f)
            print(f'  Loaded: {path}')
        except Exception as e:
            print(f'  WARN: Cannot load {path}: {e}')
    else:
        print(f'  MISSING: {path}')

combined_path = f'experiments/evaluation/results_n1000_combined_{rundate}.json'
with open(combined_path, 'w') as f:
    json.dump({'rundate': rundate, 'results': results}, f, indent=2)
print(f'\nCombined results written to: {combined_path}')
"

# Print final summary
log "=== FINAL SUMMARY ==="
python3 -c "
import json, os, glob

rundate_files = sorted(glob.glob('experiments/evaluation/results_n1000_multiseed_*.json'))
if not rundate_files:
    print('[WARN] No multiseed result file found')
else:
    path = rundate_files[-1]
    try:
        with open(path) as f:
            r = json.load(f)
        print(f'\nMain eval result: {path}')
        attacks_data = r.get('attacks', r.get('results', {}))
        for atk in ['FGSM', 'PGD', 'Square']:
            data = attacks_data.get(atk, {})
            tpr = data.get('tpr', data.get('TPR', data.get('detection_rate', '?')))
            ci = data.get('ci_95', data.get('wilson_ci', '?'))
            print(f'  {atk}: TPR={tpr}  CI={ci}')
        lat = r.get('latency', {})
        if lat:
            print(f\"  Latency: mean={lat.get('mean_ms','?')}ms  p95={lat.get('p95_ms','?')}ms\")
    except Exception as e:
        print(f'[WARN] Error reading result: {e}')
"

log "=== ALL PHASES COMPLETE ==="
echo ""
echo -e "${GREEN}${BOLD}Run complete. Results in: $PRISM_DIR/experiments/${NC}"
echo "Logs in: $LOGDIR"
echo ""
echo "To copy results back to local machine:"
echo "  scp -P 32208 -i ~/.thunder/keys/uqa02qde.pem -r ubuntu@64.247.206.140:$PRISM_DIR/experiments/ ./remote_results/"
echo "  scp -P 32208 -i ~/.thunder/keys/uqa02qde.pem -r ubuntu@64.247.206.140:$PRISM_DIR/logs/ ./remote_logs/"
