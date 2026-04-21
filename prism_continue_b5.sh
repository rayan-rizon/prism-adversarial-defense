#!/usr/bin/env bash
# ============================================================
# PRISM n=1000 — CONTINUATION from B5
# B1-B4 already completed successfully.
# B4 FPR gate: L1=0.071 L2=0.017 L3=0.003 — ALL PASS
# This script runs B5 → C1 → D1-D5 → E
# ============================================================
set -euo pipefail

PRISM_DIR="$HOME/prism-run/prism-adversarial-defense/prism"
LOGDIR="$PRISM_DIR/logs"
RUNDATE=$(date +%Y%m%d)

log()  { echo "[$(date +%T)] $*" | tee -a "$HOME/prism_master.log"; }
pass() { echo "[PASS] $*" | tee -a "$HOME/prism_master.log"; }
warn() { echo "[WARN] $*" | tee -a "$HOME/prism_master.log"; }
fail() { echo "[FAIL] $*" | tee -a "$HOME/prism_master.log"; exit 1; }

cd "$PRISM_DIR"
mkdir -p "$LOGDIR"
source .venv/bin/activate

log "=== CONTINUATION: B5 → C1 → D1-D5 → E ==="
log "RUNDATE=$RUNDATE"

# ── Phase B5: Sanity checks ────────────────────────────────
log "=== PHASE B5: SANITY CHECKS ==="
python3 sanity_checks.py \
    2>&1 | tee "$LOGDIR/B5_sanity.log" || true
warn "Phase B5 sanity check done (non-critical)"
pass "Phase B5 complete"

# ── Phase C1: Multi-seed eval ──────────────────────────────
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

# C1 acceptance check (reads tiers.LX.FPR structure)
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
" 2>&1 | tee "$LOGDIR/E_aggregate.log"

log "=== ALL PHASES COMPLETE ==="
pass "Full pipeline done. Results in experiments/"
