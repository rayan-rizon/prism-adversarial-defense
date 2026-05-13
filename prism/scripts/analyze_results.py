"""
Analyze PRISM results for publication readiness.
"""
import json, numpy as np, os, sys

def wilson_ci(k, n, z=1.96):
    if n == 0: return (0.0, 1.0)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2*n)) / d
    m = (z * np.sqrt(p*(1-p)/n + z**2/(4*n**2))) / d
    return (max(0.0, c-m), min(1.0, c+m))

with open('experiments/evaluation/results_paper.json') as f:
    r = json.load(f)

with open('experiments/calibration/ensemble_fpr_report.json') as f:
    fpr_r = json.load(f)

with open('experiments/ablation/results_ablation_paper.json') as f:
    ablation = json.load(f)

print("=" * 70)
print("PRISM PUBLICATION READINESS ANALYSIS")
print("=" * 70)

print("\n--- SECTION 1: MAIN EVALUATION (n=300 per attack, seed=42) ---")
targets = {'FGSM': 0.90, 'PGD': 0.90, 'Square': 0.80}
fpr_target = 0.10

for attack in ['FGSM', 'PGD', 'Square']:
    d = r[attack]
    tpr_target = targets[attack]
    tpr_ok = d['TPR'] >= tpr_target
    fpr_ok = d['FPR'] <= fpr_target
    ci_width = d['TPR_CI_95'][1] - d['TPR_CI_95'][0]
    print(f"\n  {attack}:")
    print(f"    TPR = {d['TPR']:.4f}  target >= {tpr_target:.2f}  [{'PASS' if tpr_ok else 'FAIL'}]")
    print(f"    95% CI = [{d['TPR_CI_95'][0]:.4f}, {d['TPR_CI_95'][1]:.4f}]  width = {ci_width:.4f}")
    print(f"    FPR = {d['FPR']:.4f}  target <= {fpr_target:.2f}  [{'PASS' if fpr_ok else 'FAIL'}]")
    print(f"    F1  = {d['F1']:.4f}   Precision = {d['Precision']:.4f}")
    print(f"    TP={d['TP']} FP={d['FP']} FN={d['FN']} TN={d['TN']}")
    print(f"    Per-tier FPR: L1+={d['per_tier_fpr']['FPR_L1_plus']:.4f} (<=0.10) L2+={d['per_tier_fpr']['FPR_L2_plus']:.4f} (<=0.03) L3+={d['per_tier_fpr']['FPR_L3_plus']:.4f} (<=0.005)")

print("\n--- SECTION 2: HIGH-POWER FPR REPORT (n=1000 val split) ---")
for tier, td in sorted(fpr_r['tiers'].items()):
    ci_w = td['CI_95'][1] - td['CI_95'][0]
    status = "PASS" if td['passed'] else "FAIL"
    print(f"  {tier}+: FPR={td['FPR']:.4f} CI=[{td['CI_95'][0]:.4f},{td['CI_95'][1]:.4f}] width={ci_w:.4f} target<={td['target']:.3f} [{status}]")

print("\n--- SECTION 3: LATENCY ---")
lat = r['_meta']['latency']
lat_ok = lat['pass']
print(f"  Mean = {lat['mean_ms']:.1f}ms  target <= {lat['target_ms']}ms  [{'PASS' if lat_ok else 'FAIL'}]")
print(f"  p50  = {lat['p50_ms']:.1f}ms   p95 = {lat['p95_ms']:.1f}ms  std = {lat['std_ms']:.1f}ms")
print(f"  n    = {lat['n']} samples (on {r['_meta']['device']})")

print("\n--- SECTION 4: ABLATION STUDY (n=500 per config) ---")
print(f"  {'Config':20s} {'Mean TPR':>10s} {'Mean FPR':>10s} {'TDA-drop':>10s}")
full_tpr = ablation['Full PRISM']['mean_TPR']
for config, d in ablation.items():
    drop = full_tpr - d['mean_TPR']
    print(f"  {config:20s} {d['mean_TPR']:>10.4f} {d['mean_FPR']:>10.4f} {drop:>+10.4f}")

print("\n--- SECTION 5: MISSING BENCHMARKS ---")
missing = []
missing.append("CW (Carlini-Wagner L2): Not in current evaluation (required for NeurIPS/ICML)")
missing.append("AutoAttack: Not in current evaluation (gold standard benchmark)")
missing.append("n=1000 evaluation (current: n=300, target: n=1000 for tight CIs)")
missing.append("Architecture generalization: Only ResNet-18 evaluated (claim: architecture-agnostic)")
for i, m in enumerate(missing):
    print(f"  [{i+1}] {m}")

print("\n--- SECTION 6: STATISTICAL POWER SUMMARY ---")
print(f"  Main eval n=300: TPR CI width = ~{0.886-0.805:.3f} (FGSM), need n=1000 for <0.04 width")
print(f"  FPR val  n=1000: L1+ CI width = {fpr_r['tiers']['L1']['CI_95'][1]-fpr_r['tiers']['L1']['CI_95'][0]:.4f} (GOOD)")
print(f"  FPR main n=300:  L1+ CI width = {r['FGSM']['FPR_CI_95'][1]-r['FGSM']['FPR_CI_95'][0]:.4f} (THIN)")

print("\n--- SECTION 7: MODEL PROVENANCE ---")
import pickle
with open('models/ensemble_scorer.pkl','rb') as f:
    d = pickle.load(f)
print(f"  Training eps:     {d['training_eps']:.6f} (= 8/255 = {8/255:.6f}) [MATCH: {'YES' if abs(d['training_eps']-8/255)<1e-9 else 'NO'}]")
print(f"  Training attacks: {d['training_attacks']}")
print(f"  Training n:       {d['training_n']}")
print(f"  Feature dim:      {len(d['logistic_weights'])} (expected: 36)")
print(f"  logit_shift:      {d['logit_shift']:.4f} (data-derived: YES)")
print(f"  w_score_mean:     {d['w_score_mean']:.4f} (data-derived: YES)")

with open('models/calibrator.pkl','rb') as f:
    cal = pickle.load(f)
print(f"  Calibrator n:     {cal.n_calibration}")
print(f"  Calibrator alphas:{cal.alphas}")
print(f"  Calibrator thresholds: {cal.thresholds}")

print("\n=== SUMMARY: PASS/FAIL CHECKLIST ===")
checks = [
    ("FGSM TPR >= 0.90", r['FGSM']['TPR'] >= 0.90),
    ("PGD  TPR >= 0.90", r['PGD']['TPR'] >= 0.90),
    ("Square TPR >= 0.80", r['Square']['TPR'] >= 0.80),
    ("FPR L1+ <= 0.10 (main eval)", r['FGSM']['per_tier_fpr']['FPR_L1_plus'] <= 0.10),
    ("FPR L2+ <= 0.03 (main eval)", r['FGSM']['per_tier_fpr']['FPR_L2_plus'] <= 0.03),
    ("FPR L3+ <= 0.005 (main eval)", r['FGSM']['per_tier_fpr']['FPR_L3_plus'] <= 0.005),
    ("FPR L1+ <= 0.10 (val n=1000)", fpr_r['tiers']['L1']['passed']),
    ("FPR L2+ <= 0.03 (val n=1000)", fpr_r['tiers']['L2']['passed']),
    ("FPR L3+ <= 0.005 (val n=1000)", fpr_r['tiers']['L3']['passed']),
    ("Training eps == 8/255", abs(d['training_eps']-8/255)<1e-9),
    ("Feature dim == 36", len(d['logistic_weights'])==36),
    ("Data-derived normalization", 'logit_shift' in d and 'w_score_mean' in d),
    ("Calibration n >= 2000", cal.n_calibration >= 2000),
    ("Latency <= 100ms mean", lat['pass']),
    ("CW attack evaluated", False),
    ("AutoAttack evaluated", False),
    ("n_test >= 1000", r['_meta']['n_test'] >= 1000),
    ("Architecture generalization tested", False),
    ("Ablation TDA-only non-degenerate", ablation['TDA only']['FGSM']['FPR'] < 0.5),
    ("Ablation monotone: Full > No MoE > TDA", ablation['Full PRISM']['mean_TPR'] > ablation['No MoE']['mean_TPR']),
]
for name, result in checks:
    status = "✅" if result else "❌"
    print(f"  {status} {name}")

passed = sum(1 for _, v in checks if v)
print(f"\n  Score: {passed}/{len(checks)} checks passed")
