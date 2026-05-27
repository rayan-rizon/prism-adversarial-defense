import json

print('=== LATENCY BREAKDOWN ===')
d = json.load(open('prism/experiments/stress/vastai_latency_breakdown.json'))
print(f'device: {d.get("gpu_name")}, n={d["n_samples"]}, split={d["split"]}')
for r in d['stages']:
    print(f'  {r["stage"]:<20s}  mean={r["mean_ms"]:>7.2f}  p50={r["p50_ms"]:>7.2f}  p95={r["p95_ms"]:>7.2f}  {r["pct_of_total"]:>5.1f}%')

print()
print('=== STRONGER CW (pooled 5 seeds x n=1000) ===')
d = json.load(open('prism/experiments/stress/vastai_stronger_cw.json'))
a = d['aggregate']
print(f'pooled clean: mean={a["clean"]["mean"]:.3f}, std={a["clean"]["std"]:.3f}, '
      f'FPR L1={a["clean"]["FPR"]["L1"]:.4f}, L2={a["clean"]["FPR"]["L2"]:.4f}, L3={a["clean"]["FPR"]["L3"]:.4f}')
for k in sorted(a):
    if not k.startswith('kappa_'):
        continue
    v = a[k]
    ci1 = v['TPR_L1_CI95']
    print(f'  {k}: succ={v["attack_success_mean"]:.3f}  L2={v["L2_mean"]:.3f}  '
          f'TPR L1={v["TPR_L1"]:.4f} [{ci1[0]:.3f},{ci1[1]:.3f}]  '
          f'L2={v["TPR_L2"]:.4f}  L3={v["TPR_L3"]:.4f}')

print()
print('=== MULTI-ATTACK RECOVERY (pooled 5 seeds x n=1000) ===')
d = json.load(open('prism/experiments/stress/vastai_recovery_multi.json'))
a = d['aggregate']
print(f'L3 threshold: {d["L3_threshold"]:.4f}')
print(f'\n  attack | n_L3   | trig  | pass  | uni (gap)        | topo (gap)        | force | oracle')
print('  ' + '-' * 95)
for attack in ['FGSM', 'Square', 'CW']:
    v = a[attack]
    if v.get('n_L3_total', 0) == 0:
        print(f'  {attack:>6s}: no L3')
        continue
    print(f'  {attack:>6s} | {v["n_L3_total"]:>4d} | '
          f'{v["L3_trigger_rate_mean"]:.3f} | '
          f'{v["pass_acc"]:.3f} | '
          f'{v["uniform_acc"]:.3f} ({v["gap_uniform_vs_pass_pp"]:+5.1f}pp)  | '
          f'{v["topology_acc"]:.3f} ({v["gap_topo_vs_pass_pp"]:+5.1f}pp)  | '
          f'{v["force_pgd_acc"]:.3f} | {v["oracle_acc"]:.3f}')
    ci = v['topology_CI95']
    print(f'         topology CI95={[round(c,3) for c in ci]}')
