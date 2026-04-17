"""
PRISM Master Pipeline — Run All Experiments in Correct Order

This script orchestrates the complete experimental pipeline needed
to produce paper-quality results.  It replaces the ad-hoc sequence
of individual scripts and ensures correct data flow.

PIPELINE ORDER (must not be reordered — each step depends on outputs of prior):

  Phase 1 — Profile (TDA self-profile from test set)
    scripts/build_profile_testset.py
      → models/reference_profiles.pkl
      → models/scorer.pkl
      → experiments/calibration/clean_scores.npy      (cal split)
      → experiments/calibration/val_scores.npy         (val split)
      → experiments/calibration/profile_scores.npy     (profile split)

  Phase 2 — Calibrate (conformal thresholds, test-set cal split)
    scripts/calibrate_testset.py
      → models/calibrator.pkl
      Verifies FPR guarantee on val split

  Phase 3 — Train ensemble scorer (logistic regression on persistence features)
    scripts/train_ensemble_scorer.py
      → models/ensemble_scorer.pkl
      Trained on CIFAR-10 TRAINING set — no test-set leakage

  Phase 4 — Train MoE experts (Topology-Aware MoE Self-Healing)
    scripts/train_experts.py
      → models/experts.pkl

  Phase 5 — Full evaluation (n=1000, standard ε, proper splits)
    experiments/evaluation/run_evaluation_full.py
      → experiments/evaluation/results_paper.json

  Phase 6 — Campaign detection
    experiments/campaign/run_campaign.py
      → experiments/campaign/results.json

  Phase 7 — Paper-quality ablation (n=500, multi-attack)
    experiments/ablation/run_ablation_paper.py
      → experiments/ablation/results_ablation_paper.json
      → experiments/ablation/results_ablation_paper.md

USAGE
-----
  cd prism/

  # Full pipeline (GPU recommended, ~2-4h on A100):
  python run_pipeline.py --all

  # Individual phases:
  python run_pipeline.py --phases 1 2        # profile + calibrate only
  python run_pipeline.py --phases 5          # eval only (needs phases 1-2)
  python run_pipeline.py --phases 5 6 7      # eval + campaign + ablation

  # Quick sanity check (CPU, ~15-30min):
  python run_pipeline.py --all --fast

  # GPU-local quick check of Phase 5 only:
  python run_pipeline.py --phases 5 --n-test 100
"""
import subprocess
import sys
import os
import argparse
import time


def run_step(args_list, description, cwd=None):
    """Run a Python script as a subprocess and check return code."""
    cwd = cwd or os.getcwd()
    print(f"\n{'='*70}")
    print(f"STEP: {description}")
    print(f"CMD:  python {' '.join(args_list)}")
    print(f"{'='*70}")

    t0   = time.time()
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    proc = subprocess.run(
        [sys.executable] + args_list,
        cwd=cwd,
        env=env,
    )
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"\n[FAILED] {description} (exit code {proc.returncode})")
        print(f"   Elapsed: {elapsed:.1f}s")
        sys.exit(proc.returncode)

    print(f"\n[OK] {description} -- done ({elapsed:.1f}s)")
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="PRISM Master Pipeline")
    parser.add_argument('--all',        action='store_true',
                        help='Run all phases 1-7')
    parser.add_argument('--phases',     nargs='+', type=int,
                        default=None,
                        help='Which phases to run (e.g. --phases 1 2 5)')
    parser.add_argument('--fast',       action='store_true',
                        help='Reduced n for quick sanity checks')
    parser.add_argument('--n-test',     type=int, default=1000,
                        help='Images per attack in full eval (default 1000)')
    parser.add_argument('--attacks',    nargs='+',
                        default=['FGSM', 'PGD', 'Square'],
                        help='Attacks for full eval and ablation')
    parser.add_argument('--data-root',  default='./data')
    args = parser.parse_args()

    if args.all:
        phases = list(range(1, 8))
    elif args.phases:
        phases = sorted(set(args.phases))
    else:
        print("ERROR: Specify --all or --phases N [N ...]")
        parser.print_help()
        sys.exit(1)

    fast     = args.fast
    n_test   = 100 if fast else args.n_test
    n_ablate = 50  if fast else 500
    n_train  = 200 if fast else 2000

    timings = {}

    # ── Phase 1: Build profile from test set ─────────────────────────────────
    if 1 in phases:
        timings[1] = run_step(
            ['scripts/build_profile_testset.py',
             '--data-root', args.data_root],
            "Phase 1: TDA profile (CIFAR-10 test split)"
        )

    # ── Phase 2: Conformal calibration ───────────────────────────────────────
    if 2 in phases:
        timings[2] = run_step(
            ['scripts/calibrate_testset.py'],
            "Phase 2: Conformal calibration (test-set cal split)"
        )

    # ── Phase 3: Train ensemble scorer ───────────────────────────────────────
    if 3 in phases:
        timings[3] = run_step(
            ['scripts/train_ensemble_scorer.py',
             '--n-train', str(n_train),
             '--data-root', args.data_root],
            "Phase 3: Persistence ensemble scorer training (FGSM+PGD, eps=8/255)"
        )
        # Phase 3.5: Re-calibrate thresholds for the ensemble composite score
        run_step(
            ['scripts/calibrate_ensemble.py',
             '--data-root', args.data_root],
            "Phase 3.5: Ensemble conformal calibration"
        )
        # Phase 3.6: High-power FPR report on val split (n=1000)
        run_step(
            ['scripts/compute_ensemble_val_fpr.py',
             '--data-root', args.data_root],
            "Phase 3.6: Ensemble val FPR report (n=1000, Wilson CI ~+/-1.5%)"
        )


    # ── Phase 4: Train MoE experts ───────────────────────────────────────────
    if 4 in phases:
        timings[4] = run_step(
            ['scripts/train_experts.py',
             '--data-root', args.data_root],
            "Phase 4: MoE expert training (TAMSH)"
        )

    # ── Phase 5: Full evaluation ──────────────────────────────────────────────
    if 5 in phases:
        atk_args = []
        for a in args.attacks:
            atk_args += [a]
        timings[5] = run_step(
            ['experiments/evaluation/run_evaluation_full.py',
             '--n-test', str(n_test),
             '--data-root', args.data_root,
             '--attacks'] + args.attacks,
            "Phase 5: Full attack evaluation (paper results)"
        )

    # ── Phase 6: Campaign detection ───────────────────────────────────────────
    if 6 in phases:
        timings[6] = run_step(
            ['experiments/campaign/run_campaign.py'],
            "Phase 6: Campaign detection experiment"
        )

    # ── Phase 7: Paper-quality ablation ──────────────────────────────────────
    if 7 in phases:
        ablate_args = ['experiments/ablation/run_ablation_paper.py',
                       '--n', str(n_ablate),
                       '--attacks'] + args.attacks
        if fast:
            ablate_args.append('--fast')
        timings[7] = run_step(
            ablate_args,
            "Phase 7: Paper-quality ablation study"
        )

    # ── Summary ──────────────────────────────────────────────────────────────
    total = sum(timings.values())
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print(f"{'='*70}")
    for phase, elapsed in timings.items():
        print(f"  Phase {phase}: {elapsed:6.1f}s")
    print(f"  TOTAL : {total:6.1f}s  ({total/60:.1f}min)")

    print("\nOutput files:")
    outputs = [
        ('models/reference_profiles.pkl', 'TDA reference profiles'),
        ('models/calibrator.pkl',          'Conformal calibrator'),
        ('models/ensemble_scorer.pkl',     'Persistence ensemble scorer'),
        ('models/experts.pkl',             'MoE expert networks'),
        ('experiments/evaluation/results_paper.json', 'Full eval results'),
        ('experiments/campaign/results.json',         'Campaign detection'),
        ('experiments/ablation/results_ablation_paper.json', 'Ablation results'),
        ('experiments/ablation/results_ablation_paper.md',   'Ablation table (paper)'),
    ]
    for path, desc in outputs:
        exists = '[OK]' if os.path.exists(path) else '[ ]'
        print(f"  {exists}  {path:45s}  {desc}")


if __name__ == '__main__':
    main()
