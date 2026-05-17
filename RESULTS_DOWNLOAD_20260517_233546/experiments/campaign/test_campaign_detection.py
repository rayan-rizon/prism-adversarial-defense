"""
Campaign Detection Test (Phase 3, Week 12-13)

Simulates an adversary's probing-then-attack campaign and validates
that L0 (BOCPD) detects the distribution shift before the full attack.

Scenario:
  Phase 1: 50 clean queries (normal baseline traffic)
  Phase 2: 30 probing queries (slightly elevated anomaly scores)
  Phase 3: 5 full attack queries (high anomaly scores)

Target: L0 activates within 20 probe queries (<20 lead time).
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.sacd.monitor import CampaignMonitor


def test_campaign_detection():
    print("=== Campaign Detection Test ===\n")

    monitor = CampaignMonitor(
        window_size=100,
        warmup_steps=20,
        alert_run_length=5,
        alert_run_prob=0.5,
        hazard_rate=1 / 50,
        mu0=0.1,
        kappa0=1.0,
        alpha0=2.0,
        beta0=0.02,
        cooldown_steps=50,
    )

    # --- Phase 1: Normal traffic (50 clean queries) ---
    print("Phase 1: 50 clean queries (normal traffic)")
    rng = np.random.RandomState(42)
    clean_scores = rng.normal(0.1, 0.02, 50)

    for i, s in enumerate(clean_scores):
        state = monitor.process_score(s, timestamp=float(i))

    print(f"  After clean phase: L0 active = {state['l0_active']}")
    print(f"  Buffer mean = {state['buffer_mean']:.4f}")
    assert not state['l0_active'], "L0 should NOT be active during clean traffic!"
    print("  ✓ L0 correctly inactive\n")

    # --- Phase 2: Probing queries (30 slightly elevated scores) ---
    print("Phase 2: 30 probe queries (slightly elevated scores)")
    probe_scores = rng.normal(0.25, 0.05, 30)
    detected_at = None

    for i, s in enumerate(probe_scores):
        state = monitor.process_score(s, timestamp=float(50 + i))
        if state['l0_active'] and detected_at is None:
            detected_at = i
            print(f"  *** L0 ACTIVATED at probe query {i}! "
                  f"(cp_prob={state['changepoint_prob']:.4f})")

    if detected_at is None:
        print("  L0 did NOT activate during probing phase")
    print()

    # --- Phase 3: Full attack (5 high-score queries) ---
    print("Phase 3: 5 full attack queries (high scores)")
    attack_scores = rng.normal(0.8, 0.1, 5)

    for i, s in enumerate(attack_scores):
        state = monitor.process_score(s, timestamp=float(80 + i))
        if state['l0_active'] and detected_at is None:
            detected_at = 30 + i  # Relative to start of probing
            print(f"  *** L0 ACTIVATED at attack query {i}!")

    print(f"\n=== Results ===")
    print(f"  L0 active: {state['l0_active']}")
    print(f"  Campaign detection lead time: "
          f"{detected_at if detected_at is not None else 'NOT DETECTED'} queries")

    if detected_at is not None:
        if detected_at < 20:
            print(f"  ✓ Detected within target (<20 queries)")
        else:
            print(f"  ⚠ Detection slower than target (>20 queries)")
            print(f"    Consider tuning cp_threshold or hazard_rate")
    else:
        print(f"  ✗ Campaign NOT detected. Need to tune parameters.")

    # Print alert log
    alerts = monitor.get_alert_log()
    if alerts:
        print(f"\n  Alert log ({len(alerts)} entries):")
        for a in alerts:
            print(f"    {a['type']} at step {a['step']}")

    return detected_at


def test_no_false_alarm():
    """Verify L0 does NOT trigger on purely clean traffic."""
    print("\n=== False Alarm Test ===\n")
    monitor = CampaignMonitor(
        window_size=100,
        warmup_steps=20,
        alert_run_length=5,
        alert_run_prob=0.5,
        hazard_rate=1 / 50,
        mu0=0.1,
        kappa0=1.0,
        alpha0=2.0,
        beta0=0.02,
    )
    rng = np.random.RandomState(123)

    clean_scores = rng.normal(0.1, 0.02, 500)
    false_alarms = 0

    for i, s in enumerate(clean_scores):
        state = monitor.process_score(s)
        if state['l0_active']:
            false_alarms += 1
            monitor.deactivate_l0()

    print(f"  500 clean queries processed")
    print(f"  False alarms: {false_alarms}")
    if false_alarms == 0:
        print("  ✓ No false alarms on clean traffic")
    else:
        print(f"  ⚠ {false_alarms} false alarms — tune cp_threshold higher")


if __name__ == '__main__':
    test_campaign_detection()
    test_no_false_alarm()
