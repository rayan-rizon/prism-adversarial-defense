#!/bin/bash
set -u

cd /workspace/prism-repo/prism/prism || exit 1

tag="${1:-ensemble_complete_lambda_scan_n50}"
interval="${2:-30}"

while true; do
  clear
  echo "PRISM ensemble-complete adaptive attack monitor"
  echo "tag=${tag}"
  date -Is
  echo

  echo "GPU: util %, mem MiB, total MiB, power W, limit W"
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,power.limit --format=csv,noheader,nounits 2>/dev/null || true
  echo

  echo "controller"
  pgrep -af 'run_vastai_ensemble_complete_two_stage.sh' 2>/dev/null || echo "two-stage controller not found"
  tmux list-windows -t prism_ec_adaptive 2>/dev/null || true
  echo

  echo "processes"
  pgrep -af 'run_adaptive_pgd.py' 2>/dev/null || echo "no adaptive python processes"
  echo

  "${PYTHON:-/workspace/prism-venv/bin/python}" - "$tag" <<'PY'
import math
import re
import sys
from pathlib import Path

tag = sys.argv[1]
logdir = Path("logs") / tag

plans = {
    "ensemble_complete_lambda_scan_n50": {
        "n": 50,
        "lambdas": ["0.0", "0.5", "1.0", "2.0", "5.0", "10.0"],
        "next_phase_units": 200,
    },
    "ensemble_complete_worst_lambda_n200": {
        "n": 200,
        "lambdas": None,
        "next_phase_units": 0,
    },
}
plan = plans.get(tag, {"n": None, "lambdas": None, "next_phase_units": 0})

progress_re = re.compile(
    r"lambda_?=?\s*|"
)
bar_re = re.compile(
    r"(?:lambda|λ)=([0-9.]+):\s+([0-9]+)%.*?\|\s*([0-9]+)/([0-9]+)\s*\[.*?,\s*([0-9.]+)(s/it|it/s)"
)

def fmt_seconds(seconds):
    if seconds is None or not math.isfinite(seconds):
        return "unknown"
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"

def parse_log(path):
    text = path.read_text(errors="replace")
    lines = text.splitlines()
    latest = None
    for line in lines:
        m = bar_re.search(line)
        if m:
            latest = m
    if latest is None:
        return None
    lam, pct, done, total, rate, unit = latest.groups()
    done = int(done)
    total = int(total)
    rate = float(rate)
    sec_per_item = rate if unit == "s/it" else 1.0 / max(rate, 1e-12)
    lambdas = plan["lambdas"]
    if lambdas and lam in lambdas:
        lam_index = lambdas.index(lam)
        phase_done = lam_index * total + done
        phase_total = len(lambdas) * total
    else:
        phase_done = done
        phase_total = plan["n"] or total
    remaining = max(0, phase_total - phase_done)
    return {
        "seed": re.sub(r"\D+", "", path.stem) or path.name,
        "lambda": lam,
        "pct": int(pct),
        "done": done,
        "total": total,
        "phase_done": phase_done,
        "phase_total": phase_total,
        "sec_per_item": sec_per_item,
        "eta": remaining * sec_per_item,
        "line": latest.string.strip(),
    }

items = []
for path in sorted(logdir.glob("ensemble_complete_adaptive_seed*.log")):
    parsed = parse_log(path)
    if parsed:
        items.append(parsed)

print("ETA")
if not items:
    print("  waiting for tqdm progress lines")
else:
    phase_eta = max(x["eta"] for x in items)
    avg_rate = sum(x["sec_per_item"] for x in items) / len(items)
    next_phase_eta = plan["next_phase_units"] * avg_rate
    print(f"  current phase remaining: {fmt_seconds(phase_eta)}")
    if next_phase_eta:
        print(f"  two-stage remaining incl. confirm: {fmt_seconds(phase_eta + next_phase_eta)}")
    print("  per-seed")
    for item in items:
        print(
            "    seed={seed} lambda={lam} image={done}/{total} "
            "phase={phase_done}/{phase_total} rate={rate:.1f}s/img eta={eta}".format(
                seed=item["seed"],
                lam=item["lambda"],
                done=item["done"],
                total=item["total"],
                phase_done=item["phase_done"],
                phase_total=item["phase_total"],
                rate=item["sec_per_item"],
                eta=fmt_seconds(item["eta"]),
            )
        )
PY
  echo

  echo "progress tail"
  for f in logs/${tag}/ensemble_complete_adaptive_seed*.log; do
    [ -e "$f" ] || continue
    echo "---- $(basename "$f") ----"
    tail -n 35 "$f" | tr '\r' '\n' | grep -E 'λ=|TPR=|Model ASR|Results saved|Skipping|Adaptive PGD' | tail -n 10 || tail -n 5 "$f"
  done
  echo

  echo "outputs"
  ls -lh experiments/evaluation/${tag} 2>/dev/null || true
  echo

  echo "Usage: /workspace/monitor_prism_ec.sh [tag] [seconds]"
  echo "Scan tag:    ensemble_complete_lambda_scan_n50"
  echo "Confirm tag: ensemble_complete_worst_lambda_n200"
  echo "Controller log: tail -f /workspace/prism_ec_two_stage.master.log"
  sleep "$interval"
done
