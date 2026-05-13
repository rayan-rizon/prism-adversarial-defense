"""
Deprecated ablation entrypoint.

The publishable CIFAR-native ablation pipeline lives in
experiments/ablation/run_ablation_paper.py. This wrapper prevents accidental
use of the obsolete ImageNet/upscaled-CIFAR ablation script.
"""
import os
import subprocess
import sys


def main():
    here = os.path.dirname(__file__)
    target = os.path.join(here, 'run_ablation_paper.py')
    cmd = [sys.executable, target] + sys.argv[1:]
    raise SystemExit(subprocess.call(cmd))


if __name__ == '__main__':
    main()
