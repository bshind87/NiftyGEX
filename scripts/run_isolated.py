"""Run a walk-forward script in a fresh Python process per walk (or per group of walks).

  PYTHONPATH=src python scripts/run_isolated.py [--walks-per-process 1] scripts/run_neural.py --models attention_gru --window 24

Apple-GPU (MPS) driver memory grows with every new tensor shape and is only returned when the process exits, so a
long run can exhaust it. Each child process trains at most --walks-per-process walks that are not yet checkpointed
(NIFTYGEX_WALK_LIMIT), saves them and exits with code 75; the launcher then starts the next process, which resumes
from the checkpoints. The last process writes the run as usual and exits 0. Training itself is unchanged.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from niftygex_eval.forecast.runner import MORE_WALKS, WALK_LIMIT_ENV


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--walks-per-process", type=int, default=1)
    parser.add_argument("--max-processes", type=int, default=200, help="safety limit on the number of processes")
    parser.add_argument("script", help="the run script, e.g. scripts/run_neural.py")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="arguments passed to the script")
    args = parser.parse_args()
    env = {**os.environ, WALK_LIMIT_ENV: str(args.walks_per_process)}
    started = time.time()
    for process in range(1, args.max_processes + 1):
        print(f"[isolated] process {process}: {args.script} {' '.join(args.args)}", flush=True)
        code = subprocess.run([sys.executable, args.script, *args.args], env=env).returncode
        if code != MORE_WALKS:
            print(f"[isolated] done after {process} processes, {(time.time() - started) / 60:.1f} min, exit {code}", flush=True)
            sys.exit(code)
    sys.exit(f"[isolated] stopped after {args.max_processes} processes")


if __name__ == "__main__":
    main()
