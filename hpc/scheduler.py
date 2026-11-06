"""Submit the exploratory GPU jobs within each partition's limits; resubmit stopped jobs until they finish.

  NIFTYGEX_PROFILE=hpc PYTHONPATH=src python hpc/scheduler.py --once [--waves calibration] [--dry-run]
  sbatch hpc/scheduler.sbatch                     (polls every 5 minutes from a CPU node, never the login node)
  bash hpc/remote.sh scheduler                    (one pass from the Mac over ssh)

Routing (hpc/hpc.yaml): short jobs to gpu-short, long jobs to gpu, courses-gpu whenever sbatch --test-only
estimates an earlier start there; a long job still pending on gpu may run as chained gpu-short pieces. A job is
done when its .done marker exists (outputs/hpc/status/), failed when its .failed marker exists (a crash: listed,
not resubmitted until --retry-failed), and otherwise submitted again while it has attempts left. Every submission
and move is logged to outputs/hpc/submissions.tsv.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from niftygex.config import REPO_ROOT
from niftygex_eval.hpc_jobs import Job, Queued, expand, plan

START = re.compile(r"to start at (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})")


def load_settings() -> dict:
    raw = yaml.safe_load((REPO_ROOT / "hpc" / "hpc.yaml").read_text())
    return {k: expand(v) if isinstance(v, str) else v for k, v in raw.items()}


def parse_time(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def our_queue(prefix: str) -> dict[str, Queued]:
    """Our running and pending jobs by name (prefix removed), with squeue's estimated start for pending ones."""
    out = subprocess.run(["squeue", "-u", os.environ["USER"], "-h", "-o", "%j|%P|%T|%i|%S"], capture_output=True, text=True, check=True).stdout
    queued = {}
    for line in out.splitlines():
        name, partition, state, slurm_id, start = line.split("|")
        if name.startswith(prefix):
            queued[name[len(prefix):]] = Queued(partition, state, slurm_id, parse_time(start))
    return queued


def test_only(scripts: dict[str, str]):
    """sbatch --test-only estimate of the start time of one job on a partition (None when SLURM gives none)."""
    def estimate(job: Job, partition: str, time_limit: str) -> datetime | None:
        result = subprocess.run(["sbatch", "--test-only", f"--partition={partition}", f"--time={time_limit}", scripts[job.name]],
                                capture_output=True, text=True)
        match = START.search(result.stderr + result.stdout)
        return parse_time(match.group(1)) if match else None
    return estimate


def run_once(hpc: dict, waves: set[str] | None, dry_run: bool, retry_failed: bool) -> bool:
    base = Path(hpc["scratch_dir"]) / "outputs" / "hpc"
    listing = json.loads((base / "jobs" / "jobs.json").read_text())
    jobs = [Job(j["name"], j["wave"], j["command"], j["kind"], j["group"]) for j in listing["jobs"] if waves is None or j["wave"] in waves]
    scripts = {j["name"]: j["script"] for j in listing["jobs"]}
    status = base / "status"
    status.mkdir(parents=True, exist_ok=True)
    if retry_failed:
        for job in jobs:
            (status / f"{job.name}.failed").unlink(missing_ok=True)
    done = {j.name for j in jobs if (status / f"{j.name}.done").exists()}
    failed = {j.name for j in jobs if (status / f"{j.name}.failed").exists()}
    attempts = {j.name: int((status / f"{j.name}.attempts").read_text()) for j in jobs if (status / f"{j.name}.attempts").exists()}
    queued = our_queue(hpc["job_prefix"])
    actions, exhausted = plan(jobs, queued, done, failed, attempts, hpc, listing.get("walk_minutes", {}), test_only(scripts), datetime.now())
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for action in actions:
        name = action.job.name
        if dry_run:
            print(f"[dry-run] would {action.kind} {name} on {action.partition} --time={action.time}" + (f" (cancel {action.cancel})" if action.cancel else ""))
            continue
        if action.cancel:
            subprocess.run(["scancel", action.cancel], check=True)
        slurm_id = subprocess.run(["sbatch", "--parsable", f"--partition={action.partition}", f"--time={action.time}", scripts[name]],
                                  capture_output=True, text=True, check=True).stdout.strip()
        attempt = attempts.get(name, 0) + 1
        (status / f"{name}.attempts").write_text(str(attempt))
        with (base / "submissions.tsv").open("a") as log:
            log.write(f"{stamp}\t{name}\t{action.kind}\t{action.partition}\t{action.time}\t{slurm_id}\t{attempt}\t{listing['commit']}\n")
        print(f"{stamp} {action.kind} {name} -> {action.partition} ({action.time}) as {slurm_id}, attempt {attempt}", flush=True)
    open_jobs = [j.name for j in jobs if j.name not in done and j.name not in failed and j.name not in exhausted]
    print(f"{stamp} done {len(done)}/{len(jobs)}, queued {len(queued)}, failed {sorted(failed)}, out of attempts {exhausted}", flush=True)
    return not actions and not queued and not open_jobs   # nothing running and nothing left that can still be submitted


def main() -> None:
    os.umask(0o077)   # status markers and logs stay private
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--loop", action="store_true")
    parser.add_argument("--interval", type=int, default=300, help="seconds between polls with --loop")
    parser.add_argument("--waves", nargs="*", help="only these waves (calibration, references, e1, e3, e2, e2_l72)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    hpc = load_settings()
    waves = set(args.waves) if args.waves else None
    finished = run_once(hpc, waves, args.dry_run, args.retry_failed)
    while args.loop and not finished:
        time.sleep(args.interval)
        finished = run_once(hpc, waves, False, False)
    sys.exit(0)


if __name__ == "__main__":
    main()
