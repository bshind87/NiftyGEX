"""Submit the exploratory GPU jobs within each partition's limits; resubmit jobs stopped at the time limit.

  NIFTYGEX_PROFILE=hpc PYTHONPATH=src python hpc/scheduler.py --once [--waves calibration] [--dry-run]
  sbatch hpc/scheduler.sbatch                     (keeps polling every 5 minutes from a CPU job, never the login node)

A job is done when its .done marker exists (outputs/hpc/status/), failed when its .failed marker exists (a crash;
listed, not resubmitted until --retry-failed), and otherwise submitted again while it has attempts left. Every
submission is logged to outputs/hpc/submissions.tsv.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from niftygex.config import REPO_ROOT
from niftygex_eval.hpc_jobs import Job, expand, plan_submissions


def load_settings() -> dict:
    raw = yaml.safe_load((REPO_ROOT / "hpc" / "hpc.yaml").read_text())
    return {k: expand(v) if isinstance(v, str) else v for k, v in raw.items()}


def our_queue(prefix: str) -> dict[str, str]:
    """Our running and pending jobs: job name (without the prefix) -> partition."""
    out = subprocess.run(["squeue", "-u", os.environ["USER"], "-h", "-o", "%j|%P|%T"], capture_output=True, text=True, check=True).stdout
    queued = {}
    for line in out.splitlines():
        name, partition, _ = line.split("|")
        if name.startswith(prefix):
            queued[name[len(prefix):]] = partition
    return queued


def run_once(hpc: dict, waves: set[str] | None, dry_run: bool, retry_failed: bool) -> bool:
    base = Path(hpc["scratch_dir"]) / "outputs" / "hpc"
    listing = json.loads((base / "jobs" / "jobs.json").read_text())
    jobs = [Job(j["name"], j["wave"], j["command"], j["calibration"]) for j in listing["jobs"] if waves is None or j["wave"] in waves]
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
    counts: dict[str, int] = {}
    for partition in queued.values():
        counts[partition] = counts.get(partition, 0) + 1
    plan, exhausted = plan_submissions(jobs, queued, done, failed, attempts, hpc["partitions"], hpc["calibration"],
                                       int(hpc["max_attempts"]), counts)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for job, partition in plan:
        if dry_run:
            print(f"[dry-run] would submit {job.name} to {partition}")
            continue
        slurm_id = subprocess.run(["sbatch", "--parsable", f"--partition={partition}", scripts[job.name]],
                                  capture_output=True, text=True, check=True).stdout.strip()
        attempt = attempts.get(job.name, 0) + 1
        (status / f"{job.name}.attempts").write_text(str(attempt))
        with (base / "submissions.tsv").open("a") as log:
            log.write(f"{stamp}\t{job.name}\t{partition}\t{slurm_id}\t{attempt}\n")
        print(f"{stamp} submitted {job.name} to {partition} as {slurm_id} (attempt {attempt})", flush=True)
    open_jobs = [j.name for j in jobs if j.name not in done and j.name not in failed and j.name not in exhausted]
    print(f"{stamp} done {len(done)}/{len(jobs)}, queued {len(queued)}, failed {sorted(failed)}, out of attempts {exhausted}", flush=True)
    return not plan and not queued and not open_jobs   # nothing running and nothing left that can still be submitted


def main() -> None:
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
