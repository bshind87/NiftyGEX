"""Submit the exploratory GPU jobs within each partition's limits; resubmit stopped jobs until they finish.

  NIFTYGEX_PROFILE=hpc PYTHONPATH=src python hpc/scheduler.py --once [--waves calibration] [--dry-run]
  sbatch hpc/scheduler.sbatch                     (polls every 5 minutes from a CPU node, never the login node)
  bash hpc/remote.sh scheduler                    (one pass from the Mac over ssh)

Routing (hpc/hpc.yaml): short jobs to gpu-short, long jobs to gpu, courses-gpu whenever sbatch --test-only
estimates an earlier start there; a long job still pending on gpu may run as chained gpu-short pieces. A job is
done when its .done marker exists (outputs/hpc/status/), failed when its .failed marker exists (a crash: listed,
not resubmitted until --retry-failed), and otherwise submitted again while it has attempts left. Every submission
and move is logged to outputs/hpc/submissions.tsv. Jobs are submitted with --exclude for the nodes our running jobs
use. A finished attempt that was SIGKILLed is logged to outputs/hpc/kills.tsv with any of our jobs that shared its
node; a SIGKILL with no shared node is flagged for RC.
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
from niftygex_eval.hpc_jobs import Job, Queued, expand, kill_report, plan

START = re.compile(r"to start at (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})")


def load_settings() -> dict:
    raw = yaml.safe_load((REPO_ROOT / "hpc" / "hpc.yaml").read_text())
    return {k: expand(v) if isinstance(v, str) else v for k, v in raw.items()}


def parse_time(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def our_queue(prefix: str) -> tuple[dict[str, Queued], tuple[str, ...]]:
    """Our running and pending jobs by name (prefix removed), with squeue's estimated start for pending ones, and the
    nodes our running jobs use (all our jobs, not only this experiment's)."""
    out = subprocess.run(["squeue", "-u", os.environ["USER"], "-h", "-o", "%j|%P|%T|%i|%S|%N"], capture_output=True, text=True, check=True).stdout
    queued, nodes = {}, set()
    for line in out.splitlines():
        name, partition, state, slurm_id, start, nodelist = line.split("|")
        if state == "RUNNING" and nodelist:
            nodes.update(expand_nodes(nodelist))
        if name.startswith(prefix):
            queued[name[len(prefix):]] = Queued(partition, state, slurm_id, parse_time(start))
    return queued, tuple(sorted(nodes))


def expand_nodes(nodelist: str) -> list[str]:
    out = subprocess.run(["scontrol", "show", "hostnames", nodelist], capture_output=True, text=True)
    return out.stdout.split() or [nodelist]


def accounting(slurm_ids: list[str]) -> dict[str, dict]:
    """sacct records (state, exit code, node, start, end) of our finished and running attempts."""
    if not slurm_ids:
        return {}
    out = subprocess.run(["sacct", "-X", "-n", "-P", "-j", ",".join(slurm_ids), "--format=JobID,State,ExitCode,NodeList,Start,End"],
                         capture_output=True, text=True).stdout
    records = {}
    for line in out.splitlines():
        slurm_id, state, exit_code, nodes, start, end = line.split("|")
        records[slurm_id] = {"state": state, "exit_code": exit_code, "nodes": nodes, "start": parse_time(start), "end": parse_time(end)}
    return records


def check_kills(base: Path) -> None:
    """Log every SIGKILLed attempt once, with our attempts that shared its node; flag kills with no shared node."""
    log = base / "submissions.tsv"
    if not log.exists():
        return
    attempts = [line.split("\t") for line in log.read_text().splitlines() if line.strip()]
    checked_path = base / "kills_checked.txt"
    checked = set(checked_path.read_text().split()) if checked_path.exists() else set()
    records = accounting([a[5] for a in attempts])
    names = {a[5]: a[1] for a in attempts}
    new_lines = []
    for slurm_id, record in records.items():
        if slurm_id in checked or record["end"] is None or record["state"] in ("RUNNING", "PENDING"):
            continue
        checked.add(slurm_id)
        report = kill_report(slurm_id, record, records)
        if report is None:
            continue
        shared = ",".join(f"{names.get(o, '?')}:{o}" for o in report["shared_with"]) or "none"
        new_lines.append(f"{datetime.now():%Y-%m-%d %H:%M:%S}\t{names.get(slurm_id, '?')}\t{slurm_id}\t{record['state']}\t{record['exit_code']}"
                         f"\t{record['nodes']}\t{shared}\n")
        if not report["shared_with"]:
            print(f"WARNING: {names.get(slurm_id)} ({slurm_id}) was SIGKILLed on {record['nodes']} with no shared node: ask RC", flush=True)
        else:
            print(f"note: {names.get(slurm_id)} ({slurm_id}) was SIGKILLed on {record['nodes']}, shared with {shared}", flush=True)
    if new_lines:
        with (base / "kills.tsv").open("a") as stream:
            stream.writelines(new_lines)
    checked_path.write_text("\n".join(sorted(checked)) + "\n")


def test_only(scripts: dict[str, str], exclude: tuple[str, ...]):
    """sbatch --test-only estimate of the start time of one job on a partition (None when SLURM gives none)."""
    def estimate(job: Job, partition: str, time_limit: str) -> datetime | None:
        result = subprocess.run(["sbatch", "--test-only", f"--partition={partition}", f"--time={time_limit}",
                                 *([f"--exclude={','.join(exclude)}"] if exclude else []), scripts[job.name]], capture_output=True, text=True)
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
    queued, busy = our_queue(hpc["job_prefix"])
    check_kills(base)
    exclude = busy if hpc.get("avoid_shared_nodes", True) else ()
    actions, exhausted = plan(jobs, queued, done, failed, attempts, hpc, listing.get("walk_minutes", {}), test_only(scripts, exclude),
                              datetime.now(), busy)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for action in actions:
        name = action.job.name
        if dry_run:
            print(f"[dry-run] would {action.kind} {name} on {action.partition} --time={action.time}" + (f" (cancel {action.cancel})" if action.cancel else ""))
            continue
        if action.cancel:
            subprocess.run(["scancel", action.cancel], check=True)
        options = [f"--partition={action.partition}", f"--time={action.time}", *([f"--exclude={','.join(action.exclude)}"] if action.exclude else [])]
        slurm_id = subprocess.run(["sbatch", "--parsable", *options, scripts[name]], capture_output=True, text=True, check=True).stdout.strip()
        attempt = attempts.get(name, 0) + 1
        (status / f"{name}.attempts").write_text(str(attempt))
        with (base / "submissions.tsv").open("a") as log:
            log.write(f"{stamp}\t{name}\t{action.kind}\t{action.partition}\t{action.time}\t{slurm_id}\t{attempt}\t{listing['commit']}"
                      f"\t{','.join(action.exclude)}\n")
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
