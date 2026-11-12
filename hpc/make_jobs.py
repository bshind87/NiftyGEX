"""Write one sbatch script per exploratory GPU job and the job list the scheduler reads (on the cluster).

  NIFTYGEX_PROFILE=hpc PYTHONPATH=src python hpc/make_jobs.py [--gpu-type v100-sxm2|h200|a100|auto] [--set v2|v3]

Scripts go to /scratch/$USER/niftygex/outputs/hpc/jobs/, with jobs.json listing them in submission priority.
The GPU type is fixed here for every job (device rule); "auto" takes h200 when sinfo shows idle H200 nodes,
else v100-sxm2. Per-walk minutes from hpc/estimates.yaml (written after calibration) go into each script, so a
chained piece never starts a walk it cannot finish. Light work only, so it may run on the login node.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml

from niftygex.config import REPO_ROOT
from niftygex_eval.config import load_exploratory_config, load_exploratory_v3_config
from niftygex_eval.hpc_jobs import expand, job_list, render_script, v3_job_list


def settings() -> dict:
    raw = yaml.safe_load((REPO_ROOT / "hpc" / "hpc.yaml").read_text())
    return {k: expand(v) if isinstance(v, str) else v for k, v in raw.items()}


def idle_h200() -> bool:
    out = subprocess.run(["sinfo", "-p", "gpu", "-h", "-o", "%G %t"], capture_output=True, text=True).stdout
    return any("h200" in line and line.split()[-1] in ("idle", "mix") for line in out.splitlines())


def main() -> None:
    hpc = settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-type", default=hpc["gpu_type"])
    parser.add_argument("--set", default="v2", choices=["v2", "v3"], help="job set: E1-E3 (v2) or the v3 checks A2 and A4")
    args = parser.parse_args()
    gpu_type = ("h200" if idle_h200() else "v100-sxm2") if args.gpu_type == "auto" else args.gpu_type
    cfg = load_exploratory_config()
    seeds = [int(s) for s in cfg["eval"]["neural"]["training"]["seeds"]]
    horizons = [int(h) for h in cfg["exploratory"]["e3"]["horizons"]]
    if args.set == "v3":
        v3 = load_exploratory_v3_config()["exploratory_v3"]
        if v3["compute"]["gpu_type"] != gpu_type:
            raise SystemExit(f"v3 registers GPU type {v3['compute']['gpu_type']}, not {gpu_type}")
        horizons = [int(h) for h in v3["a4_e3_training_budget"]["horizons"]]
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    directory = Path(hpc["scratch_dir"]) / "outputs" / "hpc" / "jobs"
    directory.mkdir(parents=True, exist_ok=True)
    listed = []
    estimates_path = REPO_ROOT / "hpc" / "estimates.yaml"
    walk_minutes = (yaml.safe_load(estimates_path.read_text()) or {}).get("walk_minutes", {}) if estimates_path.exists() else {}
    for job in (v3_job_list(seeds, horizons) if args.set == "v3" else job_list(seeds, horizons)):
        path = directory / f"{job.name}.sbatch"
        path.write_text(render_script(job, hpc, gpu_type, walk_minutes.get(job.group)))
        listed.append({"name": job.name, "wave": job.wave, "command": job.command, "kind": job.kind, "group": job.group, "script": str(path)})
    (directory / "jobs.json").write_text(json.dumps({"gpu_type": gpu_type, "set": args.set, "commit": commit, "walk_minutes": walk_minutes, "jobs": listed},
                                                    indent=2) + "\n")
    waves = {}
    for job in listed:
        waves[job["wave"]] = waves.get(job["wave"], 0) + 1
    print(f"{len(listed)} job scripts for GPU type {gpu_type} at commit {commit} in {directory}: {waves}")


if __name__ == "__main__":
    main()
