"""Explorer HPC job list, SLURM scripts and the submission plan for the exploratory neural runs (docs/hpc.md).

One job per experiment, model, horizon and seed; every job trains all walks of one run with per-walk checkpoints,
so a job stopped at its time limit (or a chained piece) is submitted again and resumes. Seeds are merged afterwards
on the Mac (scripts/merge_seeds.py). The scheduler routes short jobs to gpu-short and long jobs to gpu, uses
courses-gpu whenever it is estimated to start sooner, and may run a long job still pending on gpu as chained
gpu-short pieces (hpc/hpc.yaml, routing).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable

# Exit codes of a stopped job (75: walk limit or deadline; 137/143: killed at the time limit): submit again.
RESUBMIT_CODES = {75, 137, 143}


@dataclass(frozen=True)
class Job:
    name: str
    wave: str
    command: str
    kind: str = "short"            # short (estimated under 2 h: gpu-short) or long (gpu, or chained gpu-short pieces)
    group: str = ""                 # estimate group in hpc/estimates.yaml (walk minutes from calibration)


def job_list(seeds: list[int], horizons: list[int]) -> list[Job]:
    """Every exploratory GPU job in priority order: calibration, L = 12 references, E1, E3, E2 (L = 72 last).

    All are short (gpu-short) after calibration (6 Nov: the longest, E2 L = 72, is estimated at 50-100 min on a V100);
    the long route (gpu, chained gpu-short pieces) stays available for jobs marked kind="long".
    """
    jobs = [Job("cal_t2_l12", "calibration", "scripts/run_neural.py --models attention_gru --walks 2 --seed 11 --label hpccal", group="t2_attention_gru"),
            Job("cal_e3_1h", "calibration", "scripts/run_intraday.py --horizon 1 --models attention_gru --walks 2 --seed 11", group="e3_attention_gru")]
    for seed in seeds:
        for model in ("attention_gru", "deepsets", "attention"):
            jobs.append(Job(f"ref_{model}_l12_s{seed}", "references", f"scripts/run_neural.py --models {model} --seed {seed} --label hpc",
                            group=f"t2_{model}"))
    for seed in seeds:
        for part in ("hybrid", "residual_attention", "residual_deepsets", "state_mlp"):
            jobs.append(Job(f"e1_{part}_s{seed}", "e1", f"scripts/run_exploratory_e1.py --parts {part} --seed {seed}", group=f"e1_{part}"))
    for horizon in horizons:
        for seed in seeds:
            for model in ("attention_gru", "attention", "deepsets"):
                jobs.append(Job(f"e3_{model}_{horizon}h_s{seed}", "e3", f"scripts/run_intraday.py --horizon {horizon} --models {model} --seed {seed}",
                                group=f"e3_{model}"))
    for seed in seeds:
        jobs.append(Job(f"e2_window24_s{seed}", "e2", f"scripts/run_neural.py --models attention_gru --window 24 --seed {seed} --label hpc",
                        group="e2_window24"))
        for days in (5, 10):
            jobs.append(Job(f"e2_multiday{days}_s{seed}", "e2", f"scripts/run_exploratory_e2.py --days {days} --seed {seed}", group=f"e2_multiday{days}"))
    for seed in seeds:
        jobs.append(Job(f"e2_window72_s{seed}", "e2_l72", f"scripts/run_neural.py --models attention_gru --window 72 --seed {seed} --label hpc",
                        group="e2_window72"))
    return jobs


def v3_job_list(seeds: list[int], horizons: list[int]) -> list[Job]:
    """The exploratory v3 GPU jobs (configs/eval_exploratory_v3.yaml): A4 first (the longer jobs), then A2.

    A4 trains E3 attention + GRU with patience 8 and at most 60 epochs; the patience-3 runs took 18-21 min per seed,
    so these are estimated at 40-75 min (short, gpu-short). A job stopped at the time limit resumes from its walk
    checkpoints. A2 trains the E1 token corrections without the IV field (about a minute per seed).
    """
    jobs = []
    for horizon in horizons:
        for seed in seeds:
            jobs.append(Job(f"v3_e3_patience8_{horizon}h_s{seed}", "v3_a4",
                            f"scripts/run_exploratory_v3.py --parts attention_gru_patience8 --horizon {horizon} --seed {seed}",
                            group="v3_e3_attention_gru_patience8"))
    for seed in seeds:
        for part in ("residual_attention_no_iv", "residual_deepsets_no_iv"):
            jobs.append(Job(f"v3_{part}_s{seed}", "v3_a2", f"scripts/run_exploratory_v3.py --parts {part} --seed {seed}", group=f"v3_{part}"))
    return jobs


def render_script(job: Job, settings: dict[str, Any], gpu_type: str, walk_minutes: float | None) -> str:
    """The sbatch script of one job; partition and time limit are set by the scheduler at submission."""
    resources, scratch, code = settings["resources"], settings["scratch_dir"], settings["code_dir"]
    status = f"{scratch}/outputs/hpc/status/{job.name}"
    margin = int(settings["deadline_margin_minutes"]) * 60
    return f"""#!/bin/bash
# {job.wave} ({job.kind}): {job.name}. Generated by hpc/make_jobs.py; partition and --time come from hpc/scheduler.py.
#SBATCH --job-name={settings['job_prefix']}{job.name}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:{gpu_type}:1
#SBATCH --cpus-per-task={resources['cpus_per_task']}
#SBATCH --mem={resources['mem']}
#SBATCH --output={scratch}/outputs/hpc/logs/%x_%j.log
umask 077                                   # the data and results stay private
# Start clean: the login shell may activate another conda env (~/.bashrc) and SLURM copies that environment in.
module purge
module load {' '.join(settings['modules'])}
eval "$(conda shell.bash hook)"
for _ in 1 2 3 4 5; do if [ -n "${{CONDA_PREFIX:-}}" ]; then conda deactivate; fi; done
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
conda activate {settings['env_dir']}
[ "$(command -v python)" = "{settings['env_dir']}/bin/python" ] || {{ echo "python is $(command -v python), not {settings['env_dir']}" >&2; exit 2; }}
# The anaconda module puts its older libstdc++ on LD_LIBRARY_PATH; the env's own libraries must come first.
export LD_LIBRARY_PATH="{settings['env_dir']}/lib${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"
cd {code}
end=$(scontrol show job "$SLURM_JOB_ID" | grep -oP 'EndTime=\\K\\S+')
export NIFTYGEX_PROFILE=hpc NIFTYGEX_DEVICE=cuda NIFTYGEX_GPU_TYPE={gpu_type} PYTHONPATH=src
export NIFTYGEX_DEADLINE=$(( $(date -d "$end" +%s) - {margin} )){f" NIFTYGEX_WALK_MINUTES={walk_minutes:.1f}" if walk_minutes else ""}
echo "job $SLURM_JOB_ID on $(hostname), partition $SLURM_JOB_PARTITION, ends $end, commit $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python {job.command}
code=$?
# Finished: .done. Crashed: .failed (not resubmitted). Stopped before its deadline or killed at the time limit:
# no marker, so the scheduler submits it again and it resumes from the walk checkpoints.
if [ $code -eq 0 ]; then
    touch {status}.done
elif ! echo " {' '.join(str(c) for c in sorted(RESUBMIT_CODES))} " | grep -q " $code "; then
    echo "$code" > {status}.failed
fi
sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,State --noheader || true
exit $code
"""


@dataclass(frozen=True)
class Queued:
    partition: str
    state: str                      # RUNNING or PENDING
    slurm_id: str
    start: datetime | None = None   # squeue's estimated start for pending jobs


@dataclass(frozen=True)
class Action:
    job: Job
    kind: str                       # submit, or move (cancel the pending gpu job, submit a chained piece)
    partition: str
    time: str
    cancel: str | None = None
    exclude: tuple[str, ...] = ()   # nodes our running jobs use (--exclude), so our jobs never share a node


def minutes(value: str) -> float:
    hours, mins, secs = (int(p) for p in value.split(":"))
    return 60 * hours + mins + secs / 60


def plan(jobs: list[Job], queued: dict[str, Queued], done: set[str], failed: set[str], attempts: dict[str, int],
         settings: dict[str, Any], walk_minutes: dict[str, float], estimate_start: Callable[[Job, str, str], datetime | None],
         now: datetime, busy_nodes: tuple[str, ...] = ()) -> tuple[list[Action], list[str]]:
    """This pass's submissions and moves (in job order), and the jobs that are out of attempts.

    A job goes to the partition with capacity and the earliest estimated start among its routes (short: gpu-short,
    courses-gpu; long: gpu, courses-gpu, and, after its first attempt, chained gpu-short pieces when its walks fit a
    piece and no short job is waiting). A long job pending on gpu moves to a chained piece when gpu-short has a free
    slot and the piece starts sooner by the configured margin.
    """
    parts = {name: p for name, p in settings["partitions"].items() if p.get("enabled", True)}
    routing = settings["routing"]
    exclude = tuple(sorted(busy_nodes)) if settings.get("avoid_shared_nodes", True) else ()
    submitted = {p: 0 for p in parts}
    running = {p: 0 for p in parts}
    for entry in queued.values():
        submitted[entry.partition] = submitted.get(entry.partition, 0) + 1
        running[entry.partition] = running.get(entry.partition, 0) + (entry.state == "RUNNING")
    open_jobs = [j for j in jobs if j.name not in done and j.name not in failed]
    exhausted = [j.name for j in open_jobs if j.name not in queued and attempts.get(j.name, 0) >= int(settings["max_attempts"])]
    waiting = [j for j in open_jobs if j.name not in queued and j.name not in exhausted]
    short_waiting = any(j.kind == "short" for j in waiting)
    chain, chain_time = routing["chain_on"], routing["chain_time"]

    def chainable(job: Job) -> bool:
        estimate = walk_minutes.get(job.group)
        return estimate is not None and estimate <= float(routing["chain_max_walk_minutes"]) and estimate < minutes(chain_time)

    def has_room(partition: str) -> bool:
        return submitted.get(partition, 0) < int(parts[partition]["max_submitted"])

    actions: list[Action] = []
    for job in waiting:
        routes = [(p, parts[p]["time"]) for p in routing[job.kind] if p in parts]
        # A long job is first queued on gpu (for the night); once it has run, it may continue as a chained piece.
        if job.kind == "long" and attempts.get(job.name, 0) > 0 and chain in parts and not short_waiting and chainable(job):
            routes.append((chain, chain_time))
        options = [(p, t) for p, t in routes if has_room(p)]
        if not options:
            continue
        starts = [(estimate_start(job, p, t) or now + timedelta(days=30), i, p, t) for i, (p, t) in enumerate(options)]
        _, _, partition, time_limit = min(starts)
        actions.append(Action(job, "submit", partition, time_limit, exclude=exclude))
        submitted[partition] = submitted.get(partition, 0) + 1
    # A long job still pending on gpu: run it as chained gpu-short pieces when a slot is free and starts sooner.
    margin = timedelta(minutes=float(routing["move_if_earlier_by_minutes"]))
    for job in open_jobs:
        entry = queued.get(job.name)
        if (entry is None or entry.state != "PENDING" or entry.partition == chain or job.kind != "long" or short_waiting
                or not chainable(job) or chain not in parts or running.get(chain, 0) >= int(parts[chain]["max_running"]) or not has_room(chain)):
            continue
        piece = estimate_start(job, chain, chain_time)
        if piece is not None and (entry.start is None or piece + margin < entry.start):
            actions.append(Action(job, "move", chain, chain_time, cancel=entry.slurm_id, exclude=exclude))
            submitted[chain] = submitted.get(chain, 0) + 1
    return actions, exhausted


def kill_report(slurm_id: str, record: dict[str, Any], records: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """For a finished attempt that ended by SIGKILL (exit code ...:9), our other attempts that overlapped it on its node."""
    if not str(record.get("exit_code", "")).endswith(":9"):
        return None
    shared = []
    for other_id, other in records.items():
        if other_id == slurm_id or other.get("start") is None or other.get("nodes") != record.get("nodes"):
            continue
        other_end = other.get("end") or datetime.max
        if other["start"] <= record["end"] and other_end >= record["start"]:
            shared.append(other_id)
    return {"shared_with": sorted(shared)}


def expand(value: str) -> str:
    return os.path.expandvars(value)
