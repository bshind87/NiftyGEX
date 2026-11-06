"""Per-seed jobs and compute bookkeeping shared by the run scripts (local Mac or the Explorer HPC cluster).

On the cluster each seed is its own SLURM job (``--seed``); scripts/merge_seeds.py later averages the seeds'
predictions exactly as one local run averages its seed models. The requested GPU type enters the run spec, so
runs on different GPU types never share a run id or a checkpoint; job ids and the GPU actually used are
recorded per walk (outside the spec, so a job resubmitted after the time limit resumes the same run).
"""
from __future__ import annotations

import os
from typing import Any

import polars as pl

from niftygex_eval.forecast.runner import compute_info
from niftygex_eval.nn.trainer import GPU_TYPE_ENV


def with_seed(neural: dict[str, Any], seed: int | None) -> dict[str, Any]:
    """The neural settings with a single training seed (a per-seed job), or unchanged."""
    if seed is None:
        return neural
    if seed not in neural["training"]["seeds"]:
        raise SystemExit(f"--seed {seed} is not one of the configured seeds {neural['training']['seeds']}")
    return {**neural, "training": {**neural["training"], "seeds": [int(seed)]}}


def spec_extras() -> dict[str, Any]:
    """Run-spec entries for cluster runs (nothing locally, so local run ids are unchanged)."""
    gpu_type = os.environ.get(GPU_TYPE_ENV)
    return {"gpu_type": gpu_type} if gpu_type else {}


def summary_extras(per_walk: pl.DataFrame) -> dict[str, Any]:
    """Where the run's walks were trained: the final process plus every SLURM job id and GPU seen."""
    out: dict[str, Any] = {"compute": compute_info()}
    for column in ("slurm_job_id", "gpu_name", "slurm_partition", "host"):
        if column in per_walk.columns:
            out[f"{column}s"] = sorted({v for v in per_walk[column].to_list() if v is not None})
    if "gpu_peak_gb" in per_walk.columns and per_walk["gpu_peak_gb"].drop_nulls().len():
        out["gpu_peak_gb_max"] = float(per_walk["gpu_peak_gb"].max())
    return out
