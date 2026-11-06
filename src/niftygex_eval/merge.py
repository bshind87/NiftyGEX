"""Merge per-seed runs (one cluster job per seed) into one run, exactly as a local run averages its seed models.

A local neural run predicts each row with the mean of its seed models' predictions and scores every walk on that
mean. Here the seeds' saved predictions are averaged row by row, every walk is rescored with the same metrics,
and the stopping epochs are averaged over seeds, as the local per-walk record does.
"""
from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import polars as pl

from niftygex_eval.forecast.metrics import regression_metrics

SEED_TAG = re.compile(r"_seed\d+$")


def group_key(spec: dict[str, Any]) -> str:
    """The spec without what differs between the seeds of one run: the seed, its tag suffix and git fields."""
    base = {k: v for k, v in spec.items() if k not in ("git_commit", "git_dirty")}
    if "training" in base:
        base["training"] = {k: v for k, v in base["training"].items() if k != "seeds"}
    if isinstance(base.get("tag"), str):
        base["tag"] = SEED_TAG.sub("", base["tag"])
    return json.dumps(base, sort_keys=True, default=str)


def seed_of(spec: dict[str, Any]) -> int | None:
    seeds = spec.get("training", {}).get("seeds")
    return int(seeds[0]) if seeds and len(seeds) == 1 else None


def merge_predictions(per_seed: dict[int, pl.DataFrame]) -> pl.DataFrame:
    """Row-wise mean of the seeds' predictions; every seed must predict exactly the same rows and outcomes."""
    seeds = sorted(per_seed)
    base = per_seed[seeds[0]].sort("snapshot_at")
    stacked = []
    for seed in seeds:
        frame = per_seed[seed].sort("snapshot_at")
        if not frame["snapshot_at"].equals(base["snapshot_at"]) or not np.array_equal(frame["y"].to_numpy(), base["y"].to_numpy()):
            raise ValueError(f"seed {seed} predicts different rows or outcomes than seed {seeds[0]}")
        stacked.append(frame["prediction"].to_numpy())
    # Same operation and dtype as NeuralForecaster.predict (np.mean over the seed models, seeds in order).
    return base.with_columns(pl.Series("prediction", np.mean(stacked, axis=0)))


def merge_walk_metrics(per_seed: dict[int, pl.DataFrame], predictions: pl.DataFrame) -> pl.DataFrame:
    """Per walk: metrics of the averaged predictions, row counts and dates from the seeds, epochs averaged."""
    seeds = sorted(per_seed)
    rows = []
    for (walk,), part in predictions.group_by(["walk_id"], maintain_order=True):
        records = [per_seed[s].filter(pl.col("walk_id") == walk).row(0, named=True) for s in seeds]
        first = records[0]
        row = {k: first[k] for k in ("walk_id", "train_rows", "test_rows", "test_start", "test_end")}
        row.update(regression_metrics(part["y"].to_numpy(), part["prediction"].to_numpy()))
        if "stop_epoch_mean" in first:
            row["stop_epoch_mean"] = float(np.mean([r["stop_epoch_mean"] for r in records]))
            row["stop_epoch_max"] = int(max(r["stop_epoch_max"] for r in records))
        row["seeds"] = len(seeds)
        row["seconds"] = float(sum(r.get("seconds") or 0.0 for r in records))
        for column in ("slurm_job_id", "gpu_name", "slurm_partition", "host"):
            if column in first:
                row[column] = ",".join(sorted({str(r[column]) for r in records if r.get(column) is not None})) or None
        if "gpu_peak_gb" in first:
            peaks = [r["gpu_peak_gb"] for r in records if r.get("gpu_peak_gb") is not None]
            row["gpu_peak_gb"] = max(peaks) if peaks else None
        rows.append(row)
    return pl.DataFrame(rows).sort("walk_id")


def merged_spec(specs: dict[int, dict[str, Any]], runs: dict[int, str]) -> dict[str, Any]:
    """The common spec with every seed, the tag without its seed suffix, and the per-seed run ids."""
    seeds = sorted(specs)
    commits = {specs[s].get("git_commit") for s in seeds}
    if len(commits) != 1:
        raise ValueError(f"the seeds were run from different commits: {commits}")
    spec = json.loads(json.dumps(specs[seeds[0]], default=str))
    spec["training"] = {**spec["training"], "seeds": seeds}
    if isinstance(spec.get("tag"), str):
        spec["tag"] = SEED_TAG.sub("", spec["tag"])
    spec["merged_from"] = {str(s): runs[s] for s in seeds}
    spec["git_dirty"] = any(bool(specs[s].get("git_dirty")) for s in seeds)
    return spec
