"""Run any forecaster through the walk-forward harness: inner tuning split, refit, test predictions, per-walk scores."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Callable

import polars as pl

from niftygex_eval.forecast.interface import Forecaster
from niftygex_eval.forecast.metrics import regression_metrics
from niftygex_eval.harness import Walk, check_split, inner_walk, split

KEEP = ["snapshot_at", "trade_date", "cycle"]
# Process isolation (scripts/run_isolated.py): a process trains at most this many new walks, then exits with
# MORE_WALKS so the launcher starts a fresh process; finished walks resume from their checkpoints.
WALK_LIMIT_ENV, MORE_WALKS = "NIFTYGEX_WALK_LIMIT", 75


def gpu_driver_memory() -> str:
    """GPU memory in GB when torch is loaded: CUDA peak allocated since the walk started, or Apple-GPU driver memory."""
    torch = sys.modules.get("torch")
    if torch is None:
        return ""
    if torch.cuda.is_available():
        return f", CUDA peak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GB"
    if torch.backends.mps.is_available():
        return f", GPU driver memory {torch.mps.driver_allocated_memory() / 2**30:.2f} GB"
    return ""


def compute_info() -> dict:
    """Where a walk ran (recorded per walk, outside the run spec, so a resubmitted job resumes the same run)."""
    torch = sys.modules.get("torch")
    cuda = torch is not None and torch.cuda.is_available()
    return {"host": os.uname().nodename, "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
            "gpu_peak_gb": torch.cuda.max_memory_allocated() / 2**30 if cuda else None}


def run_walks(table: pl.DataFrame, walks: list[Walk], make_model: Callable[[], Forecaster], outcome: str, *,
              inner_validation_cycles: int, inner_embargo_cycles: int, keep: list[str] | None = None,
              collect: Callable[[Forecaster, Walk, pl.DataFrame], None] | None = None,
              checkpoint: Path | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Fresh model per walk; returns (test predictions with ``y`` and ``prediction``, per-walk metrics).

    ``table`` must already be restricted to rows with a finite outcome. ``keep`` adds columns (for example a
    regime flag) to the prediction rows for later breakdowns. ``collect(model, walk, test)`` is called after each
    walk's predictions, so a caller can keep extra outputs such as attention weights.

    With ``checkpoint``, each finished walk's predictions and metrics are saved there, and a restart skips the
    walks already saved (``collect`` outputs are not checkpointed, so the two cannot be combined).
    """
    if checkpoint is not None and collect is not None:
        raise ValueError("checkpointed runs cannot use collect")
    limit = int(os.environ[WALK_LIMIT_ENV]) if os.environ.get(WALK_LIMIT_ENV) else None
    if limit is not None and checkpoint is None:
        raise ValueError(f"{WALK_LIMIT_ENV} needs a checkpointed run")
    predictions, per_walk, trained = [], [], 0
    for walk in walks:
        saved = None if checkpoint is None else (checkpoint / f"walk_{walk.walk_id}.parquet", checkpoint / f"walk_{walk.walk_id}_metrics.parquet")
        if saved is not None and all(path.exists() for path in saved):
            predictions.append(pl.read_parquet(saved[0]))
            per_walk.append(pl.read_parquet(saved[1]).row(0, named=True))
            continue
        train, test = split(table, walk)
        check_split(train, test, walk)
        inner = inner_walk(walk, inner_validation_cycles, inner_embargo_cycles)
        if inner is None or not test.height:
            continue
        if limit is not None and trained >= limit:
            print(f"walk limit {limit} reached; continuing in a new process", flush=True)
            sys.exit(MORE_WALKS)
        started = time.time()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        print(f"walk {walk.walk_id} start{gpu_driver_memory()}", flush=True)
        inner_train, valid = split(train, inner)
        check_split(inner_train, valid, inner)
        model = make_model()
        model.tune(inner_train, valid, outcome)
        model.fit(train, outcome)
        y, prediction = test[outcome].to_numpy(), model.predict(test)
        if collect is not None:
            collect(model, walk, test)
        predictions.append(test.select(*KEEP, *(keep or [])).with_columns(
            pl.lit(walk.walk_id).alias("walk_id"), pl.Series("y", y), pl.Series("prediction", prediction)))
        per_walk.append({"walk_id": walk.walk_id, "train_rows": train.height, "test_rows": test.height,
                         "test_start": test["trade_date"].min(), "test_end": test["trade_date"].max(),
                         **regression_metrics(y, prediction), **getattr(model, "summary", dict)(), "seconds": time.time() - started, **compute_info()})
        trained += 1
        del model
        print(f"walk {walk.walk_id} end, {per_walk[-1]['seconds'] / 60:.1f} min{gpu_driver_memory()}", flush=True)
        if saved is not None:
            saved[0].parent.mkdir(parents=True, exist_ok=True)
            predictions[-1].write_parquet(saved[0])
            pl.DataFrame([per_walk[-1]]).write_parquet(saved[1])
    return pl.concat(predictions), pl.DataFrame(per_walk)
