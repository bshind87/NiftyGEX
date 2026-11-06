"""Snapshot grid and as-of chain selection (only rows with available_at <= snapshot_at)."""
from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from typing import Any

import polars as pl

from niftygex.clean.session import parse_time
from niftygex.exposure.selection import TARGETS

TIMEZONE = "Asia/Kolkata"


def snapshot_grid(trading_dates: Iterable[date], grid: dict[str, Any]) -> pl.DataFrame:
    """One row per (snapshot_at, target): each target's ``<target>_daily`` time on every trading day."""
    times = {target: parse_time(grid[f"{target}_daily"]) for target in TARGETS}
    rows: list[tuple[date, int, str]] = []
    for day in sorted(set(trading_dates)):
        rows.extend((day, value.hour * 60 + value.minute, target) for target, value in times.items())
    frame = pl.DataFrame(rows, schema={"trade_date": pl.Date, "minute": pl.Int64, "target": pl.String}, orient="row")
    return frame.with_columns(
        (pl.col("trade_date").cast(pl.Datetime("us")) + pl.duration(minutes=pl.col("minute")))
        .dt.replace_time_zone(TIMEZONE).alias("snapshot_at")
    ).drop("minute").sort(["snapshot_at", "target"])


def sequence_grid(trading_dates: Iterable[date], grid: dict[str, Any]) -> pl.DataFrame:
    """Token-history snapshots: one row per (snapshot_at, target) every ``sequence_step_minutes`` from
    ``sequence_start`` up to and including the target's daily snapshot ``prediction_at``.

    ``lag`` counts steps back from the prediction snapshot (0 = the prediction snapshot itself).
    """
    start, step = parse_time(grid["sequence_start"]), int(grid["sequence_step_minutes"])
    start_minute = start.hour * 60 + start.minute
    rows: list[tuple[date, str, int, int, int]] = []
    for target in TARGETS:
        end = parse_time(grid[f"{target}_daily"])
        end_minute = end.hour * 60 + end.minute
        if (end_minute - start_minute) % step or end_minute < start_minute:
            raise ValueError(f"{target}: the sequence must step from {start} to {end} in {step}-minute increments")
        minutes = list(range(start_minute, end_minute + 1, step))
        for day in sorted(set(trading_dates)):
            rows.extend((day, target, m, end_minute, (end_minute - m) // step) for m in minutes)
    frame = pl.DataFrame(rows, schema={"trade_date": pl.Date, "target": pl.String, "minute": pl.Int64, "end": pl.Int64, "lag": pl.Int32}, orient="row")
    stamp = lambda column: (pl.col("trade_date").cast(pl.Datetime("us")) + pl.duration(minutes=pl.col(column))).dt.replace_time_zone(TIMEZONE)
    return frame.with_columns(stamp("minute").alias("snapshot_at"), stamp("end").alias("prediction_at")).select(
        "snapshot_at", "target", "trade_date", "prediction_at", "lag").sort(["snapshot_at", "target"])


def as_of(
    rows: pl.DataFrame,
    snapshots: pl.DataFrame,
    keys: list[str],
    *,
    max_staleness_minutes: int | None,
) -> pl.DataFrame:
    """Latest row per (snapshot_at, *keys) on the snapshot's trade date with available_at <= snapshot_at.

    With ``max_staleness_minutes``, rows older than that before the snapshot are not used.
    """
    joined = rows.join(snapshots.select("trade_date", "snapshot_at").unique(), on="trade_date").filter(
        pl.col("available_at") <= pl.col("snapshot_at")
    )
    if max_staleness_minutes is not None:
        joined = joined.filter(pl.col("available_at") >= pl.col("snapshot_at") - pl.duration(minutes=max_staleness_minutes))
    return joined.sort("available_at").group_by(["snapshot_at", *keys], maintain_order=True).last()


def intraday_grid(trading_dates: Iterable[date], snapshots: dict[str, Any], target: str) -> pl.DataFrame:
    """Exploratory E3: one row per (snapshot_at, target) every ``step_minutes`` from ``first`` to ``last`` (inclusive)."""
    first, last, step = parse_time(snapshots["first"]), parse_time(snapshots["last"]), int(snapshots["step_minutes"])
    start, end = first.hour * 60 + first.minute, last.hour * 60 + last.minute
    if (end - start) % step or end < start:
        raise ValueError(f"intraday snapshots must step from {first} to {last} in {step}-minute increments")
    rows = [(day, minute, target) for day in sorted(set(trading_dates)) for minute in range(start, end + 1, step)]
    frame = pl.DataFrame(rows, schema={"trade_date": pl.Date, "minute": pl.Int64, "target": pl.String}, orient="row")
    return frame.with_columns(
        (pl.col("trade_date").cast(pl.Datetime("us")) + pl.duration(minutes=pl.col("minute"))).dt.replace_time_zone(TIMEZONE).alias("snapshot_at")
    ).drop("minute").sort(["snapshot_at", "target"])
