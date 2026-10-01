"""Snapshot grid and as-of chain selection (only rows with available_at <= snapshot_at)."""
from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from typing import Any

import polars as pl

from niftygex.clean.session import parse_time

TIMEZONE = "Asia/Kolkata"


def snapshot_grid(trading_dates: Iterable[date], expiry_dates: Iterable[date], grid: dict[str, Any]) -> pl.DataFrame:
    """One row per (snapshot_at, target).

    T2 and T3 run every trading day; T1 runs only on days that are an option expiry date.
    """
    expiries = set(expiry_dates)
    rows: list[tuple[date, int, str]] = []
    t2, t3 = parse_time(grid["t2_daily"]), parse_time(grid["t3_daily"])
    t1 = [parse_time(value) for value in grid["t1_expiry_days"]]
    for day in sorted(set(trading_dates)):
        rows.append((day, t2.hour * 60 + t2.minute, "t2"))
        rows.append((day, t3.hour * 60 + t3.minute, "t3"))
        if day in expiries:
            rows.extend((day, value.hour * 60 + value.minute, "t1") for value in t1)
    frame = pl.DataFrame(rows, schema={"trade_date": pl.Date, "minute": pl.Int64, "target": pl.String}, orient="row")
    return frame.with_columns(
        (pl.col("trade_date").cast(pl.Datetime("us")) + pl.duration(minutes=pl.col("minute")))
        .dt.replace_time_zone(TIMEZONE).alias("snapshot_at")
    ).drop("minute").sort(["snapshot_at", "target"])


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
