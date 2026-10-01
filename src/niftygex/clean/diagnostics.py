"""Report payloads for the cleaning stage."""
from __future__ import annotations

from collections.abc import Sequence
from datetime import time
from typing import Any

import polars as pl

from niftygex.clean.dedup import decided_by


def _quantiles(series: pl.Series) -> dict[str, float | None]:
    series = series.drop_nulls()
    if not series.len():
        return {"count": 0, "median": None, "p95": None, "max": None}
    return {
        "count": series.len(),
        "median": float(series.median()),
        "p95": float(series.quantile(0.95, interpolation="linear")),
        "max": float(series.max()),
    }


def duplicate_summary(dropped: pl.DataFrame, order: Sequence[tuple[str, bool]], *, top: int = 15) -> dict[str, Any]:
    """Size of value differences and clustering of conflicting duplicates by date and source file."""
    if not dropped.height:
        return {"dropped_rows": 0, "exact_rows": 0, "conflicting_rows": 0}
    conflicting = dropped.filter(~pl.col("exact"))
    payload: dict[str, Any] = {
        "dropped_rows": dropped.height,
        "exact_rows": dropped.height - conflicting.height,
        "conflicting_rows": conflicting.height,
    }
    if not conflicting.height:
        return payload
    conflicting = conflicting.with_columns(
        decided_by(conflicting, order),
        pl.col("datetime").dt.date().alias("trade_date"),
        (pl.col("source_file") == pl.col("kept_source_file")).alias("same_file"),
    )
    payload["conflicts_within_same_source_file"] = int(conflicting["same_file"].sum())
    payload["conflicts_across_source_files"] = int((~conflicting["same_file"]).sum())
    payload["decided_by"] = dict(conflicting.group_by("decided_by").len().sort("len", descending=True).iter_rows())
    payload["absolute_differences"] = {
        column: _quantiles((conflicting[f"kept_{column}"] - conflicting[column]).abs())
        for column in ("close", "volume", "open_interest")
        if f"kept_{column}" in conflicting.columns
    }
    by_date = conflicting.group_by("trade_date").len().sort(["len", "trade_date"], descending=[True, False])
    by_file = conflicting.group_by("source_file").len().sort(["len", "source_file"], descending=[True, False])
    by_month = conflicting.group_by(pl.col("trade_date").dt.strftime("%Y-%m").alias("month")).len().sort("month")
    payload["dates_with_conflicts"] = by_date.height
    payload["top_dates"] = [{"date": d, "rows": n} for d, n in by_date.head(top).iter_rows()]
    payload["by_month"] = dict(by_month.iter_rows())
    payload["source_files_with_conflicts"] = by_file.height
    payload["top_source_files"] = [{"file": f, "rows": n} for f, n in by_file.head(top).iter_rows()]
    return payload


def completeness(days: pl.DataFrame, source: str, expected: int) -> dict[str, Any]:
    column = f"{source}_bars"
    incomplete = days.filter((pl.col(column) > 0) & (pl.col(column) < expected))
    over = days.filter(pl.col(column) > expected)
    missing = days.filter(pl.col(column) == 0)
    return {
        "trading_days": days.height,
        "complete_days": days.filter(pl.col(column) == expected).height,
        "incomplete_days": incomplete.height,
        "missing_days": missing.height,
        "days_over_expected": over.height,
        "incomplete": [{"date": d, "bars": n} for d, n in incomplete.select("trading_date", column).iter_rows()],
        "missing": missing["trading_date"].to_list(),
    }


def missing_labels(frame: pl.DataFrame, dates: list, session_open: time, bar_minutes: int, expected: int) -> dict[str, list[str]]:
    """Missing HH:MM labels on the given dates for one source."""
    labels = [
        f"{(session_open.hour * 60 + session_open.minute + i * bar_minutes) // 60:02d}:"
        f"{(session_open.hour * 60 + session_open.minute + i * bar_minutes) % 60:02d}"
        for i in range(expected)
    ]
    present = (
        frame.filter(pl.col("datetime").dt.date().is_in(dates))
        .group_by(pl.col("datetime").dt.date().alias("d"))
        .agg(pl.col("datetime").dt.strftime("%H:%M").unique().alias("labels"))
    )
    result = {}
    for day, have in present.iter_rows():
        result[str(day)] = [label for label in labels if label not in set(have)]
    return result


def merge_label_counts(total: dict[str, int], current: dict[str, int]) -> None:
    for label, count in current.items():
        total[label] = total.get(label, 0) + count
