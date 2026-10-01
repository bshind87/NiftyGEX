"""Participant and EOD OI cleaning with strict next-trading-day availability."""
from __future__ import annotations

from datetime import date, time
from pathlib import Path
from typing import Any

import polars as pl

from niftygex.clean.dedup import drop_conflicting_keys
from niftygex.clean.session import filter_excluded_dates, minute_of_day
from niftygex.io.availability import add_next_day_availability

PARTICIPANT_KEYS = ["observation_date", "Client Type"]


def clean_participant_oi(
    path: Path,
    *,
    start: date,
    end: date,
    excluded_dates: list[date],
    trading_dates: list[date],
    available_time: time,
) -> tuple[pl.DataFrame, dict[str, Any], pl.DataFrame]:
    """Participant OI is published after the close; it is available at ``available_time`` on the next trading day.

    Exact duplicate rows collapse; keys with conflicting duplicates are dropped and reported, so a
    participant sign is never chosen by row order.
    """
    frame = (
        pl.read_parquet(path)
        .with_row_index("source_row")
        .with_columns(pl.col("date").cast(pl.Date).alias("observation_date"))
        .drop("date")
    )
    stats: dict[str, Any] = {"raw_rows": frame.height}
    frame = frame.filter(pl.col("observation_date").is_between(start, end, closed="both"))
    stats["after_date_range"] = frame.height
    frame = filter_excluded_dates(frame, excluded_dates, column="observation_date")
    stats["after_excluded_dates"] = frame.height
    stats["excluded_date_rows_removed"] = stats["after_date_range"] - stats["after_excluded_dates"]
    frame, dedup_stats, removed = drop_conflicting_keys(frame, PARTICIPANT_KEYS)
    stats.update({key: value for key, value in dedup_stats.items() if key != "before_dedup"})
    frame = add_next_day_availability(frame, "observation_date", trading_dates, available_time).drop("source_row")
    stats["observation_dates_not_trading_days"] = sorted(
        set(frame["observation_date"].to_list()) - set(trading_dates)
    )
    shared = frame.group_by("available_date").agg(pl.col("observation_date").n_unique().alias("n")).filter(pl.col("n") > 1)
    stats["available_dates_with_multiple_observations"] = shared.height
    return frame.sort(PARTICIPANT_KEYS), stats, removed


def eod_oi_rows(frame: pl.DataFrame, eod_bar: time) -> pl.DataFrame:
    """Option OI observed on the configured EOD bar (the bhavcopy proxy)."""
    return frame.filter(minute_of_day() == eod_bar.hour * 60 + eod_bar.minute).select(
        pl.col("trade_date").alias("observation_date"),
        pl.col("datetime").alias("observed_bar"),
        "expiry_date", "strike_price", "right", "open_interest", "missing_option_oi",
    )


def finalize_eod_oi(frame: pl.DataFrame, trading_dates: list[date], available_time: time) -> pl.DataFrame:
    return add_next_day_availability(frame, "observation_date", trading_dates, available_time).sort(
        ["observation_date", "expiry_date", "strike_price", "right"]
    )
