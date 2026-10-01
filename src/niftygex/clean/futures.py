"""Front-month NIFTY futures cleaning with a once-per-trading-day contract selection."""
from __future__ import annotations

from datetime import date, time
from pathlib import Path
from typing import Any

import polars as pl

from niftygex.clean.dedup import FUTURES_ORDER, deduplicate
from niftygex.clean.session import (
    filter_date_range,
    filter_excluded_dates,
    filter_session,
    normalize_timestamp,
    session_removals,
)

FUTURES_KEYS = ["datetime", "expiry_date"]
ROLL_RULES = ("nearest_expiry_not_expired",)


def normalize_futures_frame(frame: pl.DataFrame, source_file: str, source_file_date: date | None) -> pl.DataFrame:
    numeric = ["open", "high", "low", "close", "volume", "open_interest"]
    return (
        frame.with_row_index("source_row")
        .with_columns(
            pl.col("expiry_date").cast(pl.String).str.to_date(strict=False),
            *[pl.col(column).cast(pl.Float64, strict=False).alias(column) for column in numeric],
            pl.lit(source_file, dtype=pl.String).alias("source_file"),
            pl.lit(source_file_date, dtype=pl.Date).alias("source_file_date"),
        )
        .pipe(normalize_timestamp)
        .with_columns(pl.col("datetime").dt.date().alias("trade_date"))
        .select(
            "datetime", "trade_date", "stock_code", "expiry_date",
            *numeric, "source_file", "source_file_date", "source_row",
        )
    )


def select_daily_contract(frame: pl.DataFrame, roll_rule: str) -> pl.DataFrame:
    """Choose one contract per trading day; return (trade_date, expiry_date).

    ``nearest_expiry_not_expired``: among contracts with at least one bar that day and
    expiry_date >= trade_date, the earliest expiry. The expiring contract stays front month on its
    expiry day. Bars missing from the chosen contract stay missing; there is no intraday fallback.
    """
    if roll_rule not in ROLL_RULES:
        raise ValueError(f"Unknown futures roll rule {roll_rule!r}; expected one of {ROLL_RULES}")
    return (
        frame.filter(pl.col("expiry_date") >= pl.col("trade_date"))
        .group_by("trade_date")
        .agg(pl.col("expiry_date").min())
        .sort("trade_date")
    )


def clean_front_month(
    sources: list[tuple[pl.DataFrame, str, date | None]],
    *,
    start: date,
    end: date,
    session_open: time,
    session_close: time,
    excluded_dates: list[date],
    roll_rule: str,
) -> tuple[pl.DataFrame, dict[str, Any], pl.DataFrame, pl.DataFrame]:
    """Return front-month bars, counts, dropped duplicates, and the per-day contract choice."""
    frame = pl.concat(
        [normalize_futures_frame(value, name, file_date) for value, name, file_date in sources],
        how="vertical_relaxed",
    )
    stats: dict[str, Any] = {"raw_rows": frame.height}
    frame = filter_date_range(frame, start, end)
    stats["after_date_range"] = frame.height
    stats["session_removals_by_label"] = session_removals(frame, session_open, session_close)
    frame = filter_session(frame, session_open, session_close)
    stats["after_session"] = frame.height
    frame = filter_excluded_dates(frame, excluded_dates)
    stats["after_excluded_dates"] = frame.height
    stats["excluded_date_rows_removed"] = stats["after_session"] - stats["after_excluded_dates"]
    frame, dedup_stats, dropped = deduplicate(frame, FUTURES_KEYS, FUTURES_ORDER)
    stats.update({key: value for key, value in dedup_stats.items() if key != "before_dedup"})
    choice = select_daily_contract(frame, roll_rule)
    front = frame.join(choice, on=["trade_date", "expiry_date"], how="semi").sort("datetime")
    stats["after_front_month"] = front.height
    stats["non_front_month_rows_removed"] = frame.height - front.height
    stats["trading_days"] = choice.height
    stats["contract_changes"] = int((choice["expiry_date"] != choice["expiry_date"].shift(1)).fill_null(False).sum())
    return front, stats, dropped, choice


def futures_files(raw_root: Path) -> list[Path]:
    return sorted((raw_root / "futures" / "NIFTY").glob("*/*.csv.gz"))
