"""Spot and India VIX cleaning."""
from __future__ import annotations

from datetime import date, time
from typing import Any

import polars as pl

from niftygex.clean.dedup import MARKET_ORDER, deduplicate
from niftygex.clean.session import (
    filter_date_range,
    filter_excluded_dates,
    filter_session,
    normalize_timestamp,
    session_removals,
)

PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]


def normalize_market_frame(frame: pl.DataFrame) -> pl.DataFrame:
    return (
        frame.with_row_index("source_row")
        .with_columns(*[pl.col(column).cast(pl.Float64, strict=False) for column in PRICE_COLUMNS])
        .pipe(normalize_timestamp)
        .with_columns(pl.col("datetime").dt.date().alias("trade_date"))
        .select("datetime", "trade_date", "stock_code", *PRICE_COLUMNS, "source_row")
    )


def end_labeled_windows(regimes: dict[str, Any], source: str) -> list[tuple[date, date]]:
    """Configured date windows in which a source's bars are end-labeled."""
    windows = (regimes.get("end_labeled_windows") or {}).get(source) or []
    return [(date.fromisoformat(str(item["start"])), date.fromisoformat(str(item["end"]))) for item in windows]


def relabel_end_labeled(frame: pl.DataFrame, windows: list[tuple[date, date]], bar_minutes: int) -> tuple[pl.DataFrame, int]:
    """Convert end labels to start labels (label - one bar) for rows dated inside the configured windows."""
    if not windows:
        return frame, 0
    day = pl.col("datetime").dt.date()
    inside = pl.any_horizontal([day.is_between(first, last, closed="both") for first, last in windows])
    count = frame.filter(inside).height
    frame = frame.with_columns(
        pl.when(inside).then(pl.col("datetime") - pl.duration(minutes=bar_minutes)).otherwise(pl.col("datetime")).alias("datetime")
    ).with_columns(pl.col("datetime").dt.date().alias("trade_date"))
    return frame, count


def clean_market_bars(
    frame: pl.DataFrame,
    *,
    start: date,
    end: date,
    session_open: time,
    session_close: time,
    excluded_dates: list[date],
    end_labeled: list[tuple[date, date]] | None = None,
    bar_minutes: int = 5,
) -> tuple[pl.DataFrame, dict[str, Any], pl.DataFrame]:
    """Intraday spot/VIX bars: label convention, date range, session, excluded dates, and deterministic dedup."""
    frame = normalize_market_frame(frame)
    stats: dict[str, Any] = {"raw_rows": frame.height}
    stats["raw_first_timestamp"] = frame["datetime"].min()
    stats["raw_last_timestamp"] = frame["datetime"].max()
    frame, stats["rows_relabeled_end_to_start"] = relabel_end_labeled(frame, end_labeled or [], bar_minutes)
    frame = filter_date_range(frame, start, end)
    stats["after_date_range"] = frame.height
    stats["session_removals_by_label"] = session_removals(frame, session_open, session_close)
    frame = filter_session(frame, session_open, session_close)
    stats["after_session"] = frame.height
    frame = filter_excluded_dates(frame, excluded_dates)
    stats["after_excluded_dates"] = frame.height
    stats["excluded_date_rows_removed"] = stats["after_session"] - stats["after_excluded_dates"]
    frame, dedup_stats, dropped = deduplicate(frame, ["datetime"], MARKET_ORDER)
    stats.update({key: value for key, value in dedup_stats.items() if key != "before_dedup"})
    stats["cleaned_first_timestamp"] = frame["datetime"].min()
    stats["cleaned_last_timestamp"] = frame["datetime"].max()
    return frame.drop("source_row").sort("datetime"), stats, dropped


def clean_daily_vix(
    frame: pl.DataFrame,
    *,
    start: date,
    end: date,
    excluded_dates: list[date],
) -> tuple[pl.DataFrame, dict[str, Any], pl.DataFrame]:
    """Daily VIX rows carry an intraday source stamp; they are keyed by observation date, not by session."""
    frame = normalize_market_frame(frame).rename({"datetime": "source_datetime", "trade_date": "observation_date"})
    stats: dict[str, Any] = {"raw_rows": frame.height}
    frame = frame.filter(pl.col("observation_date").is_between(start, end, closed="both"))
    stats["after_date_range"] = frame.height
    frame = filter_excluded_dates(frame, excluded_dates, column="observation_date")
    stats["after_excluded_dates"] = frame.height
    stats["excluded_date_rows_removed"] = stats["after_date_range"] - stats["after_excluded_dates"]
    frame, dedup_stats, dropped = deduplicate(frame, ["observation_date"], MARKET_ORDER)
    stats.update({key: value for key, value in dedup_stats.items() if key != "before_dedup"})
    return frame.drop("source_row").sort("observation_date"), stats, dropped
