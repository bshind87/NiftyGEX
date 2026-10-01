"""Curate one NIFTY option-bar file with auditable row-count statistics."""
from __future__ import annotations

from datetime import date, time
from pathlib import Path
from typing import Any

import polars as pl

from niftygex.clean.dedup import OPTION_ORDER, deduplicate
from niftygex.clean.session import (
    filter_date_range,
    filter_excluded_dates,
    filter_session,
    normalize_timestamp,
    session_removals,
)

OPTION_KEYS = ["datetime", "expiry_date", "strike_price", "right"]
OPTION_COLUMNS = [
    "datetime", "trade_date", "stock_code", "expiry_date", "right", "strike_price",
    "open", "high", "low", "close", "volume", "open_interest",
    "source_file", "source_file_date", "source_row",
]


def normalize_option_frame(frame: pl.DataFrame, source_file: str | None, source_file_date: date | None) -> pl.DataFrame:
    numeric = ["strike_price", "open", "high", "low", "close", "volume", "open_interest"]
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
        .select(OPTION_COLUMNS)
    )


def clean_option_frame(
    frame: pl.DataFrame,
    *,
    start: date,
    end: date,
    session_open: time,
    session_close: time,
    excluded_dates: list[date],
    source_file: str | None = None,
    source_file_date: date | None = None,
) -> tuple[pl.DataFrame, dict[str, Any], pl.DataFrame]:
    """Return cleaned rows, counts, and dropped duplicate rows (with kept values) for diagnostics."""
    frame = normalize_option_frame(frame, source_file, source_file_date)
    stats: dict[str, Any] = {"raw_rows": frame.height}
    frame = filter_date_range(frame, start, end)
    stats["after_date_range"] = frame.height
    stats["session_removals_by_label"] = session_removals(frame, session_open, session_close)
    frame = filter_session(frame, session_open, session_close)
    stats["after_session"] = frame.height
    frame = filter_excluded_dates(frame, excluded_dates)
    stats["after_excluded_dates"] = frame.height
    stats["excluded_date_rows_removed"] = stats["after_session"] - stats["after_excluded_dates"]
    frame, dedup_stats, dropped = deduplicate(frame, OPTION_KEYS, OPTION_ORDER)
    stats.update({key: value for key, value in dedup_stats.items() if key != "before_dedup"})
    frame = frame.filter(pl.col("close").is_not_null() & pl.col("volume").is_not_null())
    stats["after_required_fields"] = frame.height
    stats["required_fields_removed"] = stats["after_dedup"] - stats["after_required_fields"]
    return frame.sort(OPTION_KEYS), stats, dropped


def clean_option_file(path: Path, **kwargs: Any) -> tuple[pl.DataFrame, dict[str, Any], pl.DataFrame]:
    return clean_option_frame(pl.read_csv(path, infer_schema_length=10000), **kwargs)
