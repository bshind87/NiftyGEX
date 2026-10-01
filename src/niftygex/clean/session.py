"""Timestamp, date-range, session, and configured-session helpers."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any

import polars as pl

TIMEZONE = "Asia/Kolkata"


def parse_ist(value: pl.Expr) -> pl.Expr:
    """Parse naive exchange timestamps and attach the IST timezone."""
    return value.str.strptime(pl.Datetime("us"), strict=False).dt.replace_time_zone(TIMEZONE)


def normalize_timestamp(frame: pl.DataFrame, column: str = "datetime") -> pl.DataFrame:
    return frame.with_columns(parse_ist(pl.col(column)).alias(column))


def minute_of_day(column: str = "datetime") -> pl.Expr:
    return pl.col(column).dt.hour().cast(pl.Int32) * 60 + pl.col(column).dt.minute().cast(pl.Int32)


def filter_date_range(frame: pl.DataFrame, start: date, end: date) -> pl.DataFrame:
    return frame.filter(pl.col("datetime").dt.date().is_between(start, end, closed="both"))


def in_session(session_open: time, session_close: time) -> pl.Expr:
    """Start labels from the first to the last regular bar, inclusive."""
    start = session_open.hour * 60 + session_open.minute
    end = session_close.hour * 60 + session_close.minute
    return minute_of_day().is_between(start, end, closed="both")


def filter_session(frame: pl.DataFrame, session_open: time, session_close: time) -> pl.DataFrame:
    return frame.filter(in_session(session_open, session_close))


def session_removals(frame: pl.DataFrame, session_open: time, session_close: time) -> dict[str, int]:
    """Counts of out-of-session rows by HH:MM start label."""
    removed = frame.filter(~in_session(session_open, session_close))
    if not removed.height:
        return {}
    counts = removed.group_by(pl.col("datetime").dt.strftime("%H:%M").alias("label")).len().sort("label")
    return dict(zip(counts["label"].to_list(), counts["len"].to_list()))


def filter_excluded_dates(frame: pl.DataFrame, excluded_dates: list[date], column: str = "datetime") -> pl.DataFrame:
    if not excluded_dates:
        return frame
    day = pl.col(column).dt.date() if frame.schema[column] != pl.Date else pl.col(column)
    return frame.filter(~day.is_in(excluded_dates))


def parse_time(value: str) -> time:
    hour, minute = (int(part) for part in str(value).split(":", 1))
    return time(hour, minute)


def session_end(session_close: time, bar_minutes: int) -> time:
    """Wall-clock end of the last regular bar."""
    return (datetime.combine(date(2000, 1, 1), session_close) + timedelta(minutes=bar_minutes)).time()


def expected_bars(session_open: time, session_close: time, bar_minutes: int) -> int:
    span = (session_close.hour * 60 + session_close.minute) - (session_open.hour * 60 + session_open.minute)
    return span // bar_minutes + 1


def _dates(regimes: dict[str, Any], key: str) -> list[date]:
    values = regimes.get(key) or []
    return [date.fromisoformat(str(item["date"] if isinstance(item, dict) else item)) for item in values]


def holiday_dates(regimes: dict[str, Any]) -> list[date]:
    return _dates(regimes, "holidays")


def special_session_dates(regimes: dict[str, Any]) -> list[date]:
    return _dates(regimes, "non_regular_sessions")


def truncated_session_dates(regimes: dict[str, Any]) -> list[date]:
    return _dates(regimes, "truncated_sessions")


def configured_quality_dates(regimes: dict[str, Any]) -> list[date]:
    return _dates(regimes, "data_quality_flags")


def session_settings(data: dict[str, Any]) -> dict[str, Any]:
    session = data["session"]
    opening = parse_time(session["open"])
    closing = parse_time(session["close"])
    bar_minutes = int(session["bar_minutes"])
    return {
        "open": opening,
        "close": closing,
        "bar_minutes": bar_minutes,
        "end": session_end(closing, bar_minutes),
        "expected_bars": expected_bars(opening, closing, bar_minutes),
        "eod_oi_bar": parse_time(session["eod_oi_bar"]),
    }
