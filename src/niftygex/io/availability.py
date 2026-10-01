"""Availability timestamps: when each curated row may first be used by a feature."""
from __future__ import annotations

from datetime import date, time

import polars as pl

from niftygex.io.calendar import next_trading_dates

TIMEZONE = "Asia/Kolkata"


def add_bar_availability(frame: pl.DataFrame, bar_minutes: int, column: str = "datetime") -> pl.DataFrame:
    """A start-labeled bar t covers [t, t + bar) and is available at t + bar."""
    return frame.with_columns((pl.col(column) + pl.duration(minutes=bar_minutes)).alias("available_at"))


def at_ist_time(dates: pl.Expr, value: time) -> pl.Expr:
    """Combine a date expression with a wall-clock IST time."""
    return (
        dates.cast(pl.Datetime("us"))
        + pl.duration(hours=value.hour, minutes=value.minute)
    ).dt.replace_time_zone(TIMEZONE)


def add_next_day_availability(
    frame: pl.DataFrame,
    observation_column: str,
    trading_dates: list[date],
    available_time: time,
) -> pl.DataFrame:
    """Daily observations become available on the next trading day strictly after the observation date."""
    mapping = next_trading_dates(frame[observation_column].unique().to_list(), trading_dates)
    lookup = pl.DataFrame(
        {"_obs": list(mapping), "available_date": list(mapping.values())},
        schema={"_obs": pl.Date, "available_date": pl.Date},
    )
    return (
        frame.join(lookup, left_on=observation_column, right_on="_obs", how="left")
        .with_columns(at_ist_time(pl.col("available_date"), available_time).alias("available_at"))
    )


def observation_close(dates: pl.Expr, session_end: time) -> pl.Expr:
    """End of the regular session on an observation date (close label + one bar)."""
    return at_ist_time(dates, session_end)
