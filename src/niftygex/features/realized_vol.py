"""Daily realized volatility from within-day 5-minute returns, and causal trailing means (shared by features and targets)."""
from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl


def daily_realized_vol(bars: pl.DataFrame, *, min_bars: int, trading_days_per_year: int, returns_per_day: int) -> pl.DataFrame:
    """Per trade date: sqrt(sum of squared within-day 5-minute log returns), with bar count and availability.

    ``bars`` must hold a single instrument per day (front-month futures are one contract per day). Days with
    fewer than ``min_bars`` valid bars get a null RV. The annualized value scales the mean squared return to
    ``returns_per_day`` returns and ``trading_days_per_year`` days.
    """
    returns = (
        bars.filter(pl.col("close").is_not_null()).sort("datetime")
        .with_columns(pl.col("close").log().diff().over("trade_date").alias("r"))
    )
    daily = returns.group_by("trade_date").agg(
        pl.len().alias("bars"),
        pl.col("r").pow(2).sum().sqrt().alias("rv"),
        (pl.col("r").pow(2).mean() * returns_per_day * trading_days_per_year).sqrt().alias("rv_annualized"),
        pl.col("available_at").max().alias("rv_available_at"),
    )
    return daily.with_columns(
        *[pl.when(pl.col("bars") >= min_bars).then(pl.col(c)).otherwise(None).alias(c) for c in ("rv", "rv_annualized")]
    ).sort("trade_date")


def trailing_mean(daily: pl.DataFrame, trading_dates: list[date], window: int, min_days: int, column: str = "rv") -> pl.DataFrame:
    """Mean of ``column`` over the last ``window`` calendar trading days ending on each date (inclusive)."""
    values = dict(daily.select("trade_date", column).iter_rows())
    rows = []
    for index, day in enumerate(trading_dates):
        window_values = [values.get(d) for d in trading_dates[max(0, index - window + 1): index + 1]]
        present = [v for v in window_values if v is not None]
        rows.append((day, float(np.mean(present)) if len(present) >= min_days else None))
    return pl.DataFrame(rows, schema={"trade_date": pl.Date, f"{column}_trailing_mean": pl.Float64}, orient="row")
