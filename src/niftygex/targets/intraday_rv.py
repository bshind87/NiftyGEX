"""Exploratory E3: realized volatility over clock windows of a session, for the intraday target and its features.

Bars are start-labeled: the bar labeled t closes at t + 5 min. The return of label t, log(close_t / close_{t-5}),
spans the clock interval (t, t + 5] and is known at t + 5. RV over a clock window (a, b] is
sqrt(sum of squared returns of labels a .. b - 5); it needs every one of those returns (both closes present),
else it is missing. Returns never cross days, so the overnight gap and a futures contract switch never enter.

At a snapshot T: the target window (T, T + h] starts with the return known at T + 5 and is known at T + h; the
backward windows (T - w, T] end with the return known at T; normals average past days only.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

STEP = 5


def return_matrix(futures: pl.DataFrame, trading_dates: list[date], first_return_minute: int, last_label_minute: int) -> np.ndarray:
    """[day, label] 5-minute log returns for labels first_return_minute .. last_label_minute (NaN where missing)."""
    labels = np.arange(first_return_minute, last_label_minute + STEP, STEP)
    position = {day: i for i, day in enumerate(trading_dates)}
    bars = futures.filter(pl.col("close").is_not_null() & (pl.col("close") > 0)).select(
        "trade_date", (pl.col("datetime").dt.hour().cast(pl.Int32) * 60 + pl.col("datetime").dt.minute().cast(pl.Int32)).alias("minute"), "close")
    closes = np.full((len(trading_dates), labels.size + 1), np.nan)   # column 0: the close of label first - 5
    for day, minute, close in bars.iter_rows():
        index = position.get(day)
        column = (minute - (first_return_minute - STEP)) // STEP
        if index is not None and 0 <= column <= labels.size and (minute - first_return_minute) % STEP == 0:
            closes[index, column] = close
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.log(closes[:, 1:] / closes[:, :-1])


def window_rv(returns: np.ndarray, first_return_minute: int, start: int, end: int) -> np.ndarray:
    """Per day: RV over the clock window (start, end] (minutes since midnight); NaN unless every return is present."""
    lo, hi = (start - first_return_minute) // STEP, (end - STEP - first_return_minute) // STEP + 1
    if lo < 0 or hi > returns.shape[1] or hi <= lo:
        return np.full(returns.shape[0], np.nan)
    block = returns[:, lo:hi]
    complete = np.isfinite(block).all(axis=1)
    return np.where(complete, np.sqrt(np.nansum(block ** 2, axis=1)), np.nan)


def past_mean(values: np.ndarray, days: int, min_days: int) -> np.ndarray:
    """Mean over the previous ``days`` entries (the current one excluded), NaN with fewer than ``min_days`` present."""
    present = np.isfinite(values)
    total = np.concatenate([[0.0], np.cumsum(np.where(present, values, 0.0))])
    count = np.concatenate([[0], np.cumsum(present)])
    index = np.arange(values.size)
    start = np.maximum(0, index - days)
    n = count[index] - count[start]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (total[index] - total[start]) / n
    return np.where(n >= min_days, mean, np.nan)


def previous(values: np.ndarray) -> np.ndarray:
    """The previous trading day's value (NaN for the first day)."""
    return np.concatenate([[np.nan], values[:-1]])


def intraday_rv_frames(futures: pl.DataFrame, trading_dates: list[date], snapshot_minutes: list[int], horizons: list[int],
                       backward_minutes: list[int], first_return_minute: int, last_label_minute: int, normal_days: int,
                       normal_min_days: int, full_day_min_returns: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(features, targets) for every (trading day, snapshot minute): backward-window and yesterday RVs, clock-window
    normals, and the forward RV ratio per horizon. Snapshot times are minutes since midnight, IST."""
    returns = return_matrix(futures, trading_dates, first_return_minute, last_label_minute)
    present = np.isfinite(returns).sum(axis=1)
    full_day = np.where(present >= full_day_min_returns, np.sqrt(np.nansum(returns ** 2, axis=1)), np.nan)
    normal = lambda rv: past_mean(rv, normal_days, normal_min_days)
    feature_rows, target_rows = [], []
    for minute in snapshot_minutes:
        features = {"minute": minute, "rv_today_so_far": window_rv(returns, first_return_minute, first_return_minute, minute),
                    "rv_yesterday": previous(full_day)}
        for w in backward_minutes:
            rv = window_rv(returns, first_return_minute, max(minute - w, first_return_minute), minute)
            features[f"rv_last_{w}m"] = rv
            features[f"rv_back_normal_{w}m"] = normal(rv)
        targets = {"minute": minute}
        for h in horizons:
            forward = window_rv(returns, first_return_minute, minute, minute + 60 * h)
            features[f"rv_yesterday_same_window_{h}h"] = previous(forward)
            features[f"rv_normal_{h}h"] = normal(forward)
            targets[f"rv_fwd_{h}h"] = forward
            targets[f"rv_normal_{h}h"] = features[f"rv_normal_{h}h"]
        feature_rows.append(features)
        target_rows.append(targets)

    def frame(rows: list[dict]) -> pl.DataFrame:
        parts = []
        for row in rows:
            columns = {k: v for k, v in row.items() if k != "minute"}
            parts.append(pl.DataFrame({"trade_date": trading_dates, "minute": [row["minute"]] * len(trading_dates),
                                       **{k: pl.Series(k, v, dtype=pl.Float64, nan_to_null=True) for k, v in columns.items()}}))
        stamped = pl.concat(parts).with_columns(
            (pl.col("trade_date").cast(pl.Datetime("us")) + pl.duration(minutes=pl.col("minute"))).dt.replace_time_zone("Asia/Kolkata").alias("snapshot_at"))
        return stamped.sort("snapshot_at")

    features = frame(feature_rows).with_columns(pl.col("snapshot_at").alias("rv_windows_available_at"))
    targets = frame(target_rows)
    for h in horizons:
        targets = targets.with_columns(
            (pl.col(f"rv_fwd_{h}h") / pl.col(f"rv_normal_{h}h")).alias(f"rv_ratio_{h}h"),
            (pl.col("snapshot_at") + pl.duration(hours=h)).alias(f"target_available_at_{h}h"),
            pl.when(pl.col(f"rv_fwd_{h}h").is_null()).then(pl.lit("incomplete_window"))
            .when(pl.col(f"rv_normal_{h}h").is_null()).then(pl.lit("normal_insufficient")).otherwise(pl.lit("ok")).alias(f"reason_{h}h"))
    targets = targets.with_columns(pl.max_horizontal([f"target_available_at_{h}h" for h in horizons]).alias("target_available_at"))
    return features, targets
