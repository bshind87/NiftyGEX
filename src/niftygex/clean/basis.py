"""Futures-spot basis per bar and classification of basis jumps."""
from __future__ import annotations

import numpy as np
import polars as pl


def basis_frame(futures: pl.DataFrame, spot: pl.DataFrame) -> pl.DataFrame:
    """Join front-month futures and cleaned spot on bar label; basis is futures close minus spot close."""
    joined = futures.select(
        "datetime", "available_at", "trade_date", "expiry_date", pl.col("close").alias("future_close"), "data_quality",
    ).join(spot.select("datetime", pl.col("close").alias("spot_close")), on="datetime", how="full", coalesce=True)
    joined = joined.with_columns(
        pl.col("trade_date").fill_null(pl.col("datetime").dt.date()),
        (pl.col("future_close") - pl.col("spot_close")).alias("basis"),
    ).with_columns((pl.col("basis") / pl.col("spot_close") * 1e4).alias("basis_bps"))
    return joined.sort("datetime")


def classify_day(values: np.ndarray, threshold: float, revert_max_bars: int, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    """Classify within-day jumps in one day's valid basis series (bps).

    Returns per-bar ``jump_class`` (``""``, ``spike``, ``event``, ``unresolved``) at the bar where a
    jump starts, and a boolean ``bad`` mask covering spike bars. A jump at bar i reverts when a bar j in
    (i, i + revert_max_bars] returns within ``tolerance * threshold`` of the pre-jump level; bars i..j-1
    are spike bars and the reversal at j is not counted as a separate jump.
    """
    n = values.size
    jump_class = np.full(n, "", dtype=object)
    bad = np.zeros(n, dtype=bool)
    i = 1
    while i < n:
        before = values[i - 1]
        if abs(values[i] - before) <= threshold:
            i += 1
            continue
        reverted_at = None
        for j in range(i + 1, min(i + revert_max_bars, n - 1) + 1):
            if abs(values[j] - before) <= tolerance * threshold:
                reverted_at = j
                break
        if reverted_at is not None:
            jump_class[i] = "spike"
            bad[i:reverted_at] = True
            i = reverted_at + 1
        elif i + revert_max_bars > n - 1:
            # The day ends before the full reversion window; the jump cannot be classified.
            jump_class[i] = "unresolved"
            i += 1
        else:
            jump_class[i] = "event"
            i += 1
    return jump_class, bad


def classify_basis(frame: pl.DataFrame, threshold: float, revert_max_bars: int, tolerance: float) -> pl.DataFrame:
    """Add ``jump_class``, ``basis_quality`` (good/bad/missing), and ``roll_day`` to a basis frame.

    Jumps are measured only between consecutive valid bars of the same trading day. The day-boundary
    step on a day whose contract differs from the previous trading day's is a roll transition.
    """
    frame = frame.sort("datetime")
    valid = frame.filter(pl.col("basis_bps").is_not_null())
    classes: list[np.ndarray] = []
    bads: list[np.ndarray] = []
    for (_,), group in valid.group_by(["trade_date"], maintain_order=True):
        jump_class, bad = classify_day(group["basis_bps"].to_numpy(), threshold, revert_max_bars, tolerance)
        classes.append(jump_class)
        bads.append(bad)
    valid = valid.select("datetime").with_columns(
        pl.Series("jump_class", np.concatenate(classes) if classes else np.array([], dtype=object), dtype=pl.String),
        pl.Series("_bad", np.concatenate(bads) if bads else np.array([], dtype=bool)),
    )
    frame = frame.join(valid, on="datetime", how="left").with_columns(
        pl.when(pl.col("jump_class") == "").then(None).otherwise(pl.col("jump_class")).alias("jump_class"),
        pl.when(pl.col("basis_bps").is_null()).then(pl.lit("missing"))
        .when(pl.col("_bad")).then(pl.lit("bad"))
        .otherwise(pl.lit("good")).alias("basis_quality"),
    ).drop("_bad")
    contracts = (
        frame.filter(pl.col("expiry_date").is_not_null())
        .group_by("trade_date").agg(pl.col("expiry_date").first())
        .sort("trade_date")
        .with_columns((pl.col("expiry_date") != pl.col("expiry_date").shift(1)).fill_null(False).alias("roll_day"))
        .select("trade_date", "roll_day")
    )
    return frame.join(contracts, on="trade_date", how="left").with_columns(pl.col("roll_day").fill_null(False)).sort("datetime")


def _lagged_returns(reference: pl.DataFrame, other: pl.DataFrame, value: str, lag: int, bar_minutes: int) -> pl.DataFrame:
    """Join ``other`` shifted by ``lag`` bars onto ``reference`` and compute same-day 5-minute returns of both."""
    shifted = other.with_columns((pl.col("datetime") + pl.duration(minutes=bar_minutes * lag)).alias("datetime"))
    return (
        reference.join(shifted, on="datetime")
        .drop_nulls(["ref", value])
        .sort("datetime")
        .with_columns(
            pl.col("datetime").dt.date().alias("day"),
            pl.col("ref").pct_change().alias("ref_ret"),
            pl.col(value).pct_change().alias("other_ret"),
        )
        .filter(pl.col("day") == pl.col("day").shift(1))
    )


def label_alignment(reference: pl.DataFrame, other: pl.DataFrame, value: str, bar_minutes: int, lags: tuple[int, ...] = (-1, 0, 1)) -> pl.DataFrame:
    """Per-day correlation of 5-minute returns between a reference series (column ``ref``) and ``other`` shifted by each lag.

    A lag of -1 means ``other``'s label t+bar matches the reference label t, i.e. ``other`` is end-labeled
    relative to a start-labeled reference. ``best_lag`` is the lag with the highest daily correlation.
    """
    daily = None
    for lag in lags:
        returns = _lagged_returns(reference, other, value, lag, bar_minutes)
        corr = returns.group_by("day").agg(pl.corr("ref_ret", "other_ret").alias(f"corr_lag{lag}"), pl.len().alias(f"n_lag{lag}"))
        daily = corr if daily is None else daily.join(corr, on="day", how="full", coalesce=True)
    columns = [f"corr_lag{lag}" for lag in lags]
    best = pl.concat_list([pl.col(c).fill_nan(None).fill_null(-2.0) for c in columns]).list.arg_max()
    return daily.with_columns(
        best.map_elements(lambda index: lags[index], return_dtype=pl.Int32).alias("best_lag"),
        pl.max_horizontal(columns).alias("best_corr"),
    ).sort("day")


def alignment_segments(daily: pl.DataFrame) -> list[dict]:
    """Contiguous runs of days sharing the same best lag."""
    segments: list[dict] = []
    for day, lag in daily.select("day", "best_lag").iter_rows():
        if segments and segments[-1]["best_lag"] == lag:
            segments[-1]["end"] = day
            segments[-1]["days"] += 1
        else:
            segments.append({"best_lag": lag, "start": day, "end": day, "days": 1})
    return segments


def atm_synthetic(options: pl.LazyFrame) -> pl.DataFrame:
    """Nearest-expiry ATM synthetic forward per bar (K + C - P at the strike minimizing |C - P|); a diagnostic reference."""
    near = (
        options.filter(pl.col("expiry_date") >= pl.col("trade_date"))
        .group_by("trade_date").agg(pl.col("expiry_date").min().alias("near_expiry"))
    )
    chain = (
        options.join(near, on="trade_date")
        .filter(pl.col("expiry_date") == pl.col("near_expiry"))
        .select("datetime", "strike_price", "right", "close")
    )
    calls = chain.filter(pl.col("right") == "Call").select("datetime", "strike_price", pl.col("close").alias("call"))
    puts = chain.filter(pl.col("right") == "Put").select("datetime", "strike_price", pl.col("close").alias("put"))
    return (
        calls.join(puts, on=["datetime", "strike_price"])
        .with_columns((pl.col("call") - pl.col("put")).abs().alias("gap"))
        .sort(["datetime", "gap", "strike_price"])
        .group_by("datetime", maintain_order=True).first()
        .select("datetime", (pl.col("strike_price") + pl.col("call") - pl.col("put")).alias("synthetic"))
        .collect()
    )


def roll_transitions(frame: pl.DataFrame) -> pl.DataFrame:
    """Basis step across each roll day boundary: last valid basis of the prior day vs first of the roll day."""
    valid = frame.filter(pl.col("basis").is_not_null())
    first = valid.group_by("trade_date").agg(
        pl.col("expiry_date").first().alias("new_expiry"), pl.col("basis").first().alias("first_basis"),
    )
    last = valid.group_by("trade_date").agg(
        pl.col("expiry_date").last().alias("old_expiry"), pl.col("basis").last().alias("last_basis"),
    ).sort("trade_date").with_columns(pl.col("trade_date").shift(-1).alias("next_day"))
    rolls = frame.filter(pl.col("roll_day")).select("trade_date").unique()
    return (
        rolls.join(first, on="trade_date", how="left")
        .join(last.select(pl.col("next_day").alias("trade_date"), "old_expiry", "last_basis"), on="trade_date", how="left")
        .with_columns((pl.col("first_basis") - pl.col("last_basis")).alias("basis_step"))
        .sort("trade_date")
    )
