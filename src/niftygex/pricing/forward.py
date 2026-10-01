"""Parity-implied synthetic forwards and quality flags."""
from __future__ import annotations

import numpy as np
import polars as pl


def parity_forward(strike, call_price, put_price, time_to_expiry, rate):
    return np.asarray(strike, float) + np.exp(rate * np.asarray(time_to_expiry, float)) * (np.asarray(call_price, float) - np.asarray(put_price, float))


def summarize(parity_values, strikes, near_atm_width: float, min_pairs: int, max_residual: float):
    """Reference single-snapshot implementation; ``estimate_forwards`` is the vectorized equivalent."""
    parity_values = np.asarray(parity_values, float)
    strikes = np.asarray(strikes, float)
    provisional = float(np.nanmedian(parity_values))
    near = np.isfinite(parity_values) & (np.abs(strikes / provisional - 1.0) <= near_atm_width)
    selected = parity_values[near]
    forward = float(np.nanmedian(selected)) if selected.size else float("nan")
    residuals = selected - forward if selected.size else np.array([])
    quality = "good"
    if selected.size < min_pairs:
        quality = "insufficient_pairs"
    elif not np.isfinite(forward) or np.nanmedian(np.abs(residuals)) > max_residual:
        quality = "residual_high"
    return forward, near, {"pair_count": int(selected.size), "residual_median_abs": float(np.nanmedian(np.abs(residuals))) if selected.size else None, "residual_max_abs": float(np.nanmax(np.abs(residuals))) if selected.size else None, "quality": quality}


def estimate_forwards(
    pairs: pl.DataFrame,
    *,
    near_atm_width: float,
    min_pairs: int,
    max_residual: float,
    key: str = "datetime",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Median near-ATM parity forward per snapshot.

    ``pairs`` has one row per (snapshot, strike) with a ``parity_forward`` column, already restricted to
    pairs where both legs traded. The near-ATM set is strikes within ``near_atm_width`` of the provisional
    (all-strike) median. Returns (forwards per snapshot, near-ATM pairs with residuals).
    """
    pairs = pairs.filter(pl.col("parity_forward").is_finite()).with_columns(
        pl.col("parity_forward").median().over(key).alias("provisional_forward")
    ).with_columns(
        ((pl.col("strike_price") / pl.col("provisional_forward") - 1.0).abs() <= near_atm_width).alias("near_atm")
    )
    near = pairs.filter(pl.col("near_atm"))
    forward = near.group_by(key).agg(pl.col("parity_forward").median().alias("forward"), pl.len().alias("forward_pair_count"))
    near = near.join(forward.select(key, "forward"), on=key).with_columns(
        (pl.col("parity_forward") - pl.col("forward")).alias("parity_residual")
    )
    residuals = near.group_by(key).agg(
        pl.col("parity_residual").abs().median().alias("forward_residual_median_abs"),
        pl.col("parity_residual").abs().max().alias("forward_residual_max_abs"),
    )
    snapshots = pairs.select(key).unique()
    forwards = (
        snapshots.join(forward, on=key, how="left")
        .join(residuals, on=key, how="left")
        .with_columns(pl.col("forward_pair_count").fill_null(0))
        .with_columns(
            pl.when(pl.col("forward_pair_count") < min_pairs).then(pl.lit("insufficient_pairs"))
            .when(pl.col("forward").is_null() | (pl.col("forward_residual_median_abs") > max_residual)).then(pl.lit("residual_high"))
            .otherwise(pl.lit("good"))
            .alias("forward_quality")
        )
        .sort(key)
    )
    return forwards, near.sort([key, "strike_price"])
