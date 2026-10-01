"""Vectorized Black-76 implied-volatility inversion."""
from __future__ import annotations

import numpy as np
import polars as pl

from niftygex.pricing.black76 import price

IV_STATUSES = ("ok", "no_forward", "nonpositive_time", "nonpositive_price", "below_intrinsic", "above_upper_bound", "not_converged", "otm_leg_missing")


def invert_with_status(
    price_observed,
    forward,
    strike,
    time_to_expiry,
    rate,
    is_call,
    *,
    iterations: int = 60,
    bounds: tuple[float, float] = (1e-5, 5.0),
    max_error_fraction: float = 0.02,
):
    """Bracketed bisection on Black-76 price; returns (iv, status) with NaN iv unless status is ``ok``.

    No-arbitrage bounds are checked before inversion: intrinsic discounted value from below and the
    discounted forward (call) or strike (put) from above.
    """
    observed, forward, strike, time_to_expiry = np.broadcast_arrays(
        *(np.asarray(value, dtype=float) for value in (price_observed, forward, strike, time_to_expiry))
    )
    is_call = np.broadcast_to(np.asarray(is_call, dtype=bool), observed.shape)
    discount = np.exp(-rate * np.where(time_to_expiry > 0, time_to_expiry, 0.0))
    intrinsic = discount * np.where(is_call, np.maximum(forward - strike, 0.0), np.maximum(strike - forward, 0.0))
    upper = discount * np.where(is_call, forward, strike)

    status = np.full(observed.shape, "ok", dtype=object)
    status[~np.isfinite(forward)] = "no_forward"
    status[(status == "ok") & ~(time_to_expiry > 0)] = "nonpositive_time"
    status[(status == "ok") & ~(observed > 0)] = "nonpositive_price"
    status[(status == "ok") & (observed < intrinsic)] = "below_intrinsic"
    status[(status == "ok") & (observed >= upper)] = "above_upper_bound"
    attempt = status == "ok"

    sigma = np.full(observed.shape, np.nan)
    if attempt.any():
        p, f, k, t, c = observed[attempt], forward[attempt], strike[attempt], time_to_expiry[attempt], is_call[attempt]
        low = np.full(p.shape, bounds[0])
        high = np.full(p.shape, bounds[1])
        for _ in range(iterations):
            mid = 0.5 * (low + high)
            model = price(f, k, t, mid, rate, c)
            low = np.where(model < p, mid, low)
            high = np.where(model >= p, mid, high)
        mid = 0.5 * (low + high)
        residual = np.abs(price(f, k, t, mid, rate, c) - p)
        converged = np.isfinite(mid) & (residual <= max_error_fraction * np.maximum(p, 1.0))
        sigma[attempt] = np.where(converged, mid, np.nan)
        sub = status[attempt]
        sub[~converged] = "not_converged"
        status[attempt] = sub
    return sigma, status


def assign_otm_iv(legs: pl.DataFrame) -> pl.DataFrame:
    """Give both legs of a strike the IV solved from its out-of-the-money leg.

    ``legs`` carries each leg's own solve as ``iv_own`` / ``iv_own_status``. The OTM leg is the call when
    strike >= forward and the put when strike < forward. Adds ``iv`` and ``iv_status`` (from the OTM leg,
    ``otm_leg_missing`` when that leg has no row on the bar), ``iv_source_right``, and ``iv_source_volume``.
    """
    is_otm = pl.when(pl.col("strike_price") >= pl.col("forward")).then(pl.col("right") == "Call").otherwise(pl.col("right") == "Put")
    source = legs.filter(is_otm).select(
        "datetime", "strike_price",
        pl.col("iv_own").alias("iv"),
        pl.col("iv_own_status").alias("iv_status"),
        pl.col("right").alias("iv_source_right"),
        pl.col("volume").alias("iv_source_volume"),
    )
    return legs.join(source, on=["datetime", "strike_price"], how="left").with_columns(
        pl.col("iv_status").fill_null("otm_leg_missing"),
        pl.col("iv_source_right").fill_null(
            pl.when(pl.col("strike_price") >= pl.col("forward")).then(pl.lit("Call")).otherwise(pl.lit("Put"))
        ),
    )


def invert(price_observed, forward, strike, time_to_expiry, rate, is_call, iterations: int = 60):
    """Backward-compatible wrapper returning (iv, valid)."""
    sigma, status = invert_with_status(price_observed, forward, strike, time_to_expiry, rate, is_call, iterations=iterations)
    return sigma, status == "ok"
