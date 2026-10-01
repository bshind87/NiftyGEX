"""Snapshot-free Greek orchestration over IV rows (no disk I/O)."""
from __future__ import annotations

import polars as pl

from niftygex.exposure.selection import greeks_input
from niftygex.pricing.black76 import delta, gamma


def add_greeks(legs: pl.DataFrame, rate: float) -> pl.DataFrame:
    """Black-76 gamma and delta on the parity forward for rows that may enter Greeks.

    Rows failing ``greeks_input()`` (not eligible, or time_to_expiry <= 0) are dropped. Gamma is per
    index point of the forward and per unit of the underlying; it is identical for both legs of a strike
    because both carry the OTM-sourced IV.
    """
    rows = legs.filter(greeks_input())
    if not rows.height:
        return rows.with_columns(pl.lit(None, dtype=pl.Float64).alias("gamma"), pl.lit(None, dtype=pl.Float64).alias("delta"))
    f, k = rows["forward"].to_numpy(), rows["strike_price"].to_numpy()
    t, sigma = rows["time_to_expiry"].to_numpy(), rows["iv"].to_numpy()
    is_call = (rows["right"] == "Call").to_numpy()
    return rows.with_columns(
        pl.Series("gamma", gamma(f, k, t, sigma, rate)),
        pl.Series("delta", delta(f, k, t, sigma, rate, is_call)),
    )
