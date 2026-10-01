"""Data validation checks on stage outputs (pure computations; the validation stage assembles the report)."""
from __future__ import annotations

from typing import Any

import polars as pl

from niftygex.io.regimes import lot_sizes_by_expiry


def lot_size_check(options: pl.LazyFrame, regimes: dict[str, Any], *, base_share: float = 0.99, mixed_share: float = 0.9) -> pl.DataFrame:
    """Per expiry: share of positive OI values divisible by the configured lot and by the expected divisor.

    Passes when the divisor divides >= ``base_share`` of values and, for exception expiries, the dominant
    lot divides >= ``mixed_share``. Expiries without positive OI (missing-OI days only) are reported, not judged.
    """
    all_expiries = options.select(pl.col("expiry_date").unique()).collect()["expiry_date"].to_list()
    observed = (
        options.filter(pl.col("open_interest") > 0)
        .select("expiry_date", pl.col("open_interest").cast(pl.Int64).alias("oi"))
        .collect()
    )
    expected = lot_sizes_by_expiry(regimes, all_expiries)
    shares = observed.join(expected, on="expiry_date").group_by("expiry_date").agg(
        pl.len().alias("positive_oi_rows"),
        (pl.col("oi") % pl.col("units_per_lot") == 0).mean().alias("share_divisible_by_lot"),
        (pl.col("oi") % pl.col("oi_divisor") == 0).mean().alias("share_divisible_by_divisor"),
    )
    return expected.join(shares, on="expiry_date", how="left").with_columns(
        (
            (pl.col("share_divisible_by_divisor") >= base_share)
            & (pl.col("lot_exception") | (pl.col("share_divisible_by_lot") >= base_share))
            & (~pl.col("lot_exception") | (pl.col("share_divisible_by_lot") >= mixed_share))
        ).alias("passed")
    ).sort("expiry_date")


def summary(values: pl.Series, levels=(0.5, 0.95, 0.99)) -> dict[str, float | int | None]:
    values = values.drop_nulls().drop_nans() if values.dtype in (pl.Float32, pl.Float64) else values.drop_nulls()
    out: dict[str, float | int | None] = {"count": values.len()}
    for level in levels:
        out[f"p{int(level * 100)}"] = float(values.quantile(level, interpolation="linear")) if values.len() else None
    out["max"] = float(values.max()) if values.len() else None
    return out
