"""Day-level data-quality flags, bar completeness, and spot bad-print detection."""
from __future__ import annotations

from datetime import date

import polars as pl

PRICE_FIELDS = ("open", "high", "low", "close")
MISSING_CODES = {"options": "missing_options", "futures": "missing_futures", "spot": "missing_spot"}
INCOMPLETE_CODES = {"options": "incomplete_option_bars", "futures": "incomplete_futures_bars", "spot": "incomplete_spot_bars"}


def flag_spot_outliers(spot: pl.DataFrame, futures: pl.DataFrame, max_deviation: float) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Null spot OHLC where |spot close / front-month futures close - 1| exceeds ``max_deviation``.

    The test is close-to-close: open/high/low extremes legitimately diverge on circuit-halt reopen bars
    and futures high/low carry their own bad prints. Per-field deviations are logged for review only.
    ``spot_quality`` is ``ok``, ``bad_print`` (nulled), or ``unchecked`` (no futures bar at that timestamp).
    Returns the updated spot frame and the logged outlier rows.
    """
    reference = futures.select("datetime", pl.col("expiry_date").alias("future_expiry"), *[pl.col(f).alias(f"future_{f}") for f in PRICE_FIELDS])
    joined = spot.join(reference, on="datetime", how="left")
    deviations = [(pl.col(f) / pl.col(f"future_{f}") - 1.0).abs().alias(f"dev_{f}") for f in PRICE_FIELDS]
    joined = joined.with_columns(deviations).with_columns(pl.col("dev_close").alias("max_relative_deviation"))
    bad = pl.col("dev_close") > max_deviation
    joined = joined.with_columns(
        pl.when(pl.col("future_close").is_null()).then(pl.lit("unchecked"))
        .when(bad).then(pl.lit("bad_print"))
        .otherwise(pl.lit("ok"))
        .alias("spot_quality")
    )
    outliers = joined.filter(pl.col("spot_quality") == "bad_print").select(
        "datetime", *PRICE_FIELDS, "future_expiry", *[f"future_{f}" for f in PRICE_FIELDS],
        *[f"dev_{f}" for f in PRICE_FIELDS], "max_relative_deviation",
    )
    cleaned = joined.with_columns(
        [pl.when(pl.col("spot_quality") == "bad_print").then(None).otherwise(pl.col(f)).alias(f) for f in PRICE_FIELDS]
    ).select(*spot.columns, "spot_quality")
    return cleaned, outliers


def bars_per_day(frame: pl.DataFrame, name: str) -> pl.DataFrame:
    """Distinct bar labels per trading day for one source."""
    return frame.group_by(pl.col("datetime").dt.date().alias("trading_date")).agg(
        pl.col("datetime").n_unique().alias(f"{name}_bars")
    )


def day_quality(
    calendar: pl.DataFrame,
    bar_counts: dict[str, pl.DataFrame],
    *,
    expected_bars: int,
    truncated_dates: list[date],
    configured_dates: list[date],
    option_oi_share: pl.DataFrame | None = None,
    min_option_oi_share: float = 0.5,
) -> pl.DataFrame:
    """One row per observed trading date with per-source bar counts and quality codes.

    ``data_quality`` is ``ok`` or a ``|``-joined list of codes. ``missing_spot`` days must be excluded
    from anything that needs spot (GEX scaling, realized volatility, targets). ``missing_option_oi`` marks
    days where fewer than ``min_option_oi_share`` of traded option rows report positive OI (the source
    records missing OI as 0).
    """
    table = calendar.filter(pl.col("observed")).select("trading_date", "next_trading_date")
    for name, counts in bar_counts.items():
        table = table.join(counts, on="trading_date", how="left").with_columns(pl.col(f"{name}_bars").fill_null(0))
    codes: list[pl.Expr] = []
    for name in bar_counts:
        bars = pl.col(f"{name}_bars")
        if name in MISSING_CODES:
            codes.append(pl.when(bars == 0).then(pl.lit(MISSING_CODES[name])))
            codes.append(pl.when((bars > 0) & (bars < expected_bars)).then(pl.lit(INCOMPLETE_CODES[name])))
    if option_oi_share is not None:
        table = table.join(option_oi_share, on="trading_date", how="left")
        missing_oi = pl.col("option_oi_share").is_not_null() & (pl.col("option_oi_share") < min_option_oi_share)
    else:
        table = table.with_columns(pl.lit(None, dtype=pl.Float64).alias("option_oi_share"))
        missing_oi = pl.lit(False)
    table = table.with_columns(missing_oi.alias("missing_option_oi"))
    codes.append(pl.when(pl.col("missing_option_oi")).then(pl.lit("missing_option_oi")))
    codes.append(pl.when(pl.col("trading_date").is_in(configured_dates)).then(pl.lit("configured_flag")))
    table = table.with_columns(
        pl.col("trading_date").is_in(truncated_dates).alias("truncated_session"),
        (pl.col("spot_bars") == 0).alias("missing_spot") if "spot" in bar_counts else pl.lit(False).alias("missing_spot"),
        pl.concat_list(codes).list.drop_nulls().alias("_codes"),
    ).with_columns(
        pl.when(pl.col("_codes").list.len() == 0).then(pl.lit("ok")).otherwise(pl.col("_codes").list.join("|")).alias("data_quality")
    ).drop("_codes")
    return table.sort("trading_date")


def option_oi_share(frame: pl.DataFrame) -> pl.DataFrame:
    """Per trade date: traded option rows and traded rows with positive OI (summable across files)."""
    traded = frame.filter(pl.col("volume") > 0)
    return traded.group_by(pl.col("trade_date").alias("trading_date")).agg(
        pl.len().alias("traded_rows"), (pl.col("open_interest") > 0).sum().alias("traded_rows_with_oi"),
    )


def attach_day_flags(frame: pl.DataFrame, days: pl.DataFrame, date_column: str = "trade_date") -> pl.DataFrame:
    flags = days.select(pl.col("trading_date").alias(date_column), "truncated_session", "missing_spot", "missing_option_oi", "data_quality")
    return frame.join(flags, on=date_column, how="left")
