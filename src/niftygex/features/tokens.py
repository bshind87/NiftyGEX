"""Token view: one row per liquid option leg of the day's nearest included expiry at each sequence snapshot.

A liquid leg is a GEX-chain row: eligible (traded, OTM-sourced IV solved, good forward, positive time to
expiry) and available within the snapshot staleness window. The expiry is fixed for the whole day
(``selection.nearest_expiry_by_day``). Tokens are ordered by strike, calls before puts, and carry a stable
field set; missing inputs stay null with flags.
"""
from __future__ import annotations

import polars as pl

KEY = ["snapshot_at", "target"]
TOKEN_FIELDS = [
    "token_index", "expiry_date", "strike_price", "right", "is_call",
    "log_moneyness", "iv", "gamma", "delta",
    "strike_gex_absolute_crore", "strike_gex_naive_crore",
    "open_interest", "open_interest_contracts", "oi_change_since_previous_close", "oi_change_since_previous_close_pct",
    "volume_today", "volume_to_oi", "days_to_expiry", "time_to_expiry",
    "available_at", "eod_oi_available_at", "oi_missing", "previous_oi_missing",
]


def build_tokens(legs: pl.DataFrame, previous_eod: pl.DataFrame, volume_today: pl.DataFrame, lots: pl.DataFrame, crore: float) -> pl.DataFrame:
    """Assemble token rows.

    ``legs``: GEX-chain rows of one expiry per day with snapshot_at, target, trade_date, contract keys, forward, iv,
    gamma, delta, open_interest, gex_absolute, gex_naive, available_at, days_to_expiry, time_to_expiry.
    ``previous_eod``: 15:25 EOD OI keyed by ``trade_date`` = its available_date.
    ``volume_today``: cumulative volume per (snapshot_at, contract) over bars available by the snapshot.
    ``lots``: expiry_date -> units_per_lot.
    """
    contract = ["expiry_date", "strike_price", "right"]
    previous = previous_eod.select("trade_date", *contract, pl.col("open_interest").alias("previous_oi"),
                                   pl.col("available_at").alias("eod_oi_available_at"))
    tokens = (
        legs.join(previous, on=["trade_date", *contract], how="left")
        .join(volume_today, on=["snapshot_at", *contract], how="left")
        .join(lots, on="expiry_date", how="left")
        .with_columns(
            (pl.col("right") == "Call").cast(pl.Int8).alias("is_call"),
            (pl.col("strike_price") / pl.col("forward")).log().alias("log_moneyness"),
            (pl.col("gex_absolute") / crore).alias("strike_gex_absolute_crore"),
            (pl.col("gex_naive") / crore).alias("strike_gex_naive_crore"),
            (pl.col("open_interest") / pl.col("units_per_lot")).alias("open_interest_contracts"),
            (pl.col("open_interest") - pl.col("previous_oi")).alias("oi_change_since_previous_close"),
            pl.when(pl.col("previous_oi") > 0).then(pl.col("open_interest") / pl.col("previous_oi") - 1.0).otherwise(None)
            .alias("oi_change_since_previous_close_pct"),
            pl.when(pl.col("open_interest") > 0).then(pl.col("volume_today") / pl.col("open_interest")).otherwise(None).alias("volume_to_oi"),
            pl.col("open_interest").is_null().alias("oi_missing"),
            pl.col("previous_oi").is_null().alias("previous_oi_missing"),
        )
        .sort([*KEY, "strike_price", "is_call"], descending=[False, False, False, True])
        .with_columns(pl.int_range(pl.len()).over(KEY).cast(pl.Int32).alias("token_index"))
    )
    return tokens.select(*KEY, "trade_date", *TOKEN_FIELDS)
