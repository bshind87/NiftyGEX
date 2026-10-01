"""Per-strike GEX under naive, absolute, and inferred sign conventions."""
from __future__ import annotations

import polars as pl

CALL_LONG, CALL_SHORT = "Option Index Call Long", "Option Index Call Short"
PUT_LONG, PUT_SHORT = "Option Index Put Long", "Option Index Put Short"


def dealer_signs(participant: pl.DataFrame, client_type: str) -> pl.DataFrame:
    """Dealer gamma sign per right for each trading day, from the previous trading day's participant OI.

    The dealer proxy ``client_type`` is long gamma in calls when its index-call long OI exceeds its short
    OI (+1), short gamma otherwise (-1), and 0 when flat; puts likewise. Rows are keyed by
    ``available_date`` (the trading day the observation becomes usable); a missing day stays missing.
    """
    dealer = participant.filter(pl.col("Client Type") == client_type)
    return dealer.select(
        pl.col("available_date").alias("trade_date"),
        pl.col("observation_date").alias("participant_observation_date"),
        pl.col("available_at").alias("participant_available_at"),
        (pl.col(CALL_LONG) - pl.col(CALL_SHORT)).sign().cast(pl.Float64).alias("dealer_sign_call"),
        (pl.col(PUT_LONG) - pl.col(PUT_SHORT)).sign().cast(pl.Float64).alias("dealer_sign_put"),
    )


def strike_gex(chain: pl.DataFrame, scale: float) -> pl.DataFrame:
    """Add rupee GEX per leg: gamma x OI (index units) x spot^2 x scale, under three sign conventions.

    ``chain`` needs gamma, open_interest, right, spot, dealer_sign_call, dealer_sign_put. Missing spot or
    missing dealer signs give null GEX for the affected convention; nothing is filled.
    """
    is_call = pl.col("right") == "Call"
    unit = pl.col("gamma") * pl.col("open_interest") * pl.col("spot") ** 2 * scale
    return chain.with_columns(unit.alias("gex_absolute")).with_columns(
        pl.when(is_call).then(pl.col("gex_absolute")).otherwise(-pl.col("gex_absolute")).alias("gex_naive"),
        (pl.col("gex_absolute") * pl.when(is_call).then(pl.col("dealer_sign_call")).otherwise(pl.col("dealer_sign_put"))).alias("gex_inferred"),
    )
