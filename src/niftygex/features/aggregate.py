"""Aggregate snapshot features: one row per (snapshot_at, target).

Every group returns ``*_available_at`` columns for the newest input it used, so the validation stage can
assert that no feature uses data stamped after its snapshot. Inputs are already restricted to rows with
``available_at <= snapshot_at`` by the exposure stage; daily sources are matched on ``available_date ==
trade_date`` (exactly the previous trading day's observation) and are never forward-filled.
"""
from __future__ import annotations

import polars as pl

KEY = ["snapshot_at", "target"]


def pct_distance(level: str, spot: str = "spot") -> pl.Expr:
    return ((pl.col(level) - pl.col(spot)) / pl.col(spot) * 100.0).alias(f"{level}_distance_pct")


def gex_features(totals: pl.DataFrame, strikes: pl.DataFrame) -> pl.DataFrame:
    """Totals under the three conventions plus concentration (HHI of absolute GEX across strikes)."""
    per_strike = strikes.group_by([*KEY, "strike_price"]).agg(pl.col("gex_absolute").sum())
    hhi = per_strike.with_columns((pl.col("gex_absolute") / pl.col("gex_absolute").sum().over(KEY)).alias("share")).group_by(KEY).agg(
        (pl.col("share") ** 2).sum().alias("gex_concentration_hhi"),
    )
    sources = strikes.group_by(KEY).agg(
        pl.col("available_at").max().alias("gex_chain_available_at"),
        pl.col("spot_available_at").max(),
        pl.col("participant_available_at").max(),
    )
    out = totals.select(
        *KEY, "trade_date", "spot",
        pl.col("gex_naive").alias("gex_total_naive"), pl.col("gex_absolute").alias("gex_total_absolute"),
        pl.col("gex_inferred").alias("gex_total_inferred"),
        "spot_available", "oi_available", "inferred_available", pl.col("expiries").alias("gex_expiries"),
        "includes_same_day_expiry", pl.col("min_days_to_expiry").alias("days_to_expiry"),
        "truncated_session", "missing_spot", "data_quality",
    ).join(hhi, on=KEY, how="left").join(sources, on=KEY, how="left")
    # A concentration over missing GEX is missing, not zero.
    return out.with_columns(
        pl.when(pl.col("gex_total_absolute").is_not_null()).then(pl.col("gex_concentration_hhi")).otherwise(None).alias("gex_concentration_hhi")
    )


def level_features(levels: pl.DataFrame) -> pl.DataFrame:
    """Distances of key levels from spot (percent); the naive flip is the level feature, inferred is a flag."""
    return levels.select(
        *KEY,
        pct_distance("call_wall"), pct_distance("put_wall"), pct_distance("max_pain"),
        pct_distance("max_gex_strike"), pct_distance("flip_naive"),
        pl.col("flip_naive").is_not_null().alias("has_flip"),
        pl.col("flip_inferred").is_not_null().alias("has_inferred_flip"),
    ).with_columns(
        # Without spot/OI inputs the flags are unknown rather than False.
        *[pl.when(pl.col("call_wall_distance_pct").is_not_null() | pl.col("flip_naive_distance_pct").is_not_null())
          .then(pl.col(flag)).otherwise(None).alias(flag) for flag in ("has_flip", "has_inferred_flip")]
    )


def _atm(frame: pl.DataFrame, group: list[str]) -> pl.DataFrame:
    return frame.sort((pl.col("strike_price") - pl.col("forward")).abs()).group_by(group, maintain_order=True).agg(
        pl.col("iv").first().alias("atm_iv"), pl.col("time_to_expiry").first(), pl.col("days_to_expiry").first(),
    )


def vol_surface_features(strikes: pl.DataFrame, skew_abs_delta: float) -> pl.DataFrame:
    """ATM IV of the nearest included expiry, 25-delta put-call skew, and term slope to the next expiry."""
    ranked = strikes.with_columns(pl.col("expiry_date").rank("dense").over(KEY).alias("expiry_rank"))
    atm = _atm(ranked.filter(pl.col("expiry_rank") <= 2), [*KEY, "expiry_rank"])
    first = atm.filter(pl.col("expiry_rank") == 1).select(*KEY, "atm_iv", pl.col("time_to_expiry"))
    second = atm.filter(pl.col("expiry_rank") == 2).select(*KEY, pl.col("atm_iv").alias("atm_iv_next"), pl.col("days_to_expiry").alias("days_to_next_expiry"))
    nearest = ranked.filter(pl.col("expiry_rank") == 1)
    put = nearest.filter(pl.col("right") == "Put").sort((pl.col("delta") + skew_abs_delta).abs()).group_by(KEY, maintain_order=True).agg(pl.col("iv").first().alias("iv_put_25d"))
    call = nearest.filter(pl.col("right") == "Call").sort((pl.col("delta") - skew_abs_delta).abs()).group_by(KEY, maintain_order=True).agg(pl.col("iv").first().alias("iv_call_25d"))
    return (
        first.join(second, on=KEY, how="left").join(put, on=KEY, how="left").join(call, on=KEY, how="left")
        .with_columns(
            (pl.col("iv_put_25d") - pl.col("iv_call_25d")).alias("skew_25d"),
            (pl.col("atm_iv_next") - pl.col("atm_iv")).alias("term_slope"),
        )
        .drop("iv_put_25d", "iv_call_25d")
    )


def positioning_features(oi_chain: pl.DataFrame, forwards: pl.DataFrame, previous_eod: pl.DataFrame, near_money_width: float) -> pl.DataFrame:
    """OI change since the previous close near the money (calls, puts) and the OI put-call ratio.

    ``previous_eod`` holds the 15:25 EOD proxy whose ``available_date`` is the snapshot's trade date. Only
    contracts present in both are compared; a missing side (e.g. missing-OI days) gives a missing feature.
    """
    chain = oi_chain.join(forwards, on=[*KEY, "expiry_date"], how="left")
    near = chain.filter(((pl.col("strike_price") / pl.col("forward")) - 1.0).abs() <= near_money_width)
    previous = previous_eod.select("trade_date", "expiry_date", "strike_price", "right",
                                   pl.col("open_interest").alias("previous_oi"), pl.col("available_at").alias("eod_oi_available_at"))
    paired = near.join(previous, on=["trade_date", "expiry_date", "strike_price", "right"], how="inner")
    flow = paired.group_by(KEY).agg(
        *[
            (
                pl.when(pl.col("open_interest").filter(pl.col("right") == right).is_null().any()
                        | pl.col("previous_oi").filter(pl.col("right") == right).is_null().any())
                .then(None)
                .otherwise(pl.col("open_interest").filter(pl.col("right") == right).sum()
                           / pl.col("previous_oi").filter(pl.col("right") == right).sum() - 1.0)
            ).alias(f"oi_change_near_money_{right.lower()}s")
            for right in ("Call", "Put")
        ],
        pl.col("eod_oi_available_at").max(),
    )
    pcr = chain.group_by(KEY).agg(
        pl.when(pl.col("open_interest").is_null().any()).then(None).otherwise(
            pl.col("open_interest").filter(pl.col("right") == "Put").sum() / pl.col("open_interest").filter(pl.col("right") == "Call").sum()
        ).alias("put_call_oi_ratio"),
        pl.col("available_at").max().alias("oi_chain_available_at"),
    )
    return pcr.join(flow, on=KEY, how="left").with_columns(
        # A ratio over a zero denominator is undefined.
        *[pl.when(pl.col(c).is_infinite()).then(None).otherwise(pl.col(c)).alias(c)
          for c in ("put_call_oi_ratio", "oi_change_near_money_calls", "oi_change_near_money_puts")]
    )


def market_state_features(snapshots: pl.DataFrame, basis: pl.DataFrame, vix_daily: pl.DataFrame, futures: pl.DataFrame,
                          *, staleness_minutes: int, intraday_returns_per_day: int, trading_days_per_year: int) -> pl.DataFrame:
    """Futures basis at the snapshot, previous-close India VIX, and futures realized volatility so far today."""
    from niftygex.exposure.snapshots import as_of

    snaps = snapshots.select("trade_date", "snapshot_at").unique()
    good_basis = basis.filter(pl.col("basis_quality") == "good").select("trade_date", "available_at", "basis_bps")
    basis_at = as_of(good_basis, snaps, [], max_staleness_minutes=staleness_minutes).select(
        "snapshot_at", pl.col("basis_bps").alias("futures_basis_bps"), pl.col("available_at").alias("basis_available_at"))
    vix = vix_daily.select(pl.col("available_date").alias("trade_date"), pl.col("close").alias("vix_previous_close"),
                           pl.col("available_at").alias("vix_available_at"))
    bars = futures.filter(pl.col("close").is_not_null()).select("trade_date", "available_at", "close")
    returns = bars.join(snaps, on="trade_date").filter(pl.col("available_at") <= pl.col("snapshot_at")).sort("available_at").with_columns(
        pl.col("close").log().diff().over("snapshot_at").alias("r")
    )
    rv = returns.group_by("snapshot_at").agg(
        (pl.col("r").pow(2).mean() * intraday_returns_per_day * trading_days_per_year).sqrt().alias("rv_intraday_so_far_annualized"),
        pl.col("r").count().alias("qa_rv_intraday_returns"),
        pl.col("available_at").max().alias("futures_rv_available_at"),
    )
    return (
        snapshots.select(*KEY, "trade_date")
        .join(basis_at, on="snapshot_at", how="left")
        .join(vix, on="trade_date", how="left")
        .join(rv, on="snapshot_at", how="left")
        .with_columns(pl.col("vix_previous_close").is_not_null().alias("vix_available"))
        .drop("trade_date")
    )


def participant_features(snapshots: pl.DataFrame, participant: pl.DataFrame, client_types: list[str]) -> pl.DataFrame:
    """Previous trading day's net index-option positions (contracts, all index options) per client type."""
    frame = snapshots.select(*KEY, "trade_date")
    for client in client_types:
        rows = participant.filter(pl.col("Client Type") == client).select(
            pl.col("available_date").alias("trade_date"),
            (pl.col("Option Index Call Long") - pl.col("Option Index Call Short")).alias(f"participant_{client.lower()}_net_index_calls"),
            (pl.col("Option Index Put Long") - pl.col("Option Index Put Short")).alias(f"participant_{client.lower()}_net_index_puts"),
        )
        frame = frame.join(rows, on="trade_date", how="left")
    return frame.drop("trade_date")


def trailing_rv_features(snapshots: pl.DataFrame, futures: pl.DataFrame, trading_dates: list, *, window: int, min_days: int,
                         min_bars: int, returns_per_day: int, trading_days_per_year: int) -> pl.DataFrame:
    """Annualized futures RV averaged over the ``window`` trading days ending the previous trading day.

    Only completed earlier days enter, so the feature is available from the previous day's close.
    """
    from niftygex.features.realized_vol import daily_realized_vol, trailing_mean

    daily = daily_realized_vol(futures, min_bars=min_bars, trading_days_per_year=trading_days_per_year, returns_per_day=returns_per_day)
    trail = trailing_mean(daily, trading_dates, window, min_days, column="rv_annualized").join(
        daily.select("trade_date", "rv_available_at"), on="trade_date", how="left")
    previous = pl.DataFrame({"trade_date": trading_dates[1:], "previous_date": trading_dates[:-1]},
                            schema={"trade_date": pl.Date, "previous_date": pl.Date})
    lagged = previous.join(trail.rename({"trade_date": "previous_date"}), on="previous_date", how="left").select(
        "trade_date", pl.col("rv_annualized_trailing_mean").alias("rv_trailing_20d_annualized_prev"),
        pl.col("rv_available_at").alias("rv_trailing_available_at"),
    )
    return snapshots.select(*KEY, "trade_date").join(lagged, on="trade_date", how="left").drop("trade_date")
