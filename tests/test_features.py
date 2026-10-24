from datetime import date, datetime
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from niftygex.features.aggregate import level_features, market_state_features, participant_features, positioning_features, vol_surface_features

IST = ZoneInfo("Asia/Kolkata")
T = datetime(2024, 2, 28, 14, 0, tzinfo=IST)
DAY = date(2024, 2, 28)


def test_oi_change_uses_previous_eod_and_missing_stays_missing() -> None:
    chain = pl.DataFrame({
        "snapshot_at": [T] * 4, "target": ["t2"] * 4, "trade_date": [DAY] * 4, "expiry_date": [date(2024, 2, 29)] * 4,
        "strike_price": [22000.0, 22000.0, 23000.0, 22100.0], "right": ["Call", "Put", "Call", "Put"],
        "open_interest": [150.0, 300.0, 999.0, 50.0], "available_at": [T] * 4,
    })
    forwards = pl.DataFrame({"snapshot_at": [T], "target": ["t2"], "expiry_date": [date(2024, 2, 29)], "forward": [22050.0]})
    eod = pl.DataFrame({
        "trade_date": [DAY] * 3, "expiry_date": [date(2024, 2, 29)] * 3, "strike_price": [22000.0, 22000.0, 22100.0],
        "right": ["Call", "Put", "Put"], "open_interest": [100.0, 200.0, None],
        "available_at": [datetime(2024, 2, 28, 9, 15, tzinfo=IST)] * 3,
    })
    result = positioning_features(chain, forwards, eod, 0.02).row(0, named=True)
    # The 23000 call is outside the near-money band; the 22100 put's previous OI is missing -> puts missing.
    assert result["oi_change_near_money_calls"] == pytest.approx(0.5)
    assert result["oi_change_near_money_puts"] is None
    assert result["put_call_oi_ratio"] == pytest.approx(350.0 / 1149.0)


def test_vol_surface_atm_skew_and_term_slope() -> None:
    rows = []
    for expiry, iv_atm in ((date(2024, 2, 29), 0.12), (date(2024, 3, 7), 0.14)):
        rows += [
            (expiry, 22000.0, "Call", iv_atm, 0.5), (expiry, 22000.0, "Put", iv_atm, -0.5),
            (expiry, 22300.0, "Call", 0.11, 0.26), (expiry, 21700.0, "Put", 0.15, -0.24),
        ]
    frame = pl.DataFrame(rows, schema=["expiry_date", "strike_price", "right", "iv", "delta"], orient="row").with_columns(
        pl.lit(T).alias("snapshot_at"), pl.lit("t2").alias("target"), pl.lit(22010.0).alias("forward"),
        pl.lit(0.01).alias("time_to_expiry"), pl.lit(1).alias("days_to_expiry"),
    )
    result = vol_surface_features(frame, 0.25).row(0, named=True)
    assert result["atm_iv"] == pytest.approx(0.12) and result["atm_iv_next"] == pytest.approx(0.14)
    assert result["term_slope"] == pytest.approx(0.02) and result["skew_25d"] == pytest.approx(0.04)


def test_level_flags_unknown_without_inputs() -> None:
    levels = pl.DataFrame({
        "snapshot_at": [T, T.replace(minute=T.minute + 1)], "target": ["t2", "t2"], "spot": [100.0, None],
        **{c: [101.0, None] for c in ("call_wall", "put_wall", "max_pain", "max_gex_strike", "flip_naive")},
        "flip_inferred": [None, None],
    }, schema_overrides={c: pl.Float64 for c in ("spot", "call_wall", "put_wall", "max_pain", "max_gex_strike", "flip_naive", "flip_inferred")})
    result = level_features(levels).sort("snapshot_at")
    assert result["has_flip"].to_list() == [True, None]
    assert result["has_inferred_flip"].to_list() == [False, None]
    assert result["call_wall_distance_pct"][0] == pytest.approx(1.0)


def test_market_state_uses_only_previous_day_vix_and_bars_before_snapshot() -> None:
    snapshots = pl.DataFrame({"snapshot_at": [T], "target": ["t2"], "trade_date": [DAY]})
    basis = pl.DataFrame({"trade_date": [DAY, DAY], "available_at": [datetime(2024, 2, 28, 13, 55, tzinfo=IST), datetime(2024, 2, 28, 14, 5, tzinfo=IST)],
                          "basis_bps": [10.0, 99.0], "basis_quality": ["good", "good"]})
    vix = pl.DataFrame({"available_date": [date(2024, 2, 29)], "close": [15.0], "available_at": [datetime(2024, 2, 29, 9, 15, tzinfo=IST)]})
    futures = pl.DataFrame({"trade_date": [DAY] * 3, "close": [100.0, 101.0, 500.0],
                            "available_at": [datetime(2024, 2, 28, 13, 50, tzinfo=IST), datetime(2024, 2, 28, 13, 55, tzinfo=IST), datetime(2024, 2, 28, 14, 5, tzinfo=IST)]})
    result = market_state_features(snapshots, basis, vix, futures, staleness_minutes=15, intraday_returns_per_day=74, trading_days_per_year=252).row(0, named=True)
    assert result["futures_basis_bps"] == 10.0
    # The VIX row becomes available the next day, so it is not this snapshot's previous close.
    assert result["vix_previous_close"] is None and result["vix_available"] is False
    assert result["qa_rv_intraday_returns"] == 1 and result["futures_rv_available_at"] <= T


def test_participant_features_previous_day_only() -> None:
    snapshots = pl.DataFrame({"snapshot_at": [T], "target": ["t2"], "trade_date": [DAY]})
    participant = pl.DataFrame({
        "Client Type": ["Pro"], "available_date": [DAY],
        "Option Index Call Long": [10.0], "Option Index Call Short": [4.0], "Option Index Put Long": [1.0], "Option Index Put Short": [3.0],
    })
    result = participant_features(snapshots, participant, ["Pro", "FII"]).row(0, named=True)
    assert result["participant_pro_net_index_calls"] == 6.0 and result["participant_pro_net_index_puts"] == -2.0
    assert result["participant_fii_net_index_calls"] is None


def test_trailing_rv_feature_uses_only_previous_days() -> None:
    from datetime import timedelta

    from niftygex.features.aggregate import trailing_rv_features

    days = [date(2024, 1, d) for d in (1, 2, 3)]
    rows = []
    for i, day in enumerate(days):
        start = datetime(day.year, day.month, day.day, 9, 15, tzinfo=IST)
        closes = [100.0, 100.0 * (1 + 0.01 * (i + 1))]  # one return per day of size ~1%, 2%, 3%
        rows += [(start + timedelta(minutes=5 * k), day, c, start + timedelta(minutes=5 * (k + 1))) for k, c in enumerate(closes)]
    futures = pl.DataFrame(rows, schema=["datetime", "trade_date", "close", "available_at"], orient="row")
    snaps = pl.DataFrame({"snapshot_at": [datetime(2024, 1, 3, 15, 15, tzinfo=IST)], "target": ["t2"], "trade_date": [date(2024, 1, 3)]})
    result = trailing_rv_features(snaps, futures, days, windows={1: 1, 2: 2}, min_bars=2, returns_per_day=1, trading_days_per_year=1).row(0, named=True)
    # Days 1 and 2 only (|log(1.01)|, |log(1.02)|); day 3's own move is excluded.
    import numpy as np
    assert result["rv_trailing_2d_annualized_prev"] == pytest.approx(np.mean([np.log(1.01), np.log(1.02)]))
    assert result["rv_trailing_1d_annualized_prev"] == pytest.approx(np.log(1.02))
    assert result["rv_trailing_available_at"] < snaps["snapshot_at"][0]


def test_tokens_order_and_flow_fields() -> None:
    from niftygex.features.tokens import TOKEN_FIELDS, build_tokens

    near, far = date(2024, 2, 29), date(2024, 3, 7)
    legs = pl.DataFrame({
        "snapshot_at": [T] * 5, "target": ["t2"] * 5, "trade_date": [DAY] * 5,
        "expiry_date": [near, near, near, far, near], "strike_price": [22100.0, 22000.0, 22000.0, 22000.0, 22100.0],
        "right": ["Put", "Put", "Call", "Call", "Call"], "forward": [22050.0] * 5, "iv": [0.12] * 5, "gamma": [0.001] * 5,
        "delta": [-0.6, -0.4, 0.6, 0.55, 0.4], "open_interest": [100.0, 200.0, 300.0, 400.0, None],
        "gex_absolute": [1e7] * 5, "gex_naive": [-1e7, -1e7, 1e7, 1e7, 1e7],
        "available_at": [T] * 5, "days_to_expiry": [1] * 5, "time_to_expiry": [0.003] * 5,
    })
    eod = pl.DataFrame({"trade_date": [DAY], "expiry_date": [near], "strike_price": [22000.0], "right": ["Call"],
                        "open_interest": [250.0], "available_at": [datetime(2024, 2, 28, 9, 15, tzinfo=IST)]})
    volume = pl.DataFrame({"snapshot_at": [T], "expiry_date": [near], "strike_price": [22000.0], "right": ["Call"], "volume_today": [600.0]})
    lots = pl.DataFrame({"expiry_date": [near, far], "units_per_lot": [50, 50]})
    tokens = build_tokens(legs.filter(pl.col("expiry_date") == near), eod, volume, lots, 1e7)
    assert tokens.select("strike_price", "right").rows() == [(22000.0, "Call"), (22000.0, "Put"), (22100.0, "Call"), (22100.0, "Put")]
    assert tokens["token_index"].to_list() == [0, 1, 2, 3]
    call = tokens.row(0, named=True)
    assert call["oi_change_since_previous_close"] == 50.0 and call["volume_to_oi"] == pytest.approx(2.0) and call["open_interest_contracts"] == 6.0
    assert tokens.row(2, named=True)["oi_missing"] is True and tokens.row(1, named=True)["previous_oi_missing"] is True
    assert [c for c in tokens.columns if c not in ("snapshot_at", "target", "trade_date")] == TOKEN_FIELDS


def test_nearest_expiry_by_day_excludes_same_day_expiry() -> None:
    from niftygex.config import load_config
    from niftygex.exposure.selection import nearest_expiry_by_day

    gex = load_config()["data"]["gex"]
    days, expiries = [date(2024, 2, 28), date(2024, 2, 29)], [date(2024, 2, 29), date(2024, 3, 7), date(2024, 4, 25)]
    pairs = pl.DataFrame({"trade_date": days}).join(pl.DataFrame({"expiry_date": expiries}), how="cross")
    nearest = nearest_expiry_by_day(pairs, "t2", gex)
    # On the expiry day itself the same-day contract is excluded for T2, so the next week's expiry is used all day.
    assert nearest.select("trade_date", "expiry_date").rows() == [(date(2024, 2, 28), date(2024, 2, 29)), (date(2024, 2, 29), date(2024, 3, 7))]
    # An expiry with no rows on a day is skipped: the next one with data is used.
    gap = pairs.filter(~((pl.col("trade_date") == date(2024, 2, 28)) & (pl.col("expiry_date") == date(2024, 2, 29))))
    assert nearest_expiry_by_day(gap, "t2", gex).sort("trade_date")["expiry_date"][0] == date(2024, 3, 7)
