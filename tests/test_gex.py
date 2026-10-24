from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest

from niftygex.exposure.gex import dealer_signs, strike_gex
from niftygex.exposure.levels import call_wall, flip_level, max_gex_strike, max_pain, put_wall
from niftygex.exposure.snapshots import as_of, snapshot_grid
from niftygex.pricing.black76 import delta, gamma
from niftygex.validation.leakage import assert_available_by

IST = ZoneInfo("Asia/Kolkata")
GRID = {"t2_daily": "15:15", "sequence_start": "09:20", "sequence_step_minutes": 5}


def test_black76_delta_put_call_relation() -> None:
    call = delta(100.0, 105.0, 30 / 365, 0.2, 0.0, True)
    put = delta(100.0, 105.0, 30 / 365, 0.2, 0.0, False)
    assert np.isclose(call - put, 1.0)


def test_atm_gamma_increases_as_expiry_approaches() -> None:
    values = [gamma(100.0, 100.0, days / 365, 0.2, 0.0) for days in (30, 7, 1)]
    assert values[0] < values[1] < values[2]


def test_snapshot_grid_one_t2_snapshot_per_trading_day() -> None:
    grid = snapshot_grid([date(2024, 2, 29), date(2024, 2, 28)], GRID)
    assert grid["target"].to_list() == ["t2", "t2"]
    assert grid["snapshot_at"].to_list() == [datetime(2024, 2, 28, 15, 15, tzinfo=IST), datetime(2024, 2, 29, 15, 15, tzinfo=IST)]
    assert grid["snapshot_at"].dtype == pl.Datetime("us", "Asia/Kolkata")


def test_as_of_uses_only_available_rows_within_staleness() -> None:
    snaps = pl.DataFrame({"trade_date": [date(2024, 2, 28)], "snapshot_at": [datetime(2024, 2, 28, 14, 0, tzinfo=IST)]})
    rows = pl.DataFrame({
        "trade_date": [date(2024, 2, 28)] * 4,
        "available_at": [datetime(2024, 2, 28, h, m, tzinfo=IST) for h, m in ((13, 30), (13, 55), (14, 0), (14, 5))],
        "strike_price": [100.0, 100.0, 105.0, 100.0],
        "value": [1, 2, 3, 4],
    })
    latest = as_of(rows, snaps, ["strike_price"], max_staleness_minutes=15).sort("strike_price")
    # 14:05 is after the snapshot; 13:30 is stale; strike 105's 14:00 row is exactly at the snapshot.
    assert latest["value"].to_list() == [2, 3]
    assert_available_by(latest, "snapshot_at")
    assert as_of(rows, snaps, ["strike_price"], max_staleness_minutes=None).filter(pl.col("strike_price") == 100.0)["value"].to_list() == [2]


def chain(spot: float | None = 20000.0, signs: tuple | None = (1.0, -1.0)) -> pl.DataFrame:
    return pl.DataFrame({
        "right": ["Call", "Put"], "gamma": [0.001, 0.001], "open_interest": [1000.0, 2000.0],
        "spot": [spot, spot],
        "dealer_sign_call": [signs[0] if signs else None] * 2, "dealer_sign_put": [signs[1] if signs else None] * 2,
    }, schema_overrides={"spot": pl.Float64, "dealer_sign_call": pl.Float64, "dealer_sign_put": pl.Float64})


def test_gex_conventions_and_spot_squared_scaling() -> None:
    result = strike_gex(chain(), 0.01)
    unit = 0.001 * 20000.0**2 * 0.01
    assert result["gex_absolute"].to_list() == pytest.approx([1000 * unit, 2000 * unit])
    assert result["gex_naive"].to_list() == pytest.approx([1000 * unit, -2000 * unit])
    assert result["gex_inferred"].to_list() == pytest.approx([1000 * unit, -2000 * unit])
    doubled = strike_gex(chain(spot=40000.0), 0.01)
    assert doubled["gex_absolute"][0] == pytest.approx(4 * result["gex_absolute"][0])


def test_inferred_gex_is_missing_without_participant_signs() -> None:
    result = strike_gex(chain(signs=None), 0.01)
    assert result["gex_inferred"].null_count() == 2 and result["gex_naive"].null_count() == 0


def test_missing_spot_gives_missing_gex() -> None:
    assert strike_gex(chain(spot=None), 0.01)["gex_absolute"].null_count() == 2


def test_dealer_signs_from_previous_day_and_no_fill() -> None:
    participant = pl.DataFrame({
        "Client Type": ["Pro", "Client", "Pro"],
        "observation_date": [date(2024, 2, 27), date(2024, 2, 27), date(2024, 2, 28)],
        "available_date": [date(2024, 2, 28), date(2024, 2, 28), date(2024, 2, 29)],
        "available_at": [datetime(2024, 2, 28, 9, 15, tzinfo=IST), datetime(2024, 2, 28, 9, 15, tzinfo=IST), datetime(2024, 2, 29, 9, 15, tzinfo=IST)],
        "Option Index Call Long": [10.0, 1.0, 5.0], "Option Index Call Short": [4.0, 9.0, 8.0],
        "Option Index Put Long": [3.0, 1.0, 6.0], "Option Index Put Short": [7.0, 1.0, 6.0],
    })
    signs = dealer_signs(participant, "Pro").sort("trade_date")
    assert signs.select("trade_date", "dealer_sign_call", "dealer_sign_put").rows() == [
        (date(2024, 2, 28), 1.0, -1.0), (date(2024, 2, 29), -1.0, 0.0),
    ]
    assert date(2024, 3, 1) not in signs["trade_date"].to_list()


def test_walls_ties_choose_closest_strike_to_spot() -> None:
    strikes = np.array([9900.0, 10000.0, 10100.0, 10200.0, 10300.0])
    call_oi = np.array([0.0, 0.0, 50.0, 50.0, 10.0])
    put_oi = np.array([80.0, 80.0, 0.0, 0.0, 0.0])
    wall = call_wall(strikes, call_oi, 10050.0)
    assert wall.value == 10100.0 and wall.tied
    floor = put_wall(strikes, put_oi, 10050.0)
    assert floor.value == 10000.0 and floor.tied
    assert call_wall(strikes, call_oi, 10400.0).value is None


def test_max_pain_known_example() -> None:
    strikes = np.array([90.0, 100.0, 110.0])
    # Payout: S=90 -> 3000, S=100 -> 2000, S=110 -> 3000, so 100 is the unique minimum.
    pain = max_pain(strikes, np.array([100.0, 100.0, 0.0]), np.array([0.0, 100.0, 100.0]), 100.0)
    assert pain.value == 100.0 and not pain.tied
    # Equal payouts at every strike are a tie resolved to the strike closest to spot.
    tie = max_pain(strikes, np.array([100.0, 0.0, 0.0]), np.array([0.0, 0.0, 100.0]), 108.0)
    assert tie.value == 110.0 and tie.tied


def test_max_gex_strike() -> None:
    assert max_gex_strike(np.array([1.0, 2.0, 3.0]), np.array([5.0, 9.0, 1.0]), 2.0).value == 2.0


def test_flip_level_between_put_and_call_concentrations() -> None:
    strikes = np.array([9500.0, 10500.0])
    common = dict(forward=np.array([10000.0, 10000.0]), time_to_expiry=np.array([7 / 365] * 2), iv=np.array([0.15, 0.15]),
                  open_interest=np.array([1000.0, 1000.0]), spot=10000.0, scale=0.01, grid_points=401)
    # Naive: put gamma (negative) dominates below 10000, call gamma (positive) above; symmetric -> flip at 10000.
    flip = flip_level(strike=strikes, sign=np.array([-1.0, 1.0]), **common)
    assert flip.crossings == 1 and flip.value == pytest.approx(10000.0, rel=2e-3)
    assert flip_level(strike=strikes, sign=np.array([1.0, 1.0]), **common).value is None


def test_sequence_grid_runs_from_first_bar_close_to_the_prediction_snapshot() -> None:
    from niftygex.exposure.snapshots import sequence_grid

    grid = sequence_grid([date(2024, 2, 28)], GRID)
    assert grid.height == 72 and grid["target"].unique().to_list() == ["t2"]
    assert grid["snapshot_at"].min() == datetime(2024, 2, 28, 9, 20, tzinfo=IST)
    assert grid["snapshot_at"].max() == datetime(2024, 2, 28, 15, 15, tzinfo=IST)
    assert (grid["snapshot_at"] <= grid["prediction_at"]).all() and grid["prediction_at"].unique().to_list() == [datetime(2024, 2, 28, 15, 15, tzinfo=IST)]
    assert grid.sort("snapshot_at")["lag"].to_list() == list(range(71, -1, -1))
    with pytest.raises(ValueError):
        sequence_grid([date(2024, 2, 28)], {**GRID, "sequence_start": "09:22"})
