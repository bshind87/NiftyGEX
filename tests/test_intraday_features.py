"""Exploratory E3 data layer: intraday grid, strictly causal expiry choice, expiry rule and lineage coverage."""
from datetime import date, datetime
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from niftygex.exposure.selection import INTRADAY_TARGET, causal_nearest_expiry, total_gex_expiries
from niftygex.exposure.snapshots import intraday_grid
from niftygex.features.lineage import INTRADAY_LINEAGE

IST = ZoneInfo("Asia/Kolkata")
GEX = {"max_days_to_expiry": 21, "same_day_expiry_in_total_gex": {INTRADAY_TARGET: False}}
DAY = date(2025, 3, 3)


def at(hour: int, minute: int) -> datetime:
    return datetime(DAY.year, DAY.month, DAY.day, hour, minute, tzinfo=IST)


def test_intraday_grid_has_13_snapshots_from_1015_to_1315() -> None:
    grid = intraday_grid([DAY, date(2025, 3, 4)], {"first": "10:15", "last": "13:15", "step_minutes": 15}, INTRADAY_TARGET)
    assert grid.height == 26 and set(grid["target"]) == {INTRADAY_TARGET}
    times = grid.filter(pl.col("trade_date") == DAY)["snapshot_at"].dt.strftime("%H:%M").to_list()
    assert times[0] == "10:15" and times[-1] == "13:15" and len(times) == 13
    with pytest.raises(ValueError):
        intraday_grid([DAY], {"first": "10:15", "last": "13:10", "step_minutes": 15}, INTRADAY_TARGET)


def test_expiry_choice_is_strictly_causal() -> None:
    near, next_week, same_day = date(2025, 3, 6), date(2025, 3, 13), DAY
    first_seen = pl.DataFrame({"trade_date": [DAY] * 3, "expiry_date": [near, next_week, same_day],
                               "first_available_at": [at(11, 0), at(9, 20), at(9, 20)]})
    snapshots = pl.DataFrame({"trade_date": [DAY] * 4, "snapshot_at": [at(10, 15), at(10, 55), at(11, 0), at(11, 15)]})
    chosen = causal_nearest_expiry(first_seen, snapshots, INTRADAY_TARGET, GEX)
    # The nearer expiry's first row of the day is available at 11:00: before then only next week's is known.
    assert chosen["expiry_date"].to_list() == [next_week, next_week, near, near]
    # The same-day expiry is excluded by the rule (configs/intraday.yaml), as for T2.
    assert same_day not in chosen["expiry_date"].to_list()
    nothing = causal_nearest_expiry(first_seen.filter(pl.col("expiry_date") == near), snapshots.head(1), INTRADAY_TARGET, GEX)
    assert nothing["expiry_date"].to_list() == [None]


def test_intraday_expiry_rule_needs_its_own_entry() -> None:
    frame = pl.DataFrame({"trade_date": [DAY, DAY], "expiry_date": [DAY, date(2025, 3, 6)]})
    assert frame.filter(total_gex_expiries(INTRADAY_TARGET, GEX))["expiry_date"].to_list() == [date(2025, 3, 6)]
    with pytest.raises(ValueError):
        total_gex_expiries(INTRADAY_TARGET, {"max_days_to_expiry": 21, "same_day_expiry_in_total_gex": {"t2": False}})


def test_every_intraday_rv_feature_has_a_lineage_entry() -> None:
    from test_intraday_targets import bars, frames  # pytest puts tests/ on the path

    futures, dates = bars(days=3)
    features, _ = frames(futures, dates)
    columns = [c for c in features.columns if c.startswith("rv_") and c != "rv_windows_available_at"]
    assert sorted(set(columns) - set(INTRADAY_LINEAGE)) == []
