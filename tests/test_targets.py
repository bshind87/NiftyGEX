from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest

from niftygex.targets.vol_regime import causal_terciles, daily_realized_vol, trailing_mean

IST = ZoneInfo("Asia/Kolkata")


def bars(day: date, closes: list[float]) -> pl.DataFrame:
    start = datetime(day.year, day.month, day.day, 9, 15, tzinfo=IST)
    times = [start + timedelta(minutes=5 * i) for i in range(len(closes))]
    return pl.DataFrame({"datetime": times, "trade_date": [day] * len(closes), "close": closes,
                         "available_at": [t + timedelta(minutes=5) for t in times]})


def test_daily_rv_within_day_returns_and_min_bars() -> None:
    frame = pl.concat([bars(date(2024, 1, 2), [100.0, 101.0, 100.0]), bars(date(2024, 1, 3), [200.0, 200.0])])
    rv = daily_realized_vol(frame, min_bars=3, trading_days_per_year=252, returns_per_day=74).sort("trade_date")
    expected = np.sqrt(np.log(101 / 100) ** 2 + np.log(100 / 101) ** 2)
    assert rv["rv"][0] == pytest.approx(expected)
    # The overnight 100 -> 200 jump is not a within-day return; the 2-bar day is below min_bars.
    assert rv["rv"][1] is None


def test_trailing_mean_requires_min_days() -> None:
    days = [date(2024, 1, d) for d in (2, 3, 4)]
    daily = pl.DataFrame({"trade_date": days, "rv": [1.0, None, 3.0]})
    result = trailing_mean(daily, days, window=2, min_days=2)
    assert result["rv_trailing_mean"].to_list() == [None, None, None]
    result = trailing_mean(daily, days, window=3, min_days=2)
    assert result["rv_trailing_mean"].to_list()[2] == pytest.approx(2.0)


def test_causal_terciles_use_only_targets_resolved_before_the_snapshot_day() -> None:
    days = [date(2024, 1, d) for d in range(1, 11)]
    targets = days[1:] + [None]
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0, 0.5, 7.0, 8.0]
    labels, lows, highs = causal_terciles(days, targets, values, min_history=3)
    # Day 4 sees only values whose target day is before it: days 1-2's targets (days 2, 3) -> 2 values < 3.
    assert labels[:4] == [None, None, None, None]
    # Day 5's history is values of days 1-3 (targets 2-4), so cutoffs come from [1, 2, 3].
    assert lows[4] == pytest.approx(np.quantile([1, 2, 3], 1 / 3)) and highs[4] == pytest.approx(np.quantile([1, 2, 3], 2 / 3))
    assert labels[4] == "high"
    # A future outlier (day 7's value 100) never moves earlier cutoffs.
    assert highs[6] == pytest.approx(np.quantile([1, 2, 3, 4, 5], 2 / 3))
