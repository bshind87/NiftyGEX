"""Exploratory E3 intraday RV target and features: window boundaries, causality and past-only normals."""
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest

from niftygex.targets.intraday_rv import intraday_rv_frames, past_mean, return_matrix, window_rv

IST = ZoneInfo("Asia/Kolkata")
FIRST, LAST = 9 * 60 + 20, 15 * 60 + 25       # first return label, last bar label
SNAPSHOTS = [10 * 60 + 15, 11 * 60, 13 * 60 + 15]


def bars(days: int = 70, seed: int = 0) -> tuple[pl.DataFrame, list[date]]:
    rng = np.random.default_rng(seed)
    dates = [d for d in (date(2024, 1, 1) + timedelta(days=i) for i in range(days * 2)) if d.weekday() < 5][:days]
    rows = []
    for day in dates:
        price = 20000.0
        for k in range(75):
            label = datetime(day.year, day.month, day.day, 9, 15, tzinfo=IST) + timedelta(minutes=5 * k)
            price *= float(np.exp(rng.normal(0, 0.001)))
            rows.append((label, day, label + timedelta(minutes=5), price, date(2024, 12, 26)))
    frame = pl.DataFrame(rows, schema={"datetime": pl.Datetime("us", "Asia/Kolkata"), "trade_date": pl.Date,
                                       "available_at": pl.Datetime("us", "Asia/Kolkata"), "close": pl.Float64, "expiry_date": pl.Date}, orient="row")
    return frame, dates


def frames(futures, dates):
    return intraday_rv_frames(futures, dates, SNAPSHOTS, [1, 2], [30, 60, 120], FIRST, LAST, 60, 45, full_day_min_returns=69)


def bump(futures: pl.DataFrame, day: date, hour: int, minute: int, factor: float = 1.05) -> pl.DataFrame:
    target = datetime(day.year, day.month, day.day, hour, minute, tzinfo=IST)
    return futures.with_columns(pl.when(pl.col("datetime") == target).then(pl.col("close") * factor).otherwise(pl.col("close")).alias("close"))


def row(frame: pl.DataFrame, day: date, hour: int, minute: int) -> dict:
    return frame.filter(pl.col("snapshot_at") == datetime(day.year, day.month, day.day, hour, minute, tzinfo=IST)).row(0, named=True)


def test_window_rv_uses_exactly_the_returns_inside_the_clock_window() -> None:
    futures, dates = bars(days=2)
    returns = return_matrix(futures, dates, FIRST, LAST)
    assert returns.shape == (2, (LAST - FIRST) // 5 + 1)
    start, end = 11 * 60, 12 * 60
    expected = np.sqrt(np.sum(returns[0, (start - FIRST) // 5:(end - 5 - FIRST) // 5 + 1] ** 2))
    assert window_rv(returns, FIRST, start, end)[0] == pytest.approx(expected)
    assert (end - start) // 5 == 12


def test_target_window_starts_after_the_snapshot_and_ends_at_t_plus_h() -> None:
    futures, dates = bars()
    day = dates[-1]
    _, base = frames(futures, dates)
    reference = row(base, day, 11, 0)
    assert reference["target_available_at_1h"] == datetime(day.year, day.month, day.day, 12, 0, tzinfo=IST)
    # The 10:50 bar closes at 10:55, before the window's base price (the 10:55 bar's close at 11:00): no effect.
    assert row(frames(bump(futures, day, 10, 50), dates)[1], day, 11, 0)["rv_fwd_1h"] == pytest.approx(reference["rv_fwd_1h"])
    # The 11:00 bar's close (known 11:05) is the window's first return: it changes the target.
    assert row(frames(bump(futures, day, 11, 0), dates)[1], day, 11, 0)["rv_fwd_1h"] != pytest.approx(reference["rv_fwd_1h"])
    # The 12:00 bar closes at 12:05, after the 1-hour window: no effect.
    assert row(frames(bump(futures, day, 12, 0), dates)[1], day, 11, 0)["rv_fwd_1h"] == pytest.approx(reference["rv_fwd_1h"])
    assert base.filter(pl.col("target_available_at") <= pl.col("snapshot_at")).height == 0


def test_features_use_only_returns_known_by_the_snapshot() -> None:
    futures, dates = bars()
    day = dates[-1]
    base, _ = frames(futures, dates)
    reference = row(base, day, 11, 0)
    after = row(frames(bump(futures, day, 11, 0), dates)[0], day, 11, 0)   # the 11:00 bar closes at 11:05
    for column in ("rv_last_30m", "rv_last_60m", "rv_last_120m", "rv_today_so_far", "rv_normal_1h", "rv_back_normal_60m"):
        assert after[column] == pytest.approx(reference[column]), column
    # The 10:55 bar closes at 11:00, the last return the snapshot can see.
    assert row(frames(bump(futures, day, 10, 55), dates)[0], day, 11, 0)["rv_last_30m"] != pytest.approx(reference["rv_last_30m"])


def test_normals_average_past_days_only() -> None:
    values = np.arange(10, dtype=float)
    assert past_mean(values, 3, 2)[5] == pytest.approx(np.mean([2.0, 3.0, 4.0]))   # days 2-4, not day 5
    assert np.isnan(past_mean(values, 3, 2)[1])
    futures, dates = bars()
    day = dates[-1]
    base, _ = frames(futures, dates)
    today = row(frames(bump(futures, day, 11, 30), dates)[0], day, 11, 0)     # today's window does not enter its normal
    assert today["rv_normal_1h"] == pytest.approx(row(base, day, 11, 0)["rv_normal_1h"])
    earlier = row(frames(bump(futures, dates[-2], 11, 30), dates)[0], day, 11, 0)
    assert earlier["rv_normal_1h"] != pytest.approx(row(base, day, 11, 0)["rv_normal_1h"])
    assert earlier["rv_yesterday_same_window_1h"] != pytest.approx(row(base, day, 11, 0)["rv_yesterday_same_window_1h"])


def test_a_missing_bar_skips_the_window_instead_of_imputing() -> None:
    futures, dates = bars()
    day = dates[-1]
    gap = futures.filter(pl.col("datetime") != datetime(day.year, day.month, day.day, 11, 30, tzinfo=IST))
    _, targets = frames(gap, dates)
    hit = row(targets, day, 11, 0)
    assert hit["rv_fwd_1h"] is None and hit["reason_1h"] == "incomplete_window"
    assert row(targets, day, 13, 15)["reason_2h"] == "ok"


def test_backward_windows_are_truncated_at_the_first_return() -> None:
    futures, dates = bars(days=2)
    features, _ = frames(futures, dates)
    early = row(features, dates[-1], 10, 15)
    assert early["rv_last_120m"] == pytest.approx(early["rv_today_so_far"])    # (09:20, 10:15]
