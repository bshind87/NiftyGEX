from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from niftygex.io.availability import add_next_day_availability, observation_close
from niftygex.io.calendar import build_trading_calendar, next_trading_dates
from niftygex.validation.leakage import LeakageError, assert_available_after

IST = ZoneInfo("Asia/Kolkata")
OPEN = time(9, 15)
SESSION_END = time(15, 30)


def calendar_dates() -> list[date]:
    # Fri 2024-09-13, then 09-16..09-18 are option-only days (no spot), 09-19 normal.
    observed = [date(2024, 9, 13), date(2024, 9, 16), date(2024, 9, 17), date(2024, 9, 18), date(2024, 9, 19)]
    return build_trading_calendar(observed, holidays=[date(2024, 9, 20)], extend_weekdays=3)["trading_date"].to_list()


def test_calendar_extends_past_last_date_by_weekday_rule_skipping_holidays() -> None:
    dates = calendar_dates()
    # 09-20 is a configured holiday, 09-21/22 are a weekend.
    assert dates[-3:] == [date(2024, 9, 23), date(2024, 9, 24), date(2024, 9, 25)]


def test_calendar_removes_configured_holidays_from_observed_dates() -> None:
    frame = build_trading_calendar([date(2024, 9, 13), date(2024, 9, 16)], holidays=[date(2024, 9, 16)], extend_weekdays=1)
    assert date(2024, 9, 16) not in frame["trading_date"].to_list()


def test_next_trading_date_is_strictly_after_observation() -> None:
    dates = calendar_dates()
    mapping = next_trading_dates([date(2024, 9, 13), date(2024, 9, 19), date(2024, 9, 14)], dates)
    assert mapping[date(2024, 9, 13)] == date(2024, 9, 16)
    assert mapping[date(2024, 9, 19)] == date(2024, 9, 23)
    # A non-trading observation date (Saturday) still maps strictly forward.
    assert mapping[date(2024, 9, 14)] == date(2024, 9, 16)


def test_next_trading_date_raises_instead_of_falling_back() -> None:
    with pytest.raises(ValueError):
        next_trading_dates([date(2024, 9, 19)], [date(2024, 9, 18), date(2024, 9, 19)])


def test_daily_availability_is_next_trading_day_open_for_every_row() -> None:
    dates = calendar_dates()
    frame = pl.DataFrame({"observation_date": [date(2024, 9, 13), date(2024, 9, 16), date(2024, 9, 19)]})
    result = add_next_day_availability(frame, "observation_date", dates, OPEN)
    assert result["available_at"].to_list() == [
        datetime(2024, 9, 16, 9, 15, tzinfo=IST),
        datetime(2024, 9, 17, 9, 15, tzinfo=IST),
        datetime(2024, 9, 23, 9, 15, tzinfo=IST),
    ]
    checked = result.with_columns(observation_close(pl.col("observation_date"), SESSION_END).alias("close"))
    assert_available_after(checked, "close")


def test_availability_check_fails_when_available_on_observation_day() -> None:
    frame = pl.DataFrame({"observation_date": [date(2024, 9, 13)]}).with_columns(
        observation_close(pl.col("observation_date"), SESSION_END).alias("close"),
        pl.lit(datetime(2024, 9, 13, 15, 30, tzinfo=IST)).alias("available_at"),
    )
    with pytest.raises(LeakageError):
        assert_available_after(frame, "close")


def test_availability_check_fails_on_missing_availability() -> None:
    frame = pl.DataFrame({"observation_date": [date(2024, 9, 13)]}).with_columns(
        observation_close(pl.col("observation_date"), SESSION_END).alias("close"),
        pl.lit(None, dtype=pl.Datetime("us", "Asia/Kolkata")).alias("available_at"),
    )
    with pytest.raises(LeakageError):
        assert_available_after(frame, "close")
