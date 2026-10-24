# Archived 2026-10-14 from tests/test_targets.py (T3 targets were removed from the data layer).
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl

IST = ZoneInfo("Asia/Kolkata")


def bars_at(day: date, hour: int, minute: int, closes: list[float]) -> pl.DataFrame:
    start = datetime(day.year, day.month, day.day, hour, minute, tzinfo=IST)
    times = [start + timedelta(minutes=5 * i) for i in range(len(closes))]
    return pl.DataFrame({"datetime": times, "trade_date": [day] * len(closes), "close": closes, "low": closes, "high": closes,
                         "available_at": [t + timedelta(minutes=5) for t in times]})


def test_t3_label_toward_strike_and_exclusions() -> None:
    from datetime import time

    from niftygex.targets.intraday_drift import t3_targets

    day, other, truncated = date(2024, 1, 2), date(2024, 1, 3), date(2020, 3, 13)
    spot = pl.concat([bars_at(d, 14, 0, [100.0 + i for i in range(18)]) for d in (day, other, truncated)])
    snaps = pl.DataFrame({
        "snapshot_at": [datetime(d.year, d.month, d.day, 14, 0, tzinfo=IST) for d in (day, other, truncated)],
        "trade_date": [day, other, truncated], "spot": [100.0, 100.0, 100.0], "max_gex_strike": [120.0, 90.0, 120.0],
        "truncated_session": [False, False, True], "missing_spot": [False, False, False],
    })
    result = t3_targets(snaps, spot, path_start=time(14, 0), path_end_label=time(15, 25), min_path_bars=16).sort("trade_date")
    # Path rises from 100 to 117: toward 120, away from 90; the truncated session is excluded.
    assert result["t3_reason"].to_list() == ["truncated_session", "ok", "ok"]
    assert result["t3_label"].to_list() == [None, True, False]
    assert (result["target_available_at"].drop_nulls() > result["snapshot_at"].filter(result["target_available_at"].is_not_null())).all()
