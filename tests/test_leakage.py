from datetime import datetime
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from niftygex.clean.session import normalize_timestamp
from niftygex.io.availability import add_bar_availability
from niftygex.validation.leakage import LeakageError, assert_available_by, rows_available_at

IST = ZoneInfo("Asia/Kolkata")


def bars() -> pl.DataFrame:
    frame = pl.DataFrame({"datetime": ["2024-09-13 15:05:00", "2024-09-13 15:10:00", "2024-09-13 15:15:00"], "close": [1.0, 2.0, 3.0]})
    return add_bar_availability(normalize_timestamp(frame), 5)


def test_bar_is_available_one_bar_after_its_start_label() -> None:
    assert bars()["available_at"].to_list()[0] == datetime(2024, 9, 13, 15, 10, tzinfo=IST)


def test_snapshot_uses_available_at_not_label() -> None:
    usable = rows_available_at(bars(), datetime(2024, 9, 13, 15, 15, tzinfo=IST))
    # The 15:15-labeled bar covers [15:15, 15:20) and is not usable by a 15:15 snapshot.
    assert usable["close"].to_list() == [1.0, 2.0]


def test_future_dated_source_fails_ordering_assertion() -> None:
    frame = bars().with_columns(pl.lit(datetime(2024, 9, 13, 15, 15, tzinfo=IST)).alias("snapshot_at"))
    with pytest.raises(LeakageError):
        assert_available_by(frame, "snapshot_at")
    assert_available_by(frame.filter(pl.col("available_at") <= pl.col("snapshot_at")), "snapshot_at")
