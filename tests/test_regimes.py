from datetime import date

import polars as pl
import pytest

from niftygex.config import load_config
from niftygex.io.regimes import lot_sizes_by_expiry, regime_marker_flags

REGIMES = load_config()["regimes"]


def test_lot_schedule_by_expiry_with_exceptions() -> None:
    table = lot_sizes_by_expiry(REGIMES, [date(2019, 2, 14), date(2021, 7, 29), date(2024, 5, 2), date(2025, 1, 2), date(2025, 1, 30), date(2026, 1, 6)])
    assert table["units_per_lot"].to_list() == [75, 50, 25, 75, 25, 65]
    assert table.filter(pl.col("expiry_date") == date(2025, 1, 30))["lot_exception"].item()


def test_expiry_before_schedule_is_rejected() -> None:
    with pytest.raises(ValueError):
        lot_sizes_by_expiry(REGIMES, [date(2018, 12, 27)])


def test_regime_marker_flag() -> None:
    frame = pl.DataFrame({"trade_date": [date(2024, 11, 19), date(2024, 11, 20)]}).with_columns(regime_marker_flags(REGIMES))
    assert frame["regime_post_weekly_expiry_rationalisation"].to_list() == [False, True]
