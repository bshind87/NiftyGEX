from datetime import date

import polars as pl
import pytest

from niftygex.config import load_config
from niftygex.exposure.selection import greeks_input, total_gex_expiries

GEX = load_config()["data"]["gex"]


def chain() -> pl.DataFrame:
    trade = date(2024, 2, 29)  # expiry day of the 2024-02-29 contract
    return pl.DataFrame({
        "trade_date": [trade] * 4,
        "expiry_date": [date(2024, 2, 29), date(2024, 3, 7), date(2024, 3, 14), date(2024, 4, 25)],
    })


def test_t2_excludes_same_day_expiry_from_total_gex() -> None:
    kept = chain().filter(total_gex_expiries("t2", GEX))["expiry_date"].to_list()
    assert kept == [date(2024, 3, 7), date(2024, 3, 14)]


def test_same_day_expiry_kept_when_configured() -> None:
    keep = {**GEX, "same_day_expiry_in_total_gex": {"t2": True}}
    kept = chain().filter(total_gex_expiries("t2", keep))["expiry_date"].to_list()
    assert kept == [date(2024, 2, 29), date(2024, 3, 7), date(2024, 3, 14)]


def test_horizon_excludes_expiries_beyond_max_days() -> None:
    assert date(2024, 4, 25) not in chain().filter(total_gex_expiries("t2", GEX))["expiry_date"].to_list()


def test_unknown_target_is_rejected() -> None:
    with pytest.raises(ValueError):
        total_gex_expiries("t1", GEX)


def test_greeks_exclude_nonpositive_time_to_expiry() -> None:
    rows = pl.DataFrame({"eligible": [True, True, True, False], "time_to_expiry": [0.001, 0.0, -0.001, 0.01]})
    assert rows.filter(greeks_input()).height == 1
