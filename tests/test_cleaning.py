from datetime import date, time

import numpy as np
import polars as pl

from niftygex.clean.basis import classify_day
from niftygex.clean.chain import clean_option_frame
from niftygex.clean.dedup import drop_conflicting_keys
from niftygex.clean.futures import clean_front_month
from niftygex.clean.quality import flag_spot_outliers
from niftygex.clean.session import expected_bars, filter_session, normalize_timestamp, session_end

WINDOW = dict(start=date(2024, 1, 1), end=date(2024, 12, 31), session_open=time(9, 15), session_close=time(15, 25), excluded_dates=[])


def option_rows(rows: list[dict]) -> pl.DataFrame:
    base = {"stock_code": "NIFTY", "expiry_date": "2024-02-29", "right": "Call", "strike_price": 22000,
            "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 10, "open_interest": 100, "date": "x"}
    return pl.DataFrame([{**base, **row} for row in rows])


def test_session_keeps_0915_through_1525_and_drops_1530() -> None:
    frame = normalize_timestamp(pl.DataFrame({"datetime": [
        "2024-02-27 09:10:00", "2024-02-27 09:15:00", "2024-02-27 15:25:00", "2024-02-27 15:30:00", "2024-02-27 15:35:00",
    ]}))
    kept = filter_session(frame, time(9, 15), time(15, 25))
    assert kept["datetime"].dt.strftime("%H:%M").to_list() == ["09:15", "15:25"]
    assert expected_bars(time(9, 15), time(15, 25), 5) == 75
    assert session_end(time(15, 25), 5) == time(15, 30)


def test_timestamps_are_ist_aware() -> None:
    frame = normalize_timestamp(pl.DataFrame({"datetime": ["2024-02-27 09:15:00"]}))
    assert frame.schema["datetime"] == pl.Datetime("us", "Asia/Kolkata")


def test_option_dedup_keeps_higher_volume_regardless_of_row_order() -> None:
    rows = [
        {"datetime": "2024-02-27 10:00:00", "volume": 50, "close": 2.0},
        {"datetime": "2024-02-27 10:00:00", "volume": 500, "close": 3.0},
    ]
    for ordered in (rows, rows[::-1]):
        cleaned, stats, dropped = clean_option_frame(option_rows(ordered), source_file="f.csv.gz", **WINDOW)
        assert cleaned["volume"].to_list() == [500.0]
        assert stats["conflicting_key_groups"] == 1
        assert dropped["kept_volume"].to_list() == [500.0]


def test_option_exact_duplicates_are_counted_separately() -> None:
    rows = [{"datetime": "2024-02-27 10:00:00"}, {"datetime": "2024-02-27 10:00:00"}]
    cleaned, stats, _ = clean_option_frame(option_rows(rows), source_file="f.csv.gz", **WINDOW)
    assert cleaned.height == 1
    assert stats["exact_duplicate_rows_removed"] == 1 and stats["conflicting_key_groups"] == 0


def futures_source(expiry: str, labels: list[str], volume: int = 10, close: float = 100.0) -> pl.DataFrame:
    return pl.DataFrame({
        "datetime": labels, "stock_code": "NIFTY", "expiry_date": expiry, "open": close, "high": close, "low": close,
        "close": close, "volume": volume, "open_interest": 1, "count": 0, "date": "x",
    })


def test_futures_dedup_prefers_later_source_file_over_volume() -> None:
    label = ["2024-02-27 10:00:00"]
    sources = [
        (futures_source("2024-02-29", label, volume=900, close=1.0), "NIFTY-fut-20240229.csv.gz", date(2024, 2, 29)),
        (futures_source("2024-02-29", label, volume=10, close=2.0), "NIFTY-fut-20240301.csv.gz", date(2024, 3, 1)),
    ]
    front, _, _, _ = clean_front_month(sources, roll_rule="nearest_expiry_not_expired", **WINDOW)
    assert front["close"].to_list() == [2.0]


def test_futures_roll_selects_one_contract_per_day_without_intraday_fallback() -> None:
    feb_labels = ["2024-02-27 10:30:00", "2024-02-27 10:55:00", "2024-02-29 15:25:00"]
    apr_labels = ["2024-02-27 10:30:00", "2024-02-27 10:50:00", "2024-02-29 15:25:00", "2024-03-01 09:15:00"]
    mar_labels = ["2024-02-29 15:25:00", "2024-03-01 09:15:00"]
    sources = [
        (futures_source("2024-02-29", feb_labels), "NIFTY-fut-20240229.csv.gz", date(2024, 2, 29)),
        (futures_source("2024-04-25", apr_labels), "NIFTY-fut-20240425.csv.gz", date(2024, 4, 25)),
        (futures_source("2024-03-28", mar_labels), "NIFTY-fut-20240328.csv.gz", date(2024, 3, 28)),
    ]
    front, stats, _, choice = clean_front_month(sources, roll_rule="nearest_expiry_not_expired", **WINDOW)
    per_day = front.group_by("trade_date").agg(pl.col("expiry_date").n_unique())
    assert per_day["expiry_date"].max() == 1
    by_day = dict(choice.iter_rows())
    # The expiring contract stays front month on its expiry day; the 10:50 April-only bar is not used.
    assert by_day[date(2024, 2, 27)] == date(2024, 2, 29)
    assert by_day[date(2024, 2, 29)] == date(2024, 2, 29)
    assert by_day[date(2024, 3, 1)] == date(2024, 3, 28)
    assert "2024-02-27 10:50" not in front["datetime"].dt.strftime("%Y-%m-%d %H:%M").to_list()
    assert stats["contract_changes"] == 1


def test_participant_dedup_is_deterministic_and_drops_conflicts() -> None:
    frame = pl.DataFrame({
        "observation_date": [date(2024, 1, 2)] * 3 + [date(2024, 1, 3)] * 2,
        "Client Type": ["FII", "FII", "Pro", "FII", "FII"],
        "value": [1.0, 1.0, 5.0, 2.0, 3.0],
        "source_row": [0, 1, 2, 3, 4],
    })
    for ordered in (frame, frame.reverse()):
        kept, stats, removed = drop_conflicting_keys(ordered, ["observation_date", "Client Type"])
        assert kept.select("observation_date", "Client Type", "value").rows() == [
            (date(2024, 1, 2), "FII", 1.0), (date(2024, 1, 2), "Pro", 5.0),
        ]
        assert stats["conflicting_key_groups"] == 1 and removed.height == 2


def test_spot_bad_print_is_nulled_and_logged() -> None:
    spot = normalize_timestamp(pl.DataFrame({
        "datetime": ["2021-10-18 12:50:00", "2021-10-18 12:55:00", "2021-10-18 13:00:00"],
        "open": [18400.0, 9588.0, 18410.0], "high": [18400.0, 9588.0, 18410.0],
        "low": [18400.0, 9588.0, 18410.0], "close": [18400.0, 9588.0, 18410.0],
    }))
    futures = spot.head(2).with_columns(
        pl.lit(date(2021, 10, 28)).alias("expiry_date"), *[pl.lit(18480.0).alias(c) for c in ("open", "high", "low", "close")]
    )
    cleaned, outliers = flag_spot_outliers(spot, futures, 0.03)
    assert cleaned["spot_quality"].to_list() == ["ok", "bad_print", "unchecked"]
    assert cleaned["close"].to_list()[1] is None
    assert outliers.height == 1


def test_spot_halt_reopen_bar_with_wide_open_is_kept() -> None:
    spot = normalize_timestamp(pl.DataFrame({
        "datetime": ["2020-03-13 10:20:00"], "open": [8555.0], "high": [9164.0], "low": [8555.0], "close": [9115.0],
    }))
    futures = spot.with_columns(
        pl.lit(date(2020, 3, 26)).alias("expiry_date"),
        pl.lit(8299.0).alias("open"), pl.lit(9355.0).alias("high"), pl.lit(8299.0).alias("low"), pl.lit(9006.0).alias("close"),
    )
    cleaned, outliers = flag_spot_outliers(spot, futures, 0.03)
    assert cleaned["spot_quality"].to_list() == ["ok"] and outliers.height == 0


def test_basis_classification_spike_event_and_unresolved() -> None:
    flat = [10.0] * 5
    spike = np.array(flat + [60.0, 61.0, 10.5] + flat)
    classes, bad = classify_day(spike, threshold=15.0, revert_max_bars=3, tolerance=0.5)
    assert list(classes).count("spike") == 1 and bad.sum() == 2 and bad[5] and bad[6]
    step = np.array(flat + [60.0] * 6)
    classes, bad = classify_day(step, threshold=15.0, revert_max_bars=3, tolerance=0.5)
    assert list(classes).count("event") == 1 and not bad.any()
    late = np.array(flat + [60.0])
    classes, _ = classify_day(late, threshold=15.0, revert_max_bars=3, tolerance=0.5)
    assert list(classes).count("unresolved") == 1


def test_end_labeled_window_is_relabeled_before_session_filter() -> None:
    from niftygex.clean.market import clean_market_bars

    raw = pl.DataFrame({
        "datetime": ["2021-06-01 09:15:00", "2021-06-01 09:20:00", "2021-06-01 15:30:00", "2019-06-03 15:30:00"],
        "stock_code": "NIFTY", "open": 1.0, "high": 1.0, "low": 1.0, "close": [1.0, 2.0, 3.0, 4.0], "volume": 0, "date": "x",
    })
    window = {**WINDOW, "start": date(2019, 1, 1), "end": date(2021, 12, 31)}
    cleaned, stats, _ = clean_market_bars(raw, end_labeled=[(date(2020, 12, 8), date(2026, 4, 23))], bar_minutes=5, **window)
    labels = cleaned["datetime"].dt.strftime("%Y-%m-%d %H:%M").to_list()
    # Inside the window: end-labeled 09:15 (pre-open) drops, 09:20 -> 09:15, 15:30 -> 15:25 (last regular bar).
    # Outside the window: a start-labeled 15:30 bar is still dropped.
    assert labels == ["2021-06-01 09:15", "2021-06-01 15:25"]
    assert cleaned["close"].to_list() == [2.0, 3.0]
    assert stats["rows_relabeled_end_to_start"] == 3


def test_day_quality_flags_missing_option_oi_from_traded_share() -> None:
    from niftygex.clean.quality import day_quality

    calendar = pl.DataFrame({
        "trading_date": [date(2022, 3, 1), date(2022, 3, 2)], "observed": [True, True],
        "next_trading_date": [date(2022, 3, 2), date(2022, 3, 3)],
    })
    counts = {name: pl.DataFrame({"trading_date": [date(2022, 3, 1), date(2022, 3, 2)], f"{name}_bars": [75, 75]})
              for name in ("options", "futures", "spot")}
    shares = pl.DataFrame({"trading_date": [date(2022, 3, 1), date(2022, 3, 2)], "option_oi_share": [0.0, 0.99]})
    days = day_quality(calendar, counts, expected_bars=75, truncated_dates=[], configured_dates=[],
                       option_oi_share=shares, min_option_oi_share=0.5)
    assert days["missing_option_oi"].to_list() == [True, False]
    assert days["data_quality"].to_list() == ["missing_option_oi", "ok"]
