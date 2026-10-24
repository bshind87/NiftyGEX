"""T1 pinning targets on fixtures (official closes are not in the data yet)."""
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from niftygex.io.official_closes import load_official_closes
from niftygex.targets.pinning import strike_steps, t1_targets

IST = ZoneInfo("Asia/Kolkata")
EXPIRY = date(2024, 2, 29)


def write(tmp_path, name: str, text: str):
    directory = tmp_path / "official_closes"
    directory.mkdir(exist_ok=True)
    (directory / name).write_text(text)
    return directory


def snapshots(day: date = EXPIRY) -> pl.DataFrame:
    times = [datetime(day.year, day.month, day.day, h, 0, tzinfo=IST) for h in (10, 12, 14)]
    return pl.DataFrame({"snapshot_at": times, "trade_date": [day] * 3, "spot": [22010.0, 22030.0, 21995.0],
                         "max_gex_strike": [22000.0, 22000.0, None], "max_pain": [22050.0] * 3},
                        schema_overrides={"max_gex_strike": pl.Float64})


def steps(day: date = EXPIRY) -> pl.DataFrame:
    times = [datetime(day.year, day.month, day.day, h, 0, tzinfo=IST) for h in (10, 12, 14)]
    chain = pl.DataFrame({"snapshot_at": times * 3, "trade_date": [day] * 9,
                          "expiry_date": [day] * 6 + [date(2024, 3, 7)] * 3, "strike_price": [21950.0] * 3 + [22000.0] * 3 + [22025.0] * 3})
    return strike_steps(chain)


def test_loader_reads_validates_and_collapses_repeats(tmp_path) -> None:
    directory = write(tmp_path, "a.csv", "date,close\n2024-02-29,22005.5\n2024-03-07,22300\n")
    write(tmp_path, "b.csv", "date,close\n2024-02-29,22005.5\n")
    closes = load_official_closes(directory)
    assert closes.select("date", "official_close").rows() == [(date(2024, 2, 29), 22005.5), (date(2024, 3, 7), 22300.0)]
    assert load_official_closes(tmp_path / "absent") is None


@pytest.mark.parametrize("text", ["day,close\n2024-02-29,1\n", "date,close\n2024-02-29,-5\n", "date,close\n29/02/2024,1\n"])
def test_loader_rejects_bad_files(tmp_path, text: str) -> None:
    with pytest.raises(Exception):
        load_official_closes(write(tmp_path, "bad.csv", text))


def test_loader_rejects_conflicting_duplicates(tmp_path) -> None:
    directory = write(tmp_path, "a.csv", "date,close\n2024-02-29,22005\n")
    write(tmp_path, "b.csv", "date,close\n2024-02-29,22006\n")
    with pytest.raises(ValueError):
        load_official_closes(directory)


def test_strike_step_uses_expiring_contract_only() -> None:
    assert set(steps()["strike_step"].to_list()) == {50.0}


def test_pinned_settlement_and_uniform_null_scale() -> None:
    closes = pl.DataFrame({"date": [EXPIRY], "official_close": [22002.0]})
    result = t1_targets(snapshots(), closes, steps(), pin_fraction=0.1, default_step=50, available_time=time(16, 0)).sort("snapshot_at")
    first = result.row(0, named=True)
    assert first["t1_nearest_strike"] == 22000.0 and first["t1_distance_points"] == 2.0
    assert first["t1_distance_fraction"] == pytest.approx(2.0 / 25.0) and first["t1_pinned"] is True
    assert first["t1_pinned_to_max_gex"] is True and first["t1_pinned_to_max_pain"] is False
    assert first["t1_null_pin_rate"] == pytest.approx(0.2) and first["t1_reason"] == "ok"
    # The 14:00 snapshot has no max-GEX strike: grid labels stay, level labels are missing.
    last = result.row(2, named=True)
    assert last["t1_reason"] == "no_max_gex_strike" and last["t1_pinned"] is True and last["t1_pinned_to_max_gex"] is None
    assert (result["target_available_at"] > result["snapshot_at"]).all()


def test_mid_step_settlement_is_maximally_unpinned() -> None:
    closes = pl.DataFrame({"date": [EXPIRY], "official_close": [22025.0]})
    result = t1_targets(snapshots(), closes, steps(), pin_fraction=0.1, default_step=50, available_time=time(16, 0))
    assert result["t1_distance_fraction"].to_list()[0] == pytest.approx(1.0) and result["t1_pinned"].to_list()[0] is False


def test_missing_source_and_missing_date_never_substitute() -> None:
    absent = t1_targets(snapshots(), None, steps(), pin_fraction=0.1, default_step=50, available_time=time(16, 0))
    assert set(absent["t1_reason"]) == {"no_official_close_source"} and absent["t1_pinned"].null_count() == 3
    other_day = pl.DataFrame({"date": [date(2024, 3, 7)], "official_close": [22000.0]})
    missing = t1_targets(snapshots(), other_day, steps(), pin_fraction=0.1, default_step=50, available_time=time(16, 0))
    assert set(missing["t1_reason"]) == {"no_official_close_for_date"} and missing["t1_distance_fraction"].null_count() == 3
