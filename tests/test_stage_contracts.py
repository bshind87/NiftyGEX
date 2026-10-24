"""Invariants on generated stage outputs. Skipped when the runtime data has not been built."""
from __future__ import annotations

import json
from datetime import time

import polars as pl
import pytest

from niftygex.clean.session import session_settings
from niftygex.config import config_hash, load_config
from niftygex.io.availability import observation_close

CONFIG = load_config()
CURATED = CONFIG["paths"]["curated_root"]
GREEKS = CONFIG["paths"]["greeks_root"]
SESSION = session_settings(CONFIG["data"])
OPEN_MIN = SESSION["open"].hour * 60 + SESSION["open"].minute
CLOSE_MIN = SESSION["close"].hour * 60 + SESSION["close"].minute

needs_clean = pytest.mark.skipif(not (CURATED / "calendar" / "trading_calendar.parquet").exists(), reason="clean stage not run")
needs_forward = pytest.mark.skipif(not (GREEKS / "iv").exists(), reason="forward stage not run")

INTRADAY = {
    "options": CURATED / "options" / "*.parquet",
    "futures": CURATED / "futures" / "nifty_front_month.parquet",
    "spot": CURATED / "market" / "nifty_spot.parquet",
    "vix_intraday": CURATED / "vix" / "intraday.parquet",
}
DAILY = {
    "option_eod_oi": CURATED / "oi" / "option_eod_oi_1525.parquet",
    "participant_oi": CURATED / "oi" / "participant_oi.parquet",
    "vix_daily": CURATED / "vix" / "daily.parquet",
}


def minute(column: str = "datetime") -> pl.Expr:
    return pl.col(column).dt.hour().cast(pl.Int32) * 60 + pl.col(column).dt.minute().cast(pl.Int32)


@needs_clean
@pytest.mark.parametrize("name", list(INTRADAY))
def test_intraday_sources_are_ist_session_bounded_and_carry_bar_availability(name: str) -> None:
    frame = pl.scan_parquet(INTRADAY[name])
    assert frame.collect_schema()["datetime"] == pl.Datetime("us", "Asia/Kolkata")
    stats = frame.select(
        (~minute().is_between(OPEN_MIN, CLOSE_MIN)).sum().alias("outside"),
        (pl.col("available_at") != pl.col("datetime") + pl.duration(minutes=SESSION["bar_minutes"])).sum().alias("bad_available"),
        pl.col("data_quality").is_null().sum().alias("no_quality"),
    ).collect()
    assert stats.row(0) == (0, 0, 0)


@needs_clean
def test_option_keys_unique_across_files() -> None:
    duplicates = (
        pl.scan_parquet(INTRADAY["options"]).group_by(["datetime", "expiry_date", "strike_price", "right"]).len()
        .filter(pl.col("len") > 1).select(pl.len()).collect().item()
    )
    assert duplicates == 0


@needs_clean
def test_futures_at_most_one_contract_per_day() -> None:
    per_day = pl.read_parquet(INTRADAY["futures"]).group_by("trade_date").agg(pl.col("expiry_date").n_unique())
    assert per_day["expiry_date"].max() == 1


@needs_clean
@pytest.mark.parametrize("name", list(DAILY))
def test_daily_sources_available_strictly_after_observation_close(name: str) -> None:
    calendar = set(pl.read_parquet(CURATED / "calendar" / "trading_calendar.parquet")["trading_date"].to_list())
    frame = pl.read_parquet(DAILY[name]).with_columns(
        observation_close(pl.col("observation_date"), SESSION["end"]).alias("observation_close")
    )
    assert frame.filter(pl.col("available_at").is_null() | (pl.col("available_at") <= pl.col("observation_close"))).height == 0
    assert frame.filter(pl.col("available_date") <= pl.col("observation_date")).height == 0
    assert set(frame["available_date"].unique().to_list()) <= calendar
    open_time = time.fromisoformat(CONFIG["data"]["availability"]["next_trading_day_time"])
    assert set(frame["available_at"].dt.time().unique().to_list()) == {open_time}


@needs_clean
def test_calendar_last_observed_day_has_next_trading_day() -> None:
    calendar = pl.read_parquet(CURATED / "calendar" / "trading_calendar.parquet")
    assert calendar.filter(pl.col("observed"))["next_trading_date"].null_count() == 0


@needs_clean
def test_missing_spot_days_are_flagged_on_option_rows() -> None:
    calendar = pl.read_parquet(CURATED / "calendar" / "trading_calendar.parquet")
    missing = calendar.filter(pl.col("missing_spot") == True)["trading_date"].to_list()  # noqa: E712
    spot_days = set(pl.read_parquet(INTRADAY["spot"])["trade_date"].unique().to_list())
    assert not set(missing) & spot_days
    flagged = (
        pl.scan_parquet(INTRADAY["options"]).filter(pl.col("trade_date").is_in(missing))
        .select((~pl.col("data_quality").str.contains("missing_spot")).sum()).collect().item()
    )
    assert flagged == 0


@needs_clean
def test_no_spot_bar_beyond_outlier_threshold_remains() -> None:
    threshold = float(CONFIG["data"]["spot_outliers"]["max_relative_deviation_from_future"])
    spot = pl.read_parquet(INTRADAY["spot"]).select("datetime", "close")
    futures = pl.read_parquet(INTRADAY["futures"]).select("datetime", pl.col("close").alias("future"))
    joined = spot.join(futures, on="datetime").drop_nulls()
    assert joined.filter((pl.col("close") / pl.col("future") - 1).abs() > threshold).height == 0


@needs_forward
def test_iv_rows_inside_configured_band_with_bar_availability() -> None:
    band = CONFIG["data"]["liquidity"]
    stats = pl.scan_parquet(GREEKS / "iv" / "**" / "*.parquet").select(
        pl.col("moneyness").min().alias("lo"), pl.col("moneyness").max().alias("hi"),
        (pl.col("available_at") != pl.col("datetime") + pl.duration(minutes=SESSION["bar_minutes"])).sum().alias("bad"),
        (pl.col("eligible") & (pl.col("volume") <= 0)).sum().alias("eligible_without_volume"),
    ).collect().row(0, named=True)
    assert stats["lo"] >= band["moneyness_min"] and stats["hi"] <= band["moneyness_max"]
    assert stats["bad"] == 0 and stats["eligible_without_volume"] == 0


@needs_forward
@pytest.mark.parametrize("stage", ["clean", "basis", "forward", "greeks", "gex", "features", "tokens", "targets"])
def test_stage_manifest_matches_current_config(stage: str) -> None:
    manifest = json.loads((CONFIG["paths"]["manifests_root"] / f"{stage}.json").read_text())
    assert manifest["config_hash"] == config_hash(CONFIG)
    assert manifest["inputs"] and manifest["outputs"]


@needs_forward
def test_iv_is_otm_sourced_and_eligible_rows_are_greeks_ready() -> None:
    iv = pl.scan_parquet(GREEKS / "iv" / "**" / "*.parquet")
    stats = iv.select(
        (pl.col("eligible") & ~(pl.col("time_to_expiry") > 0)).sum().alias("eligible_nonpositive_time"),
        (pl.col("eligible") & (pl.col("iv_status") != "ok")).sum().alias("eligible_bad_status"),
        (pl.col("iv_status") == "below_intrinsic").sum().alias("otm_below_intrinsic"),
        (
            (pl.col("iv_source_right") == "Call") & (pl.col("strike_price") < pl.col("forward"))
            | (pl.col("iv_source_right") == "Put") & (pl.col("strike_price") >= pl.col("forward"))
        ).sum().alias("wrong_source_side"),
    ).collect().row(0, named=True)
    assert stats == {"eligible_nonpositive_time": 0, "eligible_bad_status": 0, "otm_below_intrinsic": 0, "wrong_source_side": 0}
    spread = (
        iv.filter(pl.col("iv").is_not_null()).group_by(["datetime", "expiry_date", "strike_price"])
        .agg(pl.col("iv").n_unique().alias("n")).filter(pl.col("n") > 1).select(pl.len()).collect().item()
    )
    assert spread == 0


EXPOSURE = CONFIG["paths"]["exposure_root"]
needs_gex = pytest.mark.skipif(not (EXPOSURE / "gex_totals" / "gex_totals.parquet").exists(), reason="gex stage not run")


@needs_clean
def test_missing_option_oi_days_have_null_not_zero_oi() -> None:
    stats = pl.scan_parquet(INTRADAY["options"]).select(
        (pl.col("missing_option_oi") & pl.col("open_interest").is_not_null()).sum().alias("oi_on_missing_days"),
        pl.col("missing_option_oi").sum().alias("rows_on_missing_days"),
    ).collect().row(0, named=True)
    assert stats["oi_on_missing_days"] == 0 and stats["rows_on_missing_days"] > 0


@needs_gex
def test_gex_totals_missing_when_inputs_missing_and_selection_rules_hold() -> None:
    totals = pl.read_parquet(EXPOSURE / "gex_totals" / "gex_totals.parquet")
    bad = totals.filter((~pl.col("spot_available") | ~pl.col("oi_available")) & pl.col("gex_absolute").is_not_null())
    assert bad.height == 0
    assert totals.filter(~pl.col("inferred_available") & pl.col("gex_inferred").is_not_null()).height == 0
    assert totals.filter((pl.col("target") == "t2") & pl.col("includes_same_day_expiry").fill_null(False)).height == 0
    assert totals.filter(pl.col("max_days_to_expiry") > CONFIG["data"]["gex"]["max_days_to_expiry"]).height == 0


@needs_gex
def test_gex_strike_rows_available_by_snapshot_and_positive_time() -> None:
    strikes = pl.scan_parquet(EXPOSURE / "gex_strikes" / "*.parquet").select(
        (pl.col("available_at") > pl.col("snapshot_at")).sum().alias("late"),
        (pl.col("time_to_expiry") <= 0).sum().alias("nonpositive_time"),
        (pl.col("spot_available_at") > pl.col("snapshot_at")).sum().alias("late_spot"),
        (pl.col("participant_available_at") > pl.col("snapshot_at")).sum().alias("late_participant"),
    ).collect().row(0, named=True)
    assert strikes == {"late": 0, "nonpositive_time": 0, "late_spot": 0, "late_participant": 0}


@needs_clean
def test_configured_lot_sizes_match_oi_divisibility() -> None:
    from niftygex.validation.checks import lot_size_check

    result = lot_size_check(pl.scan_parquet(INTRADAY["options"]), CONFIG["regimes"])
    judged = result.filter(pl.col("positive_oi_rows").is_not_null())
    assert judged.height > 300
    assert judged.filter(~pl.col("passed")).height == 0, judged.filter(~pl.col("passed")).to_dicts()


@needs_forward
def test_manifests_record_git_commit() -> None:
    for stage in ("clean", "forward", "greeks", "gex", "features", "tokens", "targets"):
        manifest = json.loads((CONFIG["paths"]["manifests_root"] / f"{stage}.json").read_text())
        assert "git_commit" in manifest and "git_dirty" in manifest, stage


FEATURES = CONFIG["paths"]["features_root"]
needs_tokens = pytest.mark.skipif(not (FEATURES / "tokens").exists(), reason="tokens stage not run")


@needs_gex
@needs_tokens
def test_prediction_snapshot_tokens_match_the_gex_chain() -> None:
    """At lag 0 the token chain is the GEX stage's chain for the day's nearest included expiry."""
    tokens = pl.scan_parquet(FEATURES / "tokens" / "*.parquet").filter(pl.col("lag") == 0).select(
        "snapshot_at", "expiry_date", "strike_price", "right", "strike_gex_absolute_crore").collect()
    chain = pl.scan_parquet(EXPOSURE / "gex_strikes" / "*.parquet").join(
        tokens.select("snapshot_at", "expiry_date").unique().lazy(), on=["snapshot_at", "expiry_date"]).select(
        "snapshot_at", "expiry_date", "strike_price", "right", (pl.col("gex_absolute") / 1e7).alias("chain_gex")).collect()
    joined = tokens.join(chain, on=["snapshot_at", "expiry_date", "strike_price", "right"], how="full", coalesce=True)
    assert joined.height == tokens.height == chain.height
    differs = joined.filter(~pl.col("strike_gex_absolute_crore").eq_missing(pl.col("chain_gex")) &
                            ((pl.col("strike_gex_absolute_crore") - pl.col("chain_gex")).abs() > 1e-9))
    assert differs.height == 0


@needs_tokens
def test_token_sequences_are_ordered_and_one_expiry_per_day() -> None:
    tokens = pl.scan_parquet(FEATURES / "tokens" / "*.parquet")
    stats = tokens.select(
        (pl.col("snapshot_at") > pl.col("prediction_at")).sum().alias("after_prediction"),
        (pl.col("available_at") > pl.col("snapshot_at")).sum().alias("late"),
        (pl.col("lag") < 0).sum().alias("negative_lag"),
    ).collect().row(0, named=True)
    assert stats == {"after_prediction": 0, "late": 0, "negative_lag": 0}
    per_day = tokens.group_by("trade_date", "target").agg(pl.col("expiry_date").n_unique().alias("n")).collect()
    assert per_day["n"].max() == 1
