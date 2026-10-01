"""Cleaning stage: raw sources -> curated Parquet, trading calendar, reports, and manifest."""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from niftygex.clean.chain import OPTION_KEYS, clean_option_file
from niftygex.clean.dedup import FUTURES_ORDER, MARKET_ORDER, OPTION_ORDER
from niftygex.clean.diagnostics import completeness, duplicate_summary, merge_label_counts, missing_labels
from niftygex.clean.futures import clean_front_month, futures_files
from niftygex.clean.market import clean_daily_vix, clean_market_bars, end_labeled_windows
from niftygex.clean.oi import clean_participant_oi, eod_oi_rows, finalize_eod_oi
from niftygex.clean.quality import attach_day_flags, bars_per_day, day_quality, flag_spot_outliers, option_oi_share
from niftygex.clean.session import (
    configured_quality_dates,
    holiday_dates,
    parse_time,
    session_settings,
    special_session_dates,
    truncated_session_dates,
)
from niftygex.config import load_config
from niftygex.io.availability import add_bar_availability, add_next_day_availability, observation_close
from niftygex.io.calendar import build_trading_calendar
from niftygex.io.parquet import describe_files, staged_outputs, write_manifest, write_parquet, write_report
from niftygex.validation.leakage import assert_available_after

STAGE = "clean"
OWNED = ["options", "futures", "market", "vix", "oi", "calendar"]
DROPPED_COLUMNS = ["datetime", "source_file", "close", "volume", "open_interest", "exact", "kept_source_file",
                   "kept_close", "kept_volume", "kept_open_interest", "source_file_date", "kept_source_file_date",
                   "source_row", "kept_source_row"]


def option_files(raw_root: Path) -> list[Path]:
    return sorted((raw_root / "options" / "NIFTY").glob("*/*.csv.gz"))


def source_file_date(path: Path) -> date | None:
    match = re.search(r"(\d{8})", path.name)
    return date(int(match.group(1)[:4]), int(match.group(1)[4:6]), int(match.group(1)[6:])) if match else None


def add_counts(total: dict[str, Any], current: dict[str, Any]) -> None:
    for key, value in current.items():
        if key == "session_removals_by_label":
            merge_label_counts(total.setdefault(key, {}), value)
        elif isinstance(value, int):
            total[key] = total.get(key, 0) + value


def finalize_bars(frame: pl.DataFrame, days: pl.DataFrame, bar_minutes: int) -> pl.DataFrame:
    frame = add_bar_availability(attach_day_flags(frame, days), bar_minutes)
    return frame.with_columns(pl.col("data_quality").fill_null("not_in_calendar"))


def main() -> None:
    config = load_config()
    data, regimes, paths = config["data"], config["regimes"], config["paths"]
    session = session_settings(data)
    start = date.fromisoformat(str(data["date_range"]["start"]))
    end = date.fromisoformat(str(data["date_range"]["end"]))
    holidays = holiday_dates(regimes)
    excluded_dates = sorted(set(holidays + special_session_dates(regimes)))
    truncated = truncated_session_dates(regimes)
    configured_quality = configured_quality_dates(regimes)
    available_time = parse_time(data["availability"]["next_trading_day_time"])
    raw_root, curated_root = paths["raw_root"], paths["curated_root"]
    window = dict(start=start, end=end, session_open=session["open"], session_close=session["close"], excluded_dates=excluded_dates)

    option_paths = option_files(raw_root)
    future_paths = futures_files(raw_root)
    spot_path = raw_root / "equity" / "NIFTY_5minute.csv.gz"
    vix_daily_path = raw_root / "india_vix" / "INDVIX_daily.csv.gz"
    vix_intraday_path = raw_root / "india_vix" / "INDVIX_5minute.csv.gz"
    participant_path = raw_root / "fiidii" / "participant_oi_2019_to_today.parquet"
    input_paths = option_paths + future_paths + [spot_path, vix_daily_path, vix_intraday_path, participant_path]

    with staged_outputs(curated_root, OWNED, config) as stage:
        # Futures: one contract per trading day.
        future_frames = [(pl.read_csv(path, infer_schema_length=10000), path.name, source_file_date(path)) for path in future_paths]
        futures, future_counts, future_dropped, contract_choice = clean_front_month(
            future_frames, roll_rule=data["futures"]["roll_rule"], **window
        )

        # Spot: clean, then null bad prints against the front-month future.
        spot, spot_counts, spot_dropped = clean_market_bars(
            pl.read_csv(spot_path), end_labeled=end_labeled_windows(regimes, "spot"), bar_minutes=session["bar_minutes"], **window
        )
        threshold = float(data["spot_outliers"]["max_relative_deviation_from_future"])
        spot, spot_outliers = flag_spot_outliers(spot, futures, threshold)
        spot_counts["bad_print_bars_nulled"] = spot_outliers.height
        spot_counts["unchecked_bars"] = spot.filter(pl.col("spot_quality") == "unchecked").height

        # Options pass 1: clean each file into a scratch area and collect bar labels per day.
        pass_one = stage / "_options_pass1"
        pass_one.mkdir()
        option_counts: dict[str, Any] = {}
        option_dropped: list[pl.DataFrame] = []
        option_labels: list[pl.DataFrame] = []
        oi_counts: list[pl.DataFrame] = []
        for path in option_paths:
            frame, counts, dropped = clean_option_file(path, source_file=path.name, source_file_date=source_file_date(path), **window)
            add_counts(option_counts, counts)
            if dropped.height:
                option_dropped.append(dropped.select([c for c in DROPPED_COLUMNS if c in dropped.columns]))
            if frame.height:
                write_parquet(frame, pass_one / f"{path.parent.name}_{path.name.removesuffix('.csv.gz')}.parquet")
                option_labels.append(frame.select("datetime").unique())
                oi_counts.append(option_oi_share(frame))
        option_bar_labels = pl.concat(option_labels).unique()

        # Trading calendar and day-level quality.
        observed = set(option_bar_labels["datetime"].dt.date().to_list()) | set(futures["trade_date"].to_list())
        calendar = build_trading_calendar(observed, holidays, int(data["calendar"]["extend_weekdays_after_last_observed"]))
        trading_dates = calendar["trading_date"].to_list()
        vix_intraday, vix_intraday_counts, vix_intraday_dropped = clean_market_bars(
            pl.read_csv(vix_intraday_path), end_labeled=end_labeled_windows(regimes, "vix_intraday"),
            bar_minutes=session["bar_minutes"], **window,
        )
        days = day_quality(
            calendar,
            {
                "options": bars_per_day(option_bar_labels, "options"),
                "futures": bars_per_day(futures, "futures"),
                "spot": bars_per_day(spot, "spot"),
                "vix_intraday": bars_per_day(vix_intraday, "vix_intraday"),
            },
            expected_bars=session["expected_bars"],
            truncated_dates=truncated,
            configured_dates=configured_quality,
            option_oi_share=pl.concat(oi_counts).group_by("trading_date").agg(pl.col("traded_rows").sum(), pl.col("traded_rows_with_oi").sum())
            .select("trading_date", (pl.col("traded_rows_with_oi") / pl.col("traded_rows")).alias("option_oi_share")),
            min_option_oi_share=float(data["quality"]["min_option_oi_share"]),
        )
        write_parquet(calendar.join(days.drop("next_trading_date"), on="trading_date", how="left"), stage / "calendar" / "trading_calendar.parquet")

        # Options pass 2: attach day flags and availability, and collect the EOD OI proxy.
        eod_frames = []
        for staged in sorted(pass_one.glob("*.parquet")):
            frame = finalize_bars(pl.read_parquet(staged), days, session["bar_minutes"]).with_columns(
                pl.when(pl.col("missing_option_oi")).then(None).otherwise(pl.col("open_interest")).alias("open_interest")
            )
            write_parquet(frame, stage / "options" / staged.name)
            eod_frames.append(eod_oi_rows(frame, session["eod_oi_bar"]))
        cross_file = (
            pl.scan_parquet(stage / "options" / "*.parquet").group_by(OPTION_KEYS).agg(pl.len().alias("n"))
            .filter(pl.col("n") > 1).select(pl.len()).collect().item()
        )
        if cross_file:
            raise SystemExit(f"{cross_file} option keys appear in more than one source file; dedup is per file")

        futures = finalize_bars(futures, days, session["bar_minutes"])
        write_parquet(futures, stage / "futures" / "nifty_front_month.parquet")
        write_parquet(contract_choice, stage / "futures" / "front_month_contracts.parquet")
        spot = finalize_bars(spot, days, session["bar_minutes"])
        write_parquet(spot, stage / "market" / "nifty_spot.parquet")
        vix_intraday = finalize_bars(vix_intraday, days, session["bar_minutes"])
        write_parquet(vix_intraday, stage / "vix" / "intraday.parquet")

        vix_daily, vix_daily_counts, vix_daily_dropped = clean_daily_vix(
            pl.read_csv(vix_daily_path), start=start, end=end, excluded_dates=excluded_dates
        )
        vix_daily = add_next_day_availability(vix_daily, "observation_date", trading_dates, available_time)
        write_parquet(vix_daily, stage / "vix" / "daily.parquet")

        participant, participant_counts, participant_removed = clean_participant_oi(
            participant_path, start=start, end=end, excluded_dates=excluded_dates,
            trading_dates=trading_dates, available_time=available_time,
        )
        write_parquet(participant, stage / "oi" / "participant_oi.parquet")
        eod = finalize_eod_oi(pl.concat(eod_frames), trading_dates, available_time)
        write_parquet(eod, stage / "oi" / "option_eod_oi_1525.parquet")

        # Availability invariant for daily sources: strictly after the observation date's close.
        availability_checks = {}
        for name, frame in (("option_eod_oi", eod), ("participant_oi", participant), ("vix_daily", vix_daily)):
            checked = frame.with_columns(observation_close(pl.col("observation_date"), session["end"]).alias("observation_close"))
            assert_available_after(checked, "observation_close")
            gap = checked.select((pl.col("available_at") - pl.col("observation_close")).dt.total_hours().alias("h"))["h"]
            availability_checks[name] = {
                "rows": frame.height,
                "rows_available_after_observation_close": frame.height,
                "min_hours_after_close": int(gap.min()),
                "max_hours_after_close": int(gap.max()),
            }
        # pass_one lives inside the staging root and is discarded when the owned directories are swapped in.

    write_reports(
        config, session=session, start=start, end=end, holidays=holidays, excluded_dates=excluded_dates,
        calendar=calendar, days=days, counts={
            "options": option_counts, "futures": future_counts, "spot": spot_counts,
            "vix_daily": vix_daily_counts, "vix_intraday": vix_intraday_counts, "participant_oi": participant_counts,
        },
        dropped={
            "options": (pl.concat(option_dropped) if option_dropped else pl.DataFrame(), OPTION_ORDER),
            "futures": (future_dropped, FUTURES_ORDER),
            "spot": (spot_dropped, MARKET_ORDER),
            "vix_intraday": (vix_intraday_dropped, MARKET_ORDER),
        },
        vix_daily_dropped=vix_daily_dropped, participant_removed=participant_removed,
        spot_outliers=spot_outliers, threshold=threshold, availability_checks=availability_checks,
        eod=eod, option_bar_labels=option_bar_labels, futures=futures, spot=spot, vix_daily=vix_daily,
        contract_choice=contract_choice,
    )
    outputs = [path for name in OWNED for path in (curated_root / name).rglob("*.parquet")]
    write_manifest(
        config, STAGE,
        inputs=describe_files(input_paths, config),
        outputs=describe_files(outputs, config),
        counts={
            "options_rows": option_counts["after_required_fields"], "futures_rows": futures.height,
            "spot_rows": spot.height, "vix_daily_rows": vix_daily.height, "vix_intraday_rows": vix_intraday.height,
            "participant_oi_rows": participant.height, "option_eod_oi_rows": eod.height,
            "trading_days_observed": int(calendar["observed"].sum()), "calendar_extension_days": int((~calendar["observed"]).sum()),
        },
        quality_flags={
            "missing_spot_days": days.filter(pl.col("missing_spot"))["trading_date"].to_list(),
            "truncated_session_days": days.filter(pl.col("truncated_session"))["trading_date"].to_list(),
            "days_with_quality_codes": days.filter(pl.col("data_quality") != "ok").height,
            "spot_bad_prints_nulled": spot_outliers.height,
        },
    )


def write_reports(config: dict[str, Any], **ctx: Any) -> None:
    session, days, calendar = ctx["session"], ctx["days"], ctx["calendar"]
    regimes = config["regimes"]
    counts = ctx["counts"]
    expected = session["expected_bars"]
    missing_spot = days.filter(pl.col("missing_spot"))["trading_date"].to_list()
    trading_dates = set(calendar.filter(pl.col("observed"))["trading_date"].to_list())
    vix_dates = set(ctx["vix_daily"]["observation_date"].to_list())

    write_report(config, "cleaning_report", {
        "timezone": config["data"]["timezone"],
        "date_range": {"start": ctx["start"], "end": ctx["end"]},
        "session": {
            "first_bar_label": session["open"].strftime("%H:%M"), "last_bar_label": session["close"].strftime("%H:%M"),
            "bar_minutes": session["bar_minutes"], "session_end": session["end"].strftime("%H:%M"),
            "expected_bars_per_day": expected,
            "rule": "bars are start-labeled; labels outside first..last bar label (inclusive) are dropped, including 15:30",
        },
        "excluded_dates": ctx["excluded_dates"],
        "non_regular_sessions": regimes.get("non_regular_sessions", []),
        "truncated_sessions": regimes.get("truncated_sessions", []),
        "configured_data_quality_flags": regimes.get("data_quality_flags", []),
        "holiday_dates_configured": ctx["holidays"],
        "calendar": {
            "rule": "union of option and futures trading dates minus configured holidays, extended by weekday rule",
            "observed_trading_days": int(calendar["observed"].sum()),
            "first_trading_date": calendar["trading_date"].min(),
            "last_observed_trading_date": calendar.filter(pl.col("observed"))["trading_date"].max(),
            "extension_dates": calendar.filter(~pl.col("observed"))["trading_date"].to_list(),
            "observed_dates_on_configured_holidays": sorted(trading_dates & set(ctx["holidays"])),
        },
        "row_counts": counts,
        "missing_spot_days": {
            "count": len(missing_spot), "dates": missing_spot,
            "action": "not imputed; data_quality includes missing_spot; exclude from GEX scaling, RV, and targets",
        },
        "missing_option_oi_days": {
            "count": int(days["missing_option_oi"].sum()),
            "dates": days.filter(pl.col("missing_option_oi"))["trading_date"].to_list(),
            "rule": f"share of traded option rows with open_interest > 0 below {config['data']['quality']['min_option_oi_share']}",
            "action": "open_interest set to null on these days in curated options and the EOD OI proxy; GEX and OI levels are missing",
            "option_oi_share_distribution_other_days": {
                "min": float(days.filter(~pl.col("missing_option_oi"))["option_oi_share"].min()),
                "median": float(days.filter(~pl.col("missing_option_oi"))["option_oi_share"].median()),
            },
        },
        "spot_days_not_in_calendar": sorted(set(ctx["spot"]["trade_date"].unique().to_list()) - trading_dates),
        "futures_roll": {
            "rule": config["data"]["futures"]["roll_rule"],
            "selection": "one contract per trading day; no intraday switching",
            "trading_days": ctx["contract_choice"].height,
            "contract_changes": counts["futures"]["contract_changes"],
            "max_contracts_per_day": int(ctx["futures"].group_by("trade_date").agg(pl.col("expiry_date").n_unique())["expiry_date"].max()),
        },
        "availability": {
            "intraday_bars": f"available_at = bar label + {session['bar_minutes']} minutes",
            "daily_sources": f"available_at = next trading day {config['data']['availability']['next_trading_day_time']} IST",
            "checks": ctx["availability_checks"],
        },
        "vix_daily_coverage": {
            "first_available_date": min(vix_dates) if vix_dates else None,
            "last_available_date": max(vix_dates) if vix_dates else None,
            "trading_dates_without_daily_vix": sorted(trading_dates - vix_dates),
        },
        "end_labeled_windows_applied": regimes.get("end_labeled_windows") or {},
        "spot_endpoint": {
            "raw_last_timestamp": counts["spot"]["raw_last_timestamp"],
            "cleaned_last_timestamp": counts["spot"]["cleaned_last_timestamp"],
            "note": "labels after 15:25 (15:30 and later) are outside the regular session and dropped",
        },
    })

    write_report(config, "duplicate_conflicts", {
        "rules": {
            "options": "keep higher volume, then higher open_interest, higher close, source_file, source_row",
            "futures": "keep later source_file_date, then higher volume, higher open_interest, higher close, source_file, source_row",
            "spot_and_vix": "keep higher volume, then earlier source_row",
            "participant_oi": "exact duplicates collapse; keys with conflicting duplicates are dropped (unavailable)",
        },
        **{name: duplicate_summary(frame, order) for name, (frame, order) in ctx["dropped"].items()},
        "vix_daily": duplicate_summary(ctx["vix_daily_dropped"], MARKET_ORDER) if "datetime" in ctx["vix_daily_dropped"].columns else {"dropped_rows": ctx["vix_daily_dropped"].height},
        "participant_oi": {
            "conflicting_rows_removed": ctx["participant_removed"].height,
            "conflicting_keys": ctx["participant_removed"].select("observation_date", "Client Type").unique().height
            if ctx["participant_removed"].height else 0,
            "rows": ctx["participant_removed"].to_dicts(),
        },
    })

    sources = {"options": ctx["option_bar_labels"], "futures": ctx["futures"], "spot": ctx["spot"]}
    payload: dict[str, Any] = {"expected_bars_per_day": expected, "definition": "distinct start labels per trading day; options use labels across all contracts"}
    for name in ("options", "futures", "spot", "vix_intraday"):
        payload[name] = completeness(days, name, expected)
        if name in sources:
            incomplete_dates = [item["date"] for item in payload[name]["incomplete"]]
            payload[name]["missing_labels"] = missing_labels(sources[name], incomplete_dates, session["open"], session["bar_minutes"], expected)
    computed = set(days.filter(pl.col("data_quality").str.contains("missing|incomplete"))["trading_date"].to_list())
    configured = set(configured_quality_dates(regimes))
    payload["configured_flags_vs_computed"] = {
        "configured_and_computed": sorted(configured & computed),
        "configured_only": sorted(configured - computed),
        "computed_only_count": len(computed - configured),
    }
    payload["day_quality_code_counts"] = dict(
        days.select(pl.col("data_quality").str.split("|")).explode("data_quality").group_by("data_quality").len().sort("data_quality").iter_rows()
    )
    write_report(config, "bar_completeness", payload)

    outliers = ctx["spot_outliers"]
    write_report(config, "spot_outliers", {
        "rule": "null spot OHLC when |spot close / front-month futures close - 1| exceeds the threshold; other field deviations are informational",
        "max_relative_deviation_from_future": ctx["threshold"],
        "bars_nulled": outliers.height,
        "bars_unchecked_no_future": counts["spot"]["unchecked_bars"],
        "rows": outliers.to_dicts(),
    })


if __name__ == "__main__":
    main()
