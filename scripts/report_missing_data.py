"""Missing-data report: every known gap in the curated data -> outputs/reports/missing_data.{json,md}."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from niftygex.config import load_config
from niftygex.io.parquet import write_report


def ranges(days: list[date], trading_days: list[date]) -> list[dict[str, Any]]:
    """Collapse dates into runs of consecutive trading days."""
    position = {day: i for i, day in enumerate(trading_days)}
    out: list[dict[str, Any]] = []
    for day in sorted(days):
        if out and day in position and out[-1]["_last"] is not None and position[day] == out[-1]["_last"] + 1:
            out[-1]["end"], out[-1]["_last"] = day, position[day]
            out[-1]["trading_days"] += 1
        else:
            out.append({"start": day, "end": day, "trading_days": 1, "_last": position.get(day)})
    return [{k: v for k, v in r.items() if k != "_last"} for r in out]


def fmt_ranges(items: list[dict[str, Any]]) -> str:
    return ", ".join(f"{r['start']}" if r["start"] == r["end"] else f"{r['start']} to {r['end']} ({r['trading_days']} days)" for r in items) or "none"


def main() -> None:
    config = load_config()
    paths = config["paths"]
    curated, reports = Path(paths["curated_root"]), Path(paths["reports_root"])
    calendar = pl.read_parquet(curated / "calendar" / "trading_calendar.parquet").filter(pl.col("observed")).sort("trading_date")
    trading = calendar["trading_date"].to_list()
    cleaning = json.loads((reports / "cleaning_report.json").read_text())
    completeness = json.loads((reports / "bar_completeness.json").read_text())
    alignment = json.loads((reports / "spot_label_alignment.json").read_text())
    outliers = json.loads((reports / "spot_outliers.json").read_text())

    def flagged(column: str) -> list[date]:
        return calendar.filter(pl.col(column))["trading_date"].to_list()

    codes = calendar.select("trading_date", pl.col("data_quality").str.split("|").alias("code")).explode("code")
    by_code = {code: part["trading_date"].to_list() for (code,), part in codes.filter(pl.col("code") != "ok").group_by(["code"])}

    options = pl.scan_parquet(curated / "options" / "*.parquet").group_by("expiry_date").agg(
        pl.col("trade_date").min().alias("first_day"), pl.col("trade_date").max().alias("last_day")).collect().sort("expiry_date")
    trading_set = set(trading)
    truncated = options.filter(pl.col("expiry_date").is_in(list(trading_set)) & (pl.col("last_day") < pl.col("expiry_date")))
    depth = options.select((pl.col("expiry_date") - pl.col("first_day")).dt.total_days().alias("d"))["d"]

    participant_days = set(pl.read_parquet(curated / "oi" / "participant_oi.parquet")["observation_date"].unique().to_list())
    vix_daily_days = set(pl.read_parquet(curated / "vix" / "daily.parquet")["observation_date"].unique().to_list())
    vix_intraday = pl.read_parquet(curated / "vix" / "intraday.parquet")
    raw = Path(paths["raw_root"])
    optional = {name: (raw / name).exists() and any((raw / name).iterdir()) for name in ("bhavcopy",)}

    payload = {
        "trading_days": len(trading),
        "first_trading_day": trading[0], "last_trading_day": trading[-1],
        "missing_spot": {"days": flagged("missing_spot"), "ranges": ranges(flagged("missing_spot"), trading),
                         "action": "not imputed; excluded from GEX spot scaling, RV and targets"},
        "missing_option_oi": {"days": len(flagged("missing_option_oi")), "ranges": ranges(flagged("missing_option_oi"), trading),
                              "action": "OI nulled; GEX, OI levels, PCR, OI-change and token OI fields missing"},
        "missing_options_days": by_code.get("missing_options", []),
        "missing_futures_days": by_code.get("missing_futures", []),
        "incomplete_bar_days": {source: completeness[source]["incomplete_days"] for source in ("options", "futures", "spot", "vix_intraday")},
        "truncated_expiry_files": truncated.with_columns(pl.col("expiry_date").cast(pl.String), pl.col("last_day").cast(pl.String)).select("expiry_date", "last_day").to_dicts(),
        "expiry_history_depth_days": {"median": int(depth.median()), "max": int(depth.max())},
        "daily_vix": {"last_available": max(vix_daily_days), "trading_days_without": len(cleaning["vix_daily_coverage"]["trading_dates_without_daily_vix"]),
                      "ranges": ranges([d for d in trading if d not in vix_daily_days], trading)},
        "intraday_vix": {"first": vix_intraday["trade_date"].min(), "last": vix_intraday["trade_date"].max(),
                         "trading_days_without": len([d for d in trading if d not in set(vix_intraday["trade_date"].unique().to_list())])},
        "participant_oi_missing_trading_days": ranges([d for d in trading if d not in participant_days], trading),
        "spot_bad_prints_nulled": [row["datetime"] for row in outliers["rows"]],
        "spot_alignment_exceptions": alignment["spot_vs_futures"]["days_not_lag0"],
        "optional_sources_present": optional,
    }
    write_report(config, "missing_data", payload)

    lines = [
        "# Missing-data report",
        "",
        "Generated by `scripts/report_missing_data.py` from curated data and stage reports; do not edit by hand.",
        "",
        f"Trading calendar: {payload['trading_days']} days, {payload['first_trading_day']} to {payload['last_trading_day']}.",
        "",
        "| Gap | Extent | Handling |",
        "|---|---|---|",
        f"| Option OI recorded as 0 (missing) | {payload['missing_option_oi']['days']} days: {fmt_ranges(payload['missing_option_oi']['ranges'])} | {payload['missing_option_oi']['action']} |",
        f"| Spot missing | {len(payload['missing_spot']['days'])} days: {fmt_ranges(payload['missing_spot']['ranges'])} | {payload['missing_spot']['action']} |",
        f"| Option data missing | {len(payload['missing_options_days'])} days: {', '.join(map(str, payload['missing_options_days'])) or 'none'} | flagged `missing_options` |",
        f"| Futures data missing | {len(payload['missing_futures_days'])} days: {', '.join(map(str, payload['missing_futures_days'])) or 'none'} | flagged `missing_futures` |",
        "| Incomplete-bar days | " + ", ".join(f"{k} {v}" for k, v in payload["incomplete_bar_days"].items()) + " | flagged `incomplete_*_bars`; rows kept |",
        "| Expiry file ends before its expiry day | " + (", ".join(f"{r['expiry_date']} (last data {r['last_day']})" for r in payload["truncated_expiry_files"]) or "none") + " | no expiry-day chain for that contract |",
        f"| Option history per expiry | median {payload['expiry_history_depth_days']['median']}, max {payload['expiry_history_depth_days']['max']} calendar days | GEX horizon set to 21 days |",
        f"| Daily India VIX | ends {payload['daily_vix']['last_available']}; {payload['daily_vix']['trading_days_without']} trading days without: {fmt_ranges(payload['daily_vix']['ranges'])} | NaN, never forward-filled |",
        f"| Intraday India VIX | {payload['intraday_vix']['first']} to {payload['intraday_vix']['last']}; {payload['intraday_vix']['trading_days_without']} trading days without | optional source |",
        f"| Participant OI | missing trading days: {fmt_ranges(payload['participant_oi_missing_trading_days'])} | NaN, never forward-filled |",
        f"| Spot bad prints | {len(payload['spot_bad_prints_nulled'])}: {', '.join(map(str, payload['spot_bad_prints_nulled']))} | OHLC nulled |",
        "| Spot/futures alignment exceptions | " + (", ".join(f"{r['day']} (best lag {r['best_lag']}, corr {r['best_corr']:.2f})" for r in payload["spot_alignment_exceptions"]) or "none") + " | erratic futures closes; no relabel |",
        f"| Bhavcopy | {'present' if optional['bhavcopy'] else 'absent'} | 15:25 OI proxy used; reconciliation skipped |",
        "",
    ]
    (reports / "missing_data.md").write_text("\n".join(lines))
    print(reports / "missing_data.md")


if __name__ == "__main__":
    main()
