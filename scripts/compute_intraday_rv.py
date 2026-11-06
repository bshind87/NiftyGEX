"""Exploratory E3 stage: intraday realized-volatility features and targets on the 15-minute grid (configs/intraday.yaml).

Front-month futures bars -> data/features/intraday_rv/rv_windows.parquet (backward windows, yesterday, clock-window
normals; all known at the snapshot) and data/targets/intraday/intraday.parquet (RV over (T, T + h] / its normal),
stored separately. Windows with a missing bar are skipped, never imputed, and counted in the report.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from niftygex.clean.session import parse_time
from niftygex.config import config_hash, load_config, load_intraday_config
from niftygex.exposure.selection import INTRADAY_TARGET
from niftygex.io.parquet import describe_files, require_upstream, staged_outputs, write_manifest, write_parquet, write_report
from niftygex.targets.intraday_rv import intraday_rv_frames

STAGE = "intraday_rv"


def minutes(value: str) -> int:
    clock = parse_time(value)
    return clock.hour * 60 + clock.minute


def main() -> None:
    config, t2_config = load_intraday_config(), load_config()
    clean_manifest = require_upstream(t2_config, "clean")
    intraday, data = config["intraday"], config["data"]
    curated = Path(config["paths"]["curated_root"])
    grid = intraday["snapshots"]
    snapshot_minutes = list(range(minutes(grid["first"]), minutes(grid["last"]) + 1, int(grid["step_minutes"])))
    horizons = [int(h) for h in intraday["horizons_hours"]]
    last_label = minutes(data["session"]["close"])
    if snapshot_minutes[-1] + 60 * max(horizons) > last_label + int(data["session"]["bar_minutes"]):
        raise SystemExit("The longest target window must end within the session")

    calendar = pl.read_parquet(curated / "calendar" / "trading_calendar.parquet").filter(pl.col("observed")).sort("trading_date")
    trading_dates = calendar["trading_date"].to_list()
    futures = pl.read_parquet(curated / "futures" / "nifty_front_month.parquet").select("datetime", "trade_date", "available_at", "close", "expiry_date")
    if futures.group_by("trade_date").agg(pl.col("expiry_date").n_unique())["expiry_date"].max() != 1:
        raise SystemExit("Intraday RV needs one futures contract per trading day")
    features, targets = intraday_rv_frames(
        futures, trading_dates, snapshot_minutes, horizons, [int(w) for w in intraday["backward_windows_minutes"]],
        minutes(intraday["first_return_minute"]), last_label, int(intraday["normal_days"]), int(intraday["normal_min_days"]),
        full_day_min_returns=int(data["targets"]["t2"]["min_bars"]) - 1)
    features = features.with_columns(pl.lit(INTRADAY_TARGET).alias("target"),
                                     (pl.col("minute") - minutes(data["session"]["open"])).alias("minutes_since_open")).drop("minute")
    targets = targets.with_columns(pl.lit(INTRADAY_TARGET).alias("target")).drop("minute")
    for h in horizons:
        early = targets.filter(pl.col(f"target_available_at_{h}h") <= pl.col("snapshot_at")).height
        if early:
            raise SystemExit(f"{early} intraday {h}h targets are available at or before their snapshot")
    if features.filter(pl.col("rv_windows_available_at") > pl.col("snapshot_at")).height:
        raise SystemExit("An intraday RV feature uses data after its snapshot")

    features_root, targets_root = Path(config["paths"]["features_root"]), Path(config["paths"]["targets_root"])
    with staged_outputs(features_root, ["intraday_rv"], config) as stage:
        write_parquet(features, stage / "intraday_rv" / "rv_windows.parquet")
    with staged_outputs(targets_root, ["intraday"], config) as stage:
        write_parquet(targets, stage / "intraday" / "intraday.parquet")

    observed_days = set(futures["trade_date"].unique().to_list())
    with_bars = targets.filter(pl.col("trade_date").is_in(list(observed_days)))
    skipped = {f"{h}h": dict(with_bars.group_by(f"reason_{h}h").len().sort(f"reason_{h}h").iter_rows()) for h in horizons}
    both = with_bars.filter(pl.all_horizontal([pl.col(f"reason_{h}h") == "ok" for h in horizons]))
    write_report(config, "intraday_rv_report", {
        "upstream_clean_config_hash": clean_manifest["config_hash"], "t2_config_hash": config_hash(t2_config),
        "definition": {
            "rv": "sqrt(sum of squared 5-minute log returns of the front-month future over a clock window (a, b]: bars labeled a .. b - 5 min; "
                  "every return must be present, else the window is skipped",
            "target": "rv_ratio_{h}h = RV over (T, T + h] / mean RV of the same clock window over the previous "
                      f"{intraday['normal_days']} trading days (at least {intraday['normal_min_days']}), past days only",
            "available": "target_available_at_{h}h = T + h; features use returns known by T and past days only",
        },
        "snapshots_per_day": len(snapshot_minutes), "trading_days": len(trading_dates), "days_with_futures_bars": len(observed_days),
        "rows": targets.height, "rows_on_days_with_bars": with_bars.height,
        "reason_counts_on_days_with_bars": skipped, "rows_with_both_horizons": both.height,
        "missing_share_features": {c: float(features[c].is_null().mean()) for c in features.columns if c.startswith("rv_")},
    })
    write_manifest(
        config, STAGE,
        inputs=describe_files([curated / "futures" / "nifty_front_month.parquet", curated / "calendar" / "trading_calendar.parquet"], config),
        outputs=describe_files([features_root / "intraday_rv" / "rv_windows.parquet", targets_root / "intraday" / "intraday.parquet"], config),
        counts={"rows": targets.height, "rows_with_both_horizons": both.height},
        quality_flags={"skipped": skipped},
        upstream={"clean": clean_manifest["config_hash"]},
    )
    print(f"intraday RV: {targets.height} rows, {both.height} with both horizons; skipped {skipped}")


if __name__ == "__main__":
    main()
