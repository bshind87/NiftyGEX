"""Features stage: exposure + curated sources -> one aggregate feature row per (snapshot_at, target).

  --grid intraday   exploratory E3: the same features on the 15-minute intraday grid from data/exposure/intraday/,
                    written to data/features/intraday_aggregate/ with its own manifest (features_intraday)
  --verify          T2 only: recompute and compare with the current output; nothing is written
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import polars as pl

from niftygex.clean.session import session_settings
from niftygex.config import load_config, load_intraday_config
from niftygex.exposure.selection import INTRADAY_TARGET, TARGETS, total_gex_expiries
from niftygex.features.aggregate import (
    KEY,
    gex_features,
    level_features,
    market_state_features,
    participant_features,
    positioning_features,
    trailing_rv_features,
    vol_surface_features,
)
from niftygex.io.parquet import describe_files, require_upstream, staged_outputs, write_manifest, write_parquet, write_report
from niftygex.io.regimes import lot_sizes_by_expiry, regime_marker_flags

STAGE = "features"


def select_for_targets(grid: pl.DataFrame, rows: pl.DataFrame, gex_cfg: dict[str, Any], targets: tuple[str, ...] = TARGETS) -> pl.DataFrame:
    joined = grid.select(*KEY).join(rows, on="snapshot_at")
    return pl.concat([joined.filter((pl.col("target") == t) & total_gex_expiries(t, gex_cfg)) for t in targets])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", choices=["t2", "intraday"], default="t2")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify and args.grid != "t2":
        raise SystemExit("--verify compares the T2 output only")
    intraday = args.grid == "intraday"
    config = load_intraday_config() if intraday else load_config()
    gex_manifest = require_upstream(config, "gex_intraday" if intraday else "gex")
    data, regimes = config["data"], config["regimes"]
    exposure, curated = Path(config["paths"]["exposure_root"]) / ("intraday" if intraday else ""), Path(config["paths"]["curated_root"])
    targets = (INTRADAY_TARGET,) if intraday else TARGETS
    gex_cfg = {**data["gex"], "same_day_expiry_in_total_gex": {INTRADAY_TARGET: bool(config["intraday"]["same_day_expiry_in_total_gex"])}} \
        if intraday else data["gex"]
    features_cfg = data["features"]
    settings = session_settings(data)

    grid = pl.read_parquet(exposure / "snapshots" / "snapshot_grid.parquet")
    strikes = pl.read_parquet(exposure / "gex_strikes" / "*.parquet")
    oi_chain = pl.read_parquet(exposure / "oi_chain" / "*.parquet")
    totals = pl.read_parquet(exposure / "gex_totals" / "gex_totals.parquet")
    levels = pl.read_parquet(exposure / "levels" / "levels.parquet")
    selected = select_for_targets(grid, strikes, gex_cfg, targets)
    selected_oi = select_for_targets(grid, oi_chain, gex_cfg, targets)
    forwards = selected.group_by([*KEY, "expiry_date"]).agg(pl.col("forward").median())
    eod = pl.read_parquet(curated / "oi" / "option_eod_oi_1525.parquet").rename({"available_date": "trade_date"})

    frame = gex_features(totals, selected)
    frame = frame.join(level_features(levels), on=KEY, how="left")
    frame = frame.join(vol_surface_features(selected, float(features_cfg["skew_abs_delta"])), on=KEY, how="left")
    frame = frame.join(positioning_features(selected_oi, forwards, eod, float(features_cfg["near_money_width"])), on=KEY, how="left")
    frame = frame.join(market_state_features(
        grid, pl.read_parquet(curated / "basis" / "nifty_basis.parquet"), pl.read_parquet(curated / "vix" / "daily.parquet"),
        pl.read_parquet(curated / "futures" / "nifty_front_month.parquet"),
        staleness_minutes=int(data["snapshot"]["max_staleness_minutes"]),
        intraday_returns_per_day=settings["expected_bars"] - 1,
        trading_days_per_year=int(data["targets"]["trading_days_per_year"]),
    ), on=KEY, how="left")
    trading_dates = pl.read_parquet(curated / "calendar" / "trading_calendar.parquet").filter(pl.col("observed")).sort("trading_date")["trading_date"].to_list()
    frame = frame.join(trailing_rv_features(
        grid, pl.read_parquet(curated / "futures" / "nifty_front_month.parquet"), trading_dates,
        windows={int(features_cfg["trailing_rv_days"]): int(features_cfg["trailing_rv_min_days"]),
                 **{int(days): int(n) for days, n in features_cfg["har_rv_windows"].items()}},
        min_bars=int(data["targets"]["t2"]["min_bars"]), returns_per_day=settings["expected_bars"] - 1,
        trading_days_per_year=int(data["targets"]["trading_days_per_year"]),
    ), on=KEY, how="left")
    frame = frame.join(participant_features(grid, pl.read_parquet(curated / "oi" / "participant_oi.parquet"),
                                            list(features_cfg["participant_client_types"])), on=KEY, how="left")

    # Calendar and regime features (known in advance).
    nearest = selected.group_by(KEY).agg(pl.col("expiry_date").min().alias("nearest_expiry"))
    expiry_dates = strikes["expiry_date"].unique().to_list()
    lots = lot_sizes_by_expiry(regimes, expiry_dates).select(pl.col("expiry_date").alias("nearest_expiry"), pl.col("units_per_lot").alias("lot_size"))
    frame = frame.join(nearest, on=KEY, how="left").join(lots, on="nearest_expiry", how="left").with_columns(
        pl.col("trade_date").is_in(expiry_dates).alias("is_expiry_day"),
        pl.col("nearest_expiry").dt.weekday().alias("nearest_expiry_weekday"),
        *regime_marker_flags(regimes),
    )
    calendar = pl.read_parquet(curated / "calendar" / "trading_calendar.parquet").select(
        pl.col("trading_date").alias("trade_date"), "missing_option_oi")
    frame = frame.join(calendar, on="trade_date", how="left")

    source_columns = sorted(c for c in frame.columns if c.endswith("_available_at"))
    frame = frame.with_columns(pl.max_horizontal(source_columns).alias("feature_max_source_available_at")).sort(KEY)

    features_root = Path(config["paths"]["features_root"])
    owned = "intraday_aggregate" if intraday else "aggregate"
    if args.verify:
        current = pl.read_parquet(features_root / owned / "aggregate.parquet").select(frame.columns)
        if not frame.equals(current):
            raise SystemExit("--verify: the aggregate features differ from the current output")
        print(f"--verify: aggregate features identical ({frame.height} rows)")
        return
    with staged_outputs(features_root, [owned], config) as stage:
        write_parquet(frame, stage / owned / "aggregate.parquet")

    flags_and_keys = {*KEY, "trade_date", *source_columns, "feature_max_source_available_at", "data_quality",
                      "nearest_expiry", "spot_available", "oi_available", "inferred_available", "vix_available",
                      "truncated_session", "missing_spot", "missing_option_oi"}
    feature_columns = [c for c in frame.columns if c not in flags_and_keys and not c.startswith("qa_")]
    year = pl.col("trade_date").dt.year()
    by_year = frame.group_by(year.alias("year")).agg(
        pl.len().alias("rows"),
        *[pl.col(c).is_not_null().mean().alias(c) for c in feature_columns],
    ).sort("year")
    write_report(config, "features_report" + ("_intraday" if intraday else ""), {
        "upstream_gex_config_hash": gex_manifest["config_hash"],
        "rows": frame.height,
        "rows_per_target": dict(frame.group_by("target").len().sort("target").iter_rows()),
        "feature_columns": feature_columns,
        "source_timestamp_columns": source_columns,
        "qa_columns": sorted(c for c in frame.columns if c.startswith("qa_")),
        "missing_share": {c: float(frame[c].is_null().mean()) for c in feature_columns},
        "missing_share_by_target": {
            t: {c: float(part[c].is_null().mean()) for c in feature_columns}
            for (t,), part in frame.group_by(["target"])
        },
        "non_null_share_by_year": by_year.to_dicts(),
        "rows_complete_core_share_by_year": frame.group_by(year.alias("year")).agg(
            pl.all_horizontal([pl.col(c).is_not_null() for c in ("gex_total_naive", "atm_iv", "futures_basis_bps", "vix_previous_close", "oi_change_near_money_calls")]).mean().alias("share")
        ).sort("year").to_dicts(),
        "notes": {
            "missing_option_oi": "GEX, levels, PCR and OI-change features are missing on missing_option_oi days (2019-07-30, 2022-01-03..2022-04-11)",
            "oi_change": "OI change also needs the previous day's EOD OI, so the first day after a missing-OI block is missing too",
            "vix": "daily VIX ends 2026-04-06; later rows have vix_previous_close missing (never forward-filled)",
            "participant": "participant OI covers all index options, not NIFTY only",
        },
    })
    write_manifest(
        config, STAGE + ("_intraday" if intraday else ""),
        inputs=describe_files(sorted(p for p in exposure.rglob("*.parquet") if intraday or "intraday" not in p.parts), config, checksums=False),
        outputs=describe_files([features_root / owned / "aggregate.parquet"], config),
        counts={"rows": frame.height, "feature_columns": len(feature_columns)},
        quality_flags={"rows_without_gex": int(frame["gex_total_naive"].is_null().sum())},
        upstream={"gex_intraday" if intraday else "gex": gex_manifest["config_hash"]},
    )


if __name__ == "__main__":
    main()
