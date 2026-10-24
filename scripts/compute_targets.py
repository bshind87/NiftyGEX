"""Targets stage (T2): next-day realized-volatility regime labels, stored separately from features."""
from __future__ import annotations

from pathlib import Path

import polars as pl

from niftygex.clean.session import session_settings
from niftygex.config import load_config
from niftygex.io.parquet import describe_files, require_upstream, staged_outputs, write_manifest, write_parquet, write_report
from niftygex.targets.vol_regime import LABELS, causal_terciles, daily_realized_vol, trailing_mean

STAGE = "targets"


def ratio_labels(frame: pl.DataFrame, value: str, min_history: int, prefix: str) -> pl.DataFrame:
    labels, lows, highs = causal_terciles(frame["trade_date"].to_list(), frame["target_date"].to_list(), frame[value].to_list(), min_history)
    return frame.with_columns(
        pl.Series(f"{prefix}_label", labels, dtype=pl.String),
        pl.Series(f"{prefix}_cutoff_low", lows, dtype=pl.Float64),
        pl.Series(f"{prefix}_cutoff_high", highs, dtype=pl.Float64),
    )


def main() -> None:
    config = load_config()
    features_manifest = require_upstream(config, "features")
    data = config["data"]
    t2 = data["targets"]["t2"]
    curated = Path(config["paths"]["curated_root"])
    settings = session_settings(data)
    rv_args = dict(min_bars=int(t2["min_bars"]), trading_days_per_year=int(data["targets"]["trading_days_per_year"]),
                   returns_per_day=settings["expected_bars"] - 1)

    calendar = pl.read_parquet(curated / "calendar" / "trading_calendar.parquet").filter(pl.col("observed")).sort("trading_date")
    trading_dates = calendar["trading_date"].to_list()
    futures = pl.read_parquet(curated / "futures" / "nifty_front_month.parquet").select("datetime", "trade_date", "available_at", "close", "expiry_date")
    if futures.group_by("trade_date").agg(pl.col("expiry_date").n_unique())["expiry_date"].max() != 1:
        raise SystemExit("Futures RV needs one contract per trading day")
    spot = pl.read_parquet(curated / "market" / "nifty_spot.parquet").select("datetime", "trade_date", "available_at", "close")
    rv_futures = daily_realized_vol(futures, **rv_args)
    rv_spot = daily_realized_vol(spot, **rv_args)
    window, min_days = int(t2["trailing_days"]), int(t2["trailing_min_days"])
    trail_futures = trailing_mean(rv_futures, trading_dates, window, min_days)
    trail_spot = trailing_mean(rv_spot, trading_dates, window, min_days)

    next_day = calendar.select(pl.col("trading_date").alias("trade_date"), pl.col("next_trading_date").alias("target_date"))
    features = pl.read_parquet(Path(config["paths"]["features_root"]) / "aggregate" / "aggregate.parquet").filter(pl.col("target") == "t2")
    frame = (
        features.select("snapshot_at", "trade_date", "atm_iv")
        .join(next_day, on="trade_date", how="left")
        .join(trail_futures, on="trade_date", how="left")
        .join(rv_futures.select(pl.col("trade_date").alias("target_date"), pl.col("rv").alias("rv_next"),
                                pl.col("rv_annualized").alias("rv_next_annualized"), pl.col("bars").alias("rv_next_bars"),
                                pl.col("rv_available_at").alias("target_available_at")), on="target_date", how="left")
        .join(trail_spot.rename({"rv_trailing_mean": "rv_spot_trailing_mean"}), on="trade_date", how="left")
        .join(rv_spot.select(pl.col("trade_date").alias("target_date"), pl.col("rv").alias("rv_spot_next")), on="target_date", how="left")
        .sort("trade_date")
        .with_columns(
            (pl.col("rv_next") / pl.col("rv_trailing_mean")).alias("rv_ratio_trailing"),
            (pl.col("rv_spot_next") / pl.col("rv_spot_trailing_mean")).alias("rv_spot_ratio_trailing"),
            (pl.col("rv_next_annualized") / pl.col("atm_iv")).alias("rv_over_atm_iv"),
        )
    )
    min_history = int(t2["cutoff_min_history"])
    frame = ratio_labels(frame, "rv_ratio_trailing", min_history, "t2")
    frame = ratio_labels(frame, "rv_spot_ratio_trailing", min_history, "t2_spot")
    frame = ratio_labels(frame, "rv_over_atm_iv", min_history, "t2_iv")
    frame = frame.with_columns(
        pl.when(pl.col("target_date").is_null()).then(pl.lit("no_next_trading_day"))
        .when(pl.col("rv_next").is_null()).then(pl.lit("next_day_rv_missing"))
        .when(pl.col("rv_trailing_mean").is_null()).then(pl.lit("trailing_rv_insufficient"))
        .when(pl.col("t2_label").is_null()).then(pl.lit("insufficient_cutoff_history"))
        .otherwise(pl.lit("ok")).alias("t2_reason")
    ).drop("atm_iv")
    if frame.filter(pl.col("target_available_at").is_not_null() & (pl.col("target_available_at") <= pl.col("snapshot_at"))).height:
        raise SystemExit("A T2 target is available at or before its snapshot")

    targets_root = Path(config["paths"]["targets_root"])
    with staged_outputs(targets_root, ["t2"], config) as stage:
        write_parquet(frame, stage / "t2" / "t2.parquet")

    year = pl.col("trade_date").dt.year().alias("year")
    labeled = frame.filter(pl.col("t2_label").is_not_null())
    both = frame.filter(pl.col("t2_label").is_not_null() & pl.col("t2_spot_label").is_not_null())
    distribution = labeled.group_by(year).agg(
        pl.len().alias("labeled"),
        *[(pl.col("t2_label") == label).mean().alias(f"share_{label}") for label in LABELS],
        pl.col("t2_cutoff_low").first().alias("cutoff_low_first"), pl.col("t2_cutoff_high").first().alias("cutoff_high_first"),
        pl.col("t2_cutoff_low").last().alias("cutoff_low_last"), pl.col("t2_cutoff_high").last().alias("cutoff_high_last"),
    ).sort("year")
    write_report(config, "targets_t2_report", {
        "upstream_features_config_hash": features_manifest["config_hash"],
        "definition": {
            "rv": "sqrt(sum of squared within-day 5-minute log returns of front-month futures, one contract per day); "
                  f"null when fewer than {t2['min_bars']} bars",
            "ratio": f"RV(next trading day) / mean RV over the {window} trading days ending on the snapshot day (min {min_days} present)",
            "label": "terciles low/mid/high with cutoffs from the expanding history of ratios whose target day is before the snapshot day "
                     f"(min {min_history} values)",
            "secondary": "t2_spot_* uses spot RV; t2_iv_* uses annualized next-day futures RV / snapshot ATM IV",
            "target_available_at": "last bar of the next trading day becomes available (15:30 IST)",
        },
        "rows": frame.height,
        "reason_counts": dict(frame.group_by("t2_reason").len().sort("t2_reason").iter_rows()),
        "label_distribution_by_year": distribution.to_dicts(),
        "label_distribution_overall": dict(labeled.group_by("t2_label").len().sort("t2_label").iter_rows()),
        "futures_vs_spot_label_differs": {
            "days_with_both_labels": both.height,
            "share_differs": float((both["t2_label"] != both["t2_spot_label"]).mean()) if both.height else None,
            "by_year": both.group_by(year).agg((pl.col("t2_label") != pl.col("t2_spot_label")).mean().alias("share_differs"), pl.len().alias("days")).sort("year").to_dicts(),
        },
        "rv_over_atm_iv_label_distribution": dict(frame.filter(pl.col("t2_iv_label").is_not_null()).group_by("t2_iv_label").len().sort("t2_iv_label").iter_rows()),
    })
    write_manifest(
        config, STAGE,
        inputs=describe_files([curated / "futures" / "nifty_front_month.parquet", curated / "market" / "nifty_spot.parquet",
                               Path(config["paths"]["features_root"]) / "aggregate" / "aggregate.parquet"], config),
        outputs=describe_files([targets_root / "t2" / "t2.parquet"], config),
        counts={"rows": frame.height, "labeled": labeled.height},
        quality_flags={"reasons": dict(frame.group_by("t2_reason").len().iter_rows())},
        upstream={"features": features_manifest["config_hash"]},
    )


if __name__ == "__main__":
    main()
