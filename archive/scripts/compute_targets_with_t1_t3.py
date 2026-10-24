"""Archived 2026-10-14: the targets stage as it was before T1 and T3 were removed (T1, T2 and T3 targets).

Frozen reference copy; it needs the archived modules (targets/pinning.py, targets/intraday_drift.py,
io/official_closes.py), the removed config keys and a snapshot grid with T1/T3 rows (see archive/README.md).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from niftygex.clean.session import parse_time, session_settings
from niftygex.config import load_config
from niftygex.io.parquet import describe_files, require_upstream, staged_outputs, write_manifest, write_parquet, write_report
from niftygex.io.official_closes import load_official_closes
from niftygex.targets.intraday_drift import t3_targets
from niftygex.targets.pinning import strike_steps, t1_targets
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

    # T3: last-90-minute move toward the snapshot's max-GEX strike.
    t3_cfg = data["targets"]["t3"]
    t3_snapshots = (
        pl.read_parquet(Path(config["paths"]["features_root"]) / "aggregate" / "aggregate.parquet").filter(pl.col("target") == "t3")
        .select("snapshot_at", "trade_date", "spot", "truncated_session", "missing_spot")
        .join(pl.read_parquet(Path(config["paths"]["exposure_root"]) / "levels" / "levels.parquet").filter(pl.col("target") == "t3")
              .select("snapshot_at", "max_gex_strike"), on="snapshot_at", how="left")
    )
    t3 = t3_targets(
        t3_snapshots, pl.read_parquet(curated / "market" / "nifty_spot.parquet"),
        path_start=parse_time(data["snapshot_grid"]["t3_daily"]), path_end_label=parse_time(t3_cfg["path_end_label"]),
        min_path_bars=int(t3_cfg["min_path_bars"]),
    )
    if t3.filter(pl.col("target_available_at").is_not_null() & (pl.col("target_available_at") <= pl.col("snapshot_at"))).height:
        raise SystemExit("A T3 target is available at or before its snapshot")

    # T1: official expiry-day settlement vs the strike grid and the snapshot's levels (optional source).
    t1_cfg = data["targets"]["t1"]
    exposure = Path(config["paths"]["exposure_root"])
    t1_snapshots = (
        pl.read_parquet(Path(config["paths"]["features_root"]) / "aggregate" / "aggregate.parquet").filter(pl.col("target") == "t1")
        .select("snapshot_at", "trade_date", "spot")
        .join(pl.read_parquet(exposure / "levels" / "levels.parquet").filter(pl.col("target") == "t1")
              .select("snapshot_at", "max_gex_strike", "max_pain"), on="snapshot_at", how="left")
    )
    t1_chain = pl.read_parquet(exposure / "gex_strikes" / "*.parquet", columns=["snapshot_at", "trade_date", "expiry_date", "strike_price"]).join(
        t1_snapshots.select("snapshot_at"), on="snapshot_at", how="semi")
    closes_dir = Path(config["paths"]["raw_root"]) / "official_closes"
    closes = load_official_closes(closes_dir)
    t1 = t1_targets(t1_snapshots, closes, strike_steps(t1_chain), pin_fraction=float(t1_cfg["pin_fraction_of_step"]),
                    default_step=float(t1_cfg["default_strike_step"]), available_time=parse_time(t1_cfg["settlement_available_time"]))
    if t1.filter(pl.col("target_available_at") <= pl.col("snapshot_at")).height:
        raise SystemExit("A T1 target is available at or before its snapshot")

    targets_root = Path(config["paths"]["targets_root"])
    with staged_outputs(targets_root, ["t1", "t2", "t3"], config) as stage:
        write_parquet(t1, stage / "t1" / "t1.parquet")
        write_parquet(frame, stage / "t2" / "t2.parquet")
        write_parquet(t3, stage / "t3" / "t3.parquet")

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
    t3_ok = t3.filter(pl.col("t3_reason") == "ok")
    write_report(config, "targets_t3_report", {
        "definition": {
            "start": "snapshot spot at 14:00 (close of the 13:55-labeled bar, available 14:00)",
            "end": f"close of the {t3_cfg['path_end_label']}-labeled bar (available 15:30)",
            "strike": "max-GEX strike (absolute convention) at the T3 snapshot",
            "label": "t3_label = |end - strike| < |start - strike|",
            "exclusions": "truncated sessions, missing-spot days, no max-GEX strike, missing end bar, "
                          f"fewer than {t3_cfg['min_path_bars']} path bars, start exactly at the strike",
        },
        "rows": t3.height,
        "reason_counts": dict(t3.group_by("t3_reason").len().sort("t3_reason").iter_rows()),
        "label_share_toward_by_year": t3_ok.group_by(year).agg(
            pl.len().alias("labeled"), pl.col("t3_label").mean().alias("share_toward"),
            pl.col("t3_touched_strike").mean().alias("share_touched_strike"),
            pl.col("t3_distance_reduction_pct").median().alias("median_distance_reduction_pct"),
        ).sort("year").to_dicts(),
        "label_share_toward_overall": float(t3_ok["t3_label"].mean()) if t3_ok.height else None,
        "start_distance_pct_median": float((t3_ok["t3_start_distance"] / t3_ok["t3_start_price"] * 100).median()) if t3_ok.height else None,
    })
    t1_days = t1.filter(pl.col("t1_distance_fraction").is_not_null()).unique("trade_date").sort("trade_date")
    t1_payload = {
        "definition": {
            "settlement": "official NSE closing index on the expiry date (data/raw/official_closes/*.csv: date, close)",
            "t1_distance_fraction": "|settlement - nearest strike| / (strike step / 2); U(0, 1) under no pinning",
            "t1_pinned": f"distance <= {t1_cfg['pin_fraction_of_step']} x strike step; null rate {2 * float(t1_cfg['pin_fraction_of_step'])}",
            "levels": "distances to the snapshot's max-GEX and max-pain strikes (10:00 / 12:00 / 14:00 on expiry days)",
            "independence": "the three snapshots of a day share one settlement; grid statistics below are per expiry day",
        },
        "official_close_source_present": closes is not None,
        "official_close_rows": 0 if closes is None else closes.height,
        "rows": t1.height,
        "reason_counts": dict(t1.group_by("t1_reason").len().sort("t1_reason").iter_rows()),
    }
    if t1_days.height:
        from scipy.stats import binomtest, kstest

        pinned = int(t1_days["t1_pinned"].sum())
        t1_payload["per_expiry_day"] = {
            "days": t1_days.height,
            "pinned_rate": pinned / t1_days.height, "null_rate": 2 * float(t1_cfg["pin_fraction_of_step"]),
            "pinned_binomial_p_greater": float(binomtest(pinned, t1_days.height, 2 * float(t1_cfg["pin_fraction_of_step"]), alternative="greater").pvalue),
            "distance_fraction_ks_vs_uniform_p": float(kstest(t1_days["t1_distance_fraction"].to_numpy(), "uniform").pvalue),
        }
    write_report(config, "targets_t1_report", t1_payload)
    write_manifest(
        config, STAGE,
        inputs=describe_files([curated / "futures" / "nifty_front_month.parquet", curated / "market" / "nifty_spot.parquet",
                               Path(config["paths"]["features_root"]) / "aggregate" / "aggregate.parquet"], config),
        outputs=describe_files([targets_root / "t1" / "t1.parquet", targets_root / "t2" / "t2.parquet", targets_root / "t3" / "t3.parquet"], config),
        counts={"rows": frame.height, "labeled": labeled.height},
        quality_flags={"reasons": dict(frame.group_by("t2_reason").len().iter_rows())},
        upstream={"features": features_manifest["config_hash"]},
    )


if __name__ == "__main__":
    main()
