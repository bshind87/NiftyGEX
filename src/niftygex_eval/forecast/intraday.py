"""Exploratory E3 forecasting table and classical rungs for the intraday target (next 1 and 2 hours of futures RV).

The table joins the intraday aggregate features, the intraday RV windows and the intraday targets (data layer,
configs/intraday.yaml) on the snapshot. Both horizons share the same rows: a row needs both outcomes.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from niftygex.config import config_hash, load_intraday_config
from niftygex.io.parquet import read_manifest
from niftygex.targets.intraday_rv import past_mean
from niftygex_eval.data import option_expiry_dates
from niftygex_eval.forecast.baselines import Linear, Persistence, TunedLightGBM
from niftygex_eval.forecast.interface import Forecaster
from niftygex_eval.harness import assign_cycles

CONVENTIONS = ("naive", "absolute", "inferred")
STAGES = ("gex_intraday", "features_intraday", "intraday_rv")


def normalize_gex_by_clock(frame: pl.DataFrame, window: int, min_periods: int) -> pl.DataFrame:
    """``gex_norm_<c>`` = total GEX / mean |total GEX| at the same clock time over the previous ``window`` trading
    days (at least ``min_periods``); the snapshot's own day never enters its scale."""
    parts = []
    for _, part in frame.sort("snapshot_at").group_by(pl.col("snapshot_at").dt.strftime("%H:%M"), maintain_order=True):
        values = part["gex_total_absolute"].abs().cast(pl.Float64).fill_null(np.nan).to_numpy()
        parts.append(part.with_columns(pl.Series("gex_scale", past_mean(values, window, min_periods), dtype=pl.Float64, nan_to_null=True)))
    out = pl.concat(parts).sort("snapshot_at")
    return out.with_columns(*[(pl.col(f"gex_total_{c}") / pl.col("gex_scale")).alias(f"gex_norm_{c}") for c in CONVENTIONS])


def intraday_table(cfg: dict[str, Any]) -> tuple[pl.DataFrame, list[str], dict[str, str]]:
    """Rows with both horizons' outcomes; returns (table, summary feature columns, upstream hashes)."""
    data_config, e3 = cfg["data_config"], cfg["exploratory"]["e3"]
    hashes = {stage: read_manifest(data_config, stage)["config_hash"] for stage in STAGES}
    if len(set(hashes.values())) != 1:
        raise SystemExit(f"Intraday manifests disagree on config_hash: {hashes}")
    # The hash covers resolved paths, so it is checked against configs/intraday.yaml only where the data was built.
    if data_config["profile"] == "local" and set(hashes.values()) != {config_hash(load_intraday_config())}:
        raise SystemExit(f"Intraday manifests do not match configs/intraday.yaml: {hashes}")
    paths = data_config["paths"]
    features = pl.read_parquet(Path(paths["features_root"]) / "intraday_aggregate" / "aggregate.parquet")
    windows = pl.read_parquet(Path(paths["features_root"]) / "intraday_rv" / "rv_windows.parquet").drop("trade_date", "target")
    targets = pl.read_parquet(Path(paths["targets_root"]) / "intraday" / "intraday.parquet").drop("trade_date", "target")
    table = features.join(windows, on="snapshot_at", how="inner").join(targets, on="snapshot_at", how="inner")
    norm = e3["gex_normalization"]
    table = normalize_gex_by_clock(table, int(norm["window"]), int(norm["min_periods"]))
    horizons = [int(h) for h in e3["horizons"]]
    table = table.with_columns(pl.col("gex_total_inferred").sign().alias("gex_sign_inferred"),
                               *[pl.col(f"rv_ratio_{h}h").log().alias(f"log_rv_ratio_{h}h") for h in horizons])
    table = table.filter(pl.all_horizontal([pl.col(f"log_rv_ratio_{h}h").is_not_null() & pl.col(f"log_rv_ratio_{h}h").is_finite() for h in horizons]))
    clock = sorted(table["minutes_since_open"].unique().to_list())
    table = table.with_columns(*[(pl.col("minutes_since_open") == m).cast(pl.Int8).alias(f"tod_{m}") for m in clock[1:]])
    table = assign_cycles(table.sort("snapshot_at"), option_expiry_dates(Path(paths["curated_root"])))
    report = json.loads((Path(paths["reports_root"]) / "features_report_intraday.json").read_text())
    excluded = set(cfg["eval"]["forecast"]["summary_exclude"])
    rv_columns = [c for c in windows.columns if c.startswith("rv_") and c != "rv_windows_available_at"]
    summary = ([c for c in report["feature_columns"] if c not in excluded] + [f"gex_norm_{c}" for c in CONVENTIONS]
               + ["gex_sign_inferred", "minutes_since_open"] + rv_columns)
    return table, summary, hashes


def for_horizon(table: pl.DataFrame, horizon: int) -> pl.DataFrame:
    """The table with this horizon's label time as ``target_available_at`` (the harness purges on it)."""
    return table.with_columns(pl.col(f"target_available_at_{horizon}h").alias("target_available_at"))


def time_of_day_columns(table: pl.DataFrame) -> list[str]:
    return [c for c in table.columns if c.startswith("tod_")]


def make_intraday_baseline(name: str, e3: dict[str, Any], horizon: int, summary: list[str], tod: list[str], forecast: dict[str, Any]) -> Forecaster:
    """Classical E3 rungs; column templates with {h} are filled with the horizon."""
    har = [c.replace("{h}", str(horizon)) for c in e3["intraday_har_columns"]]
    if name == "persistence":
        return Persistence(f"rv_last_{60 * horizon}m", f"rv_back_normal_{60 * horizon}m")
    if name == "intraday_har":
        return Linear("intraday_har", har, tod)
    if name == "har_gex":
        return Linear("har_gex", har, tod + list(e3["gex_columns"]), alphas=list(e3["ridge_alphas"]))
    if name == "lightgbm":
        return TunedLightGBM(summary, dict(forecast["lightgbm"]), dict(forecast["tuning"]["grid"]))
    raise ValueError(f"Unknown intraday baseline {name!r}")
