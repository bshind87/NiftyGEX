"""The forecasting table: T2 features and targets, normalized GEX, and the log volatility-ratio outcome."""
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from niftygex_eval.data import modeling_table


def trailing_mean_abs(values: list[float | None], window: int, min_periods: int) -> list[float | None]:
    """Causal trailing mean over the last ``window`` rows (inclusive), ignoring missing values."""
    out: list[float | None] = []
    for i in range(len(values)):
        present = [v for v in values[max(0, i - window + 1): i + 1] if v is not None and np.isfinite(v)]
        out.append(float(np.mean(present)) if len(present) >= min_periods else None)
    return out


def normalize_gex(frame: pl.DataFrame, conventions: list[str], window: int, min_periods: int) -> pl.DataFrame:
    """``gex_norm_<c>`` = total GEX / causal trailing mean of absolute total GEX, removing the OI growth trend."""
    frame = frame.sort("snapshot_at")
    scale = pl.Series("gex_scale", trailing_mean_abs(frame["gex_total_absolute"].to_list(), window, min_periods), dtype=pl.Float64)
    return frame.with_columns(scale).with_columns(
        *[(pl.col(f"gex_total_{c}") / pl.col("gex_scale")).alias(f"gex_norm_{c}") for c in conventions])


def forecast_table(cfg: dict[str, Any]) -> tuple[pl.DataFrame, str, list[str], dict[str, str]]:
    """Returns (rows with a finite outcome, outcome column, summary feature columns, upstream data hashes).

    The summary columns are the aggregate features minus ``forecast.summary_exclude`` (trend proxies such as
    raw rupee GEX totals), plus normalized GEX and the inferred sign.
    """
    forecast = cfg["eval"]["forecast"]
    table, feature_columns, hashes = modeling_table(cfg["data_config"], forecast["target"])
    norm = forecast["gex_normalization"]
    conventions = list(norm["conventions"])
    table = normalize_gex(table, conventions, int(norm["window"]), int(norm["min_periods"]))
    outcome = f"log_{forecast['outcome']}"
    table = table.with_columns(pl.col("gex_total_inferred").sign().alias("gex_sign_inferred"),
                               pl.col(forecast["outcome"]).log().alias(outcome))
    table = table.filter(pl.col(outcome).is_not_null() & pl.col(outcome).is_finite())
    excluded = set(forecast["summary_exclude"])
    summary = [c for c in feature_columns if c not in excluded] + [f"gex_norm_{c}" for c in conventions] + ["gex_sign_inferred"]
    return table, outcome, summary, hashes
