"""T1 pinning targets: where does the official expiry-day settlement land relative to the strike grid and key levels?"""
from __future__ import annotations

from datetime import time

import polars as pl

from niftygex.io.availability import at_ist_time


def strike_steps(chain: pl.DataFrame) -> pl.DataFrame:
    """Per T1 snapshot: smallest spacing of the expiring contract's listed strikes (same-day expiry only)."""
    same_day = chain.filter(pl.col("expiry_date") == pl.col("trade_date")).select("snapshot_at", "strike_price").unique()
    return same_day.sort("snapshot_at", "strike_price").with_columns(
        pl.col("strike_price").diff().over("snapshot_at").alias("gap")
    ).filter(pl.col("gap") > 0).group_by("snapshot_at").agg(pl.col("gap").min().alias("strike_step"))


def t1_targets(snapshots: pl.DataFrame, closes: pl.DataFrame | None, steps: pl.DataFrame, *, pin_fraction: float,
               default_step: float, available_time: time) -> pl.DataFrame:
    """One row per T1 snapshot.

    ``snapshots``: snapshot_at, trade_date, spot, max_gex_strike, max_pain. ``closes``: date, official_close
    (None when the optional source is absent). Under no pinning the settlement's position within a strike
    step is uniform, so ``t1_distance_fraction`` (distance to the nearest strike / half a step) is U(0, 1)
    and ``t1_pinned`` (distance <= pin_fraction x step) has null rate 2 x pin_fraction.
    """
    frame = snapshots.join(steps, on="snapshot_at", how="left").with_columns(
        pl.col("strike_step").fill_null(default_step), pl.lit(2 * pin_fraction).alias("t1_null_pin_rate"),
        at_ist_time(pl.col("trade_date"), available_time).alias("target_available_at"),
    )
    if closes is None:
        frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias("official_close"))
    else:
        frame = frame.join(closes.select(pl.col("date").alias("trade_date"), "official_close"), on="trade_date", how="left")
    settle, step = pl.col("official_close"), pl.col("strike_step")
    frame = frame.with_columns(
        ((settle / step).round(0) * step).alias("t1_nearest_strike"),
    ).with_columns(
        (settle - pl.col("t1_nearest_strike")).abs().alias("t1_distance_points"),
    ).with_columns(
        (pl.col("t1_distance_points") / (step / 2)).clip(0.0, 1.0).alias("t1_distance_fraction"),
        (pl.col("t1_distance_points") <= pin_fraction * step).alias("t1_pinned"),
        ((settle - pl.col("max_gex_strike")) / settle * 100).alias("t1_distance_to_max_gex_pct"),
        ((settle - pl.col("max_pain")) / settle * 100).alias("t1_distance_to_max_pain_pct"),
        ((settle - pl.col("max_gex_strike")).abs() <= pin_fraction * step).alias("t1_pinned_to_max_gex"),
        ((settle - pl.col("max_pain")).abs() <= pin_fraction * step).alias("t1_pinned_to_max_pain"),
        ((pl.col("spot") - pl.col("max_gex_strike")).abs() / pl.col("spot") * 100).alias("t1_snapshot_distance_to_max_gex_pct"),
    ).with_columns(
        pl.when(pl.lit(closes is None)).then(pl.lit("no_official_close_source"))
        .when(settle.is_null()).then(pl.lit("no_official_close_for_date"))
        .when(pl.col("max_gex_strike").is_null()).then(pl.lit("no_max_gex_strike"))
        .otherwise(pl.lit("ok")).alias("t1_reason")
    )
    label_columns = ["t1_nearest_strike", "t1_distance_points", "t1_distance_fraction", "t1_pinned", "t1_distance_to_max_gex_pct",
                     "t1_distance_to_max_pain_pct", "t1_pinned_to_max_gex", "t1_pinned_to_max_pain"]
    # Levels-based labels need the strike; grid-based labels only need the settlement.
    frame = frame.with_columns(
        *[pl.when(pl.col("t1_reason").is_in(["ok", "no_max_gex_strike"])).then(pl.col(c)).otherwise(None).alias(c)
          for c in label_columns if "max_gex" not in c and "max_pain" not in c],
        *[pl.when(pl.col("t1_reason") == "ok").then(pl.col(c)).otherwise(None).alias(c) for c in label_columns if "max_gex" in c or "max_pain" in c],
    )
    return frame.select(
        "snapshot_at", "trade_date", "spot", "max_gex_strike", "max_pain", "strike_step", "official_close",
        *label_columns, "t1_snapshot_distance_to_max_gex_pct", "t1_null_pin_rate", "t1_reason", "target_available_at",
    ).sort("snapshot_at")
