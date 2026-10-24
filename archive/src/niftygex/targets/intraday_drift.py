"""T3 intraday-drift targets: does the index move toward the max-GEX strike over the last 90 minutes?"""
from __future__ import annotations

from datetime import time

import polars as pl


def t3_targets(
    snapshots: pl.DataFrame,
    spot: pl.DataFrame,
    *,
    path_start: time,
    path_end_label: time,
    min_path_bars: int,
) -> pl.DataFrame:
    """One row per T3 snapshot.

    ``snapshots`` has snapshot_at, trade_date, spot (the snapshot spot, i.e. the close available at
    snapshot_at), max_gex_strike, truncated_session, missing_spot. The path is the spot bars labeled
    ``path_start`` .. ``path_end_label``; the end price is the close of the ``path_end_label`` bar.
    ``t3_label`` is true when |end - strike| < |start - strike|. Truncated sessions are excluded.
    """
    start_min = path_start.hour * 60 + path_start.minute
    end_min = path_end_label.hour * 60 + path_end_label.minute
    minute = pl.col("datetime").dt.hour().cast(pl.Int32) * 60 + pl.col("datetime").dt.minute().cast(pl.Int32)
    path = spot.filter(minute.is_between(start_min, end_min) & pl.col("close").is_not_null())
    per_day = path.group_by("trade_date").agg(
        pl.len().alias("t3_path_bars"),
        pl.col("close").filter(minute == end_min).first().alias("t3_end_price"),
        pl.col("available_at").filter(minute == end_min).first().alias("target_available_at"),
        pl.col("low").min().alias("t3_path_low"),
        pl.col("high").max().alias("t3_path_high"),
    )
    frame = snapshots.select(
        "snapshot_at", "trade_date", pl.col("spot").alias("t3_start_price"), pl.col("max_gex_strike").alias("t3_strike"),
        "truncated_session", "missing_spot",
    ).join(per_day, on="trade_date", how="left").with_columns(
        (pl.col("t3_start_price") - pl.col("t3_strike")).abs().alias("t3_start_distance"),
        (pl.col("t3_end_price") - pl.col("t3_strike")).abs().alias("t3_end_distance"),
    ).with_columns(
        pl.when(pl.col("truncated_session")).then(pl.lit("truncated_session"))
        .when(pl.col("missing_spot").fill_null(False)).then(pl.lit("missing_spot"))
        .when(pl.col("t3_strike").is_null()).then(pl.lit("no_max_gex_strike"))
        .when(pl.col("t3_start_price").is_null()).then(pl.lit("no_snapshot_spot"))
        .when(pl.col("t3_end_price").is_null()).then(pl.lit("missing_end_bar"))
        .when(pl.col("t3_path_bars").fill_null(0) < min_path_bars).then(pl.lit("insufficient_path_bars"))
        .when(pl.col("t3_start_distance") == 0).then(pl.lit("start_at_strike"))
        .otherwise(pl.lit("ok")).alias("t3_reason")
    )
    ok = pl.col("t3_reason") == "ok"
    return frame.with_columns(
        pl.when(ok).then(pl.col("t3_end_distance") < pl.col("t3_start_distance")).otherwise(None).alias("t3_label"),
        pl.when(ok).then((pl.col("t3_start_distance") - pl.col("t3_end_distance")) / pl.col("t3_start_price") * 100.0)
        .otherwise(None).alias("t3_distance_reduction_pct"),
        pl.when(ok).then((pl.col("t3_path_low") <= pl.col("t3_strike")) & (pl.col("t3_path_high") >= pl.col("t3_strike")))
        .otherwise(None).alias("t3_touched_strike"),
    ).drop("truncated_session", "missing_spot").sort("snapshot_at")
