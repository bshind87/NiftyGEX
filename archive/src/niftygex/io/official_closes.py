"""Optional official NSE NIFTY closing-index input for T1 (data/raw/official_closes/*.csv)."""
from __future__ import annotations

from pathlib import Path

import polars as pl

REQUIRED = ("date", "close")


def load_official_closes(directory: Path) -> pl.DataFrame | None:
    """Read every CSV under ``directory`` with columns ``date`` (YYYY-MM-DD) and ``close`` (index points).

    Returns None when the directory is absent or empty. Raises on a missing column, an unparseable or
    non-positive value, or the same date with two different closes; identical repeats collapse.
    """
    if not directory.exists():
        return None
    files = sorted(directory.glob("*.csv"))
    if not files:
        return None
    frames = []
    for path in files:
        frame = pl.read_csv(path, infer_schema_length=0)
        missing = [c for c in REQUIRED if c not in frame.columns]
        if missing:
            raise ValueError(f"{path.name}: missing columns {missing}; expected {REQUIRED}")
        frames.append(frame.select(
            pl.col("date").str.strip_chars().str.to_date("%Y-%m-%d", strict=True).alias("date"),
            pl.col("close").str.strip_chars().cast(pl.Float64, strict=True).alias("official_close"),
            pl.lit(path.name).alias("source_file"),
        ))
    closes = pl.concat(frames)
    if closes.filter(pl.col("official_close") <= 0).height:
        raise ValueError("Official closes must be positive")
    conflicts = closes.group_by("date").agg(pl.col("official_close").n_unique().alias("n")).filter(pl.col("n") > 1)
    if conflicts.height:
        raise ValueError(f"Conflicting official closes on {conflicts.height} dates, e.g. {conflicts['date'][0]}")
    return closes.unique(subset=["date"], keep="first").sort("date")
