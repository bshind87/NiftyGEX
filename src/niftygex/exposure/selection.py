"""Row and expiry selection rules for Greeks and total GEX."""
from __future__ import annotations

from typing import Any

import polars as pl

# Prediction targets built by the data layer; every stage loops over this tuple.
TARGETS = ("t2",)
# Exploratory E3 intraday target, built only by the stages' --grid intraday mode (configs/intraday.yaml).
INTRADAY_TARGET = "t_intraday"


def greeks_input() -> pl.Expr:
    """Option rows that may enter Greeks: eligible with strictly positive time to expiry."""
    return pl.col("eligible") & (pl.col("time_to_expiry") > 0)


def nearest_expiry_by_day(pairs: pl.DataFrame, target: str, gex: dict[str, Any]) -> pl.DataFrame:
    """For each trade date, the nearest expiry that ``total_gex_expiries`` includes for ``target``.

    ``pairs`` lists the (trade_date, expiry_date) combinations that have Greek rows, so an expiry missing from
    the source on a day (a truncated file, an unlisted weekly) is skipped. Token sequences use one expiry for
    the whole day, so a model's history never switches contracts intraday.
    """
    return (pairs.select("trade_date", "expiry_date").unique().filter(total_gex_expiries(target, gex))
            .group_by("trade_date").agg(pl.col("expiry_date").min())
            .with_columns(pl.lit(target).alias("target")).sort("trade_date"))


def total_gex_expiries(target: str, gex: dict[str, Any]) -> pl.Expr:
    """Expiries that enter total GEX for a target's snapshots.

    An expiry enters when 0 <= days to expiry <= ``max_days_to_expiry``. Contracts expiring on the
    snapshot's trade date are kept only when ``same_day_expiry_in_total_gex[target]`` is true (T2 excludes them).
    """
    if target not in gex["same_day_expiry_in_total_gex"]:
        raise ValueError(f"Unknown target {target!r}; expected one of {sorted(gex['same_day_expiry_in_total_gex'])}")
    days = (pl.col("expiry_date") - pl.col("trade_date")).dt.total_days()
    include = days.is_between(0, int(gex["max_days_to_expiry"]), closed="both")
    if not bool(gex["same_day_expiry_in_total_gex"][target]):
        include = include & (days > 0)
    return include


def causal_nearest_expiry(first_seen: pl.DataFrame, snapshots: pl.DataFrame, target: str, gex: dict[str, Any]) -> pl.DataFrame:
    """Per snapshot: the nearest expiry that ``total_gex_expiries`` includes and that has a Greek row available at or
    before the snapshot (strictly causal; an expiry whose first row of the day arrives later is not chosen yet).

    ``first_seen``: trade_date, expiry_date, first_available_at (earliest Greek row of that expiry on that day).
    ``snapshots``: trade_date, snapshot_at. Snapshots without any such expiry get a null ``expiry_date``.
    """
    candidates = snapshots.select("trade_date", "snapshot_at").join(first_seen, on="trade_date", how="inner").filter(
        (pl.col("first_available_at") <= pl.col("snapshot_at")) & total_gex_expiries(target, gex))
    chosen = candidates.group_by("snapshot_at").agg(pl.col("expiry_date").min())
    return snapshots.select("trade_date", "snapshot_at").join(chosen, on="snapshot_at", how="left").sort("snapshot_at")
