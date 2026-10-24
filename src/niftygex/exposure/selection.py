"""Row and expiry selection rules for Greeks and total GEX."""
from __future__ import annotations

from typing import Any

import polars as pl

# Prediction targets built by the data layer; every stage loops over this tuple.
TARGETS = ("t2",)


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
    if target not in TARGETS:
        raise ValueError(f"Unknown target {target!r}; expected one of {TARGETS}")
    days = (pl.col("expiry_date") - pl.col("trade_date")).dt.total_days()
    include = days.is_between(0, int(gex["max_days_to_expiry"]), closed="both")
    if not bool(gex["same_day_expiry_in_total_gex"][target]):
        include = include & (days > 0)
    return include
