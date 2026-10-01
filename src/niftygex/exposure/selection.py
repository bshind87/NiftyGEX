"""Row and expiry selection rules for Greeks and total GEX."""
from __future__ import annotations

from typing import Any

import polars as pl

TARGETS = ("t1", "t2", "t3")


def greeks_input() -> pl.Expr:
    """Option rows that may enter Greeks: eligible with strictly positive time to expiry."""
    return pl.col("eligible") & (pl.col("time_to_expiry") > 0)


def total_gex_expiries(target: str, gex: dict[str, Any]) -> pl.Expr:
    """Expiries that enter total GEX for a target's snapshots.

    An expiry enters when 0 <= days to expiry <= ``max_days_to_expiry``. Contracts expiring on the
    snapshot's trade date are kept only when ``same_day_expiry_in_total_gex[target]`` is true
    (T1 keeps them for expiry-day pinning; T2 excludes them).
    """
    if target not in TARGETS:
        raise ValueError(f"Unknown target {target!r}; expected one of {TARGETS}")
    days = (pl.col("expiry_date") - pl.col("trade_date")).dt.total_days()
    include = days.is_between(0, int(gex["max_days_to_expiry"]), closed="both")
    if not bool(gex["same_day_expiry_in_total_gex"][target]):
        include = include & (days > 0)
    return include
