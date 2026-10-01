"""Lookups over the manually maintained regime calendar (configs/regimes.yaml)."""
from __future__ import annotations

from datetime import date
from typing import Any

import polars as pl


def _date(value: Any) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value))


def lot_sizes_by_expiry(regimes: dict[str, Any], expiries: list[date]) -> pl.DataFrame:
    """Units per lot and expected OI divisor for each contract expiry.

    The base schedule is keyed by ``effective_from_expiry``; ``lot_size_exceptions`` override single expiries
    whose OI mixes lots from before and after a change.
    """
    schedule = sorted((_date(item["effective_from_expiry"]), int(item["units_per_lot"])) for item in regimes["lot_sizes"])
    exceptions = {_date(item["expiry"]): item for item in regimes.get("lot_size_exceptions") or []}
    rows = []
    for expiry in sorted(set(expiries)):
        base = [lot for start, lot in schedule if start <= expiry]
        if not base:
            raise ValueError(f"No lot size configured for expiry {expiry}")
        exception = exceptions.get(expiry)
        lot = int(exception["units_per_lot"]) if exception else base[-1]
        divisor = int(exception["oi_divisor"]) if exception else lot
        rows.append((expiry, lot, divisor, exception is not None))
    return pl.DataFrame(rows, schema={"expiry_date": pl.Date, "units_per_lot": pl.Int64, "oi_divisor": pl.Int64, "lot_exception": pl.Boolean}, orient="row")


def regime_marker_flags(regimes: dict[str, Any], column: str = "trade_date") -> list[pl.Expr]:
    """One boolean ``regime_post_<name>`` column per configured marker: trade date on or after the marker date."""
    return [
        (pl.col(column) >= _date(marker["date"])).alias(f"regime_post_{marker['name']}")
        for marker in regimes.get("regime_markers") or []
    ]
