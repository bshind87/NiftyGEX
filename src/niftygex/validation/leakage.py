"""Timestamp-ordering checks. All comparisons use available_at, never bar labels."""
from __future__ import annotations

from datetime import datetime

import polars as pl


class LeakageError(AssertionError):
    """Raised when a row would be used before it is available."""


def rows_available_at(frame: pl.DataFrame, snapshot: datetime, column: str = "available_at") -> pl.DataFrame:
    """Rows usable by a snapshot at time T: available_at <= T."""
    return frame.filter(pl.col(column) <= snapshot)


def assert_available_by(frame: pl.DataFrame, snapshot_column: str, available_column: str = "available_at") -> None:
    """Every row joined into a snapshot must have been available at or before that snapshot."""
    late = frame.filter(pl.col(available_column).is_null() | (pl.col(available_column) > pl.col(snapshot_column)))
    if late.height:
        raise LeakageError(f"{late.height} rows have {available_column} after {snapshot_column}")


def assert_available_after(frame: pl.DataFrame, observed_end_column: str, available_column: str = "available_at") -> None:
    """Daily sources must become available strictly after the observation date's close."""
    early = frame.filter(pl.col(available_column).is_null() | (pl.col(available_column) <= pl.col(observed_end_column)))
    if early.height:
        raise LeakageError(f"{early.height} rows have {available_column} <= {observed_end_column}")
