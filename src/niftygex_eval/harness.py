"""Expiry-aligned walk-forward splits with an embargo; rows are never shuffled."""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from datetime import date

import polars as pl


@dataclass(frozen=True)
class Walk:
    walk_id: int
    train_cycles: tuple[date, ...]
    embargo_cycles: tuple[date, ...]
    test_cycles: tuple[date, ...]


def assign_cycles(frame: pl.DataFrame, expiry_dates: list[date], column: str = "trade_date") -> pl.DataFrame:
    """Add ``cycle``: the first option expiry on or after each row's trade date."""
    expiries = sorted(set(expiry_dates))
    mapping = {}
    for day in frame[column].unique().to_list():
        index = bisect_left(expiries, day)
        mapping[day] = expiries[index] if index < len(expiries) else None
    lookup = pl.DataFrame({column: list(mapping), "cycle": list(mapping.values())}, schema={column: pl.Date, "cycle": pl.Date})
    return frame.join(lookup, on=column, how="left")


def build_walks(cycles: list[date], *, test_cycles: int, embargo_cycles: int, min_train_cycles: int,
                train_window_cycles: int | None = None) -> list[Walk]:
    """Consecutive, non-overlapping test blocks; each trains on cycles strictly before its embargo."""
    if embargo_cycles < 1:
        raise ValueError("At least one expiry cycle of embargo is required")
    # Rows after the last known expiry have no cycle and never enter a walk.
    ordered = sorted(c for c in set(cycles) if c is not None)
    walks: list[Walk] = []
    start = min_train_cycles + embargo_cycles
    while start < len(ordered):
        test = ordered[start: start + test_cycles]
        embargo = ordered[start - embargo_cycles: start]
        train_end = start - embargo_cycles
        train_start = 0 if train_window_cycles is None else max(0, train_end - train_window_cycles)
        walks.append(Walk(len(walks), tuple(ordered[train_start:train_end]), tuple(embargo), tuple(test)))
        start += test_cycles
    return walks


def split(frame: pl.DataFrame, walk: Walk) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Train and test rows for a walk, in time order.

    Training rows must also have their target available before the first test snapshot, so a label
    that resolves inside the test period can never be learned from (purge on ``target_available_at``).
    """
    test = frame.filter(pl.col("cycle").is_in(walk.test_cycles)).sort("snapshot_at")
    train = frame.filter(pl.col("cycle").is_in(walk.train_cycles)).sort("snapshot_at")
    if test.height:
        first_test = test["snapshot_at"].min()
        train = train.filter(pl.col("target_available_at") < first_test)
    return train, test


def check_split(train: pl.DataFrame, test: pl.DataFrame, walk: Walk) -> None:
    """Raise if any test row leaks into training or ordering/embargo rules are broken."""
    overlap = train.join(test, on="snapshot_at", how="semi").height
    if overlap:
        raise AssertionError(f"walk {walk.walk_id}: {overlap} test rows appear in training")
    if train.height and test.height:
        if train["snapshot_at"].max() >= test["snapshot_at"].min():
            raise AssertionError(f"walk {walk.walk_id}: training extends into the test period")
        if train["target_available_at"].max() >= test["snapshot_at"].min():
            raise AssertionError(f"walk {walk.walk_id}: a training label resolves inside the test period")
        if set(train["cycle"].unique().to_list()) & set(walk.embargo_cycles):
            raise AssertionError(f"walk {walk.walk_id}: embargo cycles used for training")


def inner_walk(walk: Walk, validation_cycles: int, embargo_cycles: int) -> Walk | None:
    """Time-ordered inner split of a walk's training cycles: inner-train, embargo, then the last validation cycles."""
    train = walk.train_cycles
    if len(train) <= validation_cycles + embargo_cycles:
        return None
    return Walk(walk.walk_id, train[: -(validation_cycles + embargo_cycles)],
                train[-(validation_cycles + embargo_cycles): -validation_cycles], train[-validation_cycles:])
