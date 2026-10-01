"""Deterministic duplicate resolution with conflict diagnostics."""
from __future__ import annotations

from collections.abc import Sequence

import polars as pl

PROVENANCE_COLUMNS = ("source_file", "source_file_date", "source_row")

# Documented keep orders (see docs/schema.md). The first row per key after sorting is kept.
OPTION_ORDER: tuple[tuple[str, bool], ...] = (
    ("volume", True),
    ("open_interest", True),
    ("close", True),
    ("source_file", False),
    ("source_row", False),
)
FUTURES_ORDER: tuple[tuple[str, bool], ...] = (
    ("source_file_date", True),
    ("volume", True),
    ("open_interest", True),
    ("close", True),
    ("source_file", False),
    ("source_row", False),
)
MARKET_ORDER: tuple[tuple[str, bool], ...] = (("volume", True), ("source_row", False))


def deduplicate(
    frame: pl.DataFrame,
    keys: Sequence[str],
    order: Sequence[tuple[str, bool]],
) -> tuple[pl.DataFrame, dict[str, int], pl.DataFrame]:
    """Keep the first row per key after sorting by ``order``.

    Returns the deduplicated frame, counts, and the dropped rows with the kept row's values attached
    (``kept_*`` columns) and an ``exact`` flag for byte-identical market values.
    """
    keys = list(keys)
    order = [(column, descending) for column, descending in order if column in frame.columns]
    columns = keys + [column for column, _ in order]
    ranked = frame.sort(
        columns,
        descending=[False] * len(keys) + [descending for _, descending in order],
        nulls_last=True,
        maintain_order=True,
    ).with_columns(pl.int_range(pl.len()).over(keys).alias("_rank"))
    kept = ranked.filter(pl.col("_rank") == 0).drop("_rank")
    dropped = ranked.filter(pl.col("_rank") > 0).drop("_rank")

    values = [column for column in frame.columns if column not in keys and column not in PROVENANCE_COLUMNS]
    if dropped.height:
        kept_values = kept.join(dropped.select(keys).unique(), on=keys, how="semi").select(
            keys + [pl.col(column).alias(f"kept_{column}") for column in values + list(PROVENANCE_COLUMNS) if column in frame.columns]
        )
        dropped = dropped.join(kept_values, on=keys, how="left").with_columns(
            pl.all_horizontal([pl.col(column).eq_missing(pl.col(f"kept_{column}")) for column in values]).alias("exact")
            if values
            else pl.lit(True).alias("exact")
        )
    else:
        dropped = dropped.with_columns(pl.lit(True).alias("exact"))

    conflicting = dropped.filter(~pl.col("exact"))
    stats = {
        "before_dedup": frame.height,
        "duplicate_key_groups": dropped.select(keys).unique().height,
        "exact_duplicate_rows_removed": dropped.height - conflicting.height,
        "conflicting_key_groups": conflicting.select(keys).unique().height,
        "conflicting_extra_rows_removed": conflicting.height,
        "after_dedup": kept.height,
    }
    return kept, stats, dropped


def decided_by(dropped: pl.DataFrame, order: Sequence[tuple[str, bool]]) -> pl.Expr:
    """Name of the first keep-order column on which a dropped row differs from the kept row."""
    expression = pl.lit("identical")
    for column, _ in reversed([item for item in order if item[0] in dropped.columns]):
        expression = pl.when(~pl.col(column).eq_missing(pl.col(f"kept_{column}"))).then(pl.lit(column)).otherwise(expression)
    return expression.alias("decided_by")


def drop_conflicting_keys(frame: pl.DataFrame, keys: Sequence[str]) -> tuple[pl.DataFrame, dict[str, int], pl.DataFrame]:
    """Collapse exact duplicates; drop every row of a key whose duplicates disagree.

    Used where no value-based preference is defensible (participant OI): a conflicting key becomes
    unavailable instead of being resolved by row order.
    """
    keys = list(keys)
    values = [column for column in frame.columns if column not in keys and column not in PROVENANCE_COLUMNS]
    exact = frame.unique(subset=keys + values, keep="first", maintain_order=True)
    counts = exact.group_by(keys).agg(pl.len().alias("_n"))
    conflicting_keys = counts.filter(pl.col("_n") > 1).select(keys)
    removed = exact.join(conflicting_keys, on=keys, how="semi")
    kept = exact.join(conflicting_keys, on=keys, how="anti").sort(keys)
    stats = {
        "before_dedup": frame.height,
        "exact_duplicate_rows_removed": frame.height - exact.height,
        "conflicting_key_groups": conflicting_keys.height,
        "conflicting_rows_removed": removed.height,
        "after_dedup": kept.height,
    }
    return kept, stats, removed
