"""Run any forecaster through the walk-forward harness: inner tuning split, refit, test predictions, per-walk scores."""
from __future__ import annotations

from typing import Callable

import polars as pl

from niftygex_eval.forecast.interface import Forecaster
from niftygex_eval.forecast.metrics import regression_metrics
from niftygex_eval.harness import Walk, check_split, inner_walk, split

KEEP = ["snapshot_at", "trade_date", "cycle"]


def run_walks(table: pl.DataFrame, walks: list[Walk], make_model: Callable[[], Forecaster], outcome: str, *,
              inner_validation_cycles: int, inner_embargo_cycles: int, keep: list[str] | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Fresh model per walk; returns (test predictions with ``y`` and ``prediction``, per-walk metrics).

    ``table`` must already be restricted to rows with a finite outcome. ``keep`` adds columns (for example a
    regime flag) to the prediction rows for later breakdowns.
    """
    predictions, per_walk = [], []
    for walk in walks:
        train, test = split(table, walk)
        check_split(train, test, walk)
        inner = inner_walk(walk, inner_validation_cycles, inner_embargo_cycles)
        if inner is None or not test.height:
            continue
        inner_train, valid = split(train, inner)
        check_split(inner_train, valid, inner)
        model = make_model()
        model.tune(inner_train, valid, outcome)
        model.fit(train, outcome)
        y, prediction = test[outcome].to_numpy(), model.predict(test)
        predictions.append(test.select(*KEEP, *(keep or [])).with_columns(
            pl.lit(walk.walk_id).alias("walk_id"), pl.Series("y", y), pl.Series("prediction", prediction)))
        per_walk.append({"walk_id": walk.walk_id, "train_rows": train.height, "test_rows": test.height,
                         "test_start": test["trade_date"].min(), "test_end": test["trade_date"].max(),
                         **regression_metrics(y, prediction)})
    return pl.concat(predictions), pl.DataFrame(per_walk)
