"""The interface every forecaster implements, from persistence to the attention + GRU model."""
from __future__ import annotations

import numpy as np
import polars as pl


class Forecaster:
    """Predicts the outcome column for each row of a modeling table.

    The runner calls ``tune`` on a time-ordered inner split of the training window (choose settings or an
    early-stopping point; the test block is never visible), then ``fit`` on the full training window with
    those settings, then ``predict`` on the test block.
    """

    name = "forecaster"

    def tune(self, train: pl.DataFrame, valid: pl.DataFrame, outcome: str) -> None:
        """Choose settings on the inner split; models without settings do nothing."""

    def fit(self, train: pl.DataFrame, outcome: str) -> "Forecaster":
        raise NotImplementedError

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        raise NotImplementedError


def matrix(frame: pl.DataFrame, columns: list[str], log: bool = False) -> np.ndarray:
    """Float design matrix with nulls as NaN; ``log`` takes logs of positive values (others become NaN)."""
    values = frame.select([pl.col(c).cast(pl.Float64) for c in columns]).to_numpy()
    if log:
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.where(values > 0, np.log(values), np.nan)
    return values
