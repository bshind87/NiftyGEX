"""HAR residual models (exploratory E1): a base forecaster learns r = y - HAR forecast; predictions are HAR + r_hat.

HAR is refitted inside every (inner) training window the runner passes in: in-sample residuals of that window
train the base model, and its out-of-sample forecasts give the residual target of the validation and test rows.
The base model sees the residual as its outcome and, optionally, the HAR forecast itself as an input column.
A neural base without refit keeps the models trained on the inner window's residuals; test predictions then add
them to the full training window's HAR forecast (the HAR run of the main ladder on the same rows).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import polars as pl

from niftygex_eval.forecast.interface import Forecaster

RESIDUAL, HAR_FORECAST = "har_residual", "har_forecast"


class ResidualForecaster(Forecaster):
    def __init__(self, name: str, base: Forecaster, make_har: Callable[[], Forecaster]) -> None:
        self.name, self.base, self.make_har = name, base, make_har

    @staticmethod
    def _with_residual(har: Forecaster, frame: pl.DataFrame, outcome: str | None) -> pl.DataFrame:
        forecast = pl.Series(HAR_FORECAST, np.asarray(har.predict(frame), dtype=np.float64))
        frame = frame.with_columns(forecast)
        return frame.with_columns((pl.col(outcome) - pl.col(HAR_FORECAST)).alias(RESIDUAL)) if outcome else frame

    def tune(self, train: pl.DataFrame, valid: pl.DataFrame, outcome: str) -> None:
        har = self.make_har().fit(train, outcome)
        self.base.tune(self._with_residual(har, train, outcome), self._with_residual(har, valid, outcome), RESIDUAL)

    def fit(self, train: pl.DataFrame, outcome: str) -> "ResidualForecaster":
        self.har_ = self.make_har().fit(train, outcome)
        self.base.fit(self._with_residual(self.har_, train, outcome), RESIDUAL)
        return self

    def predict_residual(self, frame: pl.DataFrame) -> np.ndarray:
        return np.asarray(self.base.predict(self._with_residual(self.har_, frame, None)))

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        return np.asarray(self.har_.predict(frame)) + self.predict_residual(frame)

    def summary(self) -> dict:
        return getattr(self.base, "summary", dict)()
