"""Classical rungs of the ladder (proposal steps 0-3): historical mean, persistence, HAR-RV, HAR + GEX, tuned LightGBM."""
from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from niftygex_eval.forecast.interface import Forecaster, matrix


class HistoricalMean(Forecaster):
    """Training-window mean of the outcome for every row."""

    name = "historical_mean"

    def fit(self, train: pl.DataFrame, outcome: str) -> "HistoricalMean":
        self.mean_ = float(train[outcome].mean())
        return self

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        return np.full(frame.height, self.mean_)


class Persistence(Forecaster):
    """Tomorrow is like today: the latest completed day's RV relative to its trailing level.

    log(previous-day RV / trailing RV); the training mean fills rows where either is missing.
    """

    name = "persistence"

    def __init__(self, recent: str, trailing: str) -> None:
        self.recent, self.trailing = recent, trailing

    def fit(self, train: pl.DataFrame, outcome: str) -> "Persistence":
        self.fallback_ = float(train[outcome].mean())
        return self

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        values = matrix(frame, [self.recent, self.trailing], log=True)
        out = values[:, 0] - values[:, 1]
        return np.where(np.isfinite(out), out, self.fallback_)


class Linear(Forecaster):
    """OLS (alpha None) or ridge on standardized columns; logs of the ``log_columns``, median imputation with
    missing indicators, all fitted on the training window only. Ridge alpha is chosen on the inner split by MSE."""

    def __init__(self, name: str, log_columns: list[str], columns: list[str], alphas: list[float] | None = None) -> None:
        self.name, self.log_columns, self.columns, self.alphas = name, log_columns, columns, alphas
        self.alpha_: float | None = None

    def _x(self, frame: pl.DataFrame) -> np.ndarray:
        parts = [matrix(frame, self.log_columns, log=True)] if self.log_columns else []
        parts += [matrix(frame, self.columns)] if self.columns else []
        return np.hstack(parts)

    def _model(self, alpha: float | None):
        regressor = LinearRegression() if alpha is None else Ridge(alpha=alpha)
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), StandardScaler(), regressor)

    def tune(self, train: pl.DataFrame, valid: pl.DataFrame, outcome: str) -> None:
        if not self.alphas:
            return
        x_train, y_train, x_valid, y_valid = self._x(train), train[outcome].to_numpy(), self._x(valid), valid[outcome].to_numpy()
        scores = {a: float(np.mean((self._model(a).fit(x_train, y_train).predict(x_valid) - y_valid) ** 2)) for a in self.alphas}
        self.alpha_ = min(scores, key=scores.get)

    def fit(self, train: pl.DataFrame, outcome: str) -> "Linear":
        self.model_ = self._model(self.alpha_).fit(self._x(train), train[outcome].to_numpy())
        return self

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        return self.model_.predict(self._x(frame))


class FallbackLinear(Forecaster):
    """HAR with extra log columns that are never filled (exploratory v3 A1, HAR-IV).

    ``levels`` lists extra-column sets from richest to plainest, ending with ``[]`` (HAR itself). Each level is an OLS
    on HAR's columns plus its extra columns, fitted on the training rows that have every one of them (positive and
    finite, since they enter as logs); a row is predicted by the richest level whose extra columns it has. HAR's own
    columns keep ``Linear``'s handling.
    """

    def __init__(self, name: str, har_columns: list[str], levels: list[list[str]]) -> None:
        if not levels or list(levels[-1]):
            raise ValueError("the last level must be [] (plain HAR), so every row gets a prediction")
        self.name, self.levels = name, [list(level) for level in levels]
        self.models = [Linear(f"{name}_level{i}", [*har_columns, *level], []) for i, level in enumerate(self.levels)]

    @staticmethod
    def has(frame: pl.DataFrame, columns: list[str]) -> np.ndarray:
        if not columns:
            return np.ones(frame.height, dtype=bool)
        values = matrix(frame, columns)
        with np.errstate(invalid="ignore"):
            return np.all(np.isfinite(values) & (values > 0), axis=1)

    def fit(self, train: pl.DataFrame, outcome: str) -> "FallbackLinear":
        for level, model in zip(self.levels, self.models):
            model.fit(train.filter(pl.Series(self.has(train, level))), outcome)
        return self

    def level_of(self, frame: pl.DataFrame) -> np.ndarray:
        """Index of the level that predicts each row (0 = richest)."""
        out = np.full(frame.height, len(self.levels) - 1)
        for i in reversed(range(len(self.levels))):
            out[self.has(frame, self.levels[i])] = i
        return out

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        level, out = self.level_of(frame), np.full(frame.height, np.nan)
        for i, model in enumerate(self.models):
            rows = level == i
            if rows.any():
                out[rows] = model.predict(frame.filter(pl.Series(rows)))
        return out


class TunedLightGBM(Forecaster):
    """LightGBM regression; parameters chosen on the inner split by MSE, then refit on the full training window."""

    name = "lightgbm"

    def __init__(self, columns: list[str], base: dict[str, Any], grid: dict[str, list[Any]]) -> None:
        self.columns, self.base = columns, base
        self.grid = [dict(zip(grid, values)) for values in product(*grid.values())]
        self.params_: dict[str, Any] = {}

    def _model(self, params: dict[str, Any]):
        from lightgbm import LGBMRegressor

        return LGBMRegressor(objective="regression", **{**self.base, **params})

    def tune(self, train: pl.DataFrame, valid: pl.DataFrame, outcome: str) -> None:
        x_train, y_train = matrix(train, self.columns), train[outcome].to_numpy()
        x_valid, y_valid = matrix(valid, self.columns), valid[outcome].to_numpy()
        scores = [float(np.mean((self._model(p).fit(x_train, y_train).predict(x_valid) - y_valid) ** 2)) for p in self.grid]
        self.params_ = self.grid[int(np.argmin(scores))]

    def fit(self, train: pl.DataFrame, outcome: str) -> "TunedLightGBM":
        self.model_ = self._model(self.params_).fit(matrix(train, self.columns), train[outcome].to_numpy())
        return self

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        return self.model_.predict(matrix(frame, self.columns))


def make_baseline(name: str, forecast: dict[str, Any], summary_columns: list[str]) -> Forecaster:
    """Build a classical forecaster from the ``forecast`` section of configs/eval.yaml."""
    har = list(forecast["har_columns"])
    if name == "historical_mean":
        return HistoricalMean()
    if name == "persistence":
        return Persistence(*forecast["persistence_columns"])
    if name == "har":
        return Linear("har", har, [])
    if name == "har_gex":
        return Linear("har_gex", har, list(forecast["gex_columns"]), alphas=list(forecast["ridge_alphas"]))
    if name == "lightgbm":
        return TunedLightGBM(summary_columns, dict(forecast["lightgbm"]), dict(forecast["tuning"]["grid"]))
    raise ValueError(f"Unknown baseline {name!r}")
