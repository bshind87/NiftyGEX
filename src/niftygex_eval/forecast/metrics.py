"""Forecast scores for a log volatility-ratio outcome: IC (primary), RMSE, QLIKE and out-of-sample R²."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import spearmanr


def ic(y: np.ndarray, prediction: np.ndarray) -> float:
    """Spearman rank correlation of prediction and outcome; NaN when the prediction is constant."""
    if y.size < 3 or np.std(prediction) == 0:
        return float("nan")
    return float(spearmanr(prediction, y).statistic)


def qlike(y: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Per-row QLIKE on variances, for log volatility ratios y (realized) and prediction (forecast).

    QLIKE = s/h - log(s/h) - 1 with s/h = (realized / forecast volatility)^2 = exp(2 (y - prediction));
    the shared trailing-RV denominator cancels. Zero for a perfect forecast; under-prediction costs more.
    """
    gap = 2.0 * (y - prediction)
    return np.exp(gap) - gap - 1.0


def regression_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    error = prediction - y
    return {"n": int(y.size), "ic": ic(y, prediction), "rmse": float(np.sqrt(np.mean(error ** 2))),
            "qlike": float(np.mean(qlike(y, prediction))), "bias": float(np.mean(error))}


def r2_vs(y: np.ndarray, prediction: np.ndarray, reference: np.ndarray) -> float:
    """Out-of-sample R² against a reference forecast: 1 - SSE(model) / SSE(reference)."""
    reference_sse = float(np.sum((reference - y) ** 2))
    return float(1.0 - np.sum((prediction - y) ** 2) / reference_sse) if reference_sse > 0 else float("nan")
