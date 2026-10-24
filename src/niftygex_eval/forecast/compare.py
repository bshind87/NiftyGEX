"""Comparison statistics: paired per-walk tests with a 95% CI, Diebold–Mariano on daily losses, per-period scores."""
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from scipy.stats import norm
from scipy.stats import t as student_t

from niftygex_eval.forecast.metrics import qlike, r2_vs, regression_metrics


def paired(per_walk_a: pl.DataFrame, per_walk_b: pl.DataFrame, metric: str, higher_is_better: bool) -> dict[str, Any]:
    """Model A against model B on a per-walk metric: mean difference, 95% CI, paired t, two-sided p, walks won."""
    joined = per_walk_a.select("walk_id", pl.col(metric).alias("a")).join(
        per_walk_b.select("walk_id", pl.col(metric).alias("b")), on="walk_id").drop_nulls().drop_nans()
    diff = (joined["a"] - joined["b"]).to_numpy()
    n = diff.size
    out: dict[str, Any] = {"metric": metric, "walks": int(n)}
    if n < 2:
        return out
    mean, sd = float(diff.mean()), float(diff.std(ddof=1))
    half = float(student_t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    t = mean / (sd / np.sqrt(n)) if sd > 0 else None
    return {**out, "mean_a": float(joined["a"].mean()), "mean_b": float(joined["b"].mean()), "mean_diff": mean,
            "ci95": [mean - half, mean + half], "t_paired": t,
            "p_two_sided": float(2 * student_t.sf(abs(t), n - 1)) if t is not None else None,
            "a_wins": int(np.sum(diff > 0) if higher_is_better else np.sum(diff < 0))}


def mean_ci(values: np.ndarray) -> list[float] | None:
    """Mean with a two-sided 95% t interval; None for fewer than two finite values."""
    values = values[np.isfinite(values)]
    if values.size < 2:
        return None
    half = float(student_t.ppf(0.975, values.size - 1) * values.std(ddof=1) / np.sqrt(values.size))
    return [float(values.mean()) - half, float(values.mean()) + half]


def diebold_mariano(predictions_a: pl.DataFrame, predictions_b: pl.DataFrame, loss: str, max_lags: int) -> dict[str, Any]:
    """DM test on pooled test-day loss differentials (A minus B; negative favours A), Newey–West variance."""
    joined = predictions_a.select("snapshot_at", "y", pl.col("prediction").alias("a")).join(
        predictions_b.select("snapshot_at", pl.col("prediction").alias("b")), on="snapshot_at").sort("snapshot_at")
    y, a, b = (joined[c].to_numpy() for c in ("y", "a", "b"))
    if loss == "squared":
        d = (a - y) ** 2 - (b - y) ** 2
    elif loss == "qlike":
        d = qlike(y, a) - qlike(y, b)
    else:
        raise ValueError(f"Unknown loss {loss!r}")
    n = d.size
    centred = d - d.mean()
    variance = float(np.dot(centred, centred) / n)
    for lag in range(1, min(max_lags, n - 1) + 1):
        weight = 1.0 - lag / (max_lags + 1)
        variance += 2.0 * weight * float(np.dot(centred[lag:], centred[:-lag]) / n)
    stat = float(d.mean() / np.sqrt(variance / n)) if variance > 0 else None
    return {"loss": loss, "days": int(n), "mean_loss_diff": float(d.mean()), "dm_stat": stat,
            "p_two_sided": float(2 * norm.sf(abs(stat))) if stat is not None else None}


def r2_by_walk(predictions: pl.DataFrame, reference: pl.DataFrame) -> pl.DataFrame:
    """Per-walk out-of-sample R² of a model against a reference model's predictions on the same rows."""
    joined = predictions.join(reference.select("snapshot_at", pl.col("prediction").alias("reference")), on="snapshot_at")
    rows = [{"walk_id": walk, "r2_vs_reference": r2_vs(part["y"].to_numpy(), part["prediction"].to_numpy(), part["reference"].to_numpy())}
            for walk, part in ((k[0], g) for k, g in joined.group_by("walk_id", maintain_order=True))]
    return pl.DataFrame(rows)


def by_period(predictions: pl.DataFrame, column: str) -> list[dict[str, Any]]:
    """Pooled scores for each value of ``column`` (for example the post-rule-change flag)."""
    return [{column: value, **regression_metrics(part["y"].to_numpy(), part["prediction"].to_numpy())}
            for (value,), part in predictions.group_by(column, maintain_order=True)]
