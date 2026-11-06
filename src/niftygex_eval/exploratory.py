"""Statistics for the exploratory experiments (configs/eval_exploratory_v2.yaml): residual IC, one-sample tests,
forecast combination, conditional IC with a walk-cluster bootstrap, and Holm over a registered family.

Everything works on results-store predictions (snapshot_at, walk_id, y, prediction) and per-walk metrics.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from scipy.stats import rankdata
from scipy.stats import t as student_t

from niftygex_eval.forecast.metrics import ic, regression_metrics
from niftygex_eval.multiple_testing import adjust_family


def one_sample(values: np.ndarray) -> dict[str, Any]:
    """Mean of per-walk values against zero: 95% CI, t, two-sided p and the number of positive walks."""
    values = values[np.isfinite(values)]
    n = values.size
    out: dict[str, Any] = {"walks": int(n)}
    if n < 2:
        return out
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    half = float(student_t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    t = mean / (sd / np.sqrt(n)) if sd > 0 else None
    return {**out, "mean": mean, "ci95": [mean - half, mean + half], "t": t,
            "p_two_sided": float(2 * student_t.sf(abs(t), n - 1)) if t is not None else None, "positive_walks": int((values > 0).sum())}


def residual_ic_by_walk(predictions: pl.DataFrame, har: pl.DataFrame) -> pl.DataFrame:
    """Per walk: IC of the predicted residual (prediction - HAR) against the actual residual (y - HAR)."""
    joined = predictions.select("snapshot_at", "walk_id", "y", "prediction").join(
        har.select("snapshot_at", pl.col("prediction").alias("har")), on="snapshot_at", how="inner")
    rows = []
    for (walk,), part in joined.group_by(["walk_id"], maintain_order=True):
        actual, predicted = (part["y"] - part["har"]).to_numpy(), (part["prediction"] - part["har"]).to_numpy()
        rows.append({"walk_id": walk, "residual_ic": ic(actual, predicted), "rows": part.height})
    return pl.DataFrame(rows).sort("walk_id")


def combine(members: dict[str, pl.DataFrame]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Equal-weight average of the members' test predictions on their common rows; returns (predictions, per-walk)."""
    names = list(members)
    joined = members[names[0]].select("snapshot_at", "walk_id", "y", pl.col("prediction").alias(names[0]))
    for name in names[1:]:
        joined = joined.join(members[name].select("snapshot_at", pl.col("prediction").alias(name)), on="snapshot_at", how="inner")
    predictions = joined.with_columns(pl.mean_horizontal(names).alias("prediction")).select("snapshot_at", "walk_id", "y", "prediction")
    per_walk = [{"walk_id": w, **regression_metrics(p["y"].to_numpy(), p["prediction"].to_numpy())}
                for (w,), p in predictions.group_by(["walk_id"], maintain_order=True)]
    return predictions, pl.DataFrame(per_walk).sort("walk_id")


def _spearman(y: np.ndarray, prediction: np.ndarray) -> float:
    if y.size < 3 or np.std(prediction) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(y), rankdata(prediction))[0, 1])


def pooled_ic_bootstrap(frame: pl.DataFrame, reps: int, seed: int) -> dict[str, Any]:
    """Pooled IC of a subset of test rows, with a 95% interval from resampling walks with replacement."""
    if frame.height < 3:
        return {"rows": frame.height, "walks": frame["walk_id"].n_unique() if frame.height else 0, "ic": None, "ci95": None}
    groups = [(p["y"].to_numpy(), p["prediction"].to_numpy()) for _, p in frame.group_by(["walk_id"])]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(reps):
        pick = rng.integers(0, len(groups), len(groups))
        y = np.concatenate([groups[i][0] for i in pick])
        draws.append(_spearman(y, np.concatenate([groups[i][1] for i in pick])))
    draws = np.array(draws)
    draws = draws[np.isfinite(draws)]
    return {"rows": frame.height, "walks": len(groups), "ic": _spearman(frame["y"].to_numpy(), frame["prediction"].to_numpy()),
            "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))] if draws.size else None}


def family(name: str, description: str, results: list[dict[str, Any]], alpha: float) -> dict[str, Any]:
    """Holm over registered tests; each result has ``test`` and ``p``."""
    return adjust_family(name, description, results, alpha)
