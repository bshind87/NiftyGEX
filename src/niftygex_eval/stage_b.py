"""Stage B: classical classifiers for T2 terciles evaluated inside the walk-forward harness."""
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from niftygex_eval.harness import Walk, check_split, split
from niftygex_eval.metrics import calibration, classification_metrics


class PriorModel:
    """Predicts the training class frequencies for every row (no-skill reference)."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PriorModel":
        self.classes_ = np.unique(y)
        self.prior_ = np.array([np.mean(y == c) for c in self.classes_])
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.tile(self.prior_, (x.shape[0], 1))


def make_model(name: str, cfg: dict[str, Any]):
    if name == "prior":
        return PriorModel()
    if name in ("logistic", "vix_only"):
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(C=float(cfg["logistic"]["C"]), max_iter=int(cfg["logistic"]["max_iter"])),
        )
    if name == "lightgbm":
        return LGBMClassifier(objective="multiclass", **cfg["lightgbm"])
    raise ValueError(f"Unknown model {name!r}")


def design(frame: pl.DataFrame, columns: list[str]) -> np.ndarray:
    return frame.select([pl.col(c).cast(pl.Float64) for c in columns]).to_numpy()


def aligned_proba(model, x: np.ndarray, n_classes: int) -> np.ndarray:
    """Probabilities in class-index order even if a class was absent from training."""
    raw = model.predict_proba(x)
    out = np.zeros((x.shape[0], n_classes))
    for column, cls in enumerate(model.classes_):
        out[:, int(cls)] = raw[:, column]
    return out


def run_model(table: pl.DataFrame, walks: list[Walk], name: str, columns: list[str], cfg: dict[str, Any]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Fit per walk on training rows only; return pooled test predictions and per-walk metrics."""
    classes = list(cfg["classes"])
    index = {label: i for i, label in enumerate(classes)}
    table = table.filter(pl.col(cfg["label"]).is_not_null()).with_columns(pl.col(cfg["label"]).replace_strict(index).alias("y"))
    predictions, per_walk = [], []
    for walk in walks:
        train, test = split(table, walk)
        check_split(train, test, walk)
        if not train.height or not test.height:
            continue
        model = make_model(name, cfg).fit(design(train, columns), train["y"].to_numpy())
        proba = aligned_proba(model, design(test, columns), len(classes))
        predictions.append(test.select("snapshot_at", "trade_date", "cycle", "y").with_columns(
            pl.lit(walk.walk_id).alias("walk_id"), *[pl.Series(f"p_{c}", proba[:, i]) for i, c in enumerate(classes)]))
        metrics = classification_metrics(test["y"].to_numpy(), proba, classes)
        per_walk.append({"walk_id": walk.walk_id, "train_rows": train.height, "test_rows": test.height,
                         "train_end": train["trade_date"].max(), "test_start": test["trade_date"].min(), "test_end": test["trade_date"].max(),
                         **{k: v for k, v in metrics.items() if k != "auc_by_class"}})
    return pl.concat(predictions), pl.DataFrame(per_walk)


def pooled_summary(predictions: pl.DataFrame, classes: list[str], bins: int, groups: dict[str, pl.Expr]) -> dict[str, Any]:
    columns = [f"p_{c}" for c in classes]
    y, proba = predictions["y"].to_numpy(), predictions.select(columns).to_numpy()
    out = {"overall": classification_metrics(y, proba, classes), "calibration": calibration(y, proba, classes, bins)}
    for name, mask in groups.items():
        part = predictions.filter(mask)
        out[name] = classification_metrics(part["y"].to_numpy(), part.select(columns).to_numpy(), classes)
    return out


def inner_walk(walk: Walk, validation_cycles: int, embargo_cycles: int) -> Walk | None:
    """Time-ordered inner split of a walk's training cycles: inner-train, embargo, then the last validation cycles."""
    train = walk.train_cycles
    if len(train) <= validation_cycles + embargo_cycles:
        return None
    return Walk(walk.walk_id, train[: -(validation_cycles + embargo_cycles)],
                train[-(validation_cycles + embargo_cycles): -validation_cycles], train[-validation_cycles:])


class PlattCalibrator:
    """Multinomial logistic regression on log-probabilities (multiclass Platt scaling)."""

    def fit(self, proba: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        self.model = LogisticRegression(C=1.0, max_iter=5000).fit(np.log(np.clip(proba, 1e-6, 1.0)), y)
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        raw = self.model.predict_proba(np.log(np.clip(proba, 1e-6, 1.0)))
        out = np.zeros_like(proba)
        for column, cls in enumerate(self.model.classes_):
            out[:, int(cls)] = raw[:, column]
        return out


class IsotonicCalibrator:
    """Per-class isotonic regression of one-vs-rest outcomes on that class's probability, renormalized."""

    def fit(self, proba: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        from sklearn.isotonic import IsotonicRegression

        self.models = [IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(proba[:, k], (y == k).astype(float))
                       for k in range(proba.shape[1])]
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        out = np.column_stack([m.predict(proba[:, k]) for k, m in enumerate(self.models)])
        totals = out.sum(axis=1, keepdims=True)
        uniform = np.full_like(out, 1.0 / out.shape[1])
        return np.where(totals > 0, out / np.where(totals > 0, totals, 1.0), uniform)


def run_tuned(table: pl.DataFrame, walks: list[Walk], columns: list[str], cfg: dict[str, Any], calibrators: list[str] | None = None) -> tuple[dict[str, tuple[pl.DataFrame, pl.DataFrame]], pl.DataFrame]:
    """Regularization chosen on an inner validation split, then calibrators fitted on that split.

    Variants: ``lightgbm_tuned`` (uncalibrated inner model) and ``lightgbm_tuned_<calibrator>``. All use the
    model fitted on inner-train only, so the calibrator never sees its own training rows.
    """
    from itertools import product

    from niftygex_eval.metrics import classification_metrics as metrics_of

    tuning = {**cfg["tuning"], **({"calibrators": calibrators} if calibrators is not None else {})}
    classes = list(cfg["classes"])
    index = {label: i for i, label in enumerate(classes)}
    table = table.filter(pl.col(cfg["label"]).is_not_null()).with_columns(pl.col(cfg["label"]).replace_strict(index).alias("y"))
    grid = [dict(zip(tuning["grid"], values)) for values in product(*tuning["grid"].values())]
    variants = ["lightgbm_tuned", *[f"lightgbm_tuned_{c}" for c in tuning["calibrators"]]]
    predictions: dict[str, list] = {v: [] for v in variants}
    per_walk: dict[str, list] = {v: [] for v in variants}
    chosen = []
    for walk in walks:
        train, test = split(table, walk)
        check_split(train, test, walk)
        inner = inner_walk(walk, int(tuning["inner_validation_cycles"]), int(tuning["inner_embargo_cycles"]))
        if inner is None or not test.height:
            continue
        inner_train, validation = split(train, inner)
        check_split(inner_train, validation, inner)
        x_train, y_train = design(inner_train, columns), inner_train["y"].to_numpy()
        x_val, y_val = design(validation, columns), validation["y"].to_numpy()
        best, best_score, best_model = None, np.inf, None
        for params in grid:
            model = LGBMClassifier(objective="multiclass", **{**cfg["lightgbm"], **params}).fit(x_train, y_train)
            score = metrics_of(y_val, aligned_proba(model, x_val, len(classes)), classes)[tuning["selection_metric"]]
            if score < best_score:
                best, best_score, best_model = params, score, model
        val_proba = aligned_proba(best_model, x_val, len(classes))
        calibrators = {"platt": PlattCalibrator(), "isotonic": IsotonicCalibrator()}
        test_raw = aligned_proba(best_model, design(test, columns), len(classes))
        outputs = {"lightgbm_tuned": test_raw}
        for name in tuning["calibrators"]:
            outputs[f"lightgbm_tuned_{name}"] = calibrators[name].fit(val_proba, y_val).transform(test_raw)
        chosen.append({"walk_id": walk.walk_id, "inner_train_rows": inner_train.height, "validation_rows": validation.height,
                       f"validation_{tuning['selection_metric']}": float(best_score), **best})
        for variant, proba in outputs.items():
            predictions[variant].append(test.select("snapshot_at", "trade_date", "cycle", "y").with_columns(
                pl.lit(walk.walk_id).alias("walk_id"), *[pl.Series(f"p_{c}", proba[:, i]) for i, c in enumerate(classes)]))
            m = classification_metrics(test["y"].to_numpy(), proba, classes)
            per_walk[variant].append({"walk_id": walk.walk_id, "train_rows": inner_train.height, "test_rows": test.height,
                                      "train_end": inner_train["trade_date"].max(), "test_start": test["trade_date"].min(),
                                      "test_end": test["trade_date"].max(), **{k: v for k, v in m.items() if k != "auc_by_class"}})
    results = {v: (pl.concat(predictions[v]), pl.DataFrame(per_walk[v])) for v in variants}
    return results, pl.DataFrame(chosen)


def paired(per_walk_a: pl.DataFrame, per_walk_b: pl.DataFrame, metric: str, higher_is_better: bool) -> dict[str, Any]:
    """Per-walk paired comparison of model A against model B on ``metric``."""
    joined = per_walk_a.select("walk_id", pl.col(metric).alias("a")).join(per_walk_b.select("walk_id", pl.col(metric).alias("b")), on="walk_id").drop_nulls()
    diff = (joined["a"] - joined["b"]).to_numpy()
    if not diff.size:
        return {"walks": 0}
    sd = float(np.std(diff, ddof=1)) if diff.size > 1 else float("nan")
    wins = int(np.sum(diff > 0) if higher_is_better else np.sum(diff < 0))
    return {"walks": int(diff.size), "mean_a": float(joined["a"].mean()), "mean_b": float(joined["b"].mean()), "mean_diff": float(diff.mean()),
            "sd_diff": sd, "t_paired": float(diff.mean() / (sd / np.sqrt(diff.size))) if diff.size > 1 and sd > 0 else None, "a_wins": wins}


def run_tuned_regression(table: pl.DataFrame, walks: list[Walk], columns: list[str], cfg: dict[str, Any], outcome: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Tuned LightGBM regression per walk: parameters by inner-validation MSE, refit on the full training window.

    Per-walk metrics include the Spearman information coefficient between prediction and outcome.
    """
    from itertools import product

    from lightgbm import LGBMRegressor
    from scipy.stats import spearmanr

    tuning = cfg["tuning"]
    grid = [dict(zip(tuning["grid"], values)) for values in product(*tuning["grid"].values())]
    table = table.filter(pl.col(outcome).is_not_null() & pl.col(outcome).is_finite())
    predictions, per_walk = [], []
    for walk in walks:
        train, test = split(table, walk)
        check_split(train, test, walk)
        inner = inner_walk(walk, int(tuning["inner_validation_cycles"]), int(tuning["inner_embargo_cycles"]))
        if inner is None or not test.height:
            continue
        inner_train, validation = split(train, inner)
        check_split(inner_train, validation, inner)
        best, best_mse = None, np.inf
        for params in grid:
            model = LGBMRegressor(objective="regression", **{**cfg["lightgbm"], **params}).fit(design(inner_train, columns), inner_train[outcome].to_numpy())
            mse = float(np.mean((model.predict(design(validation, columns)) - validation[outcome].to_numpy()) ** 2))
            if mse < best_mse:
                best, best_mse = params, mse
        model = LGBMRegressor(objective="regression", **{**cfg["lightgbm"], **best}).fit(design(train, columns), train[outcome].to_numpy())
        y, prediction = test[outcome].to_numpy(), model.predict(design(test, columns))
        ic = spearmanr(prediction, y).statistic if np.std(prediction) > 0 else float("nan")
        predictions.append(test.select("snapshot_at", "trade_date", "cycle", outcome).with_columns(
            pl.lit(walk.walk_id).alias("walk_id"), pl.Series("prediction", prediction)))
        per_walk.append({"walk_id": walk.walk_id, "train_rows": train.height, "test_rows": test.height,
                         "test_start": test["trade_date"].min(), "test_end": test["trade_date"].max(),
                         "ic": float(ic), "rmse": float(np.sqrt(np.mean((prediction - y) ** 2))),
                         "validation_mse": best_mse, **best})
    return pl.concat(predictions), pl.DataFrame(per_walk)
