"""Probabilistic classification metrics: macro one-vs-rest AUC, multiclass Brier, log loss, calibration."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((y.size, n_classes))
    out[np.arange(y.size), y] = 1.0
    return out


def classification_metrics(y: np.ndarray, proba: np.ndarray, classes: list[str]) -> dict[str, Any]:
    """``y`` holds class indices; ``proba`` rows sum to one."""
    k = len(classes)
    if y.size == 0:
        return {"n": 0}
    truth = one_hot(y, k)
    out: dict[str, Any] = {
        "n": int(y.size),
        "brier": float(np.mean(np.sum((proba - truth) ** 2, axis=1))),
        "log_loss": float(log_loss(y, proba, labels=list(range(k)))),
        "accuracy": float(np.mean(proba.argmax(axis=1) == y)),
    }
    present = np.unique(y)
    out["auc_macro_ovr"] = float(roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=list(range(k)))) if present.size == k else None
    out["auc_by_class"] = {
        name: (float(roc_auc_score(truth[:, i], proba[:, i])) if 0 < truth[:, i].sum() < y.size else None)
        for i, name in enumerate(classes)
    }
    return out


def calibration(y: np.ndarray, proba: np.ndarray, classes: list[str], bins: int) -> dict[str, Any]:
    """Class-wise reliability tables and expected calibration error (equal-width probability bins)."""
    truth = one_hot(y, len(classes))
    edges = np.linspace(0.0, 1.0, bins + 1)
    result: dict[str, Any] = {}
    for i, name in enumerate(classes):
        index = np.clip(np.digitize(proba[:, i], edges[1:-1]), 0, bins - 1)
        rows, ece = [], 0.0
        for b in range(bins):
            mask = index == b
            if not mask.any():
                continue
            mean_p, freq = float(proba[mask, i].mean()), float(truth[mask, i].mean())
            ece += mask.mean() * abs(mean_p - freq)
            rows.append({"bin": f"[{edges[b]:.1f}, {edges[b + 1]:.1f})", "n": int(mask.sum()), "mean_predicted": mean_p, "observed_frequency": freq})
        result[name] = {"ece": float(ece), "reliability": rows}
    result["ece_mean"] = float(np.mean([result[name]["ece"] for name in classes]))
    return result
