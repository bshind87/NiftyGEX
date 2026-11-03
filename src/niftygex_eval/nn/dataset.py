"""Token history and volatility state as model tensors, scaled per walk on training days only.

``tokens[d, t, k, f]``: day d (a row of the forecast table), snapshot t (index -1 is the 15:15 prediction
snapshot, lag 0), contract slot k (the token's ``token_index``), feature f. ``mask`` marks real contracts.
Missing values and padding become 0 after scaling (the training mean); token missing-input flags are features.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl


@dataclass
class TokenTensors:
    tokens: np.ndarray   # [D, L, N, F] float32, transformed but not scaled; NaN where missing
    mask: np.ndarray     # [D, L, N] bool, True = real contract
    features: list[str]


def signed_log(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


def load_token_tensors(tokens_root: Path, prediction_at: pl.Series, window: int, features: list[str],
                       signed_log_features: list[str]) -> TokenTensors:
    """Tokens of the last ``window`` snapshots before each prediction snapshot, in the order of ``prediction_at``."""
    days = pl.DataFrame({"prediction_at": prediction_at}).with_row_index("day")
    rows = (pl.scan_parquet(Path(tokens_root) / "*.parquet").filter(pl.col("lag") < window)
            .select("prediction_at", "lag", "token_index", *features).collect()
            .join(days, on="prediction_at", how="inner"))
    slots = int(rows["token_index"].max()) + 1 if rows.height else 1
    tokens = np.full((days.height, window, slots, len(features)), np.nan, dtype=np.float32)
    mask = np.zeros((days.height, window, slots), dtype=bool)
    d, t, k = rows["day"].to_numpy(), window - 1 - rows["lag"].to_numpy(), rows["token_index"].to_numpy()
    values = rows.select([pl.col(c).cast(pl.Float64) for c in features]).to_numpy()
    for j, name in enumerate(features):
        if name in signed_log_features:
            values[:, j] = signed_log(values[:, j])
    tokens[d, t, k] = values.astype(np.float32)
    mask[d, t, k] = True
    return TokenTensors(tokens, mask, list(features))


def state_matrix(table: pl.DataFrame, columns: list[str], log_columns: list[str]) -> np.ndarray:
    """[D, S] volatility-state features in table order; logs of positive levels, NaN where missing."""
    values = table.select([pl.col(c).cast(pl.Float64) for c in columns]).to_numpy()
    for j, name in enumerate(columns):
        if name in log_columns:
            with np.errstate(divide="ignore", invalid="ignore"):
                values[:, j] = np.where(values[:, j] > 0, np.log(values[:, j]), np.nan)
    return values.astype(np.float32)


@dataclass
class Scaler:
    """Per-feature mean and standard deviation from training days; applied to every split of the walk."""
    token_mean: np.ndarray
    token_std: np.ndarray
    state_mean: np.ndarray
    state_std: np.ndarray

    @classmethod
    def fit(cls, tokens: np.ndarray, mask: np.ndarray, state: np.ndarray) -> "Scaler":
        real = tokens[mask]
        token_mean, token_std = np.nanmean(real, axis=0), np.nanstd(real, axis=0)
        state_mean, state_std = np.nanmean(state, axis=0), np.nanstd(state, axis=0)
        safe = lambda std: np.where(np.isfinite(std) & (std > 0), std, 1.0)
        return cls(np.nan_to_num(token_mean), safe(token_std), np.nan_to_num(state_mean), safe(state_std))

    def transform(self, tokens: np.ndarray, mask: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scaled = (tokens - self.token_mean) / self.token_std
        scaled = np.where(mask[..., None] & np.isfinite(scaled), scaled, 0.0).astype(np.float32)
        scaled_state = (state - self.state_mean) / self.state_std
        return scaled, np.nan_to_num(scaled_state, nan=0.0).astype(np.float32)
