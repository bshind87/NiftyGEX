"""Token history and volatility state as model tensors, scaled per walk on training days only.

``tokens[d, t, k, f]``: day d (a row of the forecast table), snapshot t (index -1 is the 15:15 prediction
snapshot, lag 0), contract slot k (the token's ``token_index``), feature f. ``mask`` marks real contracts.
Missing values and padding become 0 after scaling (the training mean); token missing-input flags are features.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
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


@dataclass
class NeuralInputs:
    """Unscaled tensors for every row of the forecast table, looked up by ``snapshot_at``."""
    tokens: np.ndarray   # [D, L, N, F]
    mask: np.ndarray     # [D, L, N]
    state: np.ndarray    # [D, S]
    row: dict            # snapshot_at -> index into the arrays


def build_inputs(table: pl.DataFrame, tokens_root: Path, neural: dict, window: int | None = None) -> NeuralInputs:
    """Token history (last ``window`` snapshots, default ``neural.window``) and volatility state for every table row."""
    window = int(window or neural["window"])
    data = load_token_tensors(tokens_root, table["snapshot_at"], window, list(neural["token_features"]), list(neural["signed_log_features"]))
    state = state_matrix(table, list(neural["state_columns"]), list(neural["log_state_columns"]))
    return NeuralInputs(data.tokens, data.mask, state, {t: i for i, t in enumerate(table["snapshot_at"].to_list())})


def _pad_slots(data: TokenTensors, slots: int) -> TokenTensors:
    extra = slots - data.tokens.shape[2]
    if extra <= 0:
        return data
    tokens = np.concatenate([data.tokens, np.full((*data.tokens.shape[:2], extra, data.tokens.shape[3]), np.nan, np.float32)], axis=2)
    mask = np.concatenate([data.mask, np.zeros((*data.mask.shape[:2], extra), dtype=bool)], axis=2)
    return TokenTensors(tokens, mask, data.features)


def build_multiday_inputs(table: pl.DataFrame, tokens_root: Path, neural: dict, days: int, trading_dates: list[date],
                          window: int | None = None) -> NeuralInputs:
    """Exploratory E2 inputs: the 15:15 chains of the previous ``days`` trading days (oldest first), then today's last
    ``window`` snapshots. Each previous day keeps its own expiry (its lag-0 tokens); a day without tokens, or before
    the first trading date, is fully masked. All of them are available before today's snapshot.
    """
    window = int(window or neural["window"])
    features, signed = list(neural["token_features"]), list(neural["signed_log_features"])
    position = {day: i for i, day in enumerate(trading_dates)}
    blocks = []
    for k in range(days, 0, -1):
        previous = [trading_dates[position[d] - k] if d in position and position[d] >= k else None for d in table["trade_date"].to_list()]
        snapshots = [None if p is None else s - timedelta(days=(d - p).days)
                     for p, d, s in zip(previous, table["trade_date"].to_list(), table["snapshot_at"].to_list())]
        blocks.append(load_token_tensors(tokens_root, pl.Series("prediction_at", snapshots, dtype=table["snapshot_at"].dtype), 1, features, signed))
    blocks.append(load_token_tensors(tokens_root, table["snapshot_at"], window, features, signed))
    slots = max(b.tokens.shape[2] for b in blocks)
    blocks = [_pad_slots(b, slots) for b in blocks]
    state = state_matrix(table, list(neural["state_columns"]), list(neural["log_state_columns"]))
    return NeuralInputs(np.concatenate([b.tokens for b in blocks], axis=1), np.concatenate([b.mask for b in blocks], axis=1), state,
                        {t: i for i, t in enumerate(table["snapshot_at"].to_list())})


class WindowedArray:
    """Row view of per-day token histories: row r is the ``window`` snapshots ending at its snapshot (oldest first).

    ``days[d, lag]`` holds day d's history in lag order (lag 0 = the day's 15:15 snapshot). Row r reads day
    ``day[r]`` at lags ``lag0[r] + window - 1`` .. ``lag0[r]``; rows with ``valid[r]`` false (and day -1) are empty.
    Indexing with an array of rows returns a dense [rows, window, ...] array, as the trainer expects.
    """

    def __init__(self, days: np.ndarray, day: np.ndarray, lag0: np.ndarray, window: int, valid: np.ndarray, fill) -> None:
        self.days, self.day, self.lag0, self.window, self.valid, self.fill = days, day, lag0, window, valid, fill
        self.shape = (day.size, window, *days.shape[2:])

    def __len__(self) -> int:
        return self.day.size

    def __getitem__(self, rows) -> np.ndarray:
        rows = np.arange(self.day.size)[rows] if not isinstance(rows, np.ndarray) else rows
        lags = self.lag0[rows][:, None] + (self.window - 1 - np.arange(self.window))[None, :]
        out = self.days[np.maximum(self.day[rows], 0)[:, None], lags]
        empty = ~self.valid[rows] | (self.day[rows] < 0)
        out[empty] = self.fill
        return out


def build_intraday_inputs(table: pl.DataFrame, tokens_root: Path, neural: dict, window: int, expiry_choice: pl.DataFrame,
                          prediction_time: str) -> tuple[NeuralInputs, dict]:
    """Exploratory E3 inputs: for an intraday snapshot T, the last ``window`` token snapshots ending at T, sliced from
    the stored per-day token history (lag = (prediction_time - T) / 5 min).

    The stored history uses one expiry per day. A row is kept only when that expiry is the strictly causal choice at
    T (``expiry_choice``: the nearest included expiry with a Greek row at or before T); otherwise its tokens are
    masked, never replaced. Returns the inputs and counts of masked rows.
    """
    features, signed = list(neural["token_features"]), list(neural["signed_log_features"])
    hour, minute = (int(p) for p in prediction_time.split(":"))
    lag0 = ((hour * 60 + minute) - (table["snapshot_at"].dt.hour().cast(pl.Int32) * 60 + table["snapshot_at"].dt.minute().cast(pl.Int32))).to_numpy() // 5
    if (lag0 < 0).any() or (lag0 + window - 1 > 71).any():
        raise ValueError("an intraday window reaches outside the stored 09:20-15:15 token history")
    tokens = (pl.scan_parquet(Path(tokens_root) / "*.parquet").filter(pl.col("lag") >= int(lag0.min()))
              .select("trade_date", "expiry_date", "lag", "token_index", *features).collect())
    if tokens.group_by("trade_date").agg(pl.col("expiry_date").n_unique())["expiry_date"].max() != 1:
        raise ValueError("the token history must use one expiry per day")
    stored = tokens.group_by("trade_date").agg(pl.col("expiry_date").first()).rename({"expiry_date": "token_expiry"})
    days = stored["trade_date"].sort().to_list()
    position = {d: i for i, d in enumerate(days)}
    lags = int(tokens["lag"].max()) + 1
    slots = int(tokens["token_index"].max()) + 1
    history = np.full((len(days), lags, slots, len(features)), np.nan, dtype=np.float32)
    present = np.zeros((len(days), lags, slots), dtype=bool)
    d = np.array([position[x] for x in tokens["trade_date"].to_list()])
    values = tokens.select([pl.col(c).cast(pl.Float64) for c in features]).to_numpy()
    for j, name in enumerate(features):
        if name in signed:
            values[:, j] = signed_log(values[:, j])
    history[d, tokens["lag"].to_numpy(), tokens["token_index"].to_numpy()] = values.astype(np.float32)
    present[d, tokens["lag"].to_numpy(), tokens["token_index"].to_numpy()] = True

    check = table.select("snapshot_at", "trade_date").join(expiry_choice.select("snapshot_at", "expiry_date"), on="snapshot_at", how="left").join(
        stored, on="trade_date", how="left")
    causal = (check["expiry_date"] == check["token_expiry"]).fill_null(False).to_numpy()
    day = np.array([position.get(x, -1) for x in table["trade_date"].to_list()])
    state = state_matrix(table, list(neural["state_columns"]), list(neural["log_state_columns"]))
    inputs = NeuralInputs(WindowedArray(history, day, lag0, window, causal, np.nan), WindowedArray(present, day, lag0, window, causal, False),
                          state, {t: i for i, t in enumerate(table["snapshot_at"].to_list())})
    counts = {"rows": table.height, "rows_without_token_day": int((day < 0).sum()),
              "rows_masked_noncausal_expiry": int(((day >= 0) & ~causal).sum())}
    return inputs, counts
