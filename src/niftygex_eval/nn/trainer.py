"""Neural models as Forecasters: per-walk scaling, Huber loss, AdamW, early stopping on the inner split, seed averaging.

``tune`` trains each seed on the inner-train rows with early stopping on the validation rows. With ``refit: true``,
``fit`` retrains on the full training window for each seed's stopping epoch (as the classical models are refit);
with ``refit: false`` the early-stopped models are used as they are. ``predict`` averages the seeds. Mini-batches are
shuffled within the training rows only; splits stay in time order.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import polars as pl
import torch

from niftygex_eval.forecast.interface import Forecaster
from niftygex_eval.nn.dataset import NeuralInputs, Scaler


def default_factory(name: str, n_token_features: int, n_state_features: int, config: dict) -> torch.nn.Module:
    from niftygex_eval.nn.models import build_model

    return build_model(name, n_token_features, n_state_features, config)


class NeuralForecaster(Forecaster):
    def __init__(self, name: str, inputs: NeuralInputs, neural: dict[str, Any], *, device: str = "cpu",
                 factory: Callable[..., torch.nn.Module] = default_factory) -> None:
        self.name, self.inputs, self.cfg, self.device, self.factory = name, inputs, neural, torch.device(device), factory
        self.training = neural["training"]
        self.epochs_: list[int] = []
        self.models_: list[torch.nn.Module] = []
        self.history_: list[dict] = []

    # ------------------------------------------------------------------ helpers
    def _rows(self, frame: pl.DataFrame) -> np.ndarray:
        return np.array([self.inputs.row[t] for t in frame["snapshot_at"].to_list()], dtype=np.int64)

    def _tensors(self, rows: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens, state = self.scaler_.transform(self.inputs.tokens[rows], self.inputs.mask[rows], self.inputs.state[rows])
        to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(self.device)
        return to(tokens), to(self.inputs.mask[rows]), to(state)

    def _new_model(self, seed: int) -> torch.nn.Module:
        torch.manual_seed(seed)
        n_features, n_state = self.inputs.tokens.shape[-1], self.inputs.state.shape[-1]
        return self.factory(self.name, n_features, n_state, dict(self.cfg["model"])).to(self.device)

    def _batches(self, n: int, generator: np.random.Generator) -> list[np.ndarray]:
        order = generator.permutation(n)
        size = int(self.training["batch_size"])
        return [order[i:i + size] for i in range(0, n, size)]

    @torch.no_grad()
    def _predict(self, model: torch.nn.Module, data: tuple[torch.Tensor, ...]) -> np.ndarray:
        model.eval()
        tokens, mask, state = data
        size = int(self.training["batch_size"]) * 4
        return np.concatenate([model(tokens[i:i + size], mask[i:i + size], state[i:i + size]).cpu().numpy()
                               for i in range(0, tokens.shape[0], size)])

    def _train(self, data: tuple[torch.Tensor, ...], y: np.ndarray, seed: int, epochs: int,
               valid: tuple[tuple[torch.Tensor, ...], np.ndarray] | None = None) -> tuple[torch.nn.Module, int, list[float]]:
        """Train for up to ``epochs``; with ``valid``, stop after ``patience`` epochs without improvement and keep the best."""
        model = self._new_model(seed)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(self.training["lr"]), weight_decay=float(self.training["weight_decay"]))
        loss_fn = torch.nn.HuberLoss(delta=float(self.training["huber_delta"]))
        target = torch.from_numpy(y.astype(np.float32)).to(self.device)
        generator = np.random.default_rng(seed)
        best, best_epoch, best_state, curve, waited = np.inf, epochs, None, [], 0
        for epoch in range(1, epochs + 1):
            model.train()
            for index in self._batches(len(y), generator):
                optimizer.zero_grad()
                index_t = torch.from_numpy(index).to(self.device)
                loss = loss_fn(model(*(part[index_t] for part in data)), target[index_t])
                loss.backward()
                optimizer.step()
            if valid is None:
                continue
            prediction = self._predict(model, valid[0])
            score = float(loss_fn(torch.from_numpy(prediction), torch.from_numpy(valid[1].astype(np.float32))))
            curve.append(score)
            if score < best - 1e-6:
                best, best_epoch, waited = score, epoch, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                waited += 1
                if waited >= int(self.training["patience"]):
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model, best_epoch, curve

    # ------------------------------------------------------------------ Forecaster
    def tune(self, train: pl.DataFrame, valid: pl.DataFrame, outcome: str) -> None:
        train_rows, valid_rows = self._rows(train), self._rows(valid)
        self.scaler_ = Scaler.fit(self.inputs.tokens[train_rows], self.inputs.mask[train_rows], self.inputs.state[train_rows])
        data, valid_data = self._tensors(train_rows), self._tensors(valid_rows)
        y, y_valid = train[outcome].to_numpy(), valid[outcome].to_numpy()
        self.epochs_, self.history_, self.models_ = [], [], []
        for seed in self.training["seeds"]:
            model, epoch, curve = self._train(data, y, int(seed), int(self.training["max_epochs"]), valid=(valid_data, y_valid))
            self.models_.append(model)
            self.epochs_.append(epoch)
            self.history_.append({"seed": int(seed), "best_epoch": epoch, "valid_curve": curve})

    def fit(self, train: pl.DataFrame, outcome: str) -> "NeuralForecaster":
        if self.models_ and not bool(self.training.get("refit", True)):
            return self  # keep the early-stopped models and the scaler fitted on their training rows
        rows = self._rows(train)
        self.scaler_ = Scaler.fit(self.inputs.tokens[rows], self.inputs.mask[rows], self.inputs.state[rows])
        data, y = self._tensors(rows), train[outcome].to_numpy()
        epochs = self.epochs_ or [int(self.training["max_epochs"])] * len(self.training["seeds"])
        self.models_ = [self._train(data, y, int(seed), epoch)[0] for seed, epoch in zip(self.training["seeds"], epochs)]
        return self

    def summary(self) -> dict[str, Any]:
        """Per-walk training record for the results store."""
        return {"stop_epoch_mean": float(np.mean(self.epochs_)) if self.epochs_ else None,
                "stop_epoch_max": int(max(self.epochs_)) if self.epochs_ else None, "seeds": len(self.models_)}

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        data = self._tensors(self._rows(frame))
        return np.mean([self._predict(model, data) for model in self.models_], axis=0)
