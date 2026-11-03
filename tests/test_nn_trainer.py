"""Trainer tests with a tiny stand-in model (independent of nn/models.py)."""
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch")

from niftygex_eval.forecast.runner import run_walks  # noqa: E402
from niftygex_eval.harness import assign_cycles, build_walks  # noqa: E402
from niftygex_eval.nn.dataset import NeuralInputs  # noqa: E402
from niftygex_eval.nn.trainer import NeuralForecaster  # noqa: E402

IST = ZoneInfo("Asia/Kolkata")
NEURAL = {"model": {}, "training": {"huber_delta": 1.0, "lr": 0.01, "weight_decay": 0.0, "batch_size": 32, "max_epochs": 60,
                                    "patience": 8, "refit": True, "seeds": [1, 2]}}


class MeanPool(torch.nn.Module):
    """Stand-in model: masked mean of the last snapshot's tokens, plus the state, through one linear layer."""

    def __init__(self, n_token_features: int, n_state_features: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(n_token_features + n_state_features, 1)

    def forward(self, tokens, mask, state):
        last, m = tokens[:, -1], mask[:, -1].unsqueeze(-1).float()
        pooled = (last * m).sum(1) / m.sum(1).clamp(min=1.0)
        return self.linear(torch.cat([pooled, state], dim=-1)).squeeze(-1)


def factory(name, n_token_features, n_state_features, config):
    return MeanPool(n_token_features, n_state_features)


def synthetic(days: int = 360, seed: int = 0):
    """The outcome is the mean of token feature 0 at the last snapshot plus noise; feature 1 is noise."""
    rng = np.random.default_rng(seed)
    dates = [d for d in (date(2022, 1, 3) + timedelta(days=i) for i in range(days * 2)) if d.weekday() < 5][:days]
    snaps = [datetime(d.year, d.month, d.day, 15, 15, tzinfo=IST) for d in dates]
    tokens = rng.normal(size=(days, 3, 10, 2)).astype(np.float32)
    mask = np.ones((days, 3, 10), dtype=bool)
    mask[:, :, 7:] = rng.random((days, 3, 3)) < 0.5
    signal = (tokens[:, -1, :, 0] * mask[:, -1]).sum(1) / mask[:, -1].sum(1)
    state = rng.normal(size=(days, 2)).astype(np.float32)
    table = pl.DataFrame({"trade_date": dates, "snapshot_at": snaps, "y": signal + 0.1 * rng.normal(size=days)}).with_columns(
        (pl.col("snapshot_at") + pl.duration(days=1)).alias("target_available_at"))
    table = assign_cycles(table, [d for d in dates if d.weekday() == 3])
    inputs = NeuralInputs(tokens, mask, state, {t: i for i, t in enumerate(snaps)})
    return table, inputs


def test_trainer_learns_the_signal_through_the_walk_forward_runner() -> None:
    table, inputs = synthetic()
    walks = build_walks(table["cycle"].unique().to_list(), test_cycles=8, embargo_cycles=1, min_train_cycles=30)
    _, per_walk = run_walks(table, walks, lambda: NeuralForecaster("stand_in", inputs, NEURAL, factory=factory), "y",
                            inner_validation_cycles=6, inner_embargo_cycles=1)
    assert per_walk.height >= 2 and per_walk["ic"].min() > 0.8


def test_scaler_is_fitted_on_training_rows_only() -> None:
    table, inputs = synthetic(days=120)
    inputs.tokens[100:] += 50.0  # a shift that exists only outside the training rows
    model = NeuralForecaster("stand_in", inputs, NEURAL, factory=factory)
    model.fit(table[:100], "y")
    real = inputs.tokens[:100][inputs.mask[:100]]
    assert model.scaler_.token_mean == pytest.approx(real.mean(0), abs=1e-4)


def test_seeds_make_training_reproducible_and_tune_sets_epochs() -> None:
    table, inputs = synthetic(days=150)
    train, valid, test = table[:90], table[95:120], table[120:]
    runs = []
    for _ in range(2):
        model = NeuralForecaster("stand_in", inputs, NEURAL, factory=factory)
        model.tune(train, valid, "y")
        model.fit(train, "y")
        runs.append(model.predict(test))
    assert len(model.epochs_) == 2 and all(1 <= e <= NEURAL["training"]["max_epochs"] for e in model.epochs_)
    assert runs[0].shape == (test.height,) and np.allclose(runs[0], runs[1])


def test_without_refit_the_early_stopped_models_are_kept() -> None:
    table, inputs = synthetic(days=150)
    neural = {**NEURAL, "training": {**NEURAL["training"], "refit": False}}
    model = NeuralForecaster("stand_in", inputs, neural, factory=factory)
    model.tune(table[:90], table[95:120], "y")
    tuned = list(model.models_)
    model.fit(table[:120], "y")
    assert model.models_ == tuned and model.predict(table[120:]).shape == (30,)


class AttentionStandIn(MeanPool):
    """Stand-in that also returns pooling weights: uniform over the last snapshot's real contracts, 2 heads."""

    def forward(self, tokens, mask, state, return_attention=False):
        out = super().forward(tokens, mask, state)
        if not return_attention:
            return out
        m = mask[:, -1:].float()
        weights = (m / m.sum(-1, keepdim=True).clamp(min=1.0)).unsqueeze(2).expand(-1, -1, 2, -1)
        return out, weights


def test_attention_weights_come_back_per_real_contract() -> None:
    table, inputs = synthetic(days=150)
    model = NeuralForecaster("stand_in", inputs, NEURAL, factory=lambda *a: AttentionStandIn(2, 2))
    model.tune(table[:90], table[95:120], "y")
    model.fit(table[:120], "y")
    frame = model.attention(table[120:])
    real = int(inputs.mask[120:, -1].sum())
    assert frame.height == real and set(frame["lag"].unique().to_list()) == {0}
    assert frame.group_by("snapshot_at").agg(pl.col("weight").sum())["weight"].to_numpy() == pytest.approx(1.0)
    assert {"weight_head_0", "weight_head_1"} <= set(frame.columns)


def test_models_without_weights_give_an_empty_attention_frame() -> None:
    table, inputs = synthetic(days=150)

    class NoWeights(MeanPool):
        def forward(self, tokens, mask, state, return_attention=False):
            out = super().forward(tokens, mask, state)
            return (out, None) if return_attention else out

    model = NeuralForecaster("stand_in", inputs, NEURAL, factory=lambda *a: NoWeights(2, 2))
    model.fit(table[:100], "y")
    assert model.attention(table[100:]).height == 0


def test_runner_collects_per_walk_outputs() -> None:
    table, inputs = synthetic()
    walks = build_walks(table["cycle"].unique().to_list(), test_cycles=8, embargo_cycles=1, min_train_cycles=30)
    seen = []
    _, per_walk = run_walks(table, walks, lambda: NeuralForecaster("stand_in", inputs, NEURAL, factory=factory), "y",
                            inner_validation_cycles=6, inner_embargo_cycles=1, collect=lambda model, walk, test: seen.append((walk.walk_id, test.height)))
    assert [w for w, _ in seen] == per_walk["walk_id"].to_list() and all(n > 0 for _, n in seen)
