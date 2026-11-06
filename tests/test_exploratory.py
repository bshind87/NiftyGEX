"""Exploratory E1-E3 eval pieces: residual wrapper, extra models, loaders, past-only normalization, resume, statistics."""
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest

from niftygex_eval.exploratory import combine, one_sample, pooled_ic_bootstrap, residual_ic_by_walk
from niftygex_eval.forecast.baselines import Linear
from niftygex_eval.forecast.interface import Forecaster
from niftygex_eval.forecast.intraday import normalize_gex_by_clock
from niftygex_eval.forecast.residual import HAR_FORECAST, RESIDUAL, ResidualForecaster
from niftygex_eval.forecast.runner import run_walks
from niftygex_eval.harness import assign_cycles, build_walks

IST = ZoneInfo("Asia/Kolkata")


def daily_table(days: int = 240, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [d for d in (date(2022, 1, 3) + timedelta(days=i) for i in range(days * 2)) if d.weekday() < 5][:days]
    a, b = rng.normal(size=days), rng.normal(size=days)
    table = pl.DataFrame({"trade_date": dates, "snapshot_at": [datetime(d.year, d.month, d.day, 15, 15, tzinfo=IST) for d in dates],
                          "a": np.exp(a), "b": b, "y": 0.8 * a + 0.5 * b + 0.05 * rng.normal(size=days)})
    table = table.with_columns((pl.col("snapshot_at") + pl.duration(days=1)).alias("target_available_at"))
    return assign_cycles(table, [d for d in dates if d.weekday() == 3])


class Recorder(Forecaster):
    """Linear on column b; records the outcome column it was trained on."""

    def __init__(self) -> None:
        self.model, self.seen = Linear("b", [], ["b"]), {}

    def tune(self, train, valid, outcome):
        self.seen["tune"] = (outcome, train[outcome].to_numpy().copy(), "y" in valid.columns)

    def fit(self, train, outcome):
        self.seen["fit"] = (outcome, train[outcome].to_numpy().copy(), train[HAR_FORECAST].to_numpy().copy())
        self.model.fit(train, outcome)
        return self

    def predict(self, frame):
        return self.model.predict(frame)


def test_residual_wrapper_trains_on_in_sample_har_residuals_and_adds_har_back() -> None:
    table = daily_table()
    train, test = table[:180], table[180:]
    base = Recorder()
    model = ResidualForecaster("residual", base, lambda: Linear("har", ["a"], []))
    model.tune(train[:120], train[130:], "y")
    model.fit(train, "y")
    outcome, residual, har_in_sample = base.seen["fit"]
    assert outcome == RESIDUAL
    assert residual == pytest.approx(train["y"].to_numpy() - har_in_sample)
    har = Linear("har", ["a"], []).fit(train, "y")
    assert har_in_sample == pytest.approx(har.predict(train))
    prediction = model.predict(test.drop("y"))          # never needs the test outcome
    assert prediction == pytest.approx(har.predict(test) + base.predict(test))
    assert np.corrcoef(prediction, test["y"].to_numpy())[0, 1] > 0.99


def test_runner_resumes_from_checkpointed_walks(tmp_path) -> None:
    table = daily_table()
    walks = build_walks(table["cycle"].unique().to_list(), test_cycles=8, embargo_cycles=1, min_train_cycles=20)
    make = lambda: Linear("ols", ["a"], ["b"])
    first = run_walks(table, walks, make, "y", inner_validation_cycles=4, inner_embargo_cycles=1, checkpoint=tmp_path)

    class Broken(Forecaster):
        def fit(self, train, outcome):
            raise AssertionError("a checkpointed walk was retrained")

    again = run_walks(table, walks, Broken, "y", inner_validation_cycles=4, inner_embargo_cycles=1, checkpoint=tmp_path)
    assert first[0].equals(again[0]) and first[1].equals(again[1])
    with pytest.raises(ValueError):
        run_walks(table, walks, make, "y", inner_validation_cycles=4, inner_embargo_cycles=1, checkpoint=tmp_path, collect=lambda *a: None)


def test_clock_normalized_gex_uses_past_days_only() -> None:
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(6)]
    rows = [(datetime(d.year, d.month, d.day, h, 15, tzinfo=IST), 10.0 * (i + 1), 1.0, -1.0) for i, d in enumerate(days) for h in (10, 11)]
    frame = pl.DataFrame(rows, schema=["snapshot_at", "gex_total_absolute", "gex_total_naive", "gex_total_inferred"], orient="row")
    out = normalize_gex_by_clock(frame, window=3, min_periods=2)
    last = out.filter(pl.col("snapshot_at") == datetime(2024, 1, 6, 10, 15, tzinfo=IST)).row(0, named=True)
    assert last["gex_scale"] == pytest.approx(np.mean([30.0, 40.0, 50.0]))          # days 3-5, not day 6 (60)
    assert last["gex_norm_naive"] == pytest.approx(1.0 / 40.0)
    assert out.filter(pl.col("snapshot_at").dt.date() == date(2024, 1, 2))["gex_scale"].null_count() == 2


def test_exploratory_statistics() -> None:
    stats = one_sample(np.array([0.1, 0.2, 0.15, 0.05, np.nan]))
    assert stats["walks"] == 4 and stats["mean"] == pytest.approx(0.125) and stats["ci95"][0] > 0 and stats["positive_walks"] == 4
    snaps = [datetime(2024, 1, 1, 15, 15, tzinfo=IST) + timedelta(days=i) for i in range(40)]
    y = np.linspace(-1, 1, 40)
    a = pl.DataFrame({"snapshot_at": snaps, "walk_id": [i // 10 for i in range(40)], "y": y, "prediction": y})
    b = a.with_columns(pl.lit(0.0).alias("prediction"))
    predictions, per_walk = combine({"a": a, "b": b})
    assert predictions["prediction"].to_numpy() == pytest.approx(y / 2) and per_walk.height == 4
    residual = residual_ic_by_walk(a.with_columns((pl.col("prediction") * 2).alias("prediction")), b)
    assert residual["residual_ic"].to_numpy() == pytest.approx(1.0)
    boot = pooled_ic_bootstrap(a, reps=50, seed=1)
    assert boot["ic"] == pytest.approx(1.0) and boot["walks"] == 4


torch = pytest.importorskip("torch")
from niftygex_eval.nn.dataset import NeuralInputs, WindowedArray, build_multiday_inputs  # noqa: E402
from niftygex_eval.nn.extra_models import build_extra_model  # noqa: E402

CONFIG = {"d_model": 16, "heads": 2, "layers": 1, "ff_dim": 32, "dropout": 0.0, "gru_hidden": 8, "head_hidden": 8}


def test_state_mlp_ignores_tokens_and_multiday_ignores_masked_days() -> None:
    torch.manual_seed(0)
    state = torch.randn(4, 3)
    model = build_extra_model("state_mlp", 5, 3, CONFIG).eval()
    tokens, mask = torch.randn(4, 2, 6, 5), torch.ones(4, 2, 6, dtype=torch.bool)
    assert torch.allclose(model(tokens, mask, state), model(torch.randn(4, 2, 6, 5), mask, state))
    multi = build_extra_model("multiday_attention_gru", 5, 3, {**CONFIG, "multiday_days": 2}).eval()
    tokens, mask = torch.randn(4, 5, 6, 5), torch.ones(4, 5, 6, dtype=torch.bool)
    mask[:, 0] = False                                   # the oldest previous day has no chain
    changed = tokens.clone()
    changed[:, 0] = torch.randn(4, 6, 5) * 100
    assert multi(tokens, mask, state).shape == (4,)
    assert torch.allclose(multi(tokens, mask, state), multi(changed, mask, state), atol=1e-5)


def test_frame_state_columns_enter_the_model() -> None:
    from niftygex_eval.nn.trainer import NeuralForecaster

    table = daily_table(days=150).with_columns(pl.col("b").alias(HAR_FORECAST))
    rng = np.random.default_rng(0)
    inputs = NeuralInputs(rng.normal(size=(150, 1, 4, 2)).astype(np.float32), np.ones((150, 1, 4), bool),
                          np.zeros((150, 0), np.float32), {t: i for i, t in enumerate(table["snapshot_at"].to_list())})
    neural = {"model": CONFIG, "training": {"huber_delta": 1.0, "lr": 0.01, "weight_decay": 0.0, "batch_size": 32, "max_epochs": 40,
                                            "patience": 8, "refit": False, "seeds": [1]}}
    model = NeuralForecaster("state_mlp", inputs, neural, factory=build_extra_model, frame_state=[HAR_FORECAST])
    model.tune(table[:90], table[95:120], "y")
    assert model.models_[0].body[0].in_features == 1
    assert np.corrcoef(model.predict(table[120:]), table[120:]["b"].to_numpy())[0, 1] > 0.9


def test_windowed_array_slices_the_window_ending_at_each_snapshot() -> None:
    days = np.arange(2 * 72, dtype=np.float32).reshape(2, 72, 1, 1)    # value = 72 * day + lag
    view = WindowedArray(days, np.array([0, 1, -1, 1]), np.array([24, 60, 24, 0]), 3, np.array([True, True, True, False]), np.nan)
    window = view[np.array([0, 1, 2, 3])][..., 0, 0]
    assert view.shape == (4, 3, 1, 1)
    assert window[0].tolist() == [26.0, 25.0, 24.0]          # oldest first, ending at the snapshot's lag
    assert window[1].tolist() == [72 + 62.0, 72 + 61.0, 72 + 60.0]
    assert np.isnan(window[2]).all() and np.isnan(window[3]).all()   # no token day; non-causal expiry


def test_multiday_inputs_use_previous_trading_days_only(tmp_path) -> None:
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(5)]
    rows = [(datetime(d.year, d.month, d.day, 15, 15, tzinfo=IST), 0, 0, float(i)) for i, d in enumerate(days)]
    pl.DataFrame(rows, schema={"prediction_at": pl.Datetime("us", "Asia/Kolkata"), "lag": pl.Int32, "token_index": pl.Int32, "iv": pl.Float64},
                 orient="row").write_parquet(tmp_path / "tokens.parquet")
    table = pl.DataFrame({"trade_date": days[2:], "snapshot_at": [r[0] for r in rows[2:]], "atm_iv": [0.1, 0.1, 0.1]})
    neural = {"window": 1, "token_features": ["iv"], "signed_log_features": [], "state_columns": ["atm_iv"], "log_state_columns": []}
    inputs = build_multiday_inputs(table, tmp_path, neural, days=2, trading_dates=days)
    # Row for day 2: previous days 0 and 1 (oldest first), then today's lag-0 snapshot.
    assert inputs.tokens[0, :, 0, 0].tolist() == [0.0, 1.0, 2.0]
    assert inputs.tokens[2, :, 0, 0].tolist() == [2.0, 3.0, 4.0]
