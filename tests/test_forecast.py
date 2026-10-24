from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest

from niftygex_eval.forecast.baselines import Linear, Persistence
from niftygex_eval.forecast.compare import diebold_mariano, paired
from niftygex_eval.forecast.interface import Forecaster
from niftygex_eval.forecast.metrics import qlike, r2_vs, regression_metrics
from niftygex_eval.forecast.runner import run_walks
from niftygex_eval.forecast.table import trailing_mean_abs
from niftygex_eval.harness import assign_cycles, build_walks

IST = ZoneInfo("Asia/Kolkata")
OUTCOME = "y_true"


def synthetic_table(days: int = 400, seed: int = 0) -> pl.DataFrame:
    """Weekday rows with weekly expiry cycles; ``signal`` predicts the outcome, labels resolve the next day."""
    rng = np.random.default_rng(seed)
    dates = [d for d in (date(2022, 1, 3) + timedelta(days=i) for i in range(days * 2)) if d.weekday() < 5][:days]
    signal = rng.normal(size=days)
    frame = pl.DataFrame({
        "trade_date": dates,
        "snapshot_at": [datetime(d.year, d.month, d.day, 15, 15, tzinfo=IST) for d in dates],
        "signal": signal,
        OUTCOME: signal + 0.5 * rng.normal(size=days),
    }).with_columns((pl.col("snapshot_at") + pl.duration(days=1)).alias("target_available_at"))
    expiries = [d for d in dates if d.weekday() == 3]
    return assign_cycles(frame, expiries)


def run(model_factory, table=None):
    table = synthetic_table() if table is None else table
    walks = build_walks(table["cycle"].unique().to_list(), test_cycles=8, embargo_cycles=1, min_train_cycles=20)
    return run_walks(table, walks, model_factory, OUTCOME, inner_validation_cycles=6, inner_embargo_cycles=1)


class Oracle(Forecaster):
    def fit(self, train, outcome):
        return self

    def predict(self, frame):
        return frame[OUTCOME].to_numpy()


class Random(Forecaster):
    def __init__(self):
        self.rng = np.random.default_rng(1)

    def fit(self, train, outcome):
        return self

    def predict(self, frame):
        return self.rng.normal(size=frame.height)


class Spy(Forecaster):
    """Records the dates it is tuned and trained on."""

    seen: list[tuple[set, set]] = []

    def tune(self, train, valid, outcome):
        self.tuned = set(train["trade_date"].to_list()) | set(valid["trade_date"].to_list())

    def fit(self, train, outcome):
        Spy.seen.append((self.tuned, set(train["trade_date"].to_list())))
        return self

    def predict(self, frame):
        self.tested = set(frame["trade_date"].to_list())
        Spy.seen[-1] = (*Spy.seen[-1], self.tested)
        return np.zeros(frame.height)


def test_oracle_scores_perfect_ic_and_random_scores_near_zero() -> None:
    _, oracle = run(Oracle)
    _, random = run(Random)
    assert oracle.height > 3
    assert oracle["ic"].min() == pytest.approx(1.0)
    assert oracle["rmse"].max() == pytest.approx(0.0)
    assert abs(random["ic"].mean()) < 0.15


def test_tuning_and_training_never_see_test_days() -> None:
    Spy.seen = []
    run(Spy)
    assert Spy.seen
    for tuned, trained, tested in Spy.seen:
        assert not (tuned & tested) and not (trained & tested)
        assert tuned <= trained and max(trained) < min(tested)


def test_linear_model_recovers_signal() -> None:
    _, per_walk = run(lambda: Linear("ols", [], ["signal"]))
    assert per_walk["ic"].mean() > 0.7


def test_persistence_is_log_ratio_with_fallback() -> None:
    frame = pl.DataFrame({"recent": [0.2, None, 0.1], "trailing": [0.1, 0.1, 0.1], OUTCOME: [0.0, 0.0, 0.3]})
    model = Persistence("recent", "trailing").fit(frame, OUTCOME)
    assert model.predict(frame) == pytest.approx([np.log(2.0), 0.1, 0.0])


def test_qlike_zero_when_perfect_and_asymmetric() -> None:
    y = np.array([0.1, -0.2])
    assert qlike(y, y) == pytest.approx([0.0, 0.0])
    # Under-predicting volatility by the same log gap costs more than over-predicting.
    assert qlike(np.array([0.0]), np.array([-0.3]))[0] > qlike(np.array([0.0]), np.array([0.3]))[0]
    assert regression_metrics(y, y)["qlike"] == pytest.approx(0.0)
    assert r2_vs(y, y, np.zeros(2)) == pytest.approx(1.0)


def test_paired_ci_and_dm_favour_better_model() -> None:
    a = pl.DataFrame({"walk_id": range(10), "ic": [0.3 + 0.01 * i for i in range(10)]})
    b = pl.DataFrame({"walk_id": range(10), "ic": [0.1] * 10})
    result = paired(a, b, "ic", higher_is_better=True)
    assert result["a_wins"] == 10 and result["ci95"][0] > 0 and result["p_two_sided"] < 0.01

    predictions, _ = run(Oracle)
    worse = predictions.with_columns(pl.col("prediction") + 1.0)
    dm = diebold_mariano(predictions, worse, "squared", max_lags=5)
    assert dm["mean_loss_diff"] < 0 and dm["p_two_sided"] < 0.01


def test_gex_normalization_trailing_mean_is_causal() -> None:
    values = [1.0, 2.0, None, 100.0]
    assert trailing_mean_abs(values, window=2, min_periods=1) == [1.0, 1.5, 2.0, 100.0]
