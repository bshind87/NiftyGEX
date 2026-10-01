import numpy as np
import polars as pl
import pytest

from niftygex_eval.metrics import calibration, classification_metrics
from niftygex_eval.stage_a import hac_ols, trailing_mean_abs

CLASSES = ["low", "mid", "high"]


def test_perfect_and_uninformative_predictions() -> None:
    y = np.array([0, 1, 2, 0, 1, 2])
    perfect = np.eye(3)[y]
    metrics = classification_metrics(y, perfect, CLASSES)
    assert metrics["auc_macro_ovr"] == 1.0 and metrics["brier"] == 0.0 and metrics["accuracy"] == 1.0
    uniform = np.full((6, 3), 1 / 3)
    metrics = classification_metrics(y, uniform, CLASSES)
    assert metrics["auc_macro_ovr"] == pytest.approx(0.5) and metrics["brier"] == pytest.approx(2 / 3)


def test_calibration_is_zero_for_calibrated_bins() -> None:
    y = np.array([0, 0, 1, 2] * 25)
    proba = np.tile([0.5, 0.25, 0.25], (100, 1))
    result = calibration(y, proba, CLASSES, bins=10)
    assert result["ece_mean"] == pytest.approx(0.0)


def test_trailing_mean_is_causal() -> None:
    values = [1.0, 2.0, None, 100.0]
    assert trailing_mean_abs(values, window=2, min_periods=1) == [1.0, 1.5, 2.0, 100.0]


def test_hac_ols_recovers_slope() -> None:
    rng = np.random.default_rng(0)
    n = 500
    x, control = rng.normal(size=n), rng.normal(size=n)
    y = 1.0 + 0.5 * x - 0.2 * control + rng.normal(scale=0.1, size=n)
    frame = pl.DataFrame({"snapshot_at": np.arange(n), "y": y, "x": x, "c": control})
    result = hac_ols(frame, "y", ["x", "c"], maxlags=5)
    assert result["terms"]["x"]["coef"] == pytest.approx(0.5, abs=0.02) and result["terms"]["x"]["p"] < 1e-6 and result["n"] == n
    assert result["terms"]["c"]["coef"] == pytest.approx(-0.2, abs=0.02)


def test_lagged_target_only_from_previous_trading_day() -> None:
    from datetime import date

    from niftygex_eval.stage_a import add_controls

    frame = pl.DataFrame({
        "snapshot_at": [1, 2, 3], "trade_date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 4)],
        "target_date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 5)], "y": [1.0, 2.0, 3.0], "is_expiry_day": [False, True, False],
    })
    lagged = add_controls(frame, "y")["y_lagged"].to_list()
    # Row 3's predecessor targets 2024-01-03, not 2024-01-04, so no lag is used.
    assert lagged == [None, 1.0, None]


def test_calibrators_fix_overconfidence_and_keep_rows_normalized() -> None:
    from niftygex_eval.stage_b import IsotonicCalibrator, PlattCalibrator

    rng = np.random.default_rng(1)
    n = 3000
    y = rng.integers(0, 3, size=n)
    # Overconfident: always 0.9 on a class that is right only 50% of the time.
    guess = np.where(rng.random(n) < 0.5, y, (y + 1) % 3)
    proba = np.full((n, 3), 0.05)
    proba[np.arange(n), guess] = 0.9
    raw = classification_metrics(y, proba, CLASSES)["brier"]
    for calibrator in (PlattCalibrator(), IsotonicCalibrator()):
        fixed = calibrator.fit(proba[:2000], y[:2000]).transform(proba[2000:])
        assert np.allclose(fixed.sum(axis=1), 1.0)
        assert classification_metrics(y[2000:], fixed, CLASSES)["brier"] < raw


def test_paired_comparison_counts_wins() -> None:
    from niftygex_eval.stage_b import paired

    a = pl.DataFrame({"walk_id": [0, 1, 2], "auc": [0.6, 0.55, 0.5]})
    b = pl.DataFrame({"walk_id": [0, 1, 2], "auc": [0.5, 0.56, 0.45]})
    result = paired(a, b, "auc", higher_is_better=True)
    assert result["a_wins"] == 2 and result["mean_diff"] == pytest.approx((0.1 - 0.01 + 0.05) / 3)


def test_placebo_detects_attraction_and_not_noise() -> None:
    from niftygex_eval.t3_placebo import placebo_summary

    rng = np.random.default_rng(3)
    n = 2000
    start = np.full(n, 100.0)
    offset = rng.choice([-1.0, 1.0], size=n) * 0.5
    strike = start + offset
    # Pure noise path: the real strike carries no information, so actual ~ mirror ~ placebo.
    end = start + rng.normal(scale=0.4, size=n)
    noise = placebo_summary(start, end, strike, 500, np.random.default_rng(4))
    assert abs(noise["hit_rate_actual"] - noise["hit_rate_mirror"]) < 0.05
    assert noise["placebo_random_side"]["p_value_actual_ge_placebo"] > 0.01
    # Attracted path: drift toward the real strike.
    end = start + 0.3 * offset + rng.normal(scale=0.4, size=n)
    pulled = placebo_summary(start, end, strike, 500, np.random.default_rng(5))
    assert pulled["hit_rate_actual"] > pulled["hit_rate_mirror"] + 0.1
    assert pulled["placebo_random_side"]["p_value_actual_ge_placebo"] < 0.01
    assert pulled["actual_vs_mirror_sign_test"]["p_value_two_sided"] < 1e-6
