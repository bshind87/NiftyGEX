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


def test_tree_shap_is_additive_to_raw_margin() -> None:
    from lightgbm import LGBMClassifier

    from niftygex_eval.stage_b import shap_summary, tree_shap

    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 3))
    y = (x[:, 0] > 0).astype(int) + (x[:, 1] > 0.5).astype(int)
    rows = pl.DataFrame({"snapshot_at": np.arange(300), "a": x[:, 0], "b": x[:, 1], "c": x[:, 2]})
    model = LGBMClassifier(n_estimators=20, verbose=-1, random_state=0).fit(x, y)
    shap = tree_shap(model, rows, ["a", "b", "c"], CLASSES, walk_id=0)
    raw = model.predict(x, raw_score=True)
    contrib = np.asarray(model.predict(x, pred_contrib=True))
    bias = contrib[:, 3::4]
    for k, name in enumerate(CLASSES):
        total = shap.filter(pl.col("output") == name).group_by("snapshot_at").agg(pl.col("shap").sum()).sort("snapshot_at")["shap"].to_numpy()
        assert np.allclose(total + bias[:, k], raw[:, k])
    summary = shap_summary(shap).filter(pl.col("output") == "all_outputs")
    assert summary["feature"][0] in ("a", "b") and summary["feature"][-1] == "c"


def test_listed_strike_comparators_and_snapped_mirror() -> None:
    from niftygex_eval.t3_listed import comparators, compare, strike_ranks

    snapshot = 1
    chain = pl.DataFrame({
        "snapshot_at": [snapshot] * 6, "strike_price": [21900.0, 21950.0, 22000.0, 22050.0, 22100.0, 22150.0],
        "gex_absolute": [5.0, 3.0, 9.0, 4.0, 2.0, 1.0], "spot": [22010.0] * 6,
    })
    ranks = strike_ranks(chain)
    assert ranks.filter(pl.col("gex_rank") == 1)["strike_price"].item() == 22000.0
    days = pl.DataFrame({"snapshot_at": [snapshot], "trade_date": [1], "t3_start_price": [22010.0], "t3_end_price": [22003.0], "t3_strike": [22000.0]})
    frame = comparators(days, ranks).row(0, named=True)
    # Ranks 2-5 are 21900 (110 away), 22050 (40), 21950 (60), 22100 (90): 22050 best matches the 10-point distance.
    assert frame["matched_rank_strike"] == 22050.0
    # Mirror of 22000 through 22010 is 22020, snapped to the listed grid -> 22000 (the max-GEX strike itself).
    assert frame["snapped_mirror_strike"] == 22000.0
    result = compare(comparators(days, ranks), "snapped_mirror_strike")
    assert result["days"] == 0 and result["excluded_days"] == 1
    matched = compare(comparators(days, ranks), "matched_rank_strike")
    assert matched["hit_rate_max_gex"] == 1.0 and matched["hit_rate_comparator"] == 0.0


def test_holm_adjustment_known_values() -> None:
    from niftygex_eval.multiple_testing import adjust_family, holm

    with_missing = holm([0.01, 0.04, 0.03, None])
    assert with_missing[:3] == pytest.approx([0.03, 0.06, 0.06]) and with_missing[3] is None
    adjusted = holm([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])
    family = adjust_family("F", "x", [{"test": "a", "p": 0.01}, {"test": "b", "p": 0.04}, {"test": "c", "p": 0.03}], 0.05)
    assert [m["reject_holm"] for m in family["members"]] == [True, False, False]
