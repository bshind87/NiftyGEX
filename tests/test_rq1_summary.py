"""RQ1 plain-language summary: verdicts follow the test statistics by fixed rules (src/niftygex_eval/rq1_summary.py)."""
from niftygex_eval.rq1_summary import ladder_improvements, rq1_summary, rung_verdicts


def row(model: str, ic: float | None, rmse: float, qlike: float) -> dict:
    return {"model": model, "ic_mean_per_walk": ic, "rmse_mean_per_walk": rmse, "qlike_mean_per_walk": qlike}


LADDER = {m: row(m, ic, r, q) for m, ic, r, q in [
    ("historical_mean", None, 0.33, 0.31), ("persistence", 0.26, 0.37, 0.38), ("har", 0.43, 0.287, 0.223), ("har_gex", 0.445, 0.289, 0.231),
    ("lightgbm", 0.448, 0.299, 0.225), ("deepsets", 0.437, 0.293, 0.197), ("attention", 0.418, 0.293, 0.197), ("attention_gru", 0.447, 0.285, 0.189)]}


def comparison(diff: float, low: float, high: float, dm_sq: float, dm_q: float) -> dict:
    return {"ic": {"mean_diff": diff, "ci95": [low, high], "p_two_sided": 0.5, "a_wins": 14, "walks": 26},
            "dm_squared": {"p_two_sided": dm_sq}, "dm_qlike": {"p_two_sided": dm_q}}


COMPARISONS = {"attention_gru_vs_persistence": comparison(0.186, 0.129, 0.242, 1e-9, 1e-9),
               "attention_gru_vs_har": comparison(0.012, -0.036, 0.060, 0.57, 0.12),
               "attention_gru_vs_har_gex": comparison(0.001, -0.050, 0.053, 0.29, 0.033),
               "attention_gru_vs_lightgbm": comparison(-0.001, -0.047, 0.045, 0.015, 0.042)}


def test_verdicts_follow_the_intervals_and_dm_tests() -> None:
    by = {r["rung"]: r for r in rung_verdicts(COMPARISONS, LADDER, 0.05)}
    assert by["persistence"]["ic_status"] == "ahead" and by["har"]["ic_status"] == "on par" and by["lightgbm"]["ic_status"] == "on par"
    assert by["har_gex"]["qlike_status"] == "lower" and by["har"]["qlike_status"] == "not significantly different"
    assert by["lightgbm"]["rmse_status"] == "lower" and by["har_gex"]["rmse_status"] == "not significantly different"


def test_summary_states_on_par_lowest_error_and_the_unmet_rule() -> None:
    s = rq1_summary(COMPARISONS, LADDER, 0.05, beats_every_rung=False)
    assert s["headline"].startswith("Attention + GRU performs as well as the best classical models")
    assert s["lowest_rmse"] and s["lowest_qlike"] and "not met" in s["rule"] and abs(s["ruled_out_above"] - 0.060) < 1e-12
    assert "ties on ranking" in s["gex"] and "beats it on forecast error" in s["gex"]
    # A model behind a rung on IC is never called "as good as the best".
    worse = {**COMPARISONS, "attention_gru_vs_har": comparison(-0.05, -0.09, -0.01, 0.5, 0.5)}
    assert "is behind HAR-RV" in rq1_summary(worse, LADDER, 0.05, beats_every_rung=False)["headline"]


def test_improvements_are_differences_of_means() -> None:
    rows = {r["model"]: r for r in ladder_improvements(LADDER)}
    assert abs(rows["attention_gru"]["ic_vs_previous"] - (0.447 - 0.418)) < 1e-12
    assert abs(rows["attention_gru"]["qlike_vs_har_pct"] - 100 * (0.189 - 0.223) / 0.223) < 1e-9
    assert rows["persistence"]["ic_vs_har"] is None and rows["har"]["rmse_vs_har_pct"] is None
