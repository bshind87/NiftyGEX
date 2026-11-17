"""Plain-language RQ1 summary and the ladder's step-by-step improvements, computed from the results files.

Shared by scripts/report_results.py and the presentation documents, so every
document states RQ1 the same way. Each verdict follows a fixed rule from the test statistics (no hand-typed claim):
  IC (ranking, the pre-registered score): "ahead" if the 95% interval of the per-walk gain is above 0, "behind" if
  below 0, otherwise "on par".
  RMSE and QLIKE (forecast error, supporting): "lower" or "higher" if the Diebold-Mariano p-value (squared loss for
  RMSE, QLIKE loss for QLIKE) is below alpha, otherwise "not significantly different".
The pre-registered answer itself (a significant IC gain over every classical rung) is reported unchanged.
"""
from __future__ import annotations

from typing import Any

ORDER = ["persistence", "har", "har_gex", "lightgbm", "deepsets", "attention", "attention_gru"]
NAMES = {"historical_mean": "historical mean", "persistence": "persistence", "har": "HAR-RV", "har_gex": "HAR + GEX",
         "lightgbm": "tuned LightGBM", "deepsets": "DeepSets", "attention": "self-attention", "attention_gru": "attention + GRU"}
CLASSICAL = ["persistence", "har", "har_gex", "lightgbm"]


def pfmt(p: float | None) -> str:
    return "—" if p is None else "< 0.001" if p < 0.001 else f"{p:.2g}"


def _join(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1] if names else ""


def ladder_improvements(ladder: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Per rung: mean IC and its gain over persistence, HAR and the previous rung; QLIKE and its change against HAR (%).

    Differences of per-walk means, descriptive; the paired tests are the RQ1 and RQ2 comparisons.
    """
    present = [m for m in ORDER if m in ladder and ladder[m].get("ic_mean_per_walk") is not None]
    ic = {m: ladder[m]["ic_mean_per_walk"] for m in present}
    q_har = ladder["har"]["qlike_mean_per_walk"] if "har" in ladder else None
    r_har = ladder["har"]["rmse_mean_per_walk"] if "har" in ladder else None
    rows = []
    for i, m in enumerate(present):
        q = ladder[m]["qlike_mean_per_walk"]
        rows.append({"model": m, "name": NAMES[m], "step": ORDER.index(m), "ic": ic[m],
                     "ic_vs_persistence": ic[m] - ic["persistence"] if "persistence" in ic and m != "persistence" else None,
                     "ic_vs_har": ic[m] - ic["har"] if "har" in ic and m not in ("har", "persistence") else None,
                     "ic_vs_previous": ic[m] - ic[present[i - 1]] if i else None, "previous": NAMES[present[i - 1]] if i else None,
                     "qlike": q, "qlike_vs_har_pct": 100 * (q - q_har) / q_har if q_har and m != "har" else None,
                     "rmse": ladder[m]["rmse_mean_per_walk"],
                     "rmse_vs_har_pct": 100 * (ladder[m]["rmse_mean_per_walk"] - r_har) / r_har if r_har and m != "har" else None})
    return rows


def rung_verdicts(comparisons: dict[str, dict[str, Any]], ladder: dict[str, dict[str, Any]], alpha: float, model: str = "attention_gru") -> list[dict]:
    """Attention + GRU against each classical rung: the IC and QLIKE verdicts by the module's rules."""
    out = []
    for rung in CLASSICAL:
        c = comparisons.get(f"{model}_vs_{rung}")
        if not c or not c["ic"].get("ci95"):
            continue
        low, high = c["ic"]["ci95"]
        ic_status = "ahead" if low > 0 else "behind" if high < 0 else "on par"
        def status(p, ours, theirs):
            return ("lower" if ours < theirs else "higher") if p is not None and p < alpha else "not significantly different"
        p_q, p_r = c["dm_qlike"].get("p_two_sided"), c["dm_squared"].get("p_two_sided")
        ours, theirs = ladder[model]["qlike_mean_per_walk"], ladder[rung]["qlike_mean_per_walk"]
        r_ours, r_theirs = ladder[model]["rmse_mean_per_walk"], ladder[rung]["rmse_mean_per_walk"]
        q_status, r_status = status(p_q, ours, theirs), status(p_r, r_ours, r_theirs)
        out.append({"rung": rung, "name": NAMES[rung], "ic_diff": c["ic"]["mean_diff"], "ci95": [low, high], "p": c["ic"].get("p_two_sided"),
                    "wins": c["ic"].get("a_wins"), "walks": c["ic"].get("walks"), "ic_status": ic_status,
                    "qlike_ours": ours, "qlike_theirs": theirs, "qlike_lower_pct": 100 * (theirs - ours) / theirs,
                    "dm_qlike_p": p_q, "qlike_status": q_status, "rmse_ours": r_ours, "rmse_theirs": r_theirs,
                    "rmse_lower_pct": 100 * (r_theirs - r_ours) / r_theirs, "dm_squared_p": p_r, "rmse_status": r_status})
    return out


def rq1_summary(comparisons: dict[str, dict[str, Any]], ladder: dict[str, dict[str, Any]], alpha: float, beats_every_rung: bool) -> dict[str, Any]:
    """Headline, GEX sentence, pre-registered sentence and the per-rung verdicts for attention + GRU."""
    rungs = rung_verdicts(comparisons, ladder, alpha)
    ahead = [r["name"] for r in rungs if r["ic_status"] == "ahead"]
    on_par = [r["name"] for r in rungs if r["ic_status"] == "on par"]
    behind = [r["name"] for r in rungs if r["ic_status"] == "behind"]
    lower = [r for r in rungs if r["qlike_status"] == "lower"]
    lower_rmse = [r for r in rungs if r["rmse_status"] == "lower"]
    models = [m for m in ladder if ladder[m].get("qlike_mean_per_walk") is not None]
    lowest_qlike = min(models, key=lambda m: ladder[m]["qlike_mean_per_walk"]) == "attention_gru"
    lowest_rmse = min(models, key=lambda m: ladder[m]["rmse_mean_per_walk"]) == "attention_gru"
    ours = ladder["attention_gru"]

    if beats_every_rung:
        headline = "Attention + GRU beats every classical model on ranking (IC), as the pre-registered rule requires."
    elif not behind and on_par:
        headline = (f"Attention + GRU performs as well as the best classical models: it ranks next-day volatility on par with {_join(on_par)} "
                    f"(IC {ours['ic_mean_per_walk']:.3f})" + (f" and clearly better than {_join(ahead)}" if ahead else "") + ".")
    else:
        headline = (f"Attention + GRU (IC {ours['ic_mean_per_walk']:.3f}) is behind {_join(behind)} on ranking" +
                    (f", on par with {_join(on_par)}" if on_par else "") + ".")
    def error_sentence(metric: str, lowest: bool, better: list[dict], key: str, pct: str, p: str) -> str:
        value = ours[f"{key}_mean_per_walk"]
        head = (f"the lowest {metric} of all {len(models)} models ({value:.3f})" if lowest else f"{metric} {value:.3f}")
        tail = (", significantly lower than " + _join([f"{r['name']} by {r[pct]:.0f}% (p {pfmt(r[p])})" for r in better]) if better
                else ", not significantly lower than any classical model")
        return head + tail
    loss = ("On forecast error it has " + error_sentence("RMSE", lowest_rmse, lower_rmse, "rmse", "rmse_lower_pct", "dm_squared_p") +
            "; and " + error_sentence("QLIKE", lowest_qlike, lower, "qlike", "qlike_lower_pct", "dm_qlike_p") + ".")

    gex = next((r for r in rungs if r["rung"] == "har_gex"), None)
    gex_sentence = None
    if gex:
        rank = {"ahead": "is ahead on ranking", "on par": "ties on ranking", "behind": "is behind on ranking"}[gex["ic_status"]]
        error = ({"lower": f"beats it on forecast error: QLIKE {gex['qlike_lower_pct']:.0f}% lower (p {pfmt(gex['dm_qlike_p'])})",
                  "higher": f"is behind on QLIKE (p {pfmt(gex['dm_qlike_p'])})",
                  "not significantly different": "is not significantly different on QLIKE"}[gex["qlike_status"]])
        error += (f"; RMSE {gex['rmse_lower_pct']:.1f}% lower" + (f" (p {pfmt(gex['dm_squared_p'])})" if gex["rmse_status"] == "lower"
                  else f", not significant (p {pfmt(gex['dm_squared_p'])})"))
        gex_sentence = (f"Against GEX (HAR + GEX) it {rank} ({gex['ic_diff']:+.3f} IC, 95% interval {gex['ci95'][0]:+.3f} to {gex['ci95'][1]:+.3f}) "
                        f"and {error}.")
    ruled_out = max(r["ci95"][1] for r in rungs if r["ic_status"] == "on par") if on_par else None
    rule = (("The pre-registered win rule (a significant IC gain over every classical model) is met." if beats_every_rung else
             "The pre-registered win rule (a significant IC gain over every classical model) is not met: the IC gains over "
             f"{_join(on_par)} are smaller than the test can detect" + (f" (gains above {ruled_out:+.3f} are ruled out)" if ruled_out is not None else "")
             + ". So the model does well, but not by the clear margin we hoped for."))
    short = ("beats every classical model" if beats_every_rung else
             "on par with the best classical models on ranking, best on forecast error" if lowest_qlike and not behind else
             "on par with the best classical models on ranking" if not behind else "behind some classical models on ranking")
    return {"headline": headline, "loss": loss, "gex": gex_sentence, "rule": rule, "short": short, "rungs": rungs,
            "beats_every_rung": beats_every_rung, "lowest_qlike": lowest_qlike, "lowest_rmse": lowest_rmse, "ruled_out_above": ruled_out}


METRICS_PLAIN = [
    ("RMSE", "typical size of a forecast's miss, in units of the target (log of tomorrow's volatility over its recent normal); lower is better"),
    ("QLIKE", "the standard error measure for volatility forecasts: it compares forecast and realized variance as a ratio and punishes "
              "under-predicting a volatile day more than over-predicting it by the same amount; lower is better"),
    ("IC", "ranking accuracy: whether the forecasts put calm and wild days in the right order (rank correlation, 1 = perfect, 0 = none); higher is better"),
]


def qlike_explained() -> list[str]:
    """What QLIKE is and why it is used, with a worked example computed by the project's own QLIKE function."""
    import numpy as np

    from niftygex_eval.forecast.metrics import qlike

    low = float(qlike(np.array([0.0]), np.array([np.log(0.8)]))[0])     # forecast volatility 20% too low
    high = float(qlike(np.array([0.0]), np.array([np.log(1.2)]))[0])    # forecast volatility 20% too high
    return [
        "What it is: QLIKE compares the variance that happened (s) with the variance forecast (h): QLIKE = s/h − log(s/h) − 1. "
        "It is 0 for a perfect forecast and grows with the miss; we report its average over test days, lower is better.",
        f"It punishes under-prediction more: forecasting tomorrow's volatility 20% too low costs {low:.3f}, 20% too high costs {high:.3f}, "
        f"so under-predicting costs about {low / high:.1f} times as much. Being caught out by a wild day is the costlier mistake for anyone managing risk.",
        "It measures misses as ratios, so a 20% miss counts the same on a calm day and on a wild day; RMSE on our log target does the same.",
        "It is the standard loss for comparing volatility forecasts: it is one of the few losses shown (Patton, 2011) to rank forecasts "
        "correctly even when the true volatility can only be measured with noise, as here by realized volatility from 5-minute returns.",
    ]
