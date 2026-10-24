"""Classical forecasting ladder (proposal steps 0-3) in the walk-forward harness.

Writes one results-store run per model and outputs/results/forecast_comparison.json (per-model scores, paired
per-walk tests and Diebold-Mariano against the reference model, scores before/after the rule-change marker).
"""
from __future__ import annotations

import json

import polars as pl

from niftygex_eval.config import eval_hash, load_eval_config
from niftygex_eval.forecast.baselines import make_baseline
from niftygex_eval.forecast.compare import by_period, diebold_mariano, mean_ci, paired, r2_by_walk
from niftygex_eval.forecast.metrics import regression_metrics
from niftygex_eval.forecast.runner import run_walks
from niftygex_eval.forecast.table import forecast_table
from niftygex_eval.harness import build_walks
from niftygex_eval.results import ResultsStore


def main() -> None:
    cfg = load_eval_config()
    evaluation = cfg["eval"]
    forecast, harness = evaluation["forecast"], evaluation["harness"]
    table, outcome, summary_columns, hashes = forecast_table(cfg)
    walks = build_walks(table["cycle"].unique().to_list(), **harness)
    tuning = forecast["tuning"]
    regime = forecast["regime_column"]
    store = ResultsStore(cfg["results_root"])
    base = {"data_config_hash": hashes["features"], "eval_config_hash": eval_hash(evaluation), "stage": "forecast",
            "target": forecast["target"], "outcome": outcome, "harness": harness}

    results: dict[str, tuple[pl.DataFrame, pl.DataFrame]] = {}
    runs: dict[str, str] = {}
    for name in forecast["models"]:
        predictions, per_walk = run_walks(
            table, walks, lambda: make_baseline(name, forecast, summary_columns), outcome,
            inner_validation_cycles=int(tuning["inner_validation_cycles"]), inner_embargo_cycles=int(tuning["inner_embargo_cycles"]),
            keep=[regime])
        results[name] = (predictions, per_walk)
        spec = {**base, "model": name, "columns": summary_columns if name == "lightgbm" else None}
        pooled = regression_metrics(predictions["y"].to_numpy(), predictions["prediction"].to_numpy())
        runs[name] = store.write_run(spec, {"predictions": predictions, "walk_metrics": per_walk}, {"pooled": pooled}).name
        print(f"{name}: walks {per_walk.height}, mean IC {per_walk['ic'].mean():.3f}, pooled IC {pooled['ic']:.3f}")

    reference = forecast["reference"]
    ref_predictions, ref_walks = results[reference]
    rows, tests = [], {}
    for name, (predictions, per_walk) in results.items():
        per_walk = per_walk.join(r2_by_walk(predictions, ref_predictions), on="walk_id", how="left")
        results[name] = (predictions, per_walk)
        rows.append({"model": name, "walks": per_walk.height, "test_days": predictions.height,
                     **{f"{m}_mean_per_walk": float(per_walk[m].mean()) for m in ("ic", "rmse", "qlike", "r2_vs_reference")},
                     "ic_ci95": mean_ci(per_walk["ic"].to_numpy()), "ic_positive_walks": int((per_walk["ic"].is_finite() & (per_walk["ic"] > 0)).sum()),
                     "pooled": regression_metrics(predictions["y"].to_numpy(), predictions["prediction"].to_numpy()),
                     "by_regime": by_period(predictions, regime)})
        if name != reference:
            tests[f"{name}_vs_{reference}"] = {
                "ic": paired(per_walk, ref_walks, "ic", higher_is_better=True),
                "qlike": paired(per_walk, ref_walks, "qlike", higher_is_better=False),
                "dm_squared": diebold_mariano(predictions, ref_predictions, "squared", int(forecast["dm_max_lags"])),
                "dm_qlike": diebold_mariano(predictions, ref_predictions, "qlike", int(forecast["dm_max_lags"])),
            }
    out = {"data_config_hash": hashes["features"], "eval_config_hash": base["eval_config_hash"], "outcome": outcome,
           "reference": reference, "walks": len(walks), "summary_columns": summary_columns, "runs": runs,
           "comparison_table": rows, "paired_vs_reference": tests,
           "note": "Exploratory ladder comparisons; the pre-registered RQ1-RQ3 tests involve the neural models (configs/eval.yaml)."}
    path = cfg["results_root"] / "forecast_comparison.json"
    path.write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(path)


if __name__ == "__main__":
    main()
