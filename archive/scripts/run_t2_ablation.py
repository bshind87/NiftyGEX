"""T2 ablation: A (vol-state/calendar), B (A + participant), C (B + GEX/levels), C1 (B + GEX + inferred sign), C2 (B + levels).

Classification: tuned LightGBM + Platt on terciles. Regression: tuned LightGBM on log(next-day RV ratio).
Writes one results-store run per (task, feature set) and outputs/results/t2_ablation.json.
"""
from __future__ import annotations

import json

import numpy as np
import polars as pl

from niftygex_eval.config import eval_hash, load_eval_config
from niftygex_eval.data import modeling_table
from niftygex_eval.harness import build_walks
from niftygex_eval.results import ResultsStore
from niftygex_eval.stage_a import add_regressors
from niftygex_eval.stage_b import paired, pooled_summary, run_tuned, run_tuned_regression, shap_summary


def main() -> None:
    cfg = load_eval_config()
    evaluation, data_config = cfg["eval"], cfg["data_config"]
    stage, harness, stage_a = evaluation["stage_b"], evaluation["harness"], evaluation["stage_a"]
    table, feature_columns, hashes = modeling_table(data_config, stage["target"])
    table = add_regressors(table, list(stage_a["conventions"]), int(stage_a["normalization_window"]), int(stage_a["normalization_min_periods"]))
    table = table.with_columns(pl.col("gex_total_inferred").sign().alias("gex_sign_inferred"))
    sets_cfg = stage["feature_sets"]
    sets = {"A": list(sets_cfg["A"])}
    sets["B"] = sets["A"] + list(sets_cfg["B_adds"])
    sets["C"] = sets["B"] + list(sets_cfg["C_adds"])
    sets["C1"] = sets["B"] + list(sets_cfg["C1_adds"])
    sets["C2"] = sets["B"] + list(sets_cfg["C2_adds"])
    shap_set = stage["shap"]["feature_set"]
    available = set(feature_columns) | {f"gex_norm_{c}" for c in stage_a["conventions"]} | {"gex_sign_inferred"}
    for name, columns in sets.items():
        if missing := set(columns) - available:
            raise SystemExit(f"Feature set {name} has unknown columns: {sorted(missing)}")

    regression = stage["regression"]
    outcome = f"log_{regression['outcome']}"
    table = table.with_columns(pl.col(regression["outcome"]).log().alias(outcome))
    labeled = table.filter(pl.col(stage["label"]).is_not_null())
    class_walks = build_walks(labeled["cycle"].unique().to_list(), **harness)
    with_outcome = table.filter(pl.col(outcome).is_not_null() & pl.col(outcome).is_finite())
    reg_walks = build_walks(with_outcome["cycle"].unique().to_list(), **harness)

    store = ResultsStore(cfg["results_root"])
    base = {"data_config_hash": hashes["features"], "eval_config_hash": eval_hash(evaluation), "stage": "t2_ablation",
            "target": stage["target"], "harness": harness}
    classes, bins = list(stage["classes"]), int(stage["calibration_bins"])
    class_results, reg_results, runs, table_rows, shap_tables = {}, {}, {}, [], {}
    for name, columns in sets.items():
        explain = name == shap_set
        tuned, chosen = run_tuned(table, class_walks, columns, stage, calibrators=["platt"], explain=explain)
        predictions, per_walk = tuned["lightgbm_tuned_platt"]
        class_results[name] = per_walk
        summary = pooled_summary(predictions, classes, bins, {})
        class_tables = {"predictions": predictions, "walk_metrics": per_walk, "chosen_params": chosen}
        if explain:
            class_tables["shap"] = tuned["shap"][0]
            shap_tables["classification"] = shap_summary(tuned["shap"][0])
            class_tables["shap_summary"] = shap_tables["classification"]
        run = store.write_run({**base, "model": f"classification_{name}", "features": columns, "params": stage["tuning"]}, class_tables, summary)
        regression_out = run_tuned_regression(table, reg_walks, columns, stage, outcome, explain=explain)
        reg_predictions, reg_walk = regression_out[0], regression_out[1]
        reg_results[name] = reg_walk
        pooled_ic = float(reg_predictions.select(pl.corr("prediction", outcome, method="spearman")).item())
        reg_tables = {"predictions": reg_predictions, "walk_metrics": reg_walk}
        if explain:
            reg_tables["shap"] = regression_out[2]
            shap_tables["regression"] = shap_summary(regression_out[2])
            reg_tables["shap_summary"] = shap_tables["regression"]
        reg_run = store.write_run({**base, "model": f"regression_{name}", "features": columns, "params": {**stage["tuning"], "outcome": outcome}},
                                  reg_tables, {"pooled_ic": pooled_ic, "mean_ic": float(reg_walk["ic"].mean())})
        runs[name] = {"classification": run.name, "regression": reg_run.name}
        ics = reg_walk["ic"].to_numpy()
        table_rows.append({
            "feature_set": name, "features": len(columns),
            "auc_pooled": summary["overall"]["auc_macro_ovr"], "auc_mean_per_walk": float(per_walk["auc_macro_ovr"].mean()),
            "brier": summary["overall"]["brier"], "brier_mean_per_walk": float(per_walk["brier"].mean()),
            "log_loss": summary["overall"]["log_loss"], "ece_mean": summary["calibration"]["ece_mean"],
            "ic_mean_per_walk": float(np.nanmean(ics)), "ic_pooled": pooled_ic,
            "ic_t_vs_zero": float(np.nanmean(ics) / (np.nanstd(ics, ddof=1) / np.sqrt(np.isfinite(ics).sum()))),
            "ic_positive_walks": int(np.sum(ics > 0)), "regression_walks": reg_walk.height,
        })
    tests = {}
    for a, b in (("B", "A"), ("C", "B"), ("C1", "B"), ("C2", "B"), ("C", "A")):
        tests[f"{a}_vs_{b}"] = {
            "auc": paired(class_results[a], class_results[b], "auc_macro_ovr", True),
            "brier": paired(class_results[a], class_results[b], "brier", False),
            "ic": paired(reg_results[a], reg_results[b], "ic", True),
        }
    per_walk = (
        class_results["A"].select("walk_id", "test_start", pl.col("auc_macro_ovr").alias("auc_A"), pl.col("brier").alias("brier_A"))
        .join(class_results["B"].select("walk_id", pl.col("auc_macro_ovr").alias("auc_B"), pl.col("brier").alias("brier_B")), on="walk_id")
        .join(class_results["C"].select("walk_id", pl.col("auc_macro_ovr").alias("auc_C"), pl.col("brier").alias("brier_C")), on="walk_id")
    )
    per_walk_ic = (
        reg_results["A"].select("walk_id", "test_start", pl.col("ic").alias("ic_A"))
        .join(reg_results["B"].select("walk_id", pl.col("ic").alias("ic_B")), on="walk_id")
        .join(reg_results["C"].select("walk_id", pl.col("ic").alias("ic_C")), on="walk_id")
    )
    payload = {"data_config_hash": hashes["features"], "eval_config_hash": eval_hash(evaluation),
               "classification_walks": len(class_walks), "regression_walks": len(reg_walks), "feature_sets": sets, "runs": runs,
               "comparison_table": table_rows, "paired_tests": tests, "shap_feature_set": shap_set,
               "shap_mean_abs": {task: frame.to_dicts() for task, frame in shap_tables.items()},
               "per_walk_classification": per_walk.to_dicts(), "per_walk_ic": per_walk_ic.to_dicts()}
    (cfg["results_root"] / "t2_ablation.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(json.dumps({"comparison_table": table_rows, "paired_tests": tests, "runs": runs}, indent=2, default=str))


if __name__ == "__main__":
    main()
