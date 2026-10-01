"""Stage B (T2): baselines, full and no-GEX LightGBM, and tuned + calibrated LightGBM inside the walk-forward harness."""
from __future__ import annotations

import json
from datetime import date

import polars as pl

from niftygex_eval.config import eval_hash, load_eval_config
from niftygex_eval.data import modeling_table
from niftygex_eval.harness import build_walks
from niftygex_eval.results import ResultsStore
from niftygex_eval.stage_b import paired, pooled_summary, run_model, run_tuned

BASE_MODELS = {"prior": "prior", "vix_only": "vix_only", "logistic": "logistic", "lightgbm": "lightgbm", "lightgbm_no_gex": "lightgbm"}


def main() -> None:
    cfg = load_eval_config()
    evaluation, data_config = cfg["eval"], cfg["data_config"]
    stage, harness = evaluation["stage_b"], evaluation["harness"]
    table, feature_columns, hashes = modeling_table(data_config, stage["target"])
    target_columns = set(pl.read_parquet(data_config["paths"]["targets_root"] / stage["target"] / f"{stage['target']}.parquet").columns) - {"snapshot_at", "trade_date"}
    columns = [c for c in feature_columns if c not in set(stage["exclude_features"])]
    no_gex = list(stage["no_gex_features"])
    for name, cols in (("full", columns), ("no_gex", no_gex)):
        if set(cols) & target_columns:
            raise SystemExit(f"Target columns in the {name} feature list: {sorted(set(cols) & target_columns)}")
        if missing := set(cols) - set(feature_columns):
            raise SystemExit(f"Unknown {name} features: {sorted(missing)}")
    labeled = table.filter(pl.col(stage["label"]).is_not_null())
    walks = build_walks(labeled["cycle"].unique().to_list(), **harness)
    walk_table = pl.DataFrame([{
        "walk_id": w.walk_id, "train_cycles": len(w.train_cycles), "train_first_cycle": w.train_cycles[0], "train_last_cycle": w.train_cycles[-1],
        "embargo_cycles": ",".join(str(c) for c in w.embargo_cycles), "test_first_cycle": w.test_cycles[0], "test_last_cycle": w.test_cycles[-1],
        "test_cycles": len(w.test_cycles)} for w in walks])
    marker = next(m for m in data_config["regimes"]["regime_markers"] if m["name"] == "weekly_expiry_rationalisation")
    marker_date = date.fromisoformat(str(marker["date"]))
    groups = {"pre_weekly_expiry_rationalisation": pl.col("trade_date") < marker_date,
              "post_weekly_expiry_rationalisation": pl.col("trade_date") >= marker_date}
    store = ResultsStore(cfg["results_root"])
    classes, bins = list(stage["classes"]), int(stage["calibration_bins"])
    base = {"data_config_hash": hashes["features"], "eval_config_hash": eval_hash(evaluation), "stage": "stage_b", "target": stage["target"], "harness": harness}

    results: dict[str, tuple[pl.DataFrame, pl.DataFrame]] = {}
    for name, kind in BASE_MODELS.items():
        model_columns = list(stage["baselines"]["vix_only"]) if kind in ("vix_only", "prior") else (no_gex if name == "lightgbm_no_gex" else columns)
        results[name] = run_model(table, walks, kind, model_columns, stage)
        summary = pooled_summary(results[name][0], classes, bins, groups)
        store.write_run({**base, "model": name, "features": model_columns, "params": stage.get("logistic" if kind == "vix_only" else kind)},
                        {"predictions": results[name][0], "walk_metrics": results[name][1], "walks": walk_table}, summary)
    tuned, chosen = run_tuned(table, walks, columns, stage)
    for name, (predictions, per_walk) in tuned.items():
        results[name] = (predictions, per_walk)
        summary = pooled_summary(predictions, classes, bins, groups)
        store.write_run({**base, "model": name, "features": columns, "params": {"base": stage["lightgbm"], "tuning": stage["tuning"]}},
                        {"predictions": predictions, "walk_metrics": per_walk, "walks": walk_table, "chosen_params": chosen}, summary)

    table_rows = []
    for name, (predictions, per_walk) in results.items():
        summary = pooled_summary(predictions, classes, bins, groups)
        table_rows.append({
            "model": name, "n": summary["overall"]["n"], "walks": per_walk.height,
            "auc_pooled": summary["overall"]["auc_macro_ovr"], "auc_mean_per_walk": float(per_walk["auc_macro_ovr"].mean()),
            "brier": summary["overall"]["brier"], "brier_mean_per_walk": float(per_walk["brier"].mean()),
            "log_loss": summary["overall"]["log_loss"], "ece_mean": summary["calibration"]["ece_mean"],
            "auc_pre_marker": summary["pre_weekly_expiry_rationalisation"]["auc_macro_ovr"],
            "auc_post_marker": summary["post_weekly_expiry_rationalisation"]["auc_macro_ovr"],
        })
    comparisons = {
        "full_vs_no_gex_auc": paired(results["lightgbm"][1], results["lightgbm_no_gex"][1], "auc_macro_ovr", True),
        "full_vs_no_gex_brier": paired(results["lightgbm"][1], results["lightgbm_no_gex"][1], "brier", False),
        **{f"{v}_vs_lightgbm_brier": paired(results[v][1], results["lightgbm"][1], "brier", False) for v in tuned},
        **{f"{v}_vs_lightgbm_auc": paired(results[v][1], results["lightgbm"][1], "auc_macro_ovr", True) for v in tuned},
        **{f"{v}_vs_prior_brier": paired(results[v][1], results["prior"][1], "brier", False) for v in tuned},
    }
    per_walk_auc = results["lightgbm"][1].select("walk_id", "test_start", pl.col("auc_macro_ovr").alias("full")).join(
        results["lightgbm_no_gex"][1].select("walk_id", pl.col("auc_macro_ovr").alias("no_gex")), on="walk_id")
    calibration = {name: pooled_summary(results[name][0], classes, bins, groups)["calibration"] for name in ("lightgbm", *tuned)}
    payload = {"data_config_hash": hashes["features"], "eval_config_hash": eval_hash(evaluation), "walks": len(walks),
               "comparison_table": table_rows, "paired": comparisons, "full_vs_no_gex_per_walk_auc": per_walk_auc.to_dicts(),
               "chosen_params": chosen.to_dicts(), "calibration": calibration}
    (cfg["results_root"] / "stage_b_t2_comparison.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(json.dumps({"comparison_table": table_rows, "paired": comparisons}, indent=2, default=str))


if __name__ == "__main__":
    main()
