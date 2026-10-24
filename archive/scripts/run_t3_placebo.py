"""T3 placebo analysis: hit rate toward the max-GEX strike vs matched-distance pseudo-strikes."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl

from niftygex.io.parquet import read_manifest
from niftygex_eval.config import eval_hash, load_eval_config
from niftygex_eval.results import ResultsStore
from niftygex_eval.t3_placebo import run_placebo


def main() -> None:
    cfg = load_eval_config()
    data_config, settings = cfg["data_config"], cfg["eval"]["t3_placebo"]
    paths = data_config["paths"]
    hashes = {stage: read_manifest(data_config, stage)["config_hash"] for stage in ("features", "targets")}
    if len(set(hashes.values())) != 1:
        raise SystemExit(f"Feature and target manifests disagree: {hashes}")
    t3 = pl.read_parquet(Path(paths["targets_root"]) / "t3" / "t3.parquet").filter(pl.col("t3_reason") == "ok")
    features = pl.read_parquet(Path(paths["features_root"]) / "aggregate" / "aggregate.parquet").filter(pl.col("target") == "t3").select(
        "snapshot_at", "is_expiry_day")
    frame = t3.join(features, on="snapshot_at", how="left")
    marker = next(m for m in data_config["regimes"]["regime_markers"] if m["name"] == "weekly_expiry_rationalisation")
    marker_date = date.fromisoformat(str(marker["date"]))
    groups = {
        "pre_weekly_expiry_rationalisation": pl.col("trade_date") < marker_date,
        "post_weekly_expiry_rationalisation": pl.col("trade_date") >= marker_date,
        "expiry_days": pl.col("is_expiry_day"),
        "non_expiry_days": ~pl.col("is_expiry_day"),
    }
    result = run_placebo(frame, int(settings["draws"]), int(settings["seed"]), int(settings["distance_buckets"]), groups)
    spec = {"stage": "t3_placebo", "target": "t3", "model": "matched_distance_placebo", "data_config_hash": hashes["targets"],
            "eval_config_hash": eval_hash(cfg["eval"]), "params": settings}
    run = ResultsStore(cfg["results_root"]).write_run(spec, {"days": frame.select("snapshot_at", "trade_date", "t3_start_price", "t3_end_price", "t3_strike", "t3_label")}, result)
    (cfg["results_root"] / "t3_placebo.json").write_text(json.dumps({"run": run.name, **result}, indent=2, default=str) + "\n")
    print(json.dumps({"run": run.name, "overall": result["overall"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
