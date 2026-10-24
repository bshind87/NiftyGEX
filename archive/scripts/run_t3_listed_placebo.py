"""T3 listed-strike placebo and continuous T3: max-GEX strike vs GEX-rank 2-5 strikes and the snapped mirror."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl

from niftygex.exposure.selection import total_gex_expiries
from niftygex.io.parquet import read_manifest
from niftygex_eval.config import eval_hash, load_eval_config
from niftygex_eval.results import ResultsStore
from niftygex_eval.t3_listed import comparators, run_listed_placebo, strike_ranks


def main() -> None:
    cfg = load_eval_config()
    data_config, settings = cfg["data_config"], cfg["eval"]["t3_listed_placebo"]
    paths = data_config["paths"]
    hashes = {stage: read_manifest(data_config, stage)["config_hash"] for stage in ("gex", "features", "targets")}
    if len(set(hashes.values())) != 1:
        raise SystemExit(f"Manifests disagree: {hashes}")
    exposure = Path(paths["exposure_root"])
    grid = pl.read_parquet(exposure / "snapshots" / "snapshot_grid.parquet").filter(pl.col("target") == "t3").select("snapshot_at")
    chain = pl.read_parquet(exposure / "gex_strikes" / "*.parquet").join(grid, on="snapshot_at").filter(
        total_gex_expiries("t3", data_config["data"]["gex"]) & pl.col("gex_absolute").is_not_null())
    ranks = strike_ranks(chain)
    days = pl.read_parquet(Path(paths["targets_root"]) / "t3" / "t3.parquet").filter(pl.col("t3_reason") == "ok").join(
        pl.read_parquet(Path(paths["features_root"]) / "aggregate" / "aggregate.parquet").filter(pl.col("target") == "t3").select("snapshot_at", "is_expiry_day"),
        on="snapshot_at", how="left")
    frame = comparators(days, ranks)
    marker = next(m for m in data_config["regimes"]["regime_markers"] if m["name"] == "weekly_expiry_rationalisation")
    marker_date = date.fromisoformat(str(marker["date"]))
    groups = {
        "expiry_days": pl.col("is_expiry_day"),
        "non_expiry_days": ~pl.col("is_expiry_day"),
        "pre_weekly_expiry_rationalisation": pl.col("trade_date") < marker_date,
        "post_weekly_expiry_rationalisation": pl.col("trade_date") >= marker_date,
    }
    result = run_listed_placebo(frame, groups, int(settings["distance_buckets"]), float(settings["max_distance_gap_pct"]))
    spec = {"stage": "t3_listed_placebo", "target": "t3", "model": "listed_strike_placebo", "data_config_hash": hashes["targets"],
            "eval_config_hash": eval_hash(cfg["eval"]), "params": settings}
    run = ResultsStore(cfg["results_root"]).write_run(spec, {"days": frame}, result)
    (cfg["results_root"] / "t3_listed_placebo.json").write_text(json.dumps({"run": run.name, **result}, indent=2, default=str) + "\n")
    print(json.dumps({"run": run.name, "overall": result["overall"], "expiry_days": result["expiry_days"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
