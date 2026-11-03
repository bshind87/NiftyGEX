"""Neural rungs of the ladder (proposal steps 4-6) in the same walk-forward harness as the classical models.

Writes one results-store run per model and outputs/results/neural_comparison.json. Options:
  --models deepsets attention   run a subset (default: neural.models)
  --window 24                   history length L for the window ablation (default: neural.window)
  --drop oi_change_since_previous_close_pct   field-drop ablation: leave these token fields out
  --save-attention              also store pooling attention weights per test day (table "attention" in each run)
  --walks 3                     development run on the last N walks only (tagged "dev"; never used by the RQ tests)

Runs other than the default configuration are tagged (window_24, drop_<fields>) and written to their own
neural_comparison_<tag>.json, so only the main run feeds the pre-registered tests.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import polars as pl
import torch

from niftygex.io.parquet import read_manifest
from niftygex_eval.config import eval_hash, load_eval_config
from niftygex_eval.forecast.compare import mean_ci
from niftygex_eval.forecast.metrics import regression_metrics
from niftygex_eval.forecast.runner import run_walks
from niftygex_eval.forecast.table import forecast_table
from niftygex_eval.harness import build_walks
from niftygex_eval.nn.dataset import build_inputs
from niftygex_eval.nn.trainer import NeuralForecaster
from niftygex_eval.results import ResultsStore


def device_for(name: str, neural: dict) -> str:
    wanted = neural.get("devices", {}).get(name, "cpu")
    return wanted if wanted != "mps" or torch.backends.mps.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--window", type=int)
    parser.add_argument("--walks", type=int, help="development run on the last N walks")
    parser.add_argument("--drop", nargs="*", default=[], help="token fields to leave out (field-drop ablation)")
    parser.add_argument("--save-attention", action="store_true")
    args = parser.parse_args()

    cfg = load_eval_config()
    evaluation, data_config = cfg["eval"], cfg["data_config"]
    neural, forecast, harness = evaluation["neural"], evaluation["forecast"], evaluation["harness"]
    table, outcome, _, hashes = forecast_table(cfg)
    tokens_hash = read_manifest(data_config, "tokens")["config_hash"]
    if tokens_hash != hashes["features"]:
        raise SystemExit(f"Token and feature manifests disagree on config_hash: {tokens_hash[:12]} vs {hashes['features'][:12]}")
    window = int(args.window or neural["window"])
    if unknown := sorted(set(args.drop) - set(neural["token_features"])):
        raise SystemExit(f"--drop: unknown token fields {unknown}")
    neural = {**neural, "token_features": [c for c in neural["token_features"] if c not in args.drop]}
    inputs = build_inputs(table, Path(data_config["paths"]["features_root"]) / "tokens", neural, window)
    walks = build_walks(table["cycle"].unique().to_list(), **harness)
    parts = ([f"window_{window}"] if window != int(evaluation["neural"]["window"]) else []) + ([f"drop_{'_'.join(args.drop)}"] if args.drop else [])
    tag = "dev" if args.walks else ("_".join(parts) or "main")
    if args.walks:
        walks = walks[-args.walks:]
    tuning = forecast["tuning"]
    store = ResultsStore(cfg["results_root"])
    base = {"data_config_hash": hashes["features"], "eval_config_hash": eval_hash(evaluation), "stage": "neural", "target": forecast["target"],
            "outcome": outcome, "harness": harness, "window": window, "tag": tag, "dropped": list(args.drop),
            "token_features": neural["token_features"], "walks_run": [w.walk_id for w in walks]}

    rows, runs = [], {}
    for name in args.models or neural["models"]:
        device, started, attention = device_for(name, neural), time.time(), []
        collect = (lambda model, walk, test: attention.append(model.attention(test).with_columns(pl.lit(walk.walk_id).alias("walk_id")))) \
            if args.save_attention and name != "deepsets" else None
        predictions, per_walk = run_walks(
            table, walks, lambda: NeuralForecaster(name, inputs, neural, device=device), outcome,
            inner_validation_cycles=int(tuning["inner_validation_cycles"]), inner_embargo_cycles=int(tuning["inner_embargo_cycles"]),
            keep=[forecast["regime_column"]], collect=collect)
        seconds = time.time() - started
        pooled = regression_metrics(predictions["y"].to_numpy(), predictions["prediction"].to_numpy())
        spec = {**base, "model": name, "device": device, "model_config": neural["model"], "training": neural["training"]}
        tables = {"predictions": predictions, "walk_metrics": per_walk, **({"attention": pl.concat(attention)} if attention else {})}
        runs[name] = store.write_run(spec, tables, {"pooled": pooled, "seconds": seconds}).name
        rows.append({"model": name, "walks": per_walk.height, "test_days": predictions.height, "seconds": round(seconds, 1),
                     "ic_mean_per_walk": float(per_walk["ic"].mean()), "ic_ci95": mean_ci(per_walk["ic"].to_numpy()),
                     "rmse_mean_per_walk": float(per_walk["rmse"].mean()), "qlike_mean_per_walk": float(per_walk["qlike"].mean()),
                     "stop_epoch_mean": float(per_walk["stop_epoch_mean"].mean()), "pooled": pooled})
        print(f"{name}: walks {per_walk.height}, mean IC {rows[-1]['ic_mean_per_walk']:.3f}, {seconds / 60:.1f} min on {device}", flush=True)

    out = {"data_config_hash": hashes["features"], "eval_config_hash": base["eval_config_hash"], "outcome": outcome, "window": window,
           "tag": tag, "dropped": list(args.drop), "walks": len(walks), "runs": runs, "comparison_table": rows}
    path = cfg["results_root"] / ("neural_comparison.json" if tag == "main" else f"neural_comparison_{tag}.json")
    path.write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(path)


if __name__ == "__main__":
    main()
