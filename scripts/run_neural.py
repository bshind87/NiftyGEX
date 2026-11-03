"""Neural rungs of the ladder (proposal steps 4-6) in the same walk-forward harness as the classical models.

Writes one results-store run per model and outputs/results/neural_comparison.json. Options:
  --models deepsets attention   run a subset (default: neural.models)
  --window 24                   history length L for the window ablation (default: neural.window)
  --walks 3                     development run on the last N walks only (tagged "dev"; never used by the RQ tests)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

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
    args = parser.parse_args()

    cfg = load_eval_config()
    evaluation, data_config = cfg["eval"], cfg["data_config"]
    neural, forecast, harness = evaluation["neural"], evaluation["forecast"], evaluation["harness"]
    table, outcome, _, hashes = forecast_table(cfg)
    tokens_hash = read_manifest(data_config, "tokens")["config_hash"]
    if tokens_hash != hashes["features"]:
        raise SystemExit(f"Token and feature manifests disagree on config_hash: {tokens_hash[:12]} vs {hashes['features'][:12]}")
    window = int(args.window or neural["window"])
    inputs = build_inputs(table, Path(data_config["paths"]["features_root"]) / "tokens", neural, window)
    walks = build_walks(table["cycle"].unique().to_list(), **harness)
    tag = "dev" if args.walks else ("main" if window == int(neural["window"]) else f"window_{window}")
    if args.walks:
        walks = walks[-args.walks:]
    tuning = forecast["tuning"]
    store = ResultsStore(cfg["results_root"])
    base = {"data_config_hash": hashes["features"], "eval_config_hash": eval_hash(evaluation), "stage": "neural", "target": forecast["target"],
            "outcome": outcome, "harness": harness, "window": window, "tag": tag, "walks_run": [w.walk_id for w in walks]}

    rows, runs = [], {}
    for name in args.models or neural["models"]:
        device, started = device_for(name, neural), time.time()
        predictions, per_walk = run_walks(
            table, walks, lambda: NeuralForecaster(name, inputs, neural, device=device), outcome,
            inner_validation_cycles=int(tuning["inner_validation_cycles"]), inner_embargo_cycles=int(tuning["inner_embargo_cycles"]),
            keep=[forecast["regime_column"]])
        seconds = time.time() - started
        pooled = regression_metrics(predictions["y"].to_numpy(), predictions["prediction"].to_numpy())
        spec = {**base, "model": name, "device": device, "model_config": neural["model"], "training": neural["training"]}
        runs[name] = store.write_run(spec, {"predictions": predictions, "walk_metrics": per_walk}, {"pooled": pooled, "seconds": seconds}).name
        rows.append({"model": name, "walks": per_walk.height, "test_days": predictions.height, "seconds": round(seconds, 1),
                     "ic_mean_per_walk": float(per_walk["ic"].mean()), "ic_ci95": mean_ci(per_walk["ic"].to_numpy()),
                     "rmse_mean_per_walk": float(per_walk["rmse"].mean()), "qlike_mean_per_walk": float(per_walk["qlike"].mean()),
                     "stop_epoch_mean": float(per_walk["stop_epoch_mean"].mean()), "pooled": pooled})
        print(f"{name}: walks {per_walk.height}, mean IC {rows[-1]['ic_mean_per_walk']:.3f}, {seconds / 60:.1f} min on {device}", flush=True)

    out = {"data_config_hash": hashes["features"], "eval_config_hash": base["eval_config_hash"], "outcome": outcome, "window": window,
           "tag": tag, "walks": len(walks), "runs": runs, "comparison_table": rows}
    path = cfg["results_root"] / ("neural_comparison.json" if tag == "main" else f"neural_comparison_{tag}.json")
    path.write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(path)


if __name__ == "__main__":
    main()
