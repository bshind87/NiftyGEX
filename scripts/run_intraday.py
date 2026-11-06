"""Exploratory E3: the forecasting ladder for the intraday target (configs/eval_exploratory_v2.yaml, section e3).

  --horizon 1 --models persistence intraday_har har_gex lightgbm    classical rungs (CPU)
  --horizon 1 --models deepsets attention attention_gru              neural rungs (patience 3, at most 30 epochs)
  --walks 2                                                         development / timing run on the last N walks (tag dev)
  --seed 11                                                         cluster job: one training seed (merged by scripts/merge_seeds.py)
  --horizon 1 --tests                                               registered E3 tests for this horizon (after all models)

Same expiry-cycle walks as T2, split by day and purged on the horizon's label time. Each model run is resumable per
walk and written to the results store (stage exploratory_e3); outputs/results/exploratory_e3_runs.json lists them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from niftygex_eval.compute import spec_extras, summary_extras, with_seed
from niftygex_eval.config import eval_hash, load_exploratory_config
from niftygex_eval.forecast.intraday import for_horizon, intraday_table, make_intraday_baseline, time_of_day_columns
from niftygex_eval.forecast.metrics import regression_metrics
from niftygex_eval.forecast.runner import run_walks
from niftygex_eval.harness import build_walks
from niftygex_eval.results import ResultsStore

CLASSICAL = ("persistence", "intraday_har", "har_gex", "lightgbm")
NEURAL = ("deepsets", "attention", "attention_gru")


def neural_config(cfg: dict) -> dict:
    """The main neural settings with E3's training, window and state (plus time of day)."""
    neural, e3 = cfg["eval"]["neural"], cfg["exploratory"]["e3"]
    return {**neural, "training": dict(e3["training"]), "window": int(e3["window"]), "devices": dict(e3["devices"]),
            "state_columns": list(neural["state_columns"]) + list(e3["extra_state_columns"])}


def device_for(name: str, neural: dict) -> str:
    from niftygex_eval.nn.trainer import resolve_device

    return resolve_device(neural["devices"].get(name, "cpu"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--models", nargs="*", default=[], choices=[*CLASSICAL, *NEURAL])
    parser.add_argument("--walks", type=int, help="development run on the last N walks (tag dev)")
    parser.add_argument("--seed", type=int, help="train this seed only (one cluster job per seed)")
    parser.add_argument("--tests", action="store_true")
    args = parser.parse_args()
    cfg = load_exploratory_config()
    evaluation, e3 = cfg["eval"], cfg["exploratory"]["e3"]
    if args.horizon not in [int(h) for h in e3["horizons"]]:
        raise SystemExit(f"--horizon must be one of {e3['horizons']}")
    if args.tests:
        from niftygex_eval.exploratory_e3 import run_tests

        run_tests(cfg, args.horizon)
        return

    table, summary, hashes = intraday_table(cfg)
    table = for_horizon(table, args.horizon)
    outcome = e3["outcome"].replace("{h}", str(args.horizon))
    walks = build_walks(table["cycle"].unique().to_list(), **evaluation["harness"])
    tag = "dev" if args.walks else "main"
    if args.walks:
        walks = walks[-args.walks:]
    forecast, tuning, tod = evaluation["forecast"], evaluation["forecast"]["tuning"], time_of_day_columns(table)
    neural = with_seed(neural_config(cfg), args.seed)
    inputs, mask_counts = None, None
    if set(args.models) & set(NEURAL):
        from niftygex_eval.nn.dataset import build_intraday_inputs

        choice = pl.read_parquet(Path(cfg["data_config"]["paths"]["exposure_root"]) / "intraday" / "expiry_choice" / "expiry_choice.parquet")
        inputs, mask_counts = build_intraday_inputs(table, Path(cfg["data_config"]["paths"]["features_root"]) / "tokens", neural,
                                                    neural["window"], choice, cfg["data_config"]["data"]["snapshot_grid"]["t2_daily"])
        print(f"intraday token rows: {mask_counts}", flush=True)
    store = ResultsStore(cfg["results_root"])
    index_path = cfg["results_root"] / "exploratory_e3_runs.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    key = f"{args.horizon}h" + ("_dev" if args.walks else "")
    for name in args.models:
        if name in NEURAL:
            from niftygex_eval.nn.trainer import NeuralForecaster

            device = device_for(name, neural)
            make = lambda: NeuralForecaster(name, inputs, neural, device=device)
            extra = {"device": device, "training": neural["training"], "model_config": neural["model"], "window": neural["window"],
                     "token_mask_counts": mask_counts}
        else:
            make = lambda: make_intraday_baseline(name, e3, args.horizon, summary, tod, forecast)
            extra = {"columns": summary if name == "lightgbm" else None}
        spec = {"stage": "exploratory_e3", "label": "exploratory", "model": name, "horizon_hours": args.horizon, "tag": tag,
                "data_config_hash": hashes["features_intraday"], "eval_config_hash": eval_hash(evaluation),
                "exploratory_config_hash": cfg["exploratory_hash"], "outcome": outcome, "harness": evaluation["harness"],
                "walks_run": [w.walk_id for w in walks], **extra, **(spec_extras() if name in NEURAL else {})}
        predictions, per_walk = run_walks(
            table, walks, make, outcome, inner_validation_cycles=int(tuning["inner_validation_cycles"]),
            inner_embargo_cycles=int(tuning["inner_embargo_cycles"]), keep=[forecast["regime_column"], "minutes_since_open"],
            checkpoint=store.checkpoint_dir(spec))
        seconds = float(per_walk["seconds"].sum())  # summed over walks: a run may span several processes
        pooled = regression_metrics(predictions["y"].to_numpy(), predictions["prediction"].to_numpy())
        run = store.write_run(spec, {"predictions": predictions, "walk_metrics": per_walk},
                              {"pooled": pooled, "seconds": seconds, **summary_extras(per_walk)}).name
        store.clear_checkpoint(spec)
        if args.seed is None or name not in NEURAL:   # per-seed cluster runs are listed by scripts/merge_seeds.py
            index.setdefault(key, {})[name] = run
            index_path.write_text(json.dumps(index, indent=2) + "\n")
        print(f"{name} ({args.horizon}h, {tag}): walks {per_walk.height}, mean IC {per_walk['ic'].mean():.3f}, pooled IC {pooled['ic']:.3f}, "
              f"{seconds / 60:.1f} min, {seconds / max(per_walk.height, 1) / 60:.1f} min per walk", flush=True)


if __name__ == "__main__":
    main()
