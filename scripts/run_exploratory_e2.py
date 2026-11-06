"""Exploratory E2: next-day target with more history (configs/eval_exploratory_v2.yaml, section e2).

  --days 5            multi-day attention + GRU: previous 5 days' 15:15 chains plus today's L = 12 (resumes per walk)
  --days 5 --seed 11  cluster job: one training seed; scripts/merge_seeds.py averages the seeds and lists the run
  --tests             registered E2 tests: L = 24, L = 72 (main-config runs) and multi-day K = 5, 10 vs attention + GRU at
                      L = 12, Holm within E2; runs not yet made are reported as missing

Same table, walks and rows as the main ladder. Runs go to the results store (stage exploratory_e2) and
outputs/results/exploratory_e2_runs.json; tests to outputs/results/exploratory_e2.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from niftygex_eval.compute import spec_extras, summary_extras, with_seed
from niftygex_eval.config import eval_hash, load_exploratory_config
from niftygex_eval.exploratory import family
from niftygex_eval.forecast.compare import diebold_mariano, paired
from niftygex_eval.forecast.metrics import regression_metrics
from niftygex_eval.forecast.runner import run_walks
from niftygex_eval.forecast.table import forecast_table
from niftygex_eval.harness import build_walks
from niftygex_eval.results import ResultsStore


def run_multiday(cfg: dict, days: int, seed: int | None = None) -> None:
    from niftygex_eval.nn.dataset import build_multiday_inputs
    from niftygex_eval.nn.extra_models import build_extra_model
    from niftygex_eval.nn.trainer import NeuralForecaster, resolve_device

    evaluation, e2 = cfg["eval"], cfg["exploratory"]["e2"]
    forecast, tuning = evaluation["forecast"], evaluation["forecast"]["tuning"]
    table, outcome, _, hashes = forecast_table(cfg)
    walks = build_walks(table["cycle"].unique().to_list(), **evaluation["harness"])
    curated = Path(cfg["data_config"]["paths"]["curated_root"])
    trading_dates = pl.read_parquet(curated / "calendar" / "trading_calendar.parquet").filter(pl.col("observed")).sort("trading_date")["trading_date"].to_list()
    window = int(e2["multiday"]["intraday_window"])
    neural = with_seed({**evaluation["neural"], "model": {**evaluation["neural"]["model"], "multiday_days": days}}, seed)
    # Memory per pass stays at the L = 12 level: the trainer splits batches over window + days snapshots.
    neural["window"] = window
    inputs = build_multiday_inputs(table, Path(cfg["data_config"]["paths"]["features_root"]) / "tokens", neural, days, trading_dates, window)
    device = resolve_device("mps")
    name = e2["multiday"]["model"]
    store = ResultsStore(cfg["results_root"])
    spec = {"stage": "exploratory_e2", "label": "exploratory", "model": name, "tag": f"multiday_{days}", "days": days, "window": window,
            "device": device, "data_config_hash": hashes["features"], "eval_config_hash": eval_hash(evaluation),
            "exploratory_config_hash": cfg["exploratory_hash"], "target": forecast["target"], "outcome": outcome,
            "harness": evaluation["harness"], "training": neural["training"], "model_config": neural["model"], **spec_extras()}
    predictions, per_walk = run_walks(
        table, walks, lambda: NeuralForecaster(name, inputs, neural, device=device, factory=build_extra_model), outcome,
        inner_validation_cycles=int(tuning["inner_validation_cycles"]), inner_embargo_cycles=int(tuning["inner_embargo_cycles"]),
        keep=[forecast["regime_column"]], checkpoint=store.checkpoint_dir(spec))
    seconds = float(per_walk["seconds"].sum())  # summed over walks: a run may span several processes
    pooled = regression_metrics(predictions["y"].to_numpy(), predictions["prediction"].to_numpy())
    run = store.write_run(spec, {"predictions": predictions, "walk_metrics": per_walk},
                          {"pooled": pooled, "seconds": seconds, **summary_extras(per_walk)}).name
    store.clear_checkpoint(spec)
    print(f"multiday_{days}: walks {per_walk.height}, mean IC {per_walk['ic'].mean():.3f}, {seconds / 60:.1f} min on {device}", flush=True)
    if seed is not None:   # per-seed cluster runs are listed by scripts/merge_seeds.py
        return
    index_path = cfg["results_root"] / "exploratory_e2_runs.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    index[f"multiday_{days}"] = run
    index_path.write_text(json.dumps(index, indent=2) + "\n")


def run_tests(cfg: dict) -> None:
    root, e2, alpha = cfg["results_root"], cfg["exploratory"]["e2"], float(cfg["exploratory"]["alpha"])
    reference = json.loads((root / cfg["exploratory"]["references"]["neural_hpc"]).read_text())   # same device and GPU type
    base_run = reference["runs"]["attention_gru"]
    candidates: dict[str, str] = {}
    for tag, filename in e2["windows"].items():
        path = root / filename
        if path.exists():
            comparison = json.loads(path.read_text())
            if comparison["eval_config_hash"] != cfg["exploratory"]["references"]["main_eval_config_hash"]:
                raise SystemExit(f"{filename} is not from the main eval config")
            candidates[tag] = comparison["runs"]["attention_gru"]
    index_path = root / "exploratory_e2_runs.json"
    candidates.update(json.loads(index_path.read_text()) if index_path.exists() else {})

    def load(run_id: str) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
        directory = root / "runs" / run_id
        return (pl.read_parquet(directory / "predictions.parquet"), pl.read_parquet(directory / "walk_metrics.parquet"),
                json.loads((directory / "run.json").read_text())["spec"])

    base = load(base_run)
    lags = int(cfg["eval"]["forecast"]["dm_max_lags"])
    results, missing, secondary = [], [], {}
    for test in e2["tests"]:
        if test["a"] not in candidates:
            missing.append(test["a"])
            continue
        run = load(candidates[test["a"]])
        if run[2]["data_config_hash"] != base[2]["data_config_hash"]:
            raise SystemExit(f"{test['a']}: data hash differs from the L = 12 run")
        if set(run[0]["snapshot_at"].to_list()) != set(base[0]["snapshot_at"].to_list()):
            raise SystemExit(f"{test['a']}: test rows differ from the L = 12 run")
        stats = paired(run[1], base[1], "ic", higher_is_better=True)
        results.append({"test": test["name"], "run_id": candidates[test["a"]], "p": stats.get("p_two_sided"), **stats})
        secondary[test["name"]] = {**{m: paired(run[1], base[1], m, higher_is_better=False) for m in ("rmse", "qlike")},
                                   "dm_squared": diebold_mariano(run[0], base[0], "squared", lags),
                                   "dm_qlike": diebold_mariano(run[0], base[0], "qlike", lags)}
    out = {"label": "exploratory", "exploratory_config_hash": cfg["exploratory_hash"], "reference_run": base_run, "candidates": candidates,
           "missing": missing, "tests": family("E2", "exploratory E2 primary tests (configs/eval_exploratory_v2.yaml e2.tests)", results, alpha),
           "secondary": secondary,
           "note": "Holm over the registered tests that have runs; missing ones are listed and the family is incomplete." if missing else None}
    path = root / "exploratory_e2.json"
    path.write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(path, "missing:", missing)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int)
    parser.add_argument("--seed", type=int, help="train this seed only (one cluster job per seed)")
    parser.add_argument("--tests", action="store_true")
    args = parser.parse_args()
    cfg = load_exploratory_config()
    if args.days is not None:
        if args.days not in cfg["exploratory"]["e2"]["multiday"]["days"]:
            raise SystemExit(f"--days must be one of {cfg['exploratory']['e2']['multiday']['days']}")
        run_multiday(cfg, args.days, args.seed)
    if args.tests:
        run_tests(cfg)


if __name__ == "__main__":
    main()
