"""Exploratory checks A1-A4 and reported results B5-B8 (configs/eval_exploratory_v3.yaml).

  --parts har_iv har_iv_plus_chain har_gex_naive har_gex_absolute har_gex_inferred     Mac, CPU (A1, B7)
  --parts residual_deepsets_no_iv residual_attention_no_iv --seed 11                    Explorer, one job per seed (A2)
  --parts attention_gru_patience8 --horizon 1 --seed 11                                 Explorer, one job per seed (A4)
  --parts tests                                                                         all tests and reported results
  --walks 2                                                                             development run on the last N walks (tag dev)

Next-day parts use the same table, walks and rows as the main ladder; A4 uses E3's intraday table and walks with
patience 8 and at most 60 epochs. Every run is resumable per walk and written to the results store (stage
exploratory_v3); outputs/results/exploratory_v3_runs.json lists them (per-seed cluster runs are listed by
scripts/merge_seeds.py). The tests write outputs/results/exploratory_v3.json. Everything here is exploratory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import t as student_t

from niftygex_eval.compute import spec_extras, summary_extras, with_seed
from niftygex_eval.config import eval_hash, load_exploratory_v3_config
from niftygex_eval.exploratory import combine, family, residual_ic_by_walk
from niftygex_eval.forecast.baselines import FallbackLinear, Linear, TunedLightGBM, make_baseline
from niftygex_eval.forecast.compare import diebold_mariano, paired
from niftygex_eval.forecast.metrics import regression_metrics
from niftygex_eval.forecast.residual import ResidualForecaster
from niftygex_eval.forecast.runner import run_walks
from niftygex_eval.forecast.table import forecast_table
from niftygex_eval.harness import build_walks
from niftygex_eval.results import ResultsStore

CPU_PARTS = ["har_iv", "har_iv_plus_chain", "har_gex_naive", "har_gex_absolute", "har_gex_inferred"]
NO_IV_PARTS = ["residual_deepsets_no_iv", "residual_attention_no_iv"]
E3_PART = "attention_gru_patience8"
PARTS = [*CPU_PARTS, *NO_IV_PARTS, E3_PART]
DEVICES = {"residual_deepsets_no_iv": "cpu", "residual_attention_no_iv": "mps", E3_PART: "mps"}
RUNS_INDEX = "exploratory_v3_runs.json"


def har_iv(cfg: dict) -> FallbackLinear:
    a1 = cfg["exploratory_v3"]["a1_har_iv"]
    return FallbackLinear(a1["model"], list(cfg["eval"]["forecast"]["har_columns"]), a1["fallback_order"])


def no_iv_neural(cfg: dict) -> dict:
    neural, drop = cfg["eval"]["neural"], list(cfg["exploratory_v3"]["a2_token_correction_no_iv"]["drop_token_features"])
    if unknown := sorted(set(drop) - set(neural["token_features"])):
        raise SystemExit(f"unknown token fields to drop: {unknown}")
    return {**neural, "token_features": [c for c in neural["token_features"] if c not in drop]}


def e3_neural(cfg: dict) -> dict:
    """E3's neural settings (window, state with time of day, devices) with the v3 A4 training budget."""
    neural, e3, a4 = cfg["eval"]["neural"], cfg["exploratory"]["e3"], cfg["exploratory_v3"]["a4_e3_training_budget"]
    return {**neural, "training": dict(a4["training"]), "window": int(e3["window"]), "devices": dict(e3["devices"]),
            "state_columns": list(neural["state_columns"]) + list(e3["extra_state_columns"])}


def device_for(part: str) -> str:
    from niftygex_eval.nn.trainer import resolve_device

    return resolve_device(DEVICES[part]) if part in DEVICES else "cpu"


def make_forecaster(part: str, cfg: dict, summary_columns: list[str], inputs, device: str, neural: dict):
    forecast, v3 = cfg["eval"]["forecast"], cfg["exploratory_v3"]
    if part == "har_iv":
        return har_iv(cfg)
    if part == "har_iv_plus_chain":
        chain = v3["a1_har_iv"]["chain_correction"]
        base = TunedLightGBM(list(chain["lightgbm_columns"]), dict(forecast["lightgbm"]), dict(forecast["tuning"]["grid"]))
        return ResidualForecaster(part, base, lambda: har_iv(cfg))
    if part.startswith("har_gex_"):
        columns = v3["reported"]["b7_gex_beyond_recent_volatility"]["by_convention"][part]
        return Linear(part, list(forecast["har_columns"]), list(columns), alphas=list(forecast["ridge_alphas"]))
    if part in NO_IV_PARTS:
        from niftygex_eval.nn.extra_models import build_extra_model
        from niftygex_eval.nn.trainer import NeuralForecaster

        name = part.removeprefix("residual_").removesuffix("_no_iv")
        factory = lambda n, n_tokens, n_state, config: build_extra_model(n, n_tokens, n_state, config)
        return ResidualForecaster(part, NeuralForecaster(name, inputs, neural, device=device, factory=factory),
                                  lambda: make_baseline("har", forecast, summary_columns))
    raise ValueError(part)


def common_spec(cfg: dict, part: str, device: str, data_hash: str) -> dict:
    return {"stage": "exploratory_v3", "label": "exploratory", "part": part, "device": device, "data_config_hash": data_hash,
            "eval_config_hash": eval_hash(cfg["eval"]), "exploratory_config_hash": cfg["exploratory_hash"],
            "exploratory_v3_config_hash": cfg["exploratory_v3_hash"], "harness": cfg["eval"]["harness"]}


def record(index_path: Path, key: str, run_id: str) -> None:
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    index_path.write_text(json.dumps({**index, key: run_id}, indent=2) + "\n")


def run_next_day(parts: list[str], cfg: dict, seed: int | None, dev_walks: int | None = None) -> None:
    from niftygex_eval.nn.dataset import NeuralInputs, build_inputs

    evaluation, forecast = cfg["eval"], cfg["eval"]["forecast"]
    table, outcome, summary_columns, hashes = forecast_table(cfg)
    walks = build_walks(table["cycle"].unique().to_list(), **evaluation["harness"])
    walks = walks[-dev_walks:] if dev_walks else walks
    neural, inputs = with_seed(no_iv_neural(cfg), seed), None
    if set(parts) & set(NO_IV_PARTS):
        full = build_inputs(table, Path(cfg["data_config"]["paths"]["features_root"]) / "tokens", neural)
        inputs = NeuralInputs(full.tokens, full.mask, full.state[:, :0], full.row)   # tokens only, an empty state vector
    store, tuning = ResultsStore(cfg["results_root"]), forecast["tuning"]
    for part in parts:
        device = device_for(part)
        spec = {**common_spec(cfg, part, device, hashes["features"]), "model": part, "target": forecast["target"], "outcome": outcome,
                **({"tag": "dev", "walks_run": [w.walk_id for w in walks]} if dev_walks else {})}
        if part == "har_iv" or part == "har_iv_plus_chain":
            spec["a1"] = cfg["exploratory_v3"]["a1_har_iv"]
        elif part.startswith("har_gex_"):
            spec["columns"] = cfg["exploratory_v3"]["reported"]["b7_gex_beyond_recent_volatility"]["by_convention"][part]
        else:
            spec.update({"a2": cfg["exploratory_v3"]["a2_token_correction_no_iv"], "token_features": neural["token_features"],
                         "training": neural["training"], "model_config": neural["model"], **spec_extras()})
        predictions, per_walk = run_walks(
            table, walks, lambda: make_forecaster(part, cfg, summary_columns, inputs, device, neural), outcome,
            inner_validation_cycles=int(tuning["inner_validation_cycles"]), inner_embargo_cycles=int(tuning["inner_embargo_cycles"]),
            keep=[forecast["regime_column"]], checkpoint=store.checkpoint_dir(spec))
        finish(store, spec, predictions, per_walk, cfg, part, seed if part in NO_IV_PARTS else None, listed=not dev_walks)


def run_e3(cfg: dict, horizon: int, seed: int | None, dev_walks: int | None = None) -> None:
    from niftygex_eval.forecast.intraday import for_horizon, intraday_table
    from niftygex_eval.nn.dataset import build_intraday_inputs
    from niftygex_eval.nn.trainer import NeuralForecaster

    evaluation, e3 = cfg["eval"], cfg["exploratory"]["e3"]
    if horizon not in [int(h) for h in cfg["exploratory_v3"]["a4_e3_training_budget"]["horizons"]]:
        raise SystemExit(f"--horizon {horizon} is not registered for A4")
    table, _, hashes = intraday_table(cfg)
    table = for_horizon(table, horizon)
    outcome = e3["outcome"].replace("{h}", str(horizon))
    walks = build_walks(table["cycle"].unique().to_list(), **evaluation["harness"])
    walks = walks[-dev_walks:] if dev_walks else walks
    neural = with_seed(e3_neural(cfg), seed)
    paths = cfg["data_config"]["paths"]
    choice = pl.read_parquet(Path(paths["exposure_root"]) / "intraday" / "expiry_choice" / "expiry_choice.parquet")
    inputs, mask_counts = build_intraday_inputs(table, Path(paths["features_root"]) / "tokens", neural, neural["window"], choice,
                                                cfg["data_config"]["data"]["snapshot_grid"]["t2_daily"])
    device = device_for(E3_PART)
    part = f"{E3_PART}_{horizon}h"
    spec = {**common_spec(cfg, part, device, hashes["features_intraday"]), "model": E3_PART, "horizon_hours": horizon, "outcome": outcome,
            "training": neural["training"], "model_config": neural["model"], "window": neural["window"], "token_mask_counts": mask_counts,
            **spec_extras(), **({"tag": "dev", "walks_run": [w.walk_id for w in walks]} if dev_walks else {})}
    store, tuning = ResultsStore(cfg["results_root"]), evaluation["forecast"]["tuning"]
    predictions, per_walk = run_walks(
        table, walks, lambda: NeuralForecaster("attention_gru", inputs, neural, device=device), outcome,
        inner_validation_cycles=int(tuning["inner_validation_cycles"]), inner_embargo_cycles=int(tuning["inner_embargo_cycles"]),
        keep=[evaluation["forecast"]["regime_column"], "minutes_since_open"], checkpoint=store.checkpoint_dir(spec))
    finish(store, spec, predictions, per_walk, cfg, part, seed, listed=not dev_walks)


def finish(store: ResultsStore, spec: dict, predictions: pl.DataFrame, per_walk: pl.DataFrame, cfg: dict, part: str, seed: int | None,
           listed: bool = True) -> None:
    seconds = float(per_walk["seconds"].sum())   # summed over walks: a run may span several processes
    pooled = regression_metrics(predictions["y"].to_numpy(), predictions["prediction"].to_numpy())
    run = store.write_run(spec, {"predictions": predictions, "walk_metrics": per_walk},
                          {"pooled": pooled, "seconds": seconds, **summary_extras(per_walk)}).name
    store.clear_checkpoint(spec)
    if seed is None and listed:   # per-seed cluster runs are listed by scripts/merge_seeds.py; dev runs are not listed
        record(Path(cfg["results_root"]) / RUNS_INDEX, part, run)
    print(f"{part}: walks {per_walk.height}, mean IC {per_walk['ic'].mean():.3f}, {seconds / 60:.1f} min on {spec['device']}", flush=True)


# ---------------------------------------------------------------------------------------------------------- tests
def load_run(root: Path, run_id: str) -> dict:
    run_dir = root / "runs" / run_id
    return {"run_id": run_id, "predictions": pl.read_parquet(run_dir / "predictions.parquet"),
            "walks": pl.read_parquet(run_dir / "walk_metrics.parquet"), "spec": json.loads((run_dir / "run.json").read_text())["spec"]}


def compare(a: dict, b: dict, lags: int) -> dict:
    """IC, RMSE and QLIKE paired per walk, plus Diebold-Mariano on squared and QLIKE losses (A minus B)."""
    return {**{m: paired(a["walks"], b["walks"], m, higher_is_better=(m == "ic")) for m in ("ic", "rmse", "qlike")},
            "dm_squared": diebold_mariano(a["predictions"], b["predictions"], "squared", lags),
            "dm_qlike": diebold_mariano(a["predictions"], b["predictions"], "qlike", lags)}


def same_rows(runs: dict[str, dict], what: str) -> None:
    rows = {frozenset(r["predictions"]["snapshot_at"].to_list()) for r in runs.values()}
    hashes = {r["spec"]["data_config_hash"] for r in runs.values()}
    if len(rows) != 1 or len(hashes) != 1:
        raise SystemExit(f"{what}: runs do not share the same test rows and data hash ({len(rows)} row sets, hashes {hashes})")


def detectable(per_walk_a: pl.DataFrame, per_walk_b: pl.DataFrame, power: float) -> dict:
    """Upper end of the 95% interval and the gain a two-sided paired t-test at 5% detects with the given power."""
    stats = paired(per_walk_a, per_walk_b, "ic", higher_is_better=True)
    n = stats["walks"]
    t_crit, t_power = float(student_t.ppf(0.975, n - 1)), float(student_t.ppf(power, n - 1))
    se = (stats["ci95"][1] - stats["mean_diff"]) / t_crit
    return {"walks": n, "mean_diff": stats["mean_diff"], "ci95": stats["ci95"], "se": se, "ci_half_width": t_crit * se,
            "ruled_out_above": stats["ci95"][1], "detectable_at_power": (t_crit + t_power) * se, "power": power}


def run_tests(cfg: dict) -> None:
    root, v3, alpha = Path(cfg["results_root"]), cfg["exploratory_v3"], float(cfg["exploratory_v3"]["alpha"])
    lags = int(cfg["eval"]["forecast"]["dm_max_lags"])
    refs = v3["references"]
    classical = json.loads((root / refs["classical"]).read_text())
    neural = json.loads((root / refs["neural"]).read_text())
    e1 = json.loads((root / refs["e1"]).read_text())
    if classical["eval_config_hash"] != refs["main_eval_config_hash"] or e1["exploratory_config_hash"] != refs["exploratory_v2_config_hash"]:
        raise SystemExit("reference runs are not from the registered main and v2 configurations")
    index_path = root / RUNS_INDEX
    explored = json.loads(index_path.read_text()) if index_path.exists() else {}

    day = {name: load_run(root, run_id) for name, run_id in {**classical["runs"], "attention_gru": neural["runs"]["attention_gru"],
                                                             "residual_deepsets": e1["runs"]["residual_deepsets"],
                                                             "residual_attention": e1["runs"]["residual_attention"]}.items()}
    day.update({p: load_run(root, explored[p]) for p in [*CPU_PARTS, *NO_IV_PARTS] if p in explored})
    same_rows(day, "next-day runs")
    combined_predictions, combined_walks = combine({m: day[m]["predictions"] for m in v3["a3_combination"]["members"]})
    day["combination"] = {"run_id": None, "predictions": combined_predictions, "walks": combined_walks, "spec": {}}

    intraday, best = {}, {}
    for h, name in zip(v3["a4_e3_training_budget"]["horizons"], refs["e3"]):
        block = json.loads((root / name).read_text())
        best[int(h)] = block["best_classical"]
        runs = {"best_classical": block["runs"][block["best_classical"]], "attention_gru_patience3": block["runs"]["attention_gru"]}
        if f"{E3_PART}_{h}h" in explored:
            runs["attention_gru_patience8"] = explored[f"{E3_PART}_{h}h"]
        intraday[int(h)] = {k: load_run(root, r) for k, r in runs.items()}
        same_rows(intraday[int(h)], f"E3 {h}h runs")

    results, pending = [], []
    for test in v3["tests"]:
        pool = intraday[int(test["horizon"])] if "horizon" in test else day
        if test["a"] not in pool or test["b"] not in pool:
            pending.append(test["name"])
            continue
        stats = paired(pool[test["a"]]["walks"], pool[test["b"]]["walks"], "ic", higher_is_better=True)
        results.append({"test": test["name"], "p": stats.get("p_two_sided"), **stats})
    tests = (family("v3", "exploratory v3 primary tests (configs/eval_exploratory_v3.yaml tests)", results, alpha) if not pending
             else {"family": "v3", "complete": False, "pending": pending, "unadjusted": results})

    secondary = {}
    for test in v3["tests"]:
        pool = intraday[int(test["horizon"])] if "horizon" in test else day
        if test["a"] in pool and test["b"] in pool:
            secondary[test["name"]] = compare(pool[test["a"]], pool[test["b"]], lags)
    pairs = [("har_iv_plus_chain", "har"), ("residual_deepsets_no_iv", "residual_deepsets"), ("residual_attention_no_iv", "residual_attention")]
    for a, b in pairs:
        if a in day:
            secondary[f"{a}_vs_{b}"] = compare(day[a], day[b], lags)
    epochs = {}
    for h, pool in intraday.items():
        if "attention_gru_patience8" in pool:
            secondary[f"e3_patience8_vs_patience3_{h}h"] = compare(pool["attention_gru_patience8"], pool["attention_gru_patience3"], lags)
        epochs[f"{h}h"] = {k: {"stop_epoch_mean": float(pool[k]["walks"]["stop_epoch_mean"].mean()),
                               "stop_epoch_max": float(pool[k]["walks"]["stop_epoch_max"].max())}
                           for k in ("attention_gru_patience3", "attention_gru_patience8") if k in pool}
    residual_ic = {p: float(residual_ic_by_walk(day[p]["predictions"], day["har"]["predictions"])["residual_ic"].mean())
                   for p in ("residual_deepsets", "residual_attention", *NO_IV_PARTS) if p in day}
    if "har_iv_plus_chain" in day and "har_iv" in day:
        residual_ic["har_iv_plus_chain (vs HAR-IV)"] = float(
            residual_ic_by_walk(day["har_iv_plus_chain"]["predictions"], day["har_iv"]["predictions"])["residual_ic"].mean())

    # A1 fallback: test rows without ATM IV are predicted by plain HAR.
    table, _, _, _ = forecast_table(cfg)
    test_rows = day["har"]["predictions"].select("snapshot_at")
    levels = v3["a1_har_iv"]["fallback_order"]
    has_iv = FallbackLinear.has(test_rows.join(table, on="snapshot_at", how="left"), levels[0])
    fallback = {"test_rows": int(test_rows.height), "rows_without_extra_columns": int((~has_iv).sum()), "columns": levels[0]}

    reported = {}
    b7 = v3["reported"]["b7_gex_beyond_recent_volatility"]
    reported["b7"] = {"har_gex_vs_har": compare(day["har_gex"], day["har"], lags),
                      **{f"{p}_vs_har": compare(day[p], day["har"], lags) for p in b7["by_convention"] if p in day}}
    b8 = v3["reported"]["b8_detectable_effect"]
    reported["b8"] = {m: detectable(day["attention_gru"]["walks"], day[m]["walks"], float(b8["power"]))
                      for m in ("persistence", "har", "har_gex", "lightgbm")}
    b6 = v3["reported"]["b6_state_only_losses"]
    reported["b6"] = {m: {k: e1["summary"][m][k] for k in b6["metrics"]} for m in [*b6["models"], "har", "lightgbm"]}
    reported["b6_tests"] = {k: v for k, v in e1["secondary"].items() if k.startswith("state_mlp_vs_")}
    reported["b5"] = e1[v3["reported"]["b5_conditional_ic"]["field"]]

    summary = {name: {m: float(run["walks"][m].mean()) for m in ("ic", "rmse", "qlike")} for name, run in day.items()}
    for h, pool in intraday.items():
        summary.update({f"e3_{h}h_{k}": {m: float(run["walks"][m].mean()) for m in ("ic", "rmse", "qlike")} for k, run in pool.items()})
    provenance = {name: {"run_id": run["run_id"], "git_commit": run["spec"].get("git_commit"), "git_dirty": run["spec"].get("git_dirty"),
                         "exploratory_v3_config_hash": run["spec"].get("exploratory_v3_config_hash"), "gpu_type": run["spec"].get("gpu_type"),
                         "device": run["spec"].get("device")}
                  for name, run in [*day.items(), *[(f"e3_{h}h_{k}", r) for h, p in intraday.items() for k, r in p.items()]] if run["run_id"]}
    out = {"label": "exploratory", "exploratory_v3_config_hash": cfg["exploratory_v3_hash"], "exploratory_config_hash": cfg["exploratory_hash"],
           "eval_config_hash": classical["eval_config_hash"], "best_classical": {f"{h}h": m for h, m in best.items()},
           "har_iv_fallback": fallback, "tests": tests, "secondary": secondary, "stop_epochs": epochs, "residual_ic": residual_ic,
           "reported": reported, "summary": summary, "runs": provenance}
    path = root / "exploratory_v3.json"
    path.write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parts", nargs="+", required=True, choices=[*PARTS, "tests"])
    parser.add_argument("--horizon", type=int, help="A4: the E3 horizon in hours")
    parser.add_argument("--seed", type=int, help="train this seed only (one cluster job per seed)")
    parser.add_argument("--walks", type=int, help="development run on the last N walks (tag dev, not listed for the tests)")
    args = parser.parse_args()
    cfg = load_exploratory_v3_config()
    day_parts = [p for p in args.parts if p in CPU_PARTS or p in NO_IV_PARTS]
    if day_parts:
        if args.seed is not None and set(day_parts) & set(CPU_PARTS):
            raise SystemExit("--seed applies to the neural parts only")
        run_next_day(day_parts, cfg, args.seed, args.walks)
    if E3_PART in args.parts:
        if args.horizon is None:
            raise SystemExit(f"{E3_PART} needs --horizon")
        run_e3(cfg, args.horizon, args.seed, args.walks)
    if "tests" in args.parts:
        run_tests(cfg)


if __name__ == "__main__":
    main()
