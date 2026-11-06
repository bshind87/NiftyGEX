"""Exploratory E1: diagnostics on the next-day target (configs/eval_exploratory_v2.yaml, section e1).

  --parts residual_lightgbm residual_deepsets residual_attention state_mlp hybrid   model runs (resume per walk)
  --parts tests     registered E1 tests, forecast combination and conditional IC from the saved predictions
  --seed 11         cluster job: one training seed (neural parts); scripts/merge_seeds.py averages the seeds and lists them

Model runs use the main ladder's table, walks and rows. Each is written to the results store (stage
exploratory_e1) and listed in outputs/results/exploratory_e1_runs.json; tests go to outputs/results/exploratory_e1.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from niftygex_eval.compute import spec_extras, summary_extras, with_seed
from niftygex_eval.config import eval_hash, load_exploratory_config
from niftygex_eval.exploratory import combine, family, one_sample, pooled_ic_bootstrap, residual_ic_by_walk
from niftygex_eval.forecast.baselines import TunedLightGBM, make_baseline
from niftygex_eval.forecast.compare import diebold_mariano, paired
from niftygex_eval.forecast.metrics import regression_metrics
from niftygex_eval.forecast.residual import HAR_FORECAST, ResidualForecaster
from niftygex_eval.forecast.runner import run_walks
from niftygex_eval.forecast.table import forecast_table
from niftygex_eval.harness import build_walks, split
from niftygex_eval.results import ResultsStore

MODEL_PARTS = ["residual_lightgbm", "residual_deepsets", "residual_attention", "state_mlp", "hybrid"]
DEVICES = {"residual_lightgbm": "cpu", "residual_deepsets": "cpu", "residual_attention": "mps", "state_mlp": "cpu", "hybrid": "mps"}
RUN_NAMES = {"hybrid": "hybrid_attention_gru"}


def make_forecaster(part: str, cfg: dict, summary_columns: list[str], inputs: dict, device: str, neural: dict):
    from niftygex_eval.nn.extra_models import build_extra_model
    from niftygex_eval.nn.trainer import NeuralForecaster

    forecast, e1 = cfg["eval"]["forecast"], cfg["exploratory"]["e1"]
    har = lambda: make_baseline("har", forecast, summary_columns)
    factory = lambda name, n_tokens, n_state, config: build_extra_model(name, n_tokens, n_state, config)
    if part == "residual_lightgbm":
        base = TunedLightGBM(list(e1["residual"]["lightgbm_columns"]), dict(forecast["lightgbm"]), dict(forecast["tuning"]["grid"]))
        return ResidualForecaster(part, base, har)
    if part in ("residual_deepsets", "residual_attention"):
        name = part.removeprefix("residual_")
        return ResidualForecaster(part, NeuralForecaster(name, inputs["tokens_only"], neural, device=device, factory=factory), har)
    if part == "state_mlp":
        return NeuralForecaster("state_mlp", inputs["state_only"], neural, device=device, factory=factory)
    if part == "hybrid":
        model = NeuralForecaster("hybrid_attention_gru", inputs["full"], neural, device=device, factory=factory, frame_state=[HAR_FORECAST])
        return ResidualForecaster("hybrid_attention_gru", model, har)
    raise ValueError(part)


def build_neural_inputs(parts: list[str], table: pl.DataFrame, cfg: dict) -> dict:
    from niftygex_eval.nn.dataset import NeuralInputs, build_inputs

    if not set(parts) & {"residual_deepsets", "residual_attention", "state_mlp", "hybrid"}:
        return {}
    neural = cfg["eval"]["neural"]
    tokens_root = Path(cfg["data_config"]["paths"]["features_root"]) / "tokens"
    out = {}
    if set(parts) & {"residual_deepsets", "residual_attention", "hybrid"}:
        full = build_inputs(table, tokens_root, neural)
        out["full"] = full
        # E1.1 neural residual models: tokens only, an empty state vector.
        out["tokens_only"] = NeuralInputs(full.tokens, full.mask, full.state[:, :0], full.row)
    if "state_mlp" in parts:
        one = build_inputs(table, tokens_root, neural, window=1)
        out["state_only"] = one
    return out


def device_for(wanted: str) -> str:
    from niftygex_eval.nn.trainer import resolve_device

    return resolve_device(wanted)


def load_run(root: Path, run_id: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    return (pl.read_parquet(root / "runs" / run_id / "predictions.parquet"), pl.read_parquet(root / "runs" / run_id / "walk_metrics.parquet"))


def run_models(parts: list[str], cfg: dict, seed: int | None = None) -> None:
    evaluation, forecast = cfg["eval"], cfg["eval"]["forecast"]
    neural = with_seed(evaluation["neural"], seed)
    table, outcome, summary_columns, hashes = forecast_table(cfg)
    walks = build_walks(table["cycle"].unique().to_list(), **evaluation["harness"])
    inputs = build_neural_inputs(parts, table, cfg)
    store, tuning = ResultsStore(cfg["results_root"]), forecast["tuning"]
    index_path = cfg["results_root"] / "exploratory_e1_runs.json"
    runs = json.loads(index_path.read_text()) if index_path.exists() else {}
    for part in parts:
        device = device_for(DEVICES[part]) if part != "residual_lightgbm" else "cpu"
        spec = {"stage": "exploratory_e1", "label": "exploratory", "model": RUN_NAMES.get(part, part), "part": part, "device": device,
                "data_config_hash": hashes["features"], "eval_config_hash": eval_hash(evaluation),
                "exploratory_config_hash": cfg["exploratory_hash"], "target": forecast["target"], "outcome": outcome,
                "harness": evaluation["harness"], "e1": cfg["exploratory"]["e1"],
                **({"training": neural["training"], "model_config": neural["model"]} if part != "residual_lightgbm" else {}), **spec_extras()}
        predictions, per_walk = run_walks(
            table, walks, lambda: make_forecaster(part, cfg, summary_columns, inputs, device, neural), outcome,
            inner_validation_cycles=int(tuning["inner_validation_cycles"]), inner_embargo_cycles=int(tuning["inner_embargo_cycles"]),
            keep=[forecast["regime_column"]], checkpoint=store.checkpoint_dir(spec))
        seconds = float(per_walk["seconds"].sum())  # summed over walks: a run may span several processes
        pooled = regression_metrics(predictions["y"].to_numpy(), predictions["prediction"].to_numpy())
        run = store.write_run(spec, {"predictions": predictions, "walk_metrics": per_walk},
                              {"pooled": pooled, "seconds": seconds, **summary_extras(per_walk)}).name
        store.clear_checkpoint(spec)
        if seed is None:   # per-seed cluster runs are listed by scripts/merge_seeds.py
            runs[part] = run
            index_path.write_text(json.dumps(runs, indent=2) + "\n")
        print(f"{part}: walks {per_walk.height}, mean IC {per_walk['ic'].mean():.3f}, {seconds / 60:.1f} min on {device}", flush=True)


def tercile_labels(table: pl.DataFrame, walks, column: str) -> pl.DataFrame:
    """Each test row's tercile of ``column`` with cutoffs from its walk's training rows (past only)."""
    rows = []
    for walk in walks:
        train, test = split(table, walk)
        values = train[column].drop_nulls().drop_nans()
        if not test.height or values.len() < 3:
            continue
        low, high = float(values.quantile(1 / 3)), float(values.quantile(2 / 3))
        rows.append(test.select("snapshot_at", pl.when(pl.col(column).is_null()).then(None).when(pl.col(column) <= low).then(pl.lit("low"))
                                .when(pl.col(column) <= high).then(pl.lit("mid")).otherwise(pl.lit("high")).alias("gex_tercile")))
    return pl.concat(rows)


def run_tests(cfg: dict) -> None:
    root, e1, alpha = cfg["results_root"], cfg["exploratory"]["e1"], float(cfg["exploratory"]["alpha"])
    classical = json.loads((root / cfg["exploratory"]["references"]["classical"]).read_text())
    neural = json.loads((root / cfg["exploratory"]["references"]["neural"]).read_text())
    if classical["eval_config_hash"] != cfg["exploratory"]["references"]["main_eval_config_hash"]:
        raise SystemExit("The classical reference runs are not from the main eval config")
    index = root / "exploratory_e1_runs.json"
    explored = json.loads(index.read_text()) if index.exists() else {}
    missing = [p for p in MODEL_PARTS if p not in explored]
    # Neural-vs-neural comparisons use the cluster references (same device and GPU type); the rest use the main runs.
    hpc_path = root / cfg["exploratory"]["references"]["neural_hpc"]
    hpc = json.loads(hpc_path.read_text()) if hpc_path.exists() else {"runs": {}}
    runs = {**classical["runs"], **neural["runs"], **{f"{m}_hpc": r for m, r in hpc["runs"].items()},
            **{RUN_NAMES.get(p, p): explored[p] for p in MODEL_PARTS if p in explored}}
    loaded = {name: load_run(root, run_id) for name, run_id in runs.items()}
    data_hashes = {json.loads((root / "runs" / r / "run.json").read_text())["spec"]["data_config_hash"] for r in runs.values()}
    if len(data_hashes) != 1:
        raise SystemExit(f"E1 runs come from different data configurations: {data_hashes}")
    rows = {name: set(p["snapshot_at"].to_list()) for name, (p, _) in loaded.items()}
    if len({frozenset(r) for r in rows.values()}) != 1:
        raise SystemExit("E1 runs do not share the same test rows")

    har_predictions = loaded["har"][0]
    residual = {name: residual_ic_by_walk(loaded[name][0], har_predictions)
                for name in ("residual_lightgbm", "residual_deepsets", "residual_attention") if name in loaded}
    combined_predictions, combined_walks = combine({m: loaded[m][0] for m in e1["combination"]["members"]})
    loaded["combination"] = (combined_predictions, combined_walks)

    # Runs still to come (the cluster's neural parts and references) leave their tests pending; Holm is applied
    # only when the whole registered family is in.
    results, pending = [], []
    for test in e1["tests"]:
        if test["a"] not in loaded or test.get("b", "har") not in loaded:
            pending.append(test["name"])
        elif test.get("vs") == "zero":
            stats = one_sample(residual[test["a"]]["residual_ic"].to_numpy())
            results.append({"test": test["name"], "p": stats.get("p_two_sided"), **stats})
        else:
            stats = paired(loaded[test["a"]][1], loaded[test["b"]][1], test["metric"], higher_is_better=True)
            results.append({"test": test["name"], "p": stats.get("p_two_sided"), **stats})
    tests = (family("E1", "exploratory E1 primary tests (configs/eval_exploratory_v2.yaml e1.tests)", results, alpha) if not pending
             else {"family": "E1", "complete": False, "pending": pending, "unadjusted": results})

    lags = int(cfg["eval"]["forecast"]["dm_max_lags"])
    secondary = {}
    for other in [m for m in ("deepsets_hpc", "attention_hpc", "attention_gru_hpc") if m in loaded and "state_mlp" in loaded]:
        secondary[f"state_mlp_vs_{other}"] = {
            **{m: paired(loaded["state_mlp"][1], loaded[other][1], m, higher_is_better=(m == "ic")) for m in ("ic", "rmse", "qlike")},
            "dm_squared": diebold_mariano(loaded["state_mlp"][0], loaded[other][0], "squared", lags),
            "dm_qlike": diebold_mariano(loaded["state_mlp"][0], loaded[other][0], "qlike", lags)}
    for member in e1["combination"]["members"]:
        secondary[f"combination_vs_{member}"] = {m: paired(combined_walks, loaded[member][1], m, higher_is_better=(m == "ic"))
                                                 for m in ("ic", "rmse", "qlike")}

    table, _, _, _ = forecast_table(cfg)
    walks = build_walks(table["cycle"].unique().to_list(), **cfg["eval"]["harness"])
    conditional_cfg = e1["conditional"]
    subsets = table.select("snapshot_at", "is_expiry_day", "has_flip", "flip_naive_distance_pct", conditional_cfg["sign_column"]).join(
        tercile_labels(table, walks, conditional_cfg["gex_tercile_column"]), on="snapshot_at", how="left")
    flip = float(conditional_cfg["flip_distance_pct"])
    definitions = {
        "expiry_day": pl.col("is_expiry_day") == True, "not_expiry_day": pl.col("is_expiry_day") == False,  # noqa: E712
        "near_flip": (pl.col("has_flip") == True) & (pl.col("flip_naive_distance_pct").abs() <= flip),  # noqa: E712
        "not_near_flip": ~((pl.col("has_flip") == True) & (pl.col("flip_naive_distance_pct").abs() <= flip)).fill_null(False),  # noqa: E712
        "gex_tercile_low": pl.col("gex_tercile") == "low", "gex_tercile_mid": pl.col("gex_tercile") == "mid",
        "gex_tercile_high": pl.col("gex_tercile") == "high",
        "dealer_long_gamma": pl.col(conditional_cfg["sign_column"]) > 0, "dealer_short_gamma": pl.col(conditional_cfg["sign_column"]) < 0,
    }
    boot = cfg["exploratory"]["bootstrap"]
    conditional = {}
    for model in [m for m in [*conditional_cfg["models"], "residual_lightgbm", "hybrid_attention_gru", "combination"] if m in loaded]:
        frame = loaded[model][0].select("snapshot_at", "walk_id", "y", "prediction").join(subsets, on="snapshot_at", how="left")
        conditional[model] = {name: pooled_ic_bootstrap(frame.filter(expr.fill_null(False)), int(boot["reps"]), int(boot["seed"]))
                              for name, expr in definitions.items()}

    summary = {}
    for name in [*runs, "combination"]:
        per_walk = loaded[name][1]
        summary[name] = {m: float(per_walk[m].mean()) for m in ("ic", "rmse", "qlike")}
        if name in residual:
            summary[name]["residual_ic"] = float(residual[name]["residual_ic"].mean())
    out = {"label": "exploratory", "exploratory_config_hash": cfg["exploratory_hash"], "eval_config_hash": classical["eval_config_hash"],
           "data_config_hash": data_hashes.pop(), "runs": runs, "missing_runs": missing, "summary": summary,
           "residual_ic_by_walk": {k: v.to_dicts() for k, v in residual.items()}, "tests": tests, "secondary": secondary,
           "conditional_ic": conditional, "subset_definitions": {k: str(v) for k, v in definitions.items()}}
    path = root / "exploratory_e1.json"
    path.write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", nargs="+", default=MODEL_PARTS, choices=[*MODEL_PARTS, "tests"])
    parser.add_argument("--seed", type=int, help="train this seed only (one cluster job per seed)")
    args = parser.parse_args()
    cfg = load_exploratory_config()
    models = [p for p in args.parts if p != "tests"]
    if models:
        run_models(models, cfg, args.seed)
    if "tests" in args.parts:
        run_tests(cfg)


if __name__ == "__main__":
    main()
