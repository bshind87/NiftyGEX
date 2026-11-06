"""Exploratory E3 tests (configs/eval_exploratory_v2.yaml e3.tests): attention + GRU vs the best classical rung per
horizon on per-walk IC (the walk is the unit), Holm across the two horizons once both exist; secondary scores."""
from __future__ import annotations

import json
from typing import Any

import polars as pl

from niftygex_eval.exploratory import family
from niftygex_eval.forecast.compare import by_period, diebold_mariano, paired
from niftygex_eval.forecast.metrics import ic, regression_metrics

CLASSICAL = ("persistence", "intraday_har", "har_gex", "lightgbm")
NEURAL = ("deepsets", "attention", "attention_gru")


def _load(root, run_id: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    directory = root / "runs" / run_id
    return pl.read_parquet(directory / "predictions.parquet"), pl.read_parquet(directory / "walk_metrics.parquet")


def horizon_results(cfg: dict, horizon: int) -> dict[str, Any]:
    root = cfg["results_root"]
    runs = json.loads((root / "exploratory_e3_runs.json").read_text()).get(f"{horizon}h", {})
    missing = [m for m in (*CLASSICAL, *NEURAL) if m not in runs]
    if missing:
        raise SystemExit(f"E3 {horizon}h runs missing: {missing}")
    loaded = {m: _load(root, r) for m, r in runs.items()}
    if len({frozenset(p["snapshot_at"].to_list()) for p, _ in loaded.values()}) != 1:
        raise SystemExit(f"E3 {horizon}h runs do not share the same test rows")
    means = {m: float(w["ic"].mean()) for m, (_, w) in loaded.items()}
    best = max(CLASSICAL, key=lambda m: means[m])
    ours_p, ours_w = loaded["attention_gru"]
    primary = paired(ours_w, loaded[best][1], "ic", higher_is_better=True)
    # Snapshots overlap within a day: Newey-West lags cover one day of 15-minute snapshots.
    lags = 13
    secondary = {m: {"ic": paired(ours_w, w, "ic", True), "rmse": paired(ours_w, w, "rmse", False), "qlike": paired(ours_w, w, "qlike", False),
                     "dm_squared": diebold_mariano(ours_p, p, "squared", lags), "dm_qlike": diebold_mariano(ours_p, p, "qlike", lags)}
                 for m, (p, w) in loaded.items() if m != "attention_gru"}
    regime = cfg["eval"]["forecast"]["regime_column"]
    by_time = {m: [{"minutes_since_open": int(k), "ic": ic(g["y"].to_numpy(), g["prediction"].to_numpy()), "rows": g.height}
                   for (k,), g in p.sort("minutes_since_open").group_by(["minutes_since_open"], maintain_order=True)]
               for m, (p, _) in loaded.items()}
    scores = {m: {"ic_mean_per_walk": means[m], "rmse_mean_per_walk": float(w["rmse"].mean()), "qlike_mean_per_walk": float(w["qlike"].mean()),
                  "pooled": regression_metrics(p["y"].to_numpy(), p["prediction"].to_numpy()), "walks": w.height,
                  "ic_positive_walks": int((w["ic"].is_finite() & (w["ic"] > 0)).sum()), "by_regime": by_period(p, regime)}
              for m, (p, w) in loaded.items()}
    return {"horizon_hours": horizon, "runs": runs, "best_classical": best, "primary": {"test": f"attention_gru_vs_best_classical_{horizon}h",
            "best_classical": best, "p": primary.get("p_two_sided"), **primary}, "secondary": secondary, "scores": scores,
            "ic_by_time_of_day": by_time, "dm_lags": lags}


def run_tests(cfg: dict, horizon: int) -> None:
    root = cfg["results_root"]
    result = {"label": "exploratory", "exploratory_config_hash": cfg["exploratory_hash"], **horizon_results(cfg, horizon)}
    (root / f"exploratory_e3_{horizon}h.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(root / f"exploratory_e3_{horizon}h.json", "best classical:", result["best_classical"])
    done = [h for h in cfg["exploratory"]["e3"]["horizons"] if (root / f"exploratory_e3_{h}h.json").exists()]
    if len(done) == len(cfg["exploratory"]["e3"]["horizons"]):
        members = [json.loads((root / f"exploratory_e3_{h}h.json").read_text())["primary"] for h in done]
        out = {"label": "exploratory", "exploratory_config_hash": cfg["exploratory_hash"],
               "tests": family("E3", "attention + GRU vs the best classical rung per horizon, Holm across horizons", members,
                               float(cfg["exploratory"]["alpha"]))}
        (root / "exploratory_e3.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
        print(root / "exploratory_e3.json")

