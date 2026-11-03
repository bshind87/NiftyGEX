"""Pre-registered RQ1-RQ3 tests (configs/eval.yaml, multiple_testing) on the stored forecast and neural runs.

RQ1: attention + GRU vs every classical rung with a defined IC; mean per-walk IC gain, 95% CI, paired t; a win
     over the ladder requires every comparison to reject at alpha (intersection-union). QLIKE, RMSE, walks won and
     Diebold-Mariano are supporting.
RQ2: DeepSets vs self-attention (interactions) and self-attention vs attention + GRU (history); Holm within the family.
RQ3: per-walk IC gain of attention + GRU over tuned LightGBM, walks after vs before the rule-change marker (Welch t).
Writes outputs/results/rq_tests.json (--tag dev reads neural_comparison_dev.json and writes rq_tests_dev.json).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import ttest_ind

from niftygex_eval.config import load_eval_config
from niftygex_eval.forecast.compare import diebold_mariano, paired
from niftygex_eval.multiple_testing import adjust_family

LADDER = ["persistence", "har", "har_gex", "lightgbm"]


def load_runs(root: Path, comparison: str) -> dict[str, tuple[pl.DataFrame, pl.DataFrame]]:
    runs = json.loads((root / comparison).read_text())["runs"]
    return {name: (pl.read_parquet(root / "runs" / run / "predictions.parquet"), pl.read_parquet(root / "runs" / run / "walk_metrics.parquet"))
            for name, run in runs.items()}


def comparison(a: tuple, b: tuple, max_lags: int) -> dict:
    (pred_a, walk_a), (pred_b, walk_b) = a, b
    return {"ic": paired(walk_a, walk_b, "ic", higher_is_better=True), "qlike": paired(walk_a, walk_b, "qlike", higher_is_better=False),
            "rmse": paired(walk_a, walk_b, "rmse", higher_is_better=False),
            "dm_squared": diebold_mariano(pred_a, pred_b, "squared", max_lags), "dm_qlike": diebold_mariano(pred_a, pred_b, "qlike", max_lags)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="main", help="neural run tag: main (default) or dev")
    tag = parser.parse_args().tag
    suffix = "" if tag == "main" else f"_{tag}"
    cfg = load_eval_config()
    evaluation, root = cfg["eval"], Path(cfg["results_root"])
    settings, forecast = evaluation["multiple_testing"], evaluation["forecast"]
    alpha, max_lags, regime = float(settings["alpha"]), int(forecast["dm_max_lags"]), forecast["regime_column"]
    classical, neural = load_runs(root, "forecast_comparison.json"), load_runs(root, f"neural_comparison{suffix}.json")
    hashes = {json.loads((root / f).read_text())["eval_config_hash"] for f in ("forecast_comparison.json", f"neural_comparison{suffix}.json")}
    if len(hashes) != 1:
        raise SystemExit(f"Forecast and neural runs used different eval configs {sorted(h[:12] for h in hashes)}; rerun scripts/run_forecast.py")
    model = "attention_gru"

    rq1 = {f"{model}_vs_{rung}": comparison(neural[model], classical[rung], max_lags) for rung in LADDER}
    members = [{"test": name, "p": c["ic"].get("p_two_sided"), "estimate": c["ic"].get("mean_diff")} for name, c in rq1.items()]
    wins_all = all(m["p"] is not None and m["p"] < alpha and m["estimate"] > 0 for m in members)

    rq2_pairs = {"attention_vs_deepsets (interactions)": ("attention", "deepsets"), "attention_gru_vs_attention (history)": (model, "attention")}
    rq2 = {name: comparison(neural[a], neural[b], max_lags) for name, (a, b) in rq2_pairs.items()}
    rq2_family = adjust_family("RQ2", settings["families"]["RQ2"],
                               [{"test": n, "p": c["ic"].get("p_two_sided"), "estimate": c["ic"].get("mean_diff"), "primary": True} for n, c in rq2.items()], alpha)

    gain = (neural[model][1].select("walk_id", pl.col("ic").alias("a"))
            .join(classical["lightgbm"][1].select("walk_id", pl.col("ic").alias("b")), on="walk_id")
            .join(neural[model][0].group_by("walk_id").agg(pl.col(regime).cast(pl.Float64).mean().alias("after_share")), on="walk_id")
            .with_columns((pl.col("a") - pl.col("b")).alias("gain"), (pl.col("after_share") >= 0.5).alias("after")))
    after, before = gain.filter(pl.col("after"))["gain"].to_numpy(), gain.filter(~pl.col("after"))["gain"].to_numpy()
    welch = ttest_ind(after, before, equal_var=False) if after.size > 1 and before.size > 1 else None
    rq3 = {"walks_after": int(after.size), "walks_before": int(before.size), "mean_gain_after": float(np.mean(after)) if after.size else None,
           "mean_gain_before": float(np.mean(before)) if before.size else None, "welch_t": float(welch.statistic) if welch else None,
           "p_two_sided": float(welch.pvalue) if welch else None, "per_walk": gain.sort("walk_id").to_dicts()}

    out = {"eval_config_hash": hashes.pop(), "alpha": alpha, "tag": tag,
           "RQ1": {"description": settings["families"]["RQ1"], "comparisons": rq1, "beats_every_rung": wins_all},
           "RQ2": {"description": settings["families"]["RQ2"], "comparisons": rq2, "holm": rq2_family},
           "RQ3": {"description": settings["families"]["RQ3"], **rq3}}
    path = root / f"rq_tests{suffix}.json"
    path.write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(path, "| RQ1 beats every rung:", wins_all)


if __name__ == "__main__":
    main()
