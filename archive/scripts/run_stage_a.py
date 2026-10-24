"""Stage A (T2): HAC regressions of the next-day RV ratio on total GEX; writes a results-store run."""
from __future__ import annotations

import json

import polars as pl

from niftygex_eval.config import eval_hash, load_eval_config
from niftygex_eval.data import modeling_table
from niftygex_eval.results import ResultsStore
from niftygex_eval.stage_a import run_stage_a

MARKER = "regime_post_weekly_expiry_rationalisation"


def main() -> None:
    cfg = load_eval_config()
    stage = cfg["eval"]["stage_a"]
    table, _, hashes = modeling_table(cfg["data_config"], stage["target"])
    results = run_stage_a(table, stage, MARKER)
    spec = {"stage": "stage_a", "target": stage["target"], "model": "ols_hac", "data_config_hash": hashes["features"],
            "eval_config_hash": eval_hash(cfg["eval"]), "params": stage}
    headline = results.filter(pl.col("is_gex_term") & pl.col("sample").is_in(["full", "pre_weekly_expiry_rationalisation", "post_weekly_expiry_rationalisation"]))
    run = ResultsStore(cfg["results_root"]).write_run(spec, {"regressions": results}, {"headline": headline.to_dicts()})
    print(json.dumps({"run": str(run), "rows": results.height}, indent=2))


if __name__ == "__main__":
    main()
