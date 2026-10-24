"""Holm correction within the pre-registered families (configs/eval.yaml) -> outputs/results/multiple_testing.json."""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from niftygex_eval.config import eval_hash, load_eval_config
from niftygex_eval.multiple_testing import adjust_family


def latest_run(root: Path, stage: str, model: str) -> str:
    run = None
    for line in (root / "index.jsonl").read_text().splitlines():
        record = json.loads(line)
        if record["stage"] == stage and record["model"] == model:
            run = record["run_id"]
    if run is None:
        raise SystemExit(f"No {stage}/{model} run in the results store")
    return run


def main() -> None:
    cfg = load_eval_config()
    root = Path(cfg["results_root"])
    settings = cfg["eval"]["multiple_testing"]
    alpha, families = float(settings["alpha"]), settings["families"]
    stage_a_run = latest_run(root, "stage_a", "ols_hac")
    regressions = pl.read_parquet(root / "runs" / stage_a_run / "regressions.parquet")
    ablation = json.loads((root / "t2_ablation.json").read_text())
    placebo = json.loads((root / "t3_placebo.json").read_text())
    listed = json.loads((root / "t3_listed_placebo.json").read_text())
    samples = ("full", "pre_weekly_expiry_rationalisation", "post_weekly_expiry_rationalisation")

    f1 = []
    for convention in ("naive", "inferred"):
        for sample in samples:
            row = regressions.filter((pl.col("spec") == "joint_magnitude_sign") & (pl.col("controls") == "full_controls")
                                     & (pl.col("term") == f"gex_sign_{convention}") & (pl.col("sample") == sample)).row(0, named=True)
            f1.append({"test": f"{convention} sign, {sample}", "estimate": row["coef"], "t": row["t"], "p": row["p"],
                       "primary": convention == "inferred" and sample == "full"})
    comparisons = ("C_vs_B", "C1_vs_B", "C2_vs_B", "B_vs_A")

    def f2(metric: str) -> list[dict]:
        return [{"test": f"{c.replace('_vs_', ' vs ')} ({metric})", "estimate": ablation["paired_tests"][c][metric]["mean_diff"],
                 "t": ablation["paired_tests"][c][metric]["t_paired"], "p": ablation["paired_tests"][c][metric]["p_two_sided"],
                 "wins": f"{ablation['paired_tests'][c][metric]['a_wins']}/{ablation['paired_tests'][c][metric]['walks']}",
                 "primary": metric == "ic"} for c in comparisons]

    f3, f3_continuous = [], []
    for slice_name in ("overall", "expiry_days"):
        mirror = placebo[slice_name]
        f3.append({"test": f"{slice_name}: max-GEX vs matched-distance mirror", "estimate": mirror["hit_rate_actual"] - mirror["hit_rate_mirror"],
                   "p": mirror["actual_vs_mirror_sign_test"]["p_value_two_sided"], "primary": True})
        for comparator, label in (("snapped_mirror", "snapped mirror"), ("matched_rank", "distance-matched GEX rank 2-5")):
            block = listed[slice_name][comparator]
            f3.append({"test": f"{slice_name}: max-GEX vs {label}", "estimate": block["hit_rate_max_gex"] - block["hit_rate_comparator"],
                       "p": block["binary_sign_test"]["p_two_sided"], "days": block["days"], "primary": True})
            f3_continuous.append({"test": f"{slice_name}: max-GEX vs {label} (distance reduction, bps)",
                                  "estimate": block["continuous"]["mean_difference_bps"], "p": block["continuous"]["paired_t_p"],
                                  "days": block["days"], "primary": False})
    result = {
        "eval_config_hash": eval_hash(cfg["eval"]),
        "sources": {"stage_a_run": stage_a_run, "t2_ablation_runs": ablation["runs"], "t3_placebo_run": placebo["run"], "t3_listed_placebo_run": listed["run"]},
        "families": [
            adjust_family("F1", families["F1"], f1, alpha),
            adjust_family("F2", families["F2"] + " [primary: IC]", f2("ic"), alpha),
            adjust_family("F2-secondary-AUC", families["F2"] + " [secondary: AUC]", f2("auc"), alpha),
            adjust_family("F2-secondary-Brier", families["F2"] + " [secondary: Brier]", f2("brier"), alpha),
            adjust_family("F3", families["F3"], f3, alpha),
            adjust_family("F3-secondary-continuous", families["F3"] + " [secondary: continuous]", f3_continuous, alpha),
            {"family": "F4", "description": families["F4"], "tests": 0, "members": [], "note": "pending official closes"},
        ],
    }
    (root / "multiple_testing.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    for family in result["families"]:
        print(family["family"], [(m["test"], m.get("p"), m.get("p_holm"), m.get("reject_holm")) for m in family["members"]])


if __name__ == "__main__":
    main()
