"""Merge per-seed cluster runs (pulled by hpc/pull_results.sh) into the local results store.

  PYTHONPATH=src python scripts/merge_seeds.py [--source outputs/hpc_pull/results]

For every group of runs that differ only by seed and that has all configured seeds, writes one merged run (seed
predictions averaged per row, walks rescored; src/niftygex_eval/merge.py) and copies the per-seed runs for
provenance. Then lists the merged runs where the test scripts look for them:
  stage neural          -> outputs/results/neural_comparison_<tag>.json (tag e.g. hpc, window_24_hpc)
  stage exploratory_e1  -> outputs/results/exploratory_e1_runs.json   (by part)
  stage exploratory_e2  -> outputs/results/exploratory_e2_runs.json   (by tag)
  stage exploratory_e3  -> outputs/results/exploratory_e3_runs.json   (by horizon, then model)
  stage exploratory_v3  -> outputs/results/exploratory_v3_runs.json   (by part)
and writes outputs/results/merge_seeds.json (merged groups and incomplete ones). Rerunning is safe.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import polars as pl

from niftygex.config import REPO_ROOT
from niftygex_eval.config import load_eval_config
from niftygex_eval.forecast.compare import mean_ci
from niftygex_eval.forecast.metrics import regression_metrics
from niftygex_eval.merge import group_key, merge_predictions, merge_walk_metrics, merged_spec, seed_of
from niftygex_eval.results import ResultsStore


def update_json(path: Path, update) -> None:
    current = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps(update(current), indent=2, default=str) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(REPO_ROOT / "outputs" / "hpc_pull" / "results"))
    args = parser.parse_args()
    cfg = load_eval_config()
    source, dest = Path(args.source), Path(cfg["results_root"])
    wanted = sorted(int(s) for s in cfg["eval"]["neural"]["training"]["seeds"])
    store = ResultsStore(dest)

    groups: dict[str, dict[int, tuple[str, dict, str]]] = {}
    for line in (source / "index.jsonl").read_text().splitlines():
        entry = json.loads(line)
        run_dir = source / "runs" / entry["run_id"]
        if not (run_dir / "run.json").exists():
            continue
        record = json.loads((run_dir / "run.json").read_text())
        seed = seed_of(record["spec"])
        if seed is None:
            continue
        previous = groups.setdefault(group_key(record["spec"]), {}).get(seed)
        if previous is None or record["created_at_utc"] > previous[2]:   # the latest run of a seed wins
            groups[group_key(record["spec"])][seed] = (entry["run_id"], record["spec"], record["created_at_utc"])

    merged, incomplete = [], []
    for key, seeds in groups.items():
        spec0 = next(iter(seeds.values()))[1]
        label = {k: spec0.get(k) for k in ("stage", "model", "tag", "part", "horizon_hours", "days", "window")}
        if sorted(seeds) != wanted:
            incomplete.append({**label, "seeds": sorted(seeds)})
            continue
        runs = {s: seeds[s][0] for s in wanted}
        for run_id in runs.values():   # keep the per-seed runs next to the merged one
            if not (dest / "runs" / run_id).exists():
                shutil.copytree(source / "runs" / run_id, dest / "runs" / run_id)
        predictions = merge_predictions({s: pl.read_parquet(source / "runs" / r / "predictions.parquet") for s, r in runs.items()})
        per_walk = merge_walk_metrics({s: pl.read_parquet(source / "runs" / r / "walk_metrics.parquet") for s, r in runs.items()}, predictions)
        spec = merged_spec({s: seeds[s][1] for s in wanted}, runs)
        pooled = regression_metrics(predictions["y"].to_numpy(), predictions["prediction"].to_numpy())
        run_id = store.write_run(spec, {"predictions": predictions, "walk_metrics": per_walk},
                                 {"pooled": pooled, "seconds": float(per_walk["seconds"].sum()), "merged_seeds": wanted}).name
        merged.append({**label, "run_id": run_id, "seed_runs": runs, "ic_mean_per_walk": float(per_walk["ic"].mean())})
        row = {"model": spec["model"], "walks": per_walk.height, "test_days": predictions.height, "seconds": float(per_walk["seconds"].sum()),
               "ic_mean_per_walk": float(per_walk["ic"].mean()), "ic_ci95": mean_ci(per_walk["ic"].to_numpy()),
               "rmse_mean_per_walk": float(per_walk["rmse"].mean()), "qlike_mean_per_walk": float(per_walk["qlike"].mean()),
               "stop_epoch_mean": float(per_walk["stop_epoch_mean"].mean()) if "stop_epoch_mean" in per_walk.columns else None, "pooled": pooled}

        stage = spec["stage"]
        if stage == "neural":
            def add(current, spec=spec, run_id=run_id, row=row):
                current.update({k: spec.get(k) for k in ("data_config_hash", "eval_config_hash", "outcome", "window", "tag", "dropped", "gpu_type")})
                current["walks"] = row["walks"]
                current.setdefault("runs", {})[spec["model"]] = run_id
                current["comparison_table"] = [r for r in current.get("comparison_table", []) if r["model"] != spec["model"]] + [row]
                return current
            update_json(dest / f"neural_comparison_{spec['tag']}.json", add)
        elif stage == "exploratory_e1":
            update_json(dest / "exploratory_e1_runs.json", lambda c, spec=spec, run_id=run_id: {**c, spec["part"]: run_id})
        elif stage == "exploratory_e2":
            update_json(dest / "exploratory_e2_runs.json", lambda c, spec=spec, run_id=run_id: {**c, spec["tag"]: run_id})
        elif stage == "exploratory_e3":
            key_h = f"{spec['horizon_hours']}h" + ("_dev" if spec.get("tag") == "dev" else "")
            def add_e3(current, key_h=key_h, spec=spec, run_id=run_id):
                current.setdefault(key_h, {})[spec["model"]] = run_id
                return current
            update_json(dest / "exploratory_e3_runs.json", add_e3)
        elif stage == "exploratory_v3":
            update_json(dest / "exploratory_v3_runs.json", lambda c, spec=spec, run_id=run_id: {**c, spec["part"]: run_id})

    (dest / "merge_seeds.json").write_text(json.dumps({"source": str(source), "seeds": wanted, "merged": merged, "incomplete": incomplete},
                                                      indent=2, default=str) + "\n")
    print(f"merged {len(merged)} runs; incomplete groups: {len(incomplete)}")
    for item in incomplete:
        print("  incomplete:", item)


if __name__ == "__main__":
    main()
