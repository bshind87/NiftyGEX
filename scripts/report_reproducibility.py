"""Reproducibility report: compare a rerun of the main pipeline (another path profile) with the original outputs.

Read-only over both profiles. Compares every data-stage Parquet file (non-float columns equal, floats within float
rounding), the stage manifests' counts and quality flags, the main forecast and neural runs (predictions,
per-walk metrics, attention weights) and the RQ test results. Writes outputs/reports/reproducibility.json (under the
original profile) and docs/findings_reproducibility.md.

    PYTHONPATH=src python scripts/report_reproducibility.py --rerun repro
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from niftygex.config import REPO_ROOT, load_config
from niftygex.io.parquet import frame_differences, write_report

STAGE_ROOTS = ["curated_root", "greeks_root", "exposure_root", "features_root", "targets_root"]
RESULT_FILES = ["forecast_comparison", "neural_comparison"]
VOLATILE = {"runs", "git_commit", "git_dirty", "created_at_utc", "data_config_hash"}


def compare(a: Path, b: Path) -> dict:
    fa, fb = pl.read_parquet(a), pl.read_parquet(b)
    if fa.equals(fb, null_equal=True):
        return {"status": "identical"}
    diff = frame_differences(fb, fa)
    return {"status": "rounding" if not diff["problems"] else "different", **diff}


def data_files(orig: dict, rerun: dict) -> dict:
    counts = {"identical": 0, "rounding": 0}
    rounding, different, missing = {}, [], []
    for key in STAGE_ROOTS:
        root_a, root_b = Path(orig["paths"][key]), Path(rerun["paths"][key])
        for a in sorted(root_a.rglob("*.parquet")):
            rel = a.relative_to(root_a)
            if "intraday" in rel.parts[0]:
                continue  # E3 data is outside the main pipeline
            b = root_b / rel
            if not b.exists():
                missing.append(f"{key}/{rel}")
                continue
            result = compare(a, b)
            if result["status"] == "different":
                different.append({"file": f"{key}/{rel}", "problems": result["problems"]})
                continue
            counts[result["status"]] += 1
            if result["status"] == "rounding":
                rounding[f"{key}/{rel}"] = result["rounding"]
    return {**counts, "rounding_detail": rounding, "different": different, "missing": missing}


def manifests(orig: dict, rerun: dict) -> dict:
    out = {}
    for b in sorted(Path(rerun["paths"]["manifests_root"]).glob("*.json")):
        a = Path(orig["paths"]["manifests_root"]) / b.name
        ma, mb = json.loads(a.read_text()), json.loads(b.read_text())
        out[b.stem] = {"counts_equal": ma.get("counts") == mb.get("counts"), "quality_flags_equal": ma.get("quality_flags") == mb.get("quality_flags"),
                       "original_commit": (ma.get("git_commit") or "")[:7], "rerun_commit": (mb.get("git_commit") or "")[:7],
                       "original_git_dirty": ma.get("git_dirty"), "rerun_git_dirty": mb.get("git_dirty")}
    return out


def runs(orig: dict, rerun: dict) -> dict:
    root_a, root_b = Path(orig["paths"]["outputs_root"]) / "results", Path(rerun["paths"]["outputs_root"]) / "results"
    out = {}
    for name in RESULT_FILES:
        ra = json.loads((root_a / f"{name}.json").read_text())["runs"]
        rb = json.loads((root_b / f"{name}.json").read_text())["runs"]
        for model, run_a in ra.items():
            pa, pb = root_a / "runs" / run_a, root_b / "runs" / rb[model]
            spec = json.loads((pb / "run.json").read_text())["spec"]
            wa, wb = pl.read_parquet(pa / "walk_metrics.parquet"), pl.read_parquet(pb / "walk_metrics.parquet")
            common = [c for c in wa.columns if c in wb.columns]
            record = {"original_run": run_a, "rerun_run": rb[model], "predictions": compare(pa / "predictions.parquet", pb / "predictions.parquet")["status"],
                      "walk_metrics_common_columns": compare_frames(wa.select(common), wb.select(common)),
                      "walk_metrics_new_columns": [c for c in wb.columns if c not in wa.columns],
                      "ic_original": wa["ic"].mean(), "ic_rerun": wb["ic"].mean(),
                      "rerun_commit": spec["git_commit"][:7], "rerun_git_dirty": spec["git_dirty"]}
            if (pa / "attention.parquet").exists():
                record["attention"] = compare(pa / "attention.parquet", pb / "attention.parquet")["status"]
            out[model] = record
    return out


def compare_frames(a: pl.DataFrame, b: pl.DataFrame) -> str:
    return "identical" if a.equals(b, null_equal=True) else ("rounding" if not frame_differences(b, a)["problems"] else "different")


def rq_tests(orig: dict, rerun: dict) -> bool:
    strip = lambda d: json.dumps({k: v for k, v in d.items() if k not in VOLATILE}, sort_keys=True, default=str)
    a = json.loads((Path(orig["paths"]["outputs_root"]) / "results" / "rq_tests.json").read_text())
    b = json.loads((Path(rerun["paths"]["outputs_root"]) / "results" / "rq_tests.json").read_text())
    return strip(a) == strip(b)


def markdown(rerun_profile: str, data: dict, mans: dict, model_runs: dict, rq_equal: bool) -> list[str]:
    commits = sorted({r["rerun_commit"] for r in model_runs.values()} | {m["rerun_commit"] for m in mans.values()})
    dirty_runs = [m for m, r in model_runs.items() if r["rerun_git_dirty"]]
    dirty_stages = [s for s, m in mans.items() if m["rerun_git_dirty"]]
    lines = ["# Reproducibility: clean-commit rerun of the main pipeline", "",
             f"Generated by `scripts/report_reproducibility.py --rerun {rerun_profile}`. The rerun wrote to the `{rerun_profile}` path profile "
             "(separate data and outputs roots) and is compared here with the original outputs.", "",
             "## Summary", "",
             f"- Rerun commit(s): {', '.join(f'`{c}`' for c in commits)}; model runs with `git_dirty = true`: {len(dirty_runs)} of {len(model_runs)}; "
             f"stage manifests with `git_dirty = true`: {', '.join(dirty_stages) if dirty_stages else 'none'}.",
             f"- Data files: {data['identical']} identical, {data['rounding']} equal within float rounding, {len(data['different'])} different, "
             f"{len(data['missing'])} missing in the rerun.",
             f"- Stage manifests: counts equal in {sum(m['counts_equal'] for m in mans.values())} of {len(mans)}, "
             f"quality flags equal in {sum(m['quality_flags_equal'] for m in mans.values())} of {len(mans)}.",
             f"- Model runs: predictions identical in {sum(r['predictions'] == 'identical' for r in model_runs.values())} of {len(model_runs)}.",
             f"- RQ test results (ignoring run ids and provenance): {'identical' if rq_equal else 'different'}.", ""]
    if data["rounding_detail"]:
        lines += ["Float-rounding differences (largest absolute difference per column):", ""]
        lines += [f"- `{f}`: " + ", ".join(f"`{c}` {v:.2g}" for c, v in cols.items()) for f, cols in data["rounding_detail"].items()] + [""]
    if dirty_stages:
        same = all(mans[s]["original_git_dirty"] for s in dirty_stages)
        lines += [f"`git_dirty = true` in the {', '.join(dirty_stages)} manifest: the validation stage writes the tracked `docs/validation_report.md` "
                  f"before its manifest, so the tree is dirty when the manifest is written; the original manifest {'records the same' if same else 'does not'}.", ""]
    lines += ["## Model runs", "", "| Model | Original run | Rerun run | Predictions | Per-walk metrics | Attention | Mean IC (original) | Mean IC (rerun) |",
              "|---|---|---|---|---|---|---|---|"]
    for model, r in model_runs.items():
        ic = lambda v: "—" if v is None or v != v else f"{v:.4f}"
        lines.append(f"| {model} | `{r['original_run']}` | `{r['rerun_run']}` | {r['predictions']} | {r['walk_metrics_common_columns']} | "
                     f"{r.get('attention', '—')} | {ic(r['ic_original'])} | {ic(r['ic_rerun'])} |")
    new_cols = sorted({c for r in model_runs.values() for c in r["walk_metrics_new_columns"]})
    if new_cols:
        lines += ["", f"Per-walk metrics are compared on the original columns; the rerun adds {', '.join(f'`{c}`' for c in new_cols)} "
                      "(compute provenance added after the original run)."]
    lines += ["", "## Stage manifests", "", "| Stage | Counts equal | Quality flags equal | Original commit | Rerun commit | Rerun git_dirty |", "|---|---|---|---|---|---|"]
    lines += [f"| {s} | {m['counts_equal']} | {m['quality_flags_equal']} | `{m['original_commit']}` | `{m['rerun_commit']}` | {m['rerun_git_dirty']} |" for s, m in mans.items()]
    return lines + [""]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rerun", default="repro", help="path profile of the rerun (configs/paths.<profile>.yaml)")
    parser.add_argument("--original", default=None, help="path profile of the original outputs (default: NIFTYGEX_PROFILE or local)")
    args = parser.parse_args()
    orig, rerun = load_config(args.original), load_config(args.rerun)
    data, mans, model_runs, rq_equal = data_files(orig, rerun), manifests(orig, rerun), runs(orig, rerun), rq_tests(orig, rerun)
    out = REPO_ROOT / "docs" / "findings_reproducibility.md"
    out.write_text("\n".join(markdown(args.rerun, data, mans, model_runs, rq_equal)))
    write_report(orig, "reproducibility", {"rerun_profile": args.rerun, "data_files": data, "manifests": mans, "runs": model_runs,
                                           "rq_tests_identical": rq_equal})
    print(out)


if __name__ == "__main__":
    main()
