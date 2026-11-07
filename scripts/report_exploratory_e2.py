"""Exploratory E2 report: next-day target with more history (docs/findings_exploratory_history.md, with a figure).

Reads outputs/results/exploratory_e2.json (scripts/run_exploratory_e2.py --tests, after the Explorer runs are merged)
and the merged runs it names. Writes outputs/reports/exploratory_e2.json and outputs/reports/figures/e2_*.png.
Everything here is exploratory.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from niftygex.config import REPO_ROOT
from niftygex.io.parquet import write_report
from niftygex_eval.config import load_exploratory_config
from niftygex_eval.figures import SERIES, TEXT_2, save

LABELS = {"window_24": "Intraday window L = 24 (2 h)", "window_72": "Intraday window L = 72 (09:20–15:15)",
          "multiday_5": "Multi-day: previous 5 days' 15:15 chains + L = 12", "multiday_10": "Multi-day: previous 10 days + L = 12"}
SHORT = {"window_24": "L = 24", "window_72": "L = 72", "multiday_5": "multi-day K = 5", "multiday_10": "multi-day K = 10"}


def f(value, spec: str = ".3f") -> str:
    return "—" if value is None or (isinstance(value, float) and not np.isfinite(value)) else format(value, spec)


def interval(ci) -> str:
    return "—" if not ci else f"{ci[0]:+.3f} to {ci[1]:+.3f}"


def run_record(root: Path, run_id: str) -> dict:
    record = json.loads((root / "runs" / run_id / "run.json").read_text())
    return {"commit": record["spec"]["git_commit"][:7], "dirty": record["spec"]["git_dirty"], "gpu_type": record["spec"].get("gpu_type"),
            "seeds": record["spec"]["training"]["seeds"], "minutes": record["summary"].get("seconds", 0) / 60}


def figure_walks(root: Path, base_run: str, candidates: dict, directory: Path) -> Path:
    base = pl.read_parquet(root / "runs" / base_run / "walk_metrics.parquet").select("walk_id", pl.col("ic").alias("base"))
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for i, (key, run_id) in enumerate(candidates.items()):
        walks = pl.read_parquet(root / "runs" / run_id / "walk_metrics.parquet").select("walk_id", "ic").join(base, on="walk_id")
        diff = (walks["ic"] - walks["base"]).to_numpy()
        jitter = np.random.default_rng(i).uniform(-0.12, 0.12, diff.size)
        ax.plot(diff, i + jitter, "o", color=SERIES[i % len(SERIES)], alpha=0.45, markersize=4)
        ax.plot(diff.mean(), i, "D", color=SERIES[i % len(SERIES)], markersize=9, markeredgecolor="white", markeredgewidth=1.5)
    ax.axvline(0, color=TEXT_2, linewidth=0.8)
    ax.set_yticks(range(len(candidates)), [SHORT[k] for k in candidates])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("IC difference vs attention + GRU at L = 12, per walk (diamond: mean over walks)")
    ax.set_title("More history for the next-day model: per-walk IC gain over L = 12 (V100, 3 seeds each)")
    return save(fig, directory, "e2_history_ic_gain")


def summary_lines(members: list[dict], secondary: dict, base_ic: float) -> list[str]:
    by = {m["test"]: m for m in members}
    rejected = [m["test"] for m in members if m.get("reject_holm")]
    lines = [f"{len(rejected)} of {len(members)} registered E2 tests reject at 5% after Holm (reference: attention + GRU at L = 12 on V100, IC {base_ic:.3f})."]
    windows = [by[t] for t in ("window_24_vs_l12", "window_72_vs_l12") if t in by]
    if windows:
        lines.append("Longer intraday history: " + "; ".join(
            f"{'L = 24' if '24' in m['test'] else 'L = 72'} IC {m['mean_a']:.3f} ({m['mean_diff']:+.3f}, 95% CI {interval(m['ci95'])}, "
            f"unadjusted p {m['p']:.2g}, Holm p {m['p_holm']:.2g})" for m in windows) + ".")
    days = [by[t] for t in ("multiday_5_vs_l12", "multiday_10_vs_l12") if t in by]
    if days:
        lines.append("Previous days' chains: " + "; ".join(
            f"K = {'5' if '_5_' in m['test'] else '10'} IC {m['mean_a']:.3f} ({m['mean_diff']:+.3f}, 95% CI {interval(m['ci95'])})" for m in days) + ".")
    worse = [k for k, v in secondary.items() if (v["dm_squared"].get("p_two_sided") or 1) < 0.05 and v["rmse"].get("mean_diff", 0) > 0]
    if worse:
        lines.append("Squared-error loss is significantly worse (Diebold-Mariano p < 0.05, unadjusted) for: " + ", ".join(worse) + ".")
    return lines


def main() -> None:
    cfg = load_exploratory_config()
    root = Path(cfg["results_root"])
    figures = Path(cfg["data_config"]["paths"]["reports_root"]) / "figures"
    e2 = json.loads((root / "exploratory_e2.json").read_text())
    members, secondary = e2["tests"]["members"], e2["secondary"]
    candidates = {key: run for key, run in e2["candidates"].items() if key in LABELS}
    base = pl.read_parquet(root / "runs" / e2["reference_run"] / "walk_metrics.parquet")
    base_ic = float(base["ic"].mean())
    records = {"reference": run_record(root, e2["reference_run"]), **{k: run_record(root, r) for k, r in candidates.items()}}
    path = figure_walks(root, e2["reference_run"], candidates, figures)

    rows = ["| Variant | Mean IC | IC diff vs L = 12 | 95% interval | p | Holm p | Holm reject (5%) | Walks won |", "|---|---|---|---|---|---|---|---|"]
    for m in members:
        key = m["test"].removesuffix("_vs_l12")
        rows.append(f"| {LABELS[key]} | {f(m['mean_a'])} | {f(m['mean_diff'], '+.3f')} | {interval(m['ci95'])} | {f(m['p'], '.2g')} | "
                    f"{f(m['p_holm'], '.2g')} | {'yes' if m['reject_holm'] else 'no'} | {m['a_wins']}/{m['walks']} |")
    loss = ["| Variant | RMSE diff | QLIKE diff | DM p (squared) | DM p (QLIKE) |", "|---|---|---|---|---|"]
    for test, v in secondary.items():
        loss.append(f"| {LABELS[test.removesuffix('_vs_l12')]} | {f(v['rmse'].get('mean_diff'), '+.4f')} | {f(v['qlike'].get('mean_diff'), '+.4f')} | "
                    f"{f(v['dm_squared'].get('p_two_sided'), '.2g')} | {f(v['dm_qlike'].get('p_two_sided'), '.2g')} |")
    runs = ["| Run | Run id | Commit | git_dirty | GPU type | Seeds | Training minutes (3 seeds, summed over walks) |", "|---|---|---|---|---|---|---|"]
    for key, rec in records.items():
        run_id = e2["reference_run"] if key == "reference" else candidates[key]
        runs.append(f"| {'Attention + GRU, L = 12 (reference)' if key == 'reference' else LABELS[key]} | `{run_id}` | `{rec['commit']}` | {rec['dirty']} | "
                    f"{rec['gpu_type']} | {rec['seeds']} | {rec['minutes']:.0f} |")
    lines = ["# Exploratory E2: next-day target with more history", "",
             "> **Exploratory.** Registered in `configs/eval_exploratory_v2.yaml` before any run; separate from the pre-registered RQ1–RQ3.",
             f"> Exploratory config hash `{e2['exploratory_config_hash'][:12]}`. Every variant and the L = 12 reference ran on Explorer V100s "
             "(device rule); L = 24 and L = 72 run under the main eval configuration, the multi-day models under the exploratory one. "
             "Same table, walks and rows as the main ladder (checked by the test script).", "",
             "Generated by `scripts/report_exploratory_e2.py`.", "",
             "## Summary", "", *[f"- {line}" for line in summary_lines(members, secondary, base_ic)], "",
             "## Primary tests (Holm within E2)", "", *rows, "",
             "## Losses (variant minus L = 12; negative is better)", "", *loss, "",
             "## Runs", "", *runs, "",
             f"![{path.stem}](../{path.relative_to(REPO_ROOT)})", ""]
    out = REPO_ROOT / "docs" / "findings_exploratory_history.md"
    out.write_text("\n".join(lines))
    write_report(cfg["data_config"], "exploratory_e2", {"tests": members, "secondary": secondary, "runs": records,
                                                        "figure": str(path.relative_to(REPO_ROOT))})
    print(out)


if __name__ == "__main__":
    main()
