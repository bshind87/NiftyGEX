"""Exploratory E3 report: next 1 and 2 hours of futures RV (docs/findings_intraday.md, with figures).

Reads outputs/results/exploratory_e3.json and exploratory_e3_{h}h.json (scripts/run_intraday.py --horizon h --tests,
after the Explorer neural runs are merged) and the intraday table for the sanity checks:
  - time-of-day pattern: share of the variance of the raw forward RV and of the target explained by the snapshot's
    clock time (the 60-day clock-window normal should remove it);
  - IC within each clock time (across days), so a high pooled IC cannot come from ranking times of day;
  - persistence's IC as the naive benchmark.
Writes outputs/reports/exploratory_e3.json and outputs/reports/figures/e3_*.png. Everything here is exploratory.
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
from niftygex_eval.forecast.intraday import intraday_table

LABELS = {"persistence": "Persistence", "intraday_har": "Intraday HAR", "har_gex": "HAR + GEX", "lightgbm": "Tuned LightGBM",
          "deepsets": "DeepSets (V100)", "attention": "Self-attention (V100)", "attention_gru": "Attention + GRU (V100)"}
ORDER = list(LABELS)


def f(value, spec: str = ".3f") -> str:
    return "—" if value is None or (isinstance(value, float) and not np.isfinite(value)) else format(value, spec)


def interval(ci) -> str:
    return "—" if not ci else f"{ci[0]:+.3f} to {ci[1]:+.3f}"


def clock(minutes: int) -> str:
    total = 9 * 60 + 15 + int(minutes)
    return f"{total // 60:02d}:{total % 60:02d}"


def time_of_day_share(values: np.ndarray, groups: np.ndarray) -> float:
    """Share of the variance explained by the clock time (between-group over total sum of squares)."""
    keep = np.isfinite(values)
    values, groups = values[keep], groups[keep]
    grand = values.mean()
    between = sum(((values[groups == g].mean() - grand) ** 2) * (groups == g).sum() for g in np.unique(groups))
    return float(between / ((values - grand) ** 2).sum())


def target_pattern(table: pl.DataFrame, horizons: list[int]) -> dict:
    """Per horizon: time-of-day share of variance for log raw forward RV and for the target, and means by clock time."""
    out = {}
    groups = table["minutes_since_open"].to_numpy()
    for h in horizons:
        raw = np.log(table[f"rv_fwd_{h}h"].to_numpy())
        target = table[f"log_rv_ratio_{h}h"].to_numpy()
        by = (table.group_by("minutes_since_open").agg(pl.col(f"rv_fwd_{h}h").log().mean().alias("raw_mean"),
                                                       pl.col(f"log_rv_ratio_{h}h").mean().alias("target_mean"),
                                                       pl.col(f"log_rv_ratio_{h}h").std().alias("target_std"), pl.len().alias("rows"))
              .sort("minutes_since_open"))
        out[f"{h}h"] = {"raw_log_rv_time_of_day_share": time_of_day_share(raw, groups), "target_time_of_day_share": time_of_day_share(target, groups),
                        "by_clock_time": by.with_columns(pl.col("minutes_since_open").map_elements(clock, return_dtype=pl.String).alias("clock")).to_dicts()}
    return out


def figure_pattern(pattern: dict, directory: Path) -> Path:
    fig, axes = plt.subplots(1, len(pattern), figsize=(10, 3.4), sharey=True)
    for ax, (h, block) in zip(np.atleast_1d(axes), pattern.items()):
        rows = block["by_clock_time"]
        x = [r["clock"] for r in rows]
        raw = np.array([r["raw_mean"] for r in rows])
        ax.plot(x, raw - raw.mean(), "-o", color=SERIES[0], markersize=4, label="log forward RV (centred)")
        ax.plot(x, [r["target_mean"] for r in rows], "-o", color=SERIES[1], markersize=4, label="target: log(RV / clock-window normal)")
        ax.axhline(0, color=TEXT_2, linewidth=0.8)
        ax.set_title(f"{h}: time of day explains {block['raw_log_rv_time_of_day_share']:.1%} of raw log RV variance,\n"
                     f"{block['target_time_of_day_share']:.2%} of the target's", fontsize=9)
        ax.tick_params(axis="x", rotation=60, labelsize=7.5)
    np.atleast_1d(axes)[0].set_ylabel("mean by snapshot time")
    np.atleast_1d(axes)[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Mean by snapshot time: raw forward RV (centred) and the clock-window-normalized target", x=0.01, ha="left", fontsize=11)
    return save(fig, directory, "e3_time_of_day_pattern")


def figure_ic_by_time(results: dict, directory: Path) -> Path:
    fig, axes = plt.subplots(1, len(results), figsize=(10, 3.6), sharey=True)
    for ax, (h, block) in zip(np.atleast_1d(axes), results.items()):
        for colour, model in zip([SERIES[0], SERIES[2], SERIES[1]], ("persistence", "intraday_har", "attention_gru")):
            rows = block["ic_by_time_of_day"].get(model, [])
            ax.plot([clock(r["minutes_since_open"]) for r in rows], [r["ic"] for r in rows], "-o", color=colour, markersize=4, label=LABELS[model])
        ax.set_title(f"{h}: IC within each snapshot time (across test days)", fontsize=9.5)
        ax.tick_params(axis="x", rotation=60, labelsize=7.5)
    np.atleast_1d(axes)[0].set_ylabel("IC")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    return save(fig, directory, "e3_ic_by_time_of_day")


def summary_lines(results: dict, holm: dict | None, pattern: dict) -> list[str]:
    lines = []
    for h, block in results.items():
        scores, best = block["scores"], block["best_classical"]
        p = block["primary"]
        member = next((m for m in (holm or {}).get("members", []) if m["test"] == p["test"]), {})
        if member.get("reject_holm"):
            verdict = "significantly higher than" if p["mean_diff"] > 0 else "significantly lower than"
        else:
            verdict = "not significantly different from"
        lines.append(f"{h}: attention + GRU is {verdict} the best classical rung after Holm across horizons.")
        lines.append(f"{h}: attention + GRU IC {scores['attention_gru']['ic_mean_per_walk']:.3f} vs the best classical rung, "
                     f"{LABELS[best]} {scores[best]['ic_mean_per_walk']:.3f} ({p.get('mean_diff', float('nan')):+.3f}, 95% CI {interval(p.get('ci95'))}, "
                     f"Holm p {f(member.get('p_holm'), '.2g')}, walks won {p.get('a_wins')}/{p.get('walks')}); persistence {scores['persistence']['ic_mean_per_walk']:.3f}.")
        within = [r["ic"] for r in block["ic_by_time_of_day"]["attention_gru"] if r["ic"] is not None]
        lines.append(f"{h}: within a single snapshot time (across days) attention + GRU's IC ranges {min(within):.3f}-{max(within):.3f}, "
                     f"so the pooled IC is not a time-of-day ranking; time of day explains {pattern[h]['raw_log_rv_time_of_day_share']:.1%} of the raw "
                     f"forward log RV's variance in this window and {pattern[h]['target_time_of_day_share']:.2%} of the target's.")
    return lines


def main() -> None:
    cfg = load_exploratory_config()
    root = Path(cfg["results_root"])
    figures = Path(cfg["data_config"]["paths"]["reports_root"]) / "figures"
    horizons = [int(h) for h in cfg["exploratory"]["e3"]["horizons"]]
    results = {f"{h}h": json.loads((root / f"exploratory_e3_{h}h.json").read_text()) for h in horizons}
    holm_path = root / "exploratory_e3.json"
    holm = json.loads(holm_path.read_text())["tests"] if holm_path.exists() else None
    table, _, _ = intraday_table(cfg)
    pattern = target_pattern(table, horizons)
    paths = [figure_pattern(pattern, figures), figure_ic_by_time(results, figures)]

    lines = ["# Exploratory E3: next 1 and 2 hours of realized volatility", "",
             "> **Exploratory.** Registered in `configs/eval_exploratory_v2.yaml` before any run; separate from the pre-registered RQ1–RQ3.",
             f"> Exploratory config hash `{cfg['exploratory_hash'][:12]}`. Target: log(front-month futures RV over (T, T + h] / its mean over "
             "the same clock window on the previous 60 trading days), 13 snapshots per day (10:15–13:15). Neural rungs on Explorer V100s; "
             "classical rungs on the Mac. The walk (not the snapshot) is the unit of the tests.", "",
             "Generated by `scripts/report_exploratory_e3.py`." + ("" if holm else " Holm across horizons pending (both horizons needed)."), "",
             "## Summary", "", *[f"- {line}" for line in summary_lines(results, holm, pattern)], ""]
    for h, block in results.items():
        scores, best = block["scores"], block["best_classical"]
        lines += [f"## {h}", "", "| Model | Mean IC per walk | Pooled IC | RMSE | QLIKE | IC > 0 walks |", "|---|---|---|---|---|---|"]
        for model in ORDER:
            s = scores[model]
            lines.append(f"| {LABELS[model]}{' (best classical)' if model == best else ''} | {f(s['ic_mean_per_walk'])} | {f(s['pooled']['ic'])} | "
                         f"{f(s['rmse_mean_per_walk'])} | {f(s['qlike_mean_per_walk'])} | {s['ic_positive_walks']}/{s['walks']} |")
        lines += ["", "| Attention + GRU vs | IC diff (95% CI) | walks won | RMSE diff | QLIKE diff | DM p (squared / QLIKE) |", "|---|---|---|---|---|---|"]
        for model, sec in block["secondary"].items():
            lines.append(f"| {LABELS[model]} | {f(sec['ic'].get('mean_diff'), '+.3f')} ({interval(sec['ic'].get('ci95'))}) | {sec['ic'].get('a_wins')}/{sec['ic'].get('walks')} | "
                         f"{f(sec['rmse'].get('mean_diff'), '+.3f')} | {f(sec['qlike'].get('mean_diff'), '+.3f')} | "
                         f"{f(sec['dm_squared'].get('p_two_sided'), '.2g')} / {f(sec['dm_qlike'].get('p_two_sided'), '.2g')} |")
        lines += ["", f"IC within each snapshot time (pooled across test days, {h}):", "",
                  "| Time | " + " | ".join(LABELS[m] for m in ORDER) + " |", "|---|" + "---|" * len(ORDER)]
        times = [r["minutes_since_open"] for r in block["ic_by_time_of_day"]["persistence"]]
        for i, minute in enumerate(times):
            lines.append(f"| {clock(minute)} | " + " | ".join(f(block["ic_by_time_of_day"][m][i]["ic"]) for m in ORDER) + " |")
        column = cfg["eval"]["forecast"]["regime_column"]
        periods = {("after" if r[column] else "before"): r for r in scores["attention_gru"].get("by_regime", []) if r.get(column) is not None}
        if periods:
            lines += ["", "Before vs after the November 2024 marker (attention + GRU, pooled): "
                      + "; ".join(f"{k}: IC {f(v.get('ic'))} on {v.get('n')} rows" for k, v in sorted(periods.items(), key=lambda kv: kv[0] != "before")), ""]
        lines.append("")
    lines += ["## Design note", "",
              "As registered, the neural rungs' state vector is the T2 volatility state recomputed at each snapshot plus time of day; "
              "it does not include the backward-window RVs (last 30 and 60 minutes, yesterday's same window, the clock-window normal) that the "
              "intraday HAR uses. Part of the gap to the classical rungs may therefore come from inputs rather than architecture; a hybrid with "
              "the intraday HAR forecast as an input (as in E1) would separate the two and is not registered.", ""]
    lines += ["## Sanity check: the time-of-day pattern", "", "| Horizon | Time of day: share of raw log RV variance | Share of target variance |", "|---|---|---|"]
    for h, block in pattern.items():
        lines.append(f"| {h} | {block['raw_log_rv_time_of_day_share']:.1%} | {block['target_time_of_day_share']:.2%} |")
    lines += ["", *[f"![{p.stem}](../{p.relative_to(REPO_ROOT)})\n" for p in paths]]
    out = REPO_ROOT / "docs" / "findings_intraday.md"
    out.write_text("\n".join(lines))
    write_report(cfg["data_config"], "exploratory_e3", {"holm": holm, "pattern": {h: {k: v for k, v in b.items() if k != "by_clock_time"} for h, b in pattern.items()},
                                                        "results": {h: {"primary": b["primary"], "best_classical": b["best_classical"]} for h, b in results.items()},
                                                        "figures": [str(p.relative_to(REPO_ROOT)) for p in paths]})
    print(out)


if __name__ == "__main__":
    main()
