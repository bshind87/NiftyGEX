"""Data-step report with figures: aggregate features and the token history.

Writes outputs/reports/data_overview.json, outputs/reports/figures/data_*.png and docs/findings_data.md from the
feature and token Parquet files and the stage reports. Rerun after any data change.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from niftygex.config import REPO_ROOT, load_config
from niftygex.io.parquet import write_report
from niftygex_eval.figures import SERIES, SURFACE, TEXT, TEXT_2, save

SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("red_gray_blue", ["#c43333", "#e34948", "#f0efec", "#2a78d6", "#1c5cab"])
EXAMPLE_DAY = date(2026, 6, 12)
KEY_FEATURES = [("atm_iv", "ATM implied volatility", 100, "%"), ("rv_trailing_20d_annualized_prev", "20-day futures RV (through yesterday)", 100, "%"),
                ("put_call_oi_ratio", "put/call OI ratio", 1, ""), ("gex_concentration_hhi", "GEX concentration (HHI)", 1, "")]


def figure_feature_coverage(report: dict, directory: Path) -> Path:
    """Share of rows with a value, per feature and year."""
    years = [row["year"] for row in report["non_null_share_by_year"]]
    columns = list(report["feature_columns"])
    values = np.array([[row[c] for row in report["non_null_share_by_year"]] for c in columns])
    fig, ax = plt.subplots(figsize=(8.5, 0.24 * len(columns) + 1.6))
    image = ax.imshow(values, aspect="auto", cmap=SEQUENTIAL, vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(years)), years)
    ax.set_yticks(range(len(columns)), columns, fontsize=8)
    ax.grid(False)
    for i, j in zip(*np.where(values < 0.95)):
        ax.text(j, i, f"{values[i, j]:.0%}", ha="center", va="center", fontsize=7, color=SURFACE if values[i, j] > 0.7 else TEXT)
    bar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    bar.set_label("share of snapshots with a value", color=TEXT_2)
    ax.set_title("Feature coverage by year (cells below 95% are labeled)")
    return save(fig, directory, "data_feature_coverage")


def figure_key_features(features: pl.DataFrame, directory: Path) -> Path:
    """Four representative features over the sample, one per panel."""
    frame = features.sort("trade_date").to_pandas()
    fig, axes = plt.subplots(len(KEY_FEATURES), 1, figsize=(10, 7.2), sharex=True)
    for ax, (column, label, factor, unit) in zip(axes, KEY_FEATURES):
        ax.plot(frame["trade_date"], frame[column] * factor, color=SERIES[0], linewidth=0.9)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel(unit)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    return save(fig, directory, "data_key_features")


def figure_tokens_by_time(per_snapshot: pl.DataFrame, directory: Path) -> Path:
    """Tokens per sequence snapshot by time of day: median and 5-95% range across days."""
    by_time = (per_snapshot.with_columns(pl.col("snapshot_at").dt.strftime("%H:%M").alias("time"))
               .group_by("time").agg(pl.col("tokens").median().alias("median"), pl.col("tokens").quantile(0.05).alias("p5"),
                                     pl.col("tokens").quantile(0.95).alias("p95")).sort("time"))
    x = np.arange(by_time.height)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.fill_between(x, by_time["p5"], by_time["p95"], color=SERIES[0], alpha=0.18, linewidth=0, label="5–95% of days")
    ax.plot(x, by_time["median"], color=SERIES[0], linewidth=2, label="median")
    ticks = [i for i, t in enumerate(by_time["time"]) if t.endswith(":20") or t == by_time["time"][-1]]
    ax.set_xticks(ticks, [by_time["time"][i] for i in ticks])
    ax.set_xlim(0, by_time.height - 1)
    ax.set_ylabel("tokens per snapshot")
    ax.set_title("Token history: contracts per snapshot through the day (09:20 to the 15:15 prediction snapshot)")
    ax.legend(loc="lower right", ncol=2)
    return save(fig, directory, "data_tokens_by_time")


def figure_token_day(tokens_day: pl.DataFrame, day: date, directory: Path) -> Path:
    """One day's token history: strike GEX (naive sign) by strike and snapshot time, with the forward."""
    pivot = tokens_day.group_by("snapshot_at", "strike_price").agg(pl.col("strike_gex_naive_crore").sum()).sort("snapshot_at", "strike_price")
    times = pivot["snapshot_at"].unique().sort()
    strikes = pivot["strike_price"].unique().sort()
    grid = np.full((strikes.len(), times.len()), np.nan)
    t_index = {t: i for i, t in enumerate(times.to_list())}
    k_index = {k: i for i, k in enumerate(strikes.to_list())}
    for t, k, v in pivot.iter_rows():
        grid[k_index[k], t_index[t]] = v
    limit = float(np.nanpercentile(np.abs(grid), 98)) or 1.0
    forward = tokens_day.group_by("snapshot_at").agg(pl.col("forward_estimate").first()).sort("snapshot_at")
    fig, ax = plt.subplots(figsize=(10, 5.2))
    image = ax.imshow(grid, aspect="auto", origin="lower", cmap=DIVERGING, norm=TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit),
                      extent=(-0.5, times.len() - 0.5, -0.5, strikes.len() - 0.5), interpolation="nearest")
    ax.plot(np.arange(forward.height), np.interp(forward["forward_estimate"].to_numpy(), strikes.to_numpy(), np.arange(strikes.len())),
            color=TEXT, linewidth=1.4, label="forward")
    ticks = [i for i, t in enumerate(times.dt.strftime("%H:%M").to_list()) if t.endswith(":20") or i == times.len() - 1]
    ax.set_xticks(ticks, [times.dt.strftime("%H:%M")[i] for i in ticks])
    step = max(1, strikes.len() // 10)
    ax.set_yticks(range(0, strikes.len(), step), [f"{k:,.0f}" for k in strikes.to_list()[::step]])
    ax.grid(False)
    ax.set_ylabel("strike")
    bar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    bar.set_label("strike GEX, naive sign (₹ crore per 1%)", color=TEXT_2)
    ax.legend(loc="upper left")
    ax.set_title(f"{day:%d %b %Y}: what the sequence model sees, one column per 5-minute snapshot (calls blue, puts red)")
    return save(fig, directory, "data_token_day")


def main() -> None:
    config = load_config()
    paths = config["paths"]
    reports = Path(paths["reports_root"])
    figures = reports / "figures"
    features_report = json.loads((reports / "features_report.json").read_text())
    tokens_report = json.loads((reports / "tokens_report.json").read_text())
    features = pl.read_parquet(Path(paths["features_root"]) / "aggregate" / "aggregate.parquet")
    tokens = pl.scan_parquet(Path(paths["features_root"]) / "tokens" / "*.parquet")
    per_snapshot = tokens.group_by("snapshot_at").agg(pl.len().alias("tokens")).collect()
    tokens_day = (tokens.filter(pl.col("trade_date") == EXAMPLE_DAY)
                  .select("snapshot_at", "strike_price", "strike_gex_naive_crore", "log_moneyness").collect()
                  .with_columns((pl.col("strike_price") / pl.col("log_moneyness").exp()).alias("forward_estimate")))

    paths_out = [figure_feature_coverage(features_report, figures), figure_key_features(features, figures),
                 figure_tokens_by_time(per_snapshot, figures), figure_token_day(tokens_day, EXAMPLE_DAY, figures)]
    prediction = tokens_report["tokens_per_prediction_snapshot"]["t2"]
    sequence = tokens_report["tokens_per_sequence_snapshot"]["t2"]
    coverage = tokens_report["sequence_coverage_by_year"]
    write_report(config, "data_overview", {
        "features": {"rows": features_report["rows"], "columns": len(features_report["feature_columns"]),
                     "complete_core_share_by_year": features_report["rows_complete_core_share_by_year"], "notes": features_report["notes"]},
        "tokens": {"rows": tokens_report["rows"], "sequence_snapshots_per_day": tokens_report["sequence_snapshots_per_day"],
                   "per_prediction_snapshot": prediction, "per_sequence_snapshot": sequence,
                   "days_with_full_sequence": tokens_report["days_with_full_sequence"], "coverage_by_year": coverage},
        "example_day": str(EXAMPLE_DAY),
        "figures": [str(p.relative_to(REPO_ROOT)) for p in paths_out],
    })

    rel = "../" + str(figures.relative_to(REPO_ROOT))
    pct = lambda v: f"{v:.1%}"
    lines = [
        "# Data: features and token history",
        "",
        f"Generated by `scripts/report_data.py` from the feature and token Parquet files and the stage reports (data config hash "
        f"`{tokens_report['config_hash'][:12]}`). Rerun after any data change.",
        "",
        "## 1. Aggregate features (the classical models' inputs)",
        "",
        f"{features_report['rows']:,} snapshots (one per trading day at 15:15) × {len(features_report['feature_columns'])} features. "
        "Definitions are in `docs/schema.md`.",
        "",
        f"![Feature coverage]({rel}/data_feature_coverage.png)",
        "",
        "Gaps follow known source issues: " + "; ".join(f"**{k}**: {v}" for k, v in features_report["notes"].items()) + ".",
        "",
        f"![Key features]({rel}/data_key_features.png)",
        "",
        "## 2. Token history (the sequence models' inputs)",
        "",
        f"{tokens_report['rows']:,} token rows: every 5-minute snapshot from 09:20 to the 15:15 prediction snapshot "
        f"({tokens_report['sequence_snapshots_per_day']} per day), one row per liquid contract of the day's nearest included expiry. "
        "`lag` counts snapshots back from 15:15 (0 = the prediction snapshot); a model with a one-hour window uses lags 0–11.",
        "",
        "| | snapshots | with tokens | median tokens | 5th pct | 95th pct | max |",
        "|---|---|---|---|---|---|---|",
        f"| Prediction snapshot (15:15) | {prediction['snapshots']:,} | {prediction['with_tokens']:,} | {prediction['median']:.0f} | "
        f"{prediction['p5']:.0f} | {prediction['p95']:.0f} | {prediction['max']} |",
        f"| All sequence snapshots | {sequence['snapshots']:,} | {sequence['with_tokens']:,} | {sequence['median']:.0f} | "
        f"{sequence['p5']:.0f} | {sequence['p95']:.0f} | {sequence['max']} |",
        "",
        f"Days with tokens at every one of their {tokens_report['sequence_snapshots_per_day']} snapshots: {tokens_report['days_with_full_sequence']:,}.",
        "",
        "| year | share of sequence snapshots with tokens | median tokens |",
        "|---|---|---|",
        *[f"| {row['year']} | {pct(row['share_with_tokens'])} | {row['median_tokens']:.0f} |" for row in coverage],
        "",
        f"![Tokens by time of day]({rel}/data_tokens_by_time.png)",
        "",
        f"![One day of token history]({rel}/data_token_day.png)",
        "",
    ]
    out = REPO_ROOT / "docs" / "findings_data.md"
    out.write_text("\n".join(lines))
    print(out)


if __name__ == "__main__":
    main()
