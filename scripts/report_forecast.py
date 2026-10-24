"""Report for the classical forecasting ladder: outputs/reports/forecast_baselines.json, figures, docs/findings_forecast.md.

Every number and figure comes from the results store (scripts/run_forecast.py) and the forecasting table.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from niftygex.config import REPO_ROOT
from niftygex.io.parquet import write_report
from niftygex_eval.config import load_eval_config
from niftygex_eval.figures import MUTED, SERIES, TEXT_2, label_line_end, save
from niftygex_eval.forecast.table import forecast_table

LABELS = {"historical_mean": "0 Historical mean", "persistence": "0 Persistence", "har": "1 HAR-RV",
          "har_gex": "2 HAR + GEX", "lightgbm": "3 Tuned LightGBM"}


def f(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def md_table(rows: list[dict], columns: list[tuple[str, str]], digits: int = 3) -> str:
    out = ["| " + " | ".join(label for _, label in columns) + " |", "|" + "---|" * len(columns)]
    return "\n".join(out + ["| " + " | ".join(f(r.get(k), digits) for k, _ in columns) + " |" for r in rows])


def figure_inputs(table: pl.DataFrame, outcome: str, directory: Path) -> Path:
    """HAR-RV terms and the outcome over time: the data the ladder sees."""
    frame = table.sort("trade_date").to_pandas()
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    for color, (column, label) in zip(SERIES, [("rv_trailing_1d_annualized_prev", "previous day"),
                                               ("rv_trailing_5d_annualized_prev", "5-day"),
                                               ("rv_trailing_20d_annualized_prev", "20-day")]):
        values = frame[column] * 100
        top.plot(frame["trade_date"], values, color=color, linewidth=1.0 if column.endswith("1d_annualized_prev") else 1.6, label=label)
    top.set_ylabel("annualized futures RV (%)")
    top.set_title("HAR-RV inputs: realized volatility through the previous day")
    top.legend(loc="upper right", ncol=3)
    bottom.plot(frame["trade_date"], frame[outcome], color=SERIES[0], linewidth=0.8)
    bottom.axhline(0, color=TEXT_2, linewidth=0.8)
    bottom.set_ylabel("log(next-day RV / 20-day mean)")
    bottom.set_title("Outcome: next-day volatility relative to its recent normal")
    bottom.xaxis.set_major_locator(mdates.YearLocator())
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    return save(fig, directory, "forecast_inputs")


def figure_ladder(rows: list[dict], directory: Path) -> Path:
    """Mean per-walk IC with its 95% interval, one row per rung."""
    rows = [r for r in rows if r["ic_ci95"] is not None]
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(rows) + 1.4))
    for i, row in enumerate(rows):
        low, high = row["ic_ci95"]
        ax.plot([low, high], [i, i], color=SERIES[0], linewidth=2, solid_capstyle="round")
        ax.plot(row["ic_mean_per_walk"], i, "o", color=SERIES[0], markersize=8, markeredgecolor="white", markeredgewidth=2)
        ax.annotate(f"{row['ic_mean_per_walk']:.3f}", (high, i), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=TEXT_2)
    ax.axvline(0, color=TEXT_2, linewidth=0.8)
    ax.set_yticks(range(len(rows)), [LABELS[r["model"]] for r in rows])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("mean per-walk IC (Spearman), 95% interval across walks")
    ax.set_title("Classical ladder: how well each model ranks calm and wild days")
    return save(fig, directory, "forecast_ladder_ic")


def figure_ic_by_walk(walk_metrics: dict[str, pl.DataFrame], directory: Path) -> Path:
    """Per-walk IC over the test periods for the rungs that make varying predictions."""
    fig, ax = plt.subplots(figsize=(10, 3.8))
    models = [m for m in ("persistence", "har", "har_gex", "lightgbm") if m in walk_metrics]
    for color, name in zip(SERIES, models):
        frame = walk_metrics[name].sort("test_start").to_pandas()
        ax.plot(frame["test_start"], frame["ic"], color=color, marker="o", markersize=4, label=LABELS[name])
        label_line_end(ax, frame["test_start"].iloc[-1], frame["ic"].iloc[-1], LABELS[name].split(" ", 1)[1])
    ax.axhline(0, color=TEXT_2, linewidth=0.8)
    ax.set_ylabel("IC in the test block")
    ax.set_title("IC per walk (each point is one ~13-week test block)")
    ax.legend(loc="lower left", ncol=len(models))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    return save(fig, directory, "forecast_ic_by_walk")


def figure_actual_vs_predicted(predictions: pl.DataFrame, name: str, days: int, directory: Path) -> Path:
    """The best rung's predictions against what happened, over the last ``days`` test days."""
    frame = predictions.sort("trade_date").tail(days).to_pandas()
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(frame["trade_date"], frame["y"], color=MUTED, linewidth=1.0, label="actual")
    ax.plot(frame["trade_date"], frame["prediction"], color=SERIES[0], linewidth=1.8, label=f"predicted ({LABELS[name].split(' ', 1)[1]})")
    ax.axhline(0, color=TEXT_2, linewidth=0.8)
    ax.set_ylabel("log volatility ratio")
    ax.set_title(f"Actual vs predicted, last {days} test days")
    ax.legend(loc="upper left", ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    return save(fig, directory, "forecast_actual_vs_predicted")


def main() -> None:
    cfg = load_eval_config()
    root = Path(cfg["results_root"])
    comparison = json.loads((root / "forecast_comparison.json").read_text())
    figures_dir = Path(cfg["data_config"]["paths"]["reports_root"]) / "figures"
    table, outcome, _, _ = forecast_table(cfg)
    runs = {name: root / "runs" / run for name, run in comparison["runs"].items()}
    walk_metrics = {name: pl.read_parquet(path / "walk_metrics.parquet") for name, path in runs.items()}
    rows = comparison["comparison_table"]
    best = max((r for r in rows if r["ic_ci95"] is not None), key=lambda r: r["ic_mean_per_walk"])["model"]
    figures = [figure_inputs(table, outcome, figures_dir), figure_ladder(rows, figures_dir), figure_ic_by_walk(walk_metrics, figures_dir),
               figure_actual_vs_predicted(pl.read_parquet(runs[best] / "predictions.parquet"), best, 250, figures_dir)]
    reference = comparison["reference"]
    tests = [{"comparison": f"{LABELS[k.split('_vs_')[0]]} vs {LABELS[reference]}", "ic_diff": v["ic"].get("mean_diff"),
              "ic_ci_low": (v["ic"].get("ci95") or [None])[0], "ic_ci_high": (v["ic"].get("ci95") or [None, None])[1],
              "ic_p": v["ic"].get("p_two_sided"), "ic_wins": v["ic"].get("a_wins"), "walks": v["ic"].get("walks"),
              "dm_squared_diff": v["dm_squared"].get("mean_loss_diff"), "dm_squared_p": v["dm_squared"].get("p_two_sided"),
              "dm_qlike_diff": v["dm_qlike"].get("mean_loss_diff"), "dm_qlike_p": v["dm_qlike"].get("p_two_sided")}
             for k, v in comparison["paired_vs_reference"].items()]
    regime = cfg["eval"]["forecast"]["regime_column"]
    regime_rows = [{"model": LABELS[r["model"]], "period": "after Nov 2024" if p[regime] else "before Nov 2024", "n": p["n"],
                    "ic": p["ic"], "rmse": p["rmse"], "qlike": p["qlike"]} for r in rows for p in r["by_regime"]]
    table_rows = [{**r, "label": LABELS[r["model"]], "ic_low": (r["ic_ci95"] or [None])[0], "ic_high": (r["ic_ci95"] or [None, None])[1]} for r in rows]

    write_report(cfg["data_config"], "forecast_baselines", {
        "eval_config_hash": comparison["eval_config_hash"], "outcome": outcome, "walks": comparison["walks"], "best_model": best,
        "models": table_rows, "paired_vs_reference": tests, "by_regime": regime_rows,
        "figures": [str(p.relative_to(REPO_ROOT)) for p in figures]})

    rel = "../" + str(figures_dir.relative_to(REPO_ROOT))
    lines = [
        "# Classical forecasting ladder (proposal steps 0–3)",
        "",
        "Generated by `scripts/report_forecast.py` from the results store; rerun `scripts/run_forecast.py` first. "
        f"Data config hash `{comparison['data_config_hash'][:12]}`, eval config hash `{comparison['eval_config_hash'][:12]}`.",
        "",
        f"**Task.** Predict `{outcome}`: next-day front-month futures RV over its trailing 20-day mean, at the 15:15 snapshot. "
        f"{comparison['walks']} expanding walk-forward walks on expiry cycles (one-cycle embargo, label purge); every model sees the same rows.",
        "",
        "## 1. Inputs",
        "",
        f"![HAR inputs and outcome]({rel}/forecast_inputs.png)",
        "",
        "## 2. Scoreboard",
        "",
        md_table(table_rows, [("label", "model"), ("walks", "walks"), ("ic_mean_per_walk", "IC mean/walk"), ("ic_low", "IC 95% low"),
                              ("ic_high", "IC 95% high"), ("ic_positive_walks", "IC>0 walks"), ("rmse_mean_per_walk", "RMSE"),
                              ("qlike_mean_per_walk", "QLIKE"), ("r2_vs_reference_mean_per_walk", f"R² vs {LABELS[reference]}")]),
        "",
        f"![Ladder IC]({rel}/forecast_ladder_ic.png)",
        "",
        f"![IC per walk]({rel}/forecast_ic_by_walk.png)",
        "",
        f"## 3. Against {LABELS[reference]} (exploratory; the pre-registered RQ tests involve the neural models)",
        "",
        md_table(tests, [("comparison", "comparison"), ("ic_diff", "IC diff"), ("ic_ci_low", "95% low"), ("ic_ci_high", "95% high"),
                         ("ic_p", "p (paired t)"), ("ic_wins", "walks won"), ("walks", "walks"), ("dm_squared_diff", "Δ squared loss"),
                         ("dm_squared_p", "DM p"), ("dm_qlike_diff", "Δ QLIKE"), ("dm_qlike_p", "DM p")], 4),
        "",
        "Δ loss is the model's mean daily loss minus HAR-RV's: negative favours the model, positive favours HAR-RV.",
        "",
        "## 4. Before and after the November 2024 rule change (pooled test days)",
        "",
        md_table(regime_rows, [("model", "model"), ("period", "period"), ("n", "days"), ("ic", "IC"), ("rmse", "RMSE"), ("qlike", "QLIKE")]),
        "",
        f"## 5. What the best rung's forecasts look like ({LABELS[best]})",
        "",
        f"![Actual vs predicted]({rel}/forecast_actual_vs_predicted.png)",
        "",
        "Historical mean predicts one constant per walk, so its per-walk IC is undefined (—) and its pooled IC only reflects "
        "differences between walks; it is the no-skill reference for RMSE and QLIKE.",
        "",
    ]
    out = REPO_ROOT / "docs" / "findings_forecast.md"
    out.write_text("\n".join(lines))
    print(out)


if __name__ == "__main__":
    main()
