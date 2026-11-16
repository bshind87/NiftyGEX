"""Results report for the full ladder (steps 0-6) and the pre-registered RQ tests: JSON, figures, docs/findings_models.md.

Reads outputs/results/forecast_comparison.json, neural_comparison.json and rq_tests.json (run run_forecast.py,
run_neural.py and run_rq_tests.py first). --tag dev reads the *_dev files and writes outputs/reports/findings_models_dev.md instead.
"""
from __future__ import annotations

import argparse
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
from niftygex_eval.rq1_summary import METRICS_PLAIN, ladder_improvements, qlike_explained, rq1_summary
from niftygex_eval.rq1_summary import NAMES as RQ1_NAMES

LABELS = {"historical_mean": "0 Historical mean", "persistence": "0 Persistence", "har": "1 HAR-RV", "har_gex": "2 HAR + GEX",
          "lightgbm": "3 Tuned LightGBM", "deepsets": "4 DeepSets", "attention": "5 Self-attention", "attention_gru": "6 Attention + GRU"}


def f(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def md_table(rows: list[dict], columns: list[tuple[str, str]], digits: int = 3) -> str:
    out = ["| " + " | ".join(label for _, label in columns) + " |", "|" + "---|" * len(columns)]
    return "\n".join(out + ["| " + " | ".join(f(r.get(k), digits) for k, _ in columns) + " |" for r in rows])


def pct(value) -> str:
    return "—" if value is None else f"{value:+.1f}%"


def signed(value, digits: int = 3) -> str:
    return "—" if value is None else f"{value:+.{digits}f}"


def figure_errors(rows: list[dict], directory: Path, suffix: str) -> Path:
    """Forecast error per model, lower is better: RMSE and QLIKE side by side (historical mean included)."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 0.45 * len(rows) + 1.4), sharey=True)
    for ax, key, title in ((axes[0], "rmse_mean_per_walk", "RMSE (lower is better)"), (axes[1], "qlike_mean_per_walk", "QLIKE (lower is better)")):
        values = [r[key] for r in rows]
        best = int(np.argmin(values))
        for i, (row, v) in enumerate(zip(rows, values)):
            color = SERIES[1] if row["model"] in ("deepsets", "attention", "attention_gru") else SERIES[0]
            ax.barh(i, v, color=color, alpha=1.0 if i == best else 0.55)
            ax.annotate(f"{v:.3f}", (v, i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8.5, color=TEXT_2)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(values) * 1.18)
    axes[0].set_yticks(range(len(rows)), [r["label"] for r in rows])
    axes[0].invert_yaxis()
    fig.suptitle("Forecast error on the same 26 test periods: classical (blue) and neural (orange); darkest bar = lowest", x=0.01, ha="left", fontsize=10)
    return save(fig, directory, f"results_errors{suffix}")


def figure_ladder(rows: list[dict], directory: Path, suffix: str) -> Path:
    rows = [r for r in rows if r.get("ic_ci95")]
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(rows) + 1.4))
    for i, row in enumerate(rows):
        color = SERIES[1] if row["model"] in ("deepsets", "attention", "attention_gru") else SERIES[0]
        low, high = row["ic_ci95"]
        ax.plot([low, high], [i, i], color=color, linewidth=2, solid_capstyle="round")
        ax.plot(row["ic_mean_per_walk"], i, "o", color=color, markersize=8, markeredgecolor="white", markeredgewidth=2)
        ax.annotate(f"{row['ic_mean_per_walk']:.3f}", (high, i), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=TEXT_2)
    ax.axvline(0, color=TEXT_2, linewidth=0.8)
    ax.set_yticks(range(len(rows)), [LABELS[r["model"]] for r in rows])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("mean per-walk IC (Spearman), 95% interval across walks")
    ax.set_title("The full ladder: classical (blue) and neural (orange) models on the same walks")
    return save(fig, directory, f"results_ladder_ic{suffix}")


def figure_ic_by_walk(walks: dict[str, pl.DataFrame], directory: Path, suffix: str) -> Path:
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for color, name in zip((SERIES[0], SERIES[2], SERIES[1]), ("har", "lightgbm", "attention_gru")):
        if name not in walks:
            continue
        frame = walks[name].sort("test_start").to_pandas()
        ax.plot(frame["test_start"], frame["ic"], color=color, marker="o", markersize=4, label=LABELS[name])
        label_line_end(ax, frame["test_start"].iloc[-1], frame["ic"].iloc[-1], LABELS[name].split(" ", 1)[1])
    ax.axhline(0, color=TEXT_2, linewidth=0.8)
    ax.set_ylabel("IC in the test block")
    ax.set_title("IC per walk: the bar to beat (HAR-RV), the best classical model, and our model")
    ax.legend(loc="lower left", ncol=3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    return save(fig, directory, f"results_ic_by_walk{suffix}")


def figure_regime(per_walk: list[dict], walk_dates: pl.DataFrame, directory: Path, suffix: str) -> Path:
    frame = pl.DataFrame(per_walk).join(walk_dates, on="walk_id").sort("test_start").to_pandas()
    fig, ax = plt.subplots(figsize=(10, 3.2))
    colors = [SERIES[1] if after else MUTED for after in frame["after"]]
    ax.bar(frame["test_start"], frame["gain"], width=60, color=colors)
    ax.axhline(0, color=TEXT_2, linewidth=0.8)
    ax.set_ylabel("IC gain over LightGBM")
    ax.set_title("RQ3: attention + GRU's IC gain over tuned LightGBM per walk (orange = after the Nov 2024 rule change)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    return save(fig, directory, f"results_regime{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="main")
    tag = parser.parse_args().tag
    suffix = "" if tag == "main" else f"_{tag}"
    cfg = load_eval_config()
    root = Path(cfg["results_root"])
    figures = Path(cfg["data_config"]["paths"]["reports_root"]) / "figures"
    classical = json.loads((root / "forecast_comparison.json").read_text())
    neural = json.loads((root / f"neural_comparison{suffix}.json").read_text())
    tests = json.loads((root / f"rq_tests{suffix}.json").read_text())
    runs = {**classical["runs"], **neural["runs"]}
    walks = {name: pl.read_parquet(root / "runs" / run / "walk_metrics.parquet") for name, run in runs.items()}
    keep = set(walks["attention_gru"]["walk_id"].to_list()) if "attention_gru" in walks else None
    if keep is not None:
        walks = {name: frame.filter(pl.col("walk_id").is_in(list(keep))) for name, frame in walks.items()}

    from niftygex_eval.forecast.compare import mean_ci

    rows = []
    for name in LABELS:
        if name not in walks:
            continue
        w = walks[name]
        rows.append({"model": name, "label": LABELS[name], "walks": w.height, "ic_mean_per_walk": float(w["ic"].mean()) if w["ic"].is_not_nan().any() else None,
                     "ic_ci95": mean_ci(w["ic"].to_numpy()), "rmse_mean_per_walk": float(w["rmse"].mean()), "qlike_mean_per_walk": float(w["qlike"].mean())})
    for r in rows:
        r["ic_low"], r["ic_high"] = (r["ic_ci95"] or [None, None])
    walk_dates = walks["attention_gru"].select("walk_id", "test_start")
    paths = [figure_ladder(rows, figures, suffix), figure_errors(rows, figures, suffix), figure_ic_by_walk(walks, figures, suffix),
             figure_regime(tests["RQ3"]["per_walk"], walk_dates, figures, suffix)]

    def test_rows(block: dict) -> list[dict]:
        out = []
        for name, c in block.items():
            ic = c["ic"]
            pair = name.split("_vs_")
            out.append({"comparison": name.replace("_vs_", " vs "), "label": " vs ".join(RQ1_NAMES.get(x, x) for x in pair), "ic_diff": ic.get("mean_diff"), "ci_low": (ic.get("ci95") or [None])[0],
                        "ci_high": (ic.get("ci95") or [None, None])[1], "p": ic.get("p_two_sided"), "wins": ic.get("a_wins"), "walks": ic.get("walks"),
                        "dm_sq_p": c["dm_squared"].get("p_two_sided"), "dm_qlike_p": c["dm_qlike"].get("p_two_sided")})
        return out

    rq1_rows, rq2_rows = test_rows(tests["RQ1"]["comparisons"]), test_rows(tests["RQ2"]["comparisons"])
    holm = {m["test"]: m for m in tests["RQ2"]["holm"]["members"]}
    for r, name in zip(rq2_rows, tests["RQ2"]["comparisons"]):
        r["p_holm"], r["reject"] = holm[name]["p_holm"], holm[name]["reject_holm"]
    rq3 = tests["RQ3"]
    ladder_by_model = {r["model"]: r for r in rows}
    summary = rq1_summary(tests["RQ1"]["comparisons"], ladder_by_model, tests["alpha"], tests["RQ1"]["beats_every_rung"]) if "attention_gru" in ladder_by_model else None
    steps = ladder_improvements(ladder_by_model) if "har" in ladder_by_model and "persistence" in ladder_by_model else []
    step_rows = [{"model": r["name"], "rmse": f"{r['rmse']:.3f}", "rmse_vs_har": pct(r["rmse_vs_har_pct"]), "qlike": f"{r['qlike']:.3f}",
                  "qlike_vs_har": pct(r["qlike_vs_har_pct"]), "ic": f"{r['ic']:.3f}", "ic_vs_persistence": signed(r["ic_vs_persistence"]),
                  "ic_vs_har": signed(r["ic_vs_har"]), "ic_vs_previous": signed(r["ic_vs_previous"]) + (f" (vs {r['previous']})" if r["previous"] else "")}
                 for r in steps]
    write_report(cfg["data_config"], f"results{suffix}", {"tag": tag, "eval_config_hash": tests["eval_config_hash"], "ladder": rows,
                                                         "rq1": rq1_rows, "rq1_beats_every_rung": tests["RQ1"]["beats_every_rung"],
                                                         "rq1_summary": summary, "improvements": steps,
                                                         "rq2": rq2_rows, "rq3": {k: v for k, v in rq3.items() if k != "per_walk"},
                                                         "figures": [str(p.relative_to(REPO_ROOT)) for p in paths]})
    rel = "../" + str(figures.relative_to(REPO_ROOT)) if tag == "main" else "figures"
    note = "" if tag == "main" else (f"\n> **Development run** on {neural['walks']} walks only, with a stand-in model; not a result.\n")
    lines = [
        "# Results: the full ladder and the pre-registered tests" + ("" if tag == "main" else " (development run)"),
        note,
        "Generated by `scripts/report_results.py` from the results store; run `scripts/run_forecast.py`, `scripts/run_neural.py` and "
        f"`scripts/run_rq_tests.py` first. Eval config hash `{tests['eval_config_hash'][:12]}`; history window L = {neural['window']} snapshots.",
        "",
        "## 1. The ladder (steps 0–6)",
        "",
        "How to read the scores:",
        "",
        *[f"- **{name}**: {text}." for name, text in METRICS_PLAIN],
        "",
        "**Why QLIKE?**",
        "",
        *[f"- {line}" for line in qlike_explained()],
        "",
        md_table(rows, [("label", "model"), ("walks", "walks"), ("rmse_mean_per_walk", "RMSE"), ("qlike_mean_per_walk", "QLIKE"),
                        ("ic_mean_per_walk", "IC mean/walk"), ("ic_low", "IC 95% low"), ("ic_high", "IC 95% high")]),
        "",
        f"![Forecast error]({rel}/results_errors{suffix}.png)",
        "",
        f"![Ladder]({rel}/results_ladder_ic{suffix}.png)",
        "",
        f"![IC per walk]({rel}/results_ic_by_walk{suffix}.png)",
        "",
        "### What each step adds",
        "",
        "Change of each model's mean score against HAR-RV (errors: negative = lower error, better) and against the step before it "
        "(IC). Differences of means over the 26 test periods, descriptive; the paired tests are in sections 2 and 3.",
        "",
        md_table(step_rows, [("model", "model"), ("rmse", "RMSE"), ("rmse_vs_har", "RMSE vs HAR"), ("qlike", "QLIKE"), ("qlike_vs_har", "QLIKE vs HAR"),
                             ("ic", "IC"), ("ic_vs_persistence", "IC gain vs persistence"), ("ic_vs_har", "IC gain vs HAR"),
                             ("ic_vs_previous", "IC gain vs previous step")]) if step_rows else "",
        "",
        "## 2. RQ1: does reading every contract predict better?",
        "",
        *([f"**{summary['headline']}** {summary['loss']}", "", summary["gex"] or "", "", summary["rule"], ""] if summary else []),
        f"Attention + GRU against every classical rung (pre-registered: a win requires every IC comparison to reject at "
        f"{tests['alpha']}; met: {'yes' if tests['RQ1']['beats_every_rung'] else 'no'}). Diebold–Mariano p-values compare daily forecast errors "
        "(squared error for RMSE, QLIKE loss for QLIKE).",
        "",
        md_table(rq1_rows, [("label", "comparison"), ("ic_diff", "IC diff"), ("ci_low", "95% low"), ("ci_high", "95% high"), ("p", "p (paired t)"),
                            ("wins", "walks won"), ("walks", "walks"), ("dm_sq_p", "DM p (squared)"), ("dm_qlike_p", "DM p (QLIKE)")], 4),
        "",
        "## 3. RQ2: if it wins, why? (Holm within the family)",
        "",
        md_table(rq2_rows, [("label", "comparison"), ("ic_diff", "IC diff"), ("ci_low", "95% low"), ("ci_high", "95% high"), ("p", "p"),
                            ("p_holm", "Holm p"), ("reject", "reject"), ("wins", "walks won"), ("walks", "walks")], 4),
        "",
        "## 4. RQ3: does it survive the rule change?",
        "",
        f"IC gain over tuned LightGBM: {f(rq3['mean_gain_before'], 4)} on {rq3['walks_before']} walks before the November 2024 marker and "
        f"{f(rq3['mean_gain_after'], 4)} on {rq3['walks_after']} walks after (Welch t {f(rq3['welch_t'], 2)}, p {f(rq3['p_two_sided'], 3)}).",
        "",
        f"![Regime]({rel}/results_regime{suffix}.png)",
        "",
    ]
    out = REPO_ROOT / "docs" / "findings_models.md" if tag == "main" else Path(cfg["data_config"]["paths"]["reports_root"]) / f"findings_models{suffix}.md"
    out.write_text("\n".join(lines))
    print(out)


if __name__ == "__main__":
    main()
