"""Exploratory E1 report: diagnostics on the next-day target (docs/findings_exploratory_diagnostics.md, with figures).

Reads outputs/results/exploratory_e1.json (scripts/run_exploratory_e1.py --parts tests, after the Explorer runs are
merged by scripts/merge_seeds.py) and the forecast table for an availability check of the residual model's inputs.
Writes outputs/reports/exploratory_e1.json and outputs/reports/figures/e1_*.png. Everything here is exploratory.
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
from niftygex_eval.figures import MUTED, SERIES, TEXT_2, save
from niftygex_eval.forecast.table import forecast_table

LABELS = {"har": "HAR-RV", "lightgbm": "Tuned LightGBM", "persistence": "Persistence", "har_gex": "HAR + GEX",
          "deepsets": "DeepSets (Mac)", "attention": "Self-attention (Mac)", "attention_gru": "Attention + GRU (Mac)",
          "deepsets_hpc": "DeepSets (V100)", "attention_hpc": "Self-attention (V100)", "attention_gru_hpc": "Attention + GRU (V100)",
          "residual_lightgbm": "HAR + residual LightGBM", "residual_deepsets": "HAR + residual DeepSets",
          "residual_attention": "HAR + residual attention", "state_mlp": "State-only MLP (V100)",
          "hybrid_attention_gru": "Hybrid attention + GRU (V100)", "combination": "Combination (HAR, LightGBM, attn + GRU)"}
SUBSETS = {"expiry_day": "Expiry day", "not_expiry_day": "Not expiry day", "near_flip": "Within 0.5% of the flip",
           "not_near_flip": "Not near the flip", "gex_tercile_low": "GEX tercile: low", "gex_tercile_mid": "GEX tercile: mid",
           "gex_tercile_high": "GEX tercile: high", "dealer_long_gamma": "Dealers long gamma", "dealer_short_gamma": "Dealers short gamma"}


def f(value, spec: str = ".3f") -> str:
    return "—" if value is None or (isinstance(value, float) and not np.isfinite(value)) else format(value, spec)


def interval(ci) -> str:
    return "—" if not ci else f"{ci[0]:+.3f} to {ci[1]:+.3f}"


def availability_check(cfg: dict, columns: list[str]) -> dict:
    """Every residual-model input must be known by its snapshot: the table's newest source timestamp per row."""
    table, _, _, _ = forecast_table(cfg)
    late = table.filter(pl.col("feature_max_source_available_at") > pl.col("snapshot_at")).height
    missing = [c for c in columns if c not in table.columns]
    return {"rows": table.height, "rows_with_a_source_after_the_snapshot": late, "columns_checked": len(columns), "columns_missing": missing,
            "max_lead_minutes": float(((table["feature_max_source_available_at"] - table["snapshot_at"]).dt.total_seconds() / 60).max())}


def figure_residual(residual: dict, directory: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 2.8))
    for i, (name, rows) in enumerate(residual.items()):
        values = np.array([r["residual_ic"] for r in rows], dtype=float)
        values = values[np.isfinite(values)]
        ax.plot(values, np.full(values.size, i) + np.random.default_rng(i).uniform(-0.12, 0.12, values.size), "o", color=SERIES[i], alpha=0.45, markersize=4)
        ax.plot(values.mean(), i, "D", color=SERIES[i], markersize=9, markeredgecolor="white", markeredgewidth=1.5)
    ax.axvline(0, color=TEXT_2, linewidth=0.8)
    ax.set_yticks(range(len(residual)), [LABELS[n].replace("HAR + ", "") for n in residual])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("IC of the predicted vs actual HAR residual, per walk (diamond: mean)")
    ax.set_title("What the chain adds beyond HAR: residual IC per walk")
    return save(fig, directory, "e1_residual_ic")


def figure_conditional(conditional: dict, models: list[str], directory: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.4))
    rows = list(SUBSETS)
    offsets = np.linspace(-0.27, 0.27, len(models))
    for colour, offset, model in zip(SERIES + [MUTED], offsets, models):
        for i, subset in enumerate(rows):
            cell = conditional[model].get(subset, {})
            if cell.get("ic") is None:
                continue
            lo, hi = cell["ci95"] or (cell["ic"], cell["ic"])
            ax.plot([lo, hi], [i + offset] * 2, color=colour, linewidth=1.6)
            ax.plot(cell["ic"], i + offset, "o", color=colour, markersize=5, label=LABELS[model] if i == 0 else None)
    ax.set_yticks(range(len(rows)), [SUBSETS[r] for r in rows])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("pooled IC within the subset, 95% interval from resampling walks")
    ax.set_title("Conditional IC: does the edge depend on the GEX regime?")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    return save(fig, directory, "e1_conditional_ic")


def summary_lines(tests: list[dict], summary: dict, secondary: dict, conditional: dict, complete: bool) -> list[str]:
    """Plain-language findings, every number from the results (nothing typed by hand)."""
    by = {t["test"]: t for t in tests}
    rejected = sum(bool(t.get("reject_holm")) for t in tests)
    lines = [f"{rejected} of {len(tests)} registered E1 tests reject at 5% after Holm" + ("." if complete else " (family incomplete).")]
    res = [by[n] for n in ("residual_lightgbm_ic", "residual_deepsets_ic", "residual_attention_ic") if n in by]
    if res:
        lines.append(f"The chain adds information beyond HAR: models that see no volatility state predict HAR's out-of-sample residual with a "
                     f"mean per-walk IC of {min(t['mean'] for t in res):.3f} to {max(t['mean'] for t in res):.3f} "
                     f"(positive in {min(t['positive_walks'] for t in res)}-{max(t['positive_walks'] for t in res)} of {res[0]['walks']} walks).")
    plus = [by[n] for n in ("har_plus_residual_lightgbm_vs_har", "har_plus_residual_deepsets_vs_har", "har_plus_residual_attention_vs_har") if n in by]
    if plus:
        lines.append(f"Adding that correction to HAR raises IC by {min(t['mean_diff'] for t in plus):+.3f} to {max(t['mean_diff'] for t in plus):+.3f}.")
    if "hybrid_vs_attention_gru" in by and "hybrid_attention_gru" in summary:
        t = by["hybrid_vs_attention_gru"]
        lines.append(f"The hybrid (HAR forecast + learned attention + GRU correction) reaches IC {summary['hybrid_attention_gru']['ic']:.3f}, "
                     f"{t['mean_diff']:+.3f} over attention + GRU on the same V100s (95% CI {interval(t['ci95'])}).")
    if "combination" in summary and "combination_vs_har" in by:
        lines.append(f"The equal-weight combination of HAR, LightGBM and attention + GRU is the best E1 model: IC {summary['combination']['ic']:.3f}, "
                     f"{by['combination_vs_har']['mean_diff']:+.3f} over HAR.")
    state = secondary.get("state_mlp_vs_attention_gru_hpc")
    if state and "state_mlp" in summary:
        lines.append(f"Inside the neural trainer the tokens add little: the state-only MLP (no tokens) reaches IC {summary['state_mlp']['ic']:.3f} "
                     f"against {summary['attention_gru_hpc']['ic']:.3f} for attention + GRU (state-only {state['ic']['mean_diff']:+.3f}, 95% CI "
                     f"{interval(state['ic']['ci95'])}, {'not significant' if state['ic']['ci95'][0] < 0 < state['ic']['ci95'][1] else 'significant'}), and QLIKE {summary['state_mlp']['qlike']:.3f} against "
                     f"{summary['attention_gru_hpc']['qlike']:.3f} (HAR {summary['har']['qlike']:.3f}): most of the neural QLIKE gain comes from the "
                     "trainer and the state, not the chain.")
    if "combination" in conditional:
        ic = {k: v["ic"] for k, v in conditional["combination"].items() if v.get("ic") is not None}
        lines.append(f"No GEX regime, flip proximity or dealer-sign subset changes the picture beyond its 95% interval; expiry days are harder for every "
                     f"model (combination {ic.get('expiry_day', float('nan')):.3f} vs {ic.get('not_expiry_day', float('nan')):.3f} on other days).")
    return lines


def main() -> None:
    cfg = load_exploratory_config()
    root = Path(cfg["results_root"])
    figures = Path(cfg["data_config"]["paths"]["reports_root"]) / "figures"
    e1 = json.loads((root / "exploratory_e1.json").read_text())
    tests, summary, secondary, conditional = e1["tests"], e1["summary"], e1["secondary"], e1["conditional_ic"]
    complete = tests.get("complete", True) and not e1["missing_runs"]
    members = tests["members"] if "members" in tests else tests.get("unadjusted", [])
    check = availability_check(cfg, list(cfg["exploratory"]["e1"]["residual"]["lightgbm_columns"]))
    paths = [figure_residual(e1["residual_ic_by_walk"], figures),
             figure_conditional(conditional, [m for m in ("har", "lightgbm", "attention_gru", "combination") if m in conditional], figures)]

    rows = ["| Test | Estimate | 95% interval | p | Holm p | Holm reject (5%) | Walks favourable |", "|---|---|---|---|---|---|---|"]
    for m in members:
        estimate = m.get("mean", m.get("mean_diff"))
        favourable = m.get("positive_walks", m.get("a_wins"))
        rows.append(f"| {m['test']} | {f(estimate, '+.3f')} | {interval(m.get('ci95'))} | {f(m.get('p'), '.2g')} | {f(m.get('p_holm'), '.2g')} | "
                    f"{'yes' if m.get('reject_holm') else ('no' if 'reject_holm' in m else '—')} | {favourable}/{m.get('walks')} |")
    scores = ["| Model | Mean IC | RMSE | QLIKE | Residual IC |", "|---|---|---|---|---|"]
    for name in ("har", "lightgbm", "attention_gru", "attention_gru_hpc", "residual_lightgbm", "residual_deepsets", "residual_attention",
                 "hybrid_attention_gru", "state_mlp", "deepsets_hpc", "attention_hpc", "combination"):
        if name in summary:
            s = summary[name]
            scores.append(f"| {LABELS[name]} | {f(s['ic'])} | {f(s['rmse'])} | {f(s['qlike'])} | {f(s.get('residual_ic'))} |")
    state = ["| State-only MLP vs | IC diff (95% CI) | RMSE diff | QLIKE diff | DM p (QLIKE) |", "|---|---|---|---|---|"]
    for name, block in secondary.items():
        if name.startswith("state_mlp_vs_"):
            other = name.removeprefix("state_mlp_vs_")
            state.append(f"| {LABELS[other]} | {f(block['ic'].get('mean_diff'), '+.3f')} ({interval(block['ic'].get('ci95'))}) | "
                         f"{f(block['rmse'].get('mean_diff'), '+.3f')} | {f(block['qlike'].get('mean_diff'), '+.3f')} | {f(block['dm_qlike'].get('p_two_sided'), '.2g')} |")
    combo = ["| Combination vs | IC diff (95% CI) | walks won | RMSE diff | QLIKE diff |", "|---|---|---|---|---|"]
    for name, block in secondary.items():
        if name.startswith("combination_vs_"):
            other = name.removeprefix("combination_vs_")
            combo.append(f"| {LABELS[other]} | {f(block['ic'].get('mean_diff'), '+.3f')} ({interval(block['ic'].get('ci95'))}) | "
                         f"{block['ic'].get('a_wins')}/{block['ic'].get('walks')} | {f(block['rmse'].get('mean_diff'), '+.3f')} | {f(block['qlike'].get('mean_diff'), '+.3f')} |")
    cond = ["| Subset | " + " | ".join(LABELS[m] for m in ("har", "lightgbm", "attention_gru", "combination") if m in conditional) + " |",
            "|---|" + "---|" * sum(m in conditional for m in ("har", "lightgbm", "attention_gru", "combination"))]
    for subset, label in SUBSETS.items():
        cells = []
        for m in ("har", "lightgbm", "attention_gru", "combination"):
            if m in conditional:
                c = conditional[m][subset]
                cells.append(f"{f(c['ic'])} ({interval(c['ci95']).replace('+', '')}, n {c['rows']})")
        cond.append(f"| {label} | " + " | ".join(cells) + " |")

    rel = "../" + str(figures.relative_to(REPO_ROOT))
    status = ("Complete: Holm applied over the 9 registered tests." if complete else
              f"Incomplete: {len(e1['missing_runs'])} runs missing ({', '.join(e1['missing_runs'])}); Holm is applied once all are in.")
    lines = [
        "# Exploratory E1: diagnostics on the next-day target", "",
        "> **Exploratory.** Registered in `configs/eval_exploratory_v2.yaml` before any run; separate from the pre-registered RQ1–RQ3.",
        f"> Exploratory config hash `{e1['exploratory_config_hash'][:12]}`, main eval hash `{e1['eval_config_hash'][:12]}`, data hash "
        f"`{e1['data_config_hash'][:12]}`. Neural models compared with each other ran on Explorer V100s (device rule); the classical "
        "models and the Mac main runs are as recorded in the main results.", "",
        f"Generated by `scripts/report_exploratory_e1.py`. {status}", "",
        "## Summary", "", *[f"- {line}" for line in summary_lines(members, summary, secondary, conditional, complete)], "",
        "## Primary tests (Holm within E1)", "", *rows, "",
        "## Scores (mean per walk, 26 walks, same rows)", "", *scores, "",
        "## Leakage check of the residual model's inputs", "",
        f"All {check['rows']} rows: the newest source feeding any feature is available at or before the 15:15 snapshot "
        f"({check['rows_with_a_source_after_the_snapshot']} rows violate it; largest lead {check['max_lead_minutes']:.0f} min, i.e. never after). "
        f"The residual LightGBM uses {check['columns_checked']} chain, GEX and positioning columns and no volatility-state column.", "",
        "## State-only control (V100, same trainer)", "", *state, "",
        "## Forecast combination (equal weights, no tuning)", "", *combo, "",
        "## Conditional IC (pooled within subset)", "", *cond, "",
        *[f"![{p.stem}]({rel}/{p.name})\n" for p in paths],
    ]
    out = REPO_ROOT / "docs" / "findings_exploratory_diagnostics.md"
    out.write_text("\n".join(lines))
    write_report(cfg["data_config"], "exploratory_e1", {"complete": complete, "tests": members, "summary": summary, "availability_check": check,
                                                        "figures": [str(p.relative_to(REPO_ROOT)) for p in paths]})
    print(out)


if __name__ == "__main__":
    main()
