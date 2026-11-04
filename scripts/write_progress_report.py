"""Midterm progress report (Word): pipeline, sources, features, token history, leakage checks and first results.

Every number comes from the stage manifests, the reports in outputs/reports/ and the results store; the figures are
the ones written by scripts/report_data.py and scripts/report_forecast.py, plus a pipeline diagram drawn here.
Run those two report scripts first. Output: docs/deliverables/NiftyGEX_Midterm_Progress_Report.docx (gitignored)
and outputs/reports/progress_report.json (the numbers used).
"""
from __future__ import annotations

import json
from pathlib import Path

from niftygex.config import REPO_ROOT, load_config
from niftygex.features.lineage import LINEAGE
from niftygex.io.parquet import write_report
from niftygex_eval.docx_report import bullets, figure, footer, load, new_document, table
from niftygex_eval.report_sections import data_gaps_section, data_sections, feature_appendix, load_data_context, pipeline_stages

OUT = REPO_ROOT / "docs" / "deliverables" / "NiftyGEX_Midterm_Progress_Report.docx"
LABELS = {"historical_mean": "Historical mean", "persistence": "Persistence", "har": "HAR-RV", "har_gex": "HAR + GEX", "lightgbm": "Tuned LightGBM"}


def main() -> None:
    config = load_config()
    reports = Path(config["paths"]["reports_root"])
    if not (reports / "forecast_baselines.json").exists():
        raise SystemExit("Run scripts/report_forecast.py first (forecast_baselines.json missing)")
    ctx = load_data_context(config)
    m, window = ctx.m, ctx.window
    forecast = load(Path(config["paths"]["outputs_root"]) / "results" / "forecast_comparison.json")
    rows = forecast["comparison_table"]
    by_model = {r["model"]: r for r in rows}
    har, best = by_model["har"], max((r for r in rows if r["ic_ci95"]), key=lambda r: r["ic_mean_per_walk"])

    doc = new_document("Long Attention, Short Gamma", "Midterm progress report: data pipeline, token history and first baselines")

    doc.add_heading("1. Summary", level=1)
    bullets(doc, [
        f"The data pipeline is complete and validated. It turns {ctx.raw['file_count']} raw files ({window['start']} to {window['end']}, "
        f"{m['clean']['counts']['trading_days_observed']:,} trading days) into features, a token history and a target. All "
        f"{ctx.leakage_checks} timestamp checks and {ctx.leakage_tests} leakage tests pass.",
        f"The classical model inputs exist: {m['features']['counts']['rows']:,} daily snapshots at 15:15 with "
        f"{m['features']['counts']['feature_columns']} features, each traced to its source in Section 5.",
        f"The deep-learning input exists: a token history of {m['tokens']['counts']['tokens']:,} rows, one per liquid option contract "
        f"at every 5-minute snapshot from 09:20 to 15:15 ({ctx.tokens['sequence_snapshots_per_day']} per day, median {ctx.sequence['median']:.0f} "
        "contracts).",
        f"The evaluation harness and the classical baselines (proposal steps 0–3) are done. On {forecast['walks']} walk-forward test "
        f"windows, HAR-RV reaches a mean IC of {har['ic_mean_per_walk']:.3f}; the best classical model, {LABELS[best['model']]}, "
        f"reaches {best['ic_mean_per_walk']:.3f}. This is the bar the attention model must beat.",
        "Next: the volatility-state vector, then the neural models (DeepSets, self-attention, attention + GRU) behind the same "
        "harness, then the pre-registered RQ1–RQ3 tests.",
    ])

    doc.add_heading("2. Project recap", level=1)
    doc.add_paragraph("Traders compress the NIFTY option chain into one number, gamma exposure (GEX). We ask whether a model that "
                      "reads every contract, with self-attention across contracts and a GRU over the recent snapshots, forecasts "
                      "next-day volatility better than GEX-based, classical and tree-based models.")
    table(doc, ["Question", "How it is tested"], [
        ["RQ1: does reading every contract predict better?", "Attention + GRU against every classical rung, on the same walks and rows"],
        ["RQ2: if it wins, why?", "Switch off cross-contract attention (DeepSets) and history (single snapshot)"],
        ["RQ3: does it survive rule changes?", "Scores before and after the November 2024 weekly-expiry rationalisation"],
    ], [2.6, 3.9])
    doc.add_paragraph("The prediction target is log(next-day realized volatility of the front-month future / its trailing 20-day "
                      "mean), measured at a 15:15 snapshot each trading day. The tests are pre-registered in configs/eval.yaml.")

    n = data_sections(doc, ctx, 3)

    doc.add_heading(f"{n}. Evaluation harness and first results", level=1)
    doc.add_paragraph(f"Every model predicts the same outcome on the same rows: {forecast['walks']} expanding walk-forward windows of "
                      "13 expiry cycles each, with tuning on an inner split of the training window only. IC (rank correlation of "
                      "prediction and outcome, per test window) is the primary score.")
    table(doc, ["Model (proposal step)", "Mean IC", "95% interval", "RMSE", "QLIKE"],
          [[f"{LABELS[r['model']]}", f"{r['ic_mean_per_walk']:.3f}" if r["ic_ci95"] else "—",
            f"{r['ic_ci95'][0]:.3f} to {r['ic_ci95'][1]:.3f}" if r["ic_ci95"] else "—",
            f"{r['rmse_mean_per_walk']:.3f}", f"{r['qlike_mean_per_walk']:.3f}"] for r in rows], [2.2, 0.9, 1.5, 0.9, 0.9], size=9)
    tests_vs = forecast["paired_vs_reference"]
    lines = [f"{LABELS[k.split('_vs_')[0]]} vs HAR-RV: IC difference {v['ic']['mean_diff']:+.3f} (95% interval {v['ic']['ci95'][0]:+.3f} to "
             f"{v['ic']['ci95'][1]:+.3f}, p {v['ic']['p_two_sided']:.2f}, wins {v['ic']['a_wins']} of {v['ic']['walks']} windows)"
             for k, v in tests_vs.items() if v["ic"].get("ci95")]
    bullets(doc, lines + ["The historical mean predicts one constant per window, so its IC is undefined; it is the RMSE/QLIKE reference."])
    figure(doc, ctx.figures / "forecast_ladder_ic.png", ctx.caption("Mean IC per test window with 95% intervals."), width=5.6)
    figure(doc, ctx.figures / "forecast_ic_by_walk.png", ctx.caption("IC in each test window."))
    doc.add_paragraph("Reading so far: HAR-RV is the bar to beat. Adding GEX (HAR + GEX) or all summary features (tuned LightGBM) "
                      "does not beat it significantly. These comparisons are exploratory; the pre-registered tests involve the "
                      "neural models.")

    data_gaps_section(doc, ctx, n + 1)

    doc.add_heading(f"{n + 2}. Next steps", level=1)
    table(doc, ["When", "Work"], [
        ["Now", "Volatility-state vector for the model head (IV, RV, calendar and total GEX)"],
        ["Early November", "Neural models behind the same harness: DeepSets, single-snapshot self-attention, attention + GRU "
                           "(2 layers × 4 heads, 64-d, about 100k parameters); sanity checks (overfit a tiny batch, permutation and padding invariance)"],
        ["November", "Pre-registered RQ1–RQ3 tests, window and field ablations, attention maps"],
        ["Dec 3 / Dec 10, Dec 17", "Final presentation, report and reproducible code"],
    ], [1.5, 5.0], size=9)

    feature_appendix(doc, ctx)
    footer(doc, f"NiftyGEX · generated by scripts/write_progress_report.py · data config {m['features']['config_hash'][:12]}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    write_report(config, "progress_report", {
        "output": str(OUT), "stages": pipeline_stages(ctx), "lineage": {c: LINEAGE[c] for c in ctx.features["feature_columns"]},
        "baselines": rows, "tokens": {"prediction": ctx.tokens["tokens_per_prediction_snapshot"]["t2"], "sequence": ctx.sequence,
                                      "rows": m["tokens"]["counts"]["tokens"]},
        "validation": {"checks": ctx.leakage_checks, "leakage_tests_passed": ctx.leakage_tests},
    })
    print(OUT)


if __name__ == "__main__":
    main()
