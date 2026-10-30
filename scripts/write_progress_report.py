"""Midterm progress report (Word): pipeline, sources, features, token history, leakage checks and first results.

Every number comes from the stage manifests, the reports in outputs/reports/ and the results store; the figures are
the ones written by scripts/report_data.py and scripts/report_forecast.py, plus a pipeline diagram drawn here.
Run those two report scripts first. Output: docs/deliverables/NiftyGEX_Midterm_Progress_Report.docx (gitignored)
and outputs/reports/progress_report.json (the numbers used).
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from matplotlib.patches import FancyBboxPatch

from niftygex.config import REPO_ROOT, load_config
from niftygex.features.lineage import LINEAGE, SOURCES, TOKEN_SOURCES
from niftygex.io.parquet import write_report
from niftygex_eval.figures import SERIES, SURFACE, TEXT, TEXT_2, save

OUT = REPO_ROOT / "docs" / "deliverables" / "NiftyGEX_Midterm_Progress_Report.docx"
HEADER_FILL = "E8EEF6"
INK = RGBColor(0x0B, 0x0B, 0x0B)
MUTED = RGBColor(0x52, 0x51, 0x4E)

LABELS = {"historical_mean": "Historical mean", "persistence": "Persistence", "har": "HAR-RV", "har_gex": "HAR + GEX", "lightgbm": "Tuned LightGBM"}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def shade(cell, fill: str) -> None:
    pr = cell._tc.get_or_add_tcPr()
    element = OxmlElement("w:shd")
    element.set(qn("w:val"), "clear")
    element.set(qn("w:color"), "auto")
    element.set(qn("w:fill"), fill)
    pr.append(element)


def table(doc, header: list[str], rows: list[list], widths: list[float], size: float = 9) -> None:
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, text in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(text)
        run.bold, run.font.size = True, Pt(size)
        shade(cell, HEADER_FILL)
    for row in rows:
        cells = t.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(value))
            run.font.size = Pt(size)
    for row in t.rows:
        for i, width in enumerate(widths):
            row.cells[i].width = Inches(width)
    doc.add_paragraph()


def bullets(doc, items: list[str]) -> None:
    for item in items:
        doc.add_paragraph(item, style="List Bullet")


def figure(doc, path: Path, caption: str, width: float = 6.3) -> None:
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph()
    run = p.add_run(caption)
    run.italic, run.font.size, run.font.color.rgb = True, Pt(9), MUTED
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER


def pipeline_diagram(stages: list[tuple[str, str]], directory: Path) -> Path:
    """Left-to-right stage flow with each stage's main output count."""
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.set_xlim(0, len(stages) * 2.0)
    ax.set_ylim(0, 3)
    ax.axis("off")
    for i, (name, detail) in enumerate(stages):
        x = i * 2.0 + 0.1
        ax.add_patch(FancyBboxPatch((x, 0.9), 1.7, 1.3, boxstyle="round,pad=0.02,rounding_size=0.12", linewidth=1.2,
                                    edgecolor=SERIES[0], facecolor=SURFACE))
        ax.text(x + 0.85, 1.85, name, ha="center", va="center", fontsize=9.5, color=TEXT, weight="bold", wrap=True)
        ax.text(x + 0.85, 1.3, detail, ha="center", va="center", fontsize=7.8, color=TEXT_2, linespacing=1.3)
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + 1.95, 1.55), xytext=(x + 1.75, 1.55), arrowprops=dict(arrowstyle="-|>", color=TEXT_2, lw=1.2))
    ax.set_title("Data pipeline: Parquet in, Parquet out; each stage refuses stale upstream configuration", fontsize=10.5, loc="left")
    return save(fig, directory, "report_pipeline")


def main() -> None:
    config = load_config()
    paths = config["paths"]
    reports, manifests = Path(paths["reports_root"]), Path(paths["manifests_root"])
    figures = reports / "figures"
    for needed in ("data_overview.json", "forecast_baselines.json"):
        if not (reports / needed).exists():
            raise SystemExit(f"Run scripts/report_data.py and scripts/report_forecast.py first ({needed} missing)")
    m = {s: load(manifests / f"{s}.json") for s in ("clean", "forward", "greeks", "gex", "features", "tokens", "targets", "validation")}
    raw = load(manifests / "raw_import_manifest.json")
    cleaning, features_r, tokens_r = load(reports / "cleaning_report.json"), load(reports / "features_report.json"), load(reports / "tokens_report.json")
    targets_r, validation, missing = load(reports / "targets_t2_report.json"), load(reports / "validation_report.json"), load(reports / "missing_data.json")
    forward_r = load(reports / "forward_report.json")
    forecast = load(Path(config["paths"]["outputs_root"]) / "results" / "forecast_comparison.json")
    options = cleaning["row_counts"]["options"]

    stages = [
        ("Raw import", f"{raw['file_count']} files\n{raw['total_size_bytes'] / 1e6:,.0f} MB"),
        ("Clean", f"{options['after_dedup']:,}\noption bars"),
        ("Forward + IV", f"{m['forward']['counts']['iv']:,}\nIV rows"),
        ("Greeks", f"{m['greeks']['counts']['greeks_rows_out']:,}\neligible legs"),
        ("GEX + levels", f"{m['gex']['counts']['snapshots']:,} snapshots\n{m['gex']['counts']['strike_rows']:,} strike rows"),
        ("Features", f"{m['features']['counts']['rows']:,} × {m['features']['counts']['feature_columns']}"),
        ("Tokens", f"{m['tokens']['counts']['tokens']:,}\ntoken rows"),
        ("Target", f"{m['targets']['counts']['rows']:,} rows\n{m['targets']['counts']['labeled']:,} labeled"),
    ]
    diagram = pipeline_diagram(stages, figures)
    window = cleaning["date_range"]
    leakage = validation["timestamp_and_leakage_checks"]
    tests = validation["leakage_test_run"]["counts"].get("passed", 0)
    rows = forecast["comparison_table"]
    by_model = {r["model"]: r for r in rows}
    har, best = by_model["har"], max((r for r in rows if r["ic_ci95"]), key=lambda r: r["ic_mean_per_walk"])
    pred, seq = tokens_r["tokens_per_prediction_snapshot"]["t2"], tokens_r["tokens_per_sequence_snapshot"]["t2"]
    iv_vix = validation["atm_iv_vs_india_vix"]["overall"]
    parity = validation["parity_residuals"]["overall"]

    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(section, side, Inches(1))
    normal = doc.styles["Normal"]
    normal.font.name, normal.font.size = "Calibri", Pt(11)
    for level, size in ((1, 15), (2, 12.5)):
        style = doc.styles[f"Heading {level}"]
        style.font.name, style.font.size, style.font.color.rgb = "Calibri", Pt(size), RGBColor(0x1C, 0x5C, 0xAB)

    title = doc.add_paragraph()
    run = title.add_run("Long Attention, Short Gamma")
    run.bold, run.font.size = True, Pt(22)
    sub = doc.add_paragraph()
    run = sub.add_run("Midterm progress report: data pipeline, token history and first baselines")
    run.font.size, run.font.color.rgb = Pt(13), MUTED
    meta = doc.add_paragraph()
    run = meta.add_run(f"Bhalchandra Shinde and Asad Mulani · CS 7150 Deep Learning, Northeastern University · {date.today():%d %B %Y}")
    run.font.size, run.font.color.rgb = Pt(10), MUTED

    doc.add_heading("1. Summary", level=1)
    bullets(doc, [
        f"The data pipeline is complete and validated. It turns {raw['file_count']} raw files ({window['start']} to {window['end']}, "
        f"{m['clean']['counts']['trading_days_observed']:,} trading days) into features, a token history and a target. All "
        f"{len(leakage)} timestamp checks and {tests} leakage tests pass.",
        f"The classical model inputs exist: {m['features']['counts']['rows']:,} daily snapshots at 15:15 with "
        f"{m['features']['counts']['feature_columns']} features, each traced to its source in Section 4.",
        f"The deep-learning input exists: a token history of {m['tokens']['counts']['tokens']:,} rows, one per liquid option contract "
        f"at every 5-minute snapshot from 09:20 to 15:15 ({tokens_r['sequence_snapshots_per_day']} per day, median {seq['median']:.0f} "
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

    doc.add_heading("3. Data sources", level=1)
    doc.add_paragraph("A private archive of NSE market data. The pipeline adapts a prototype co-developed with Sunil Kandakur "
                      "(see PROVENANCE.md). Raw files are copied once with checksums and never modified.")
    table(doc, ["Dataset", "Files", "Size (MB)", "Coverage", "Granularity", "Used for"],
          [[SOURCES[k][0], v["file_count"], f"{v['total_size_bytes'] / 1e6:,.1f}", f"{v['date_min']} to {v['date_max']}", SOURCES[k][1], SOURCES[k][2]]
           for k, v in raw["sources"].items()] + [["Regime calendar", "config", "—", "lot sizes, expiry weekdays, rule changes",
                                                   "event dates", "Lot size, regime flag"]],
          [1.3, 0.5, 0.65, 1.3, 0.9, 1.85], size=8.5)
    doc.add_paragraph(f"Project window: {window['start']} to {window['end']}. Timestamps are IST; bars are start-labeled "
                      "(a bar labeled t covers t to t + 5 minutes and is usable at t + 5 minutes).")

    doc.add_heading("4. Pipeline", level=1)
    figure(doc, diagram, "Figure 1. Stages and their main outputs (counts from the stage manifests).")
    table(doc, ["Stage", "What it does", "Output"], [
        ["Clean", f"Session filter (09:15–15:25), holidays, deduplication ({options['exact_duplicate_rows_removed'] + options['conflicting_extra_rows_removed']:,} "
                  "duplicate rows removed), one futures contract per day, spot relabel for the end-labeled window, missing-OI and missing-spot flags",
         f"{options['raw_rows']:,} → {options['after_dedup']:,} option bars"],
        ["Forward + IV", f"Put–call parity forward per snapshot; Black-76 IV from the out-of-the-money leg ({forward_r['iv_ok_rate']:.1%} of legs solve)",
         f"{m['forward']['counts']['iv']:,} IV rows"],
        ["Greeks", "Black-76 gamma and delta for eligible legs (traded, solved, positive time to expiry)", f"{m['greeks']['counts']['greeks_rows_out']:,} legs"],
        ["GEX + levels", "GEX = gamma × OI × spot² × 0.01 under naive, absolute and participant-inferred signs; call/put walls, max pain, "
                         "max-GEX strike, flip level", f"{m['gex']['counts']['snapshots']:,} snapshots"],
        ["Features", "40 aggregate features per 15:15 snapshot (Section 5)", f"{m['features']['counts']['rows']:,} × {m['features']['counts']['feature_columns']}"],
        ["Tokens", "Per-contract rows of the day's nearest expiry at 72 snapshots per day (Section 6)", f"{m['tokens']['counts']['tokens']:,} rows"],
        ["Target", "Next-day futures RV over its trailing 20-day mean; stored apart from the features", f"{m['targets']['counts']['rows']:,} rows"],
        ["Validation", "Parity residuals, IV checks, ATM IV vs VIX, lot sizes, timestamp and leakage checks", f"{len(leakage)} checks, all pass"],
    ], [1.0, 4.0, 1.5], size=8.5)

    doc.add_heading("5. Features and their sources", level=1)
    doc.add_paragraph(f"{m['features']['counts']['rows']:,} snapshots (one per trading day at 15:15) × "
                      f"{m['features']['counts']['feature_columns']} features. Every feature uses only data available by the "
                      "snapshot; daily sources enter from 09:15 on the next trading day. The full per-feature list is in Appendix A.")
    groups: dict[str, list[str]] = {}
    for c in features_r["feature_columns"]:
        groups.setdefault(LINEAGE[c][0], []).append(c)
    table(doc, ["Group", "Features", "Source", "Available at the snapshot"],
          [[g, str(len(cols)), ", ".join(dict.fromkeys(part.strip() for c in cols for part in LINEAGE[c][1].replace(" − ", " + ").split(" + "))),
            "; ".join(dict.fromkeys(LINEAGE[c][2] for c in cols))]
           for g, cols in groups.items()], [1.3, 0.7, 2.2, 2.3], size=8.5)
    figure(doc, figures / "data_feature_coverage.png", "Figure 2. Share of snapshots with a value, per feature and year. The 2022 "
           "gaps are the source's missing option OI (Jan–Apr 2022); the 2026 VIX gap is the daily VIX file ending in April.", width=4.4)
    figure(doc, figures / "data_key_features.png", "Figure 3. Four representative features over the sample.")

    doc.add_heading("6. Token history (the deep-learning input)", level=1)
    doc.add_paragraph("Each token is one liquid option contract of the day's nearest included expiry (same-day expiries excluded). "
                      f"Tokens exist at every 5-minute snapshot from 09:20 to the 15:15 prediction snapshot ({tokens_r['sequence_snapshots_per_day']} "
                      "per day), so the GRU's window length is a model setting: the last 12 snapshots (one hour) by default, and 24 or 72 "
                      "for the window ablation. Each day uses one expiry, so a history never switches contracts.")
    table(doc, ["", "Snapshots", "With tokens", "Median", "5th pct", "95th pct", "Max"], [
        ["Prediction snapshot (15:15)", f"{pred['snapshots']:,}", f"{pred['with_tokens']:,}", f"{pred['median']:.0f}", f"{pred['p5']:.0f}", f"{pred['p95']:.0f}", pred["max"]],
        ["All sequence snapshots", f"{seq['snapshots']:,}", f"{seq['with_tokens']:,}", f"{seq['median']:.0f}", f"{seq['p5']:.0f}", f"{seq['p95']:.0f}", seq["max"]],
    ], [1.9, 0.85, 0.85, 0.65, 0.65, 0.7, 0.5], size=8.5)
    doc.add_paragraph(f"{tokens_r['days_with_full_sequence']:,} days have tokens at all {tokens_r['sequence_snapshots_per_day']} snapshots; the "
                      "rest are days without option data, a truncated session, or days whose bars end early.")
    table(doc, ["Token field", "Source"], [[a, b] for a, b in TOKEN_SOURCES], [2.6, 3.9], size=8.5)
    figure(doc, figures / "data_tokens_by_time.png", "Figure 4. Contracts per snapshot through the day.")
    figure(doc, figures / "data_token_day.png", "Figure 5. One day of token history (12 June 2026): strike GEX by strike and time. "
           "This strike × time picture is what the sequence model reads.")

    doc.add_heading("7. Target", level=1)
    reasons = targets_r["reason_counts"]
    doc.add_paragraph(f"{targets_r['rows']:,} target rows; {reasons.get('ok', 0):,} are usable for tercile labels, and the regression "
                      "outcome is available on every row with a next-day RV and a trailing mean. The target becomes known at 15:30 on "
                      "the next trading day, always after its snapshot.")
    table(doc, ["Reason", "Rows"], [[k, f"{v:,}"] for k, v in sorted(reasons.items())], [3.0, 1.0], size=8.5)

    doc.add_heading("8. Leakage controls and validation", level=1)
    bullets(doc, [
        "A feature at snapshot T uses only rows with available_at ≤ T; checks compare availability times, never bar labels.",
        "Daily sources (VIX, participant OI, end-of-day OI) enter from 09:15 on the next trading day and are never forward-filled.",
        "Targets are stored apart from features and become known after the snapshot.",
        "The walk-forward harness splits by expiry cycle with a one-cycle embargo and purges training rows whose label resolves "
        "inside the test window; rows are never shuffled.",
        f"Validation: {len(leakage)} timestamp checks and {tests} leakage tests pass. Parity residual median {parity['p50']:.2f} index points; "
        f"ATM IV tracks India VIX with correlation {iv_vix['correlation']:.2f} over {iv_vix['days']:,} days.",
    ])

    doc.add_heading("9. Evaluation harness and first results", level=1)
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
    figure(doc, figures / "forecast_ladder_ic.png", "Figure 6. Mean IC per test window with 95% intervals.", width=5.6)
    figure(doc, figures / "forecast_ic_by_walk.png", "Figure 7. IC in each test window.")
    doc.add_paragraph("Reading so far: HAR-RV is the bar to beat. Adding GEX (HAR + GEX) or all summary features (tuned LightGBM) "
                      "does not beat it significantly. These comparisons are exploratory; the pre-registered tests involve the "
                      "neural models.")

    doc.add_heading("10. Known data gaps", level=1)
    table(doc, ["Gap", "Extent", "Handling"], [
        ["Option OI recorded as 0", f"{missing['missing_option_oi']['days']} days (2019-07-30; 2022-01-03 to 2022-04-11)", "OI set to missing; GEX and OI features missing"],
        ["Spot missing", f"{len(missing['missing_spot']['days'])} days", "Not imputed; excluded where spot is needed"],
        ["Option / futures days missing", f"{missing['missing_options_days']} option days, {missing['missing_futures_days']} futures day", "Flagged"],
        ["Daily India VIX", f"ends {missing['daily_vix']['last_available']}", "Missing, never forward-filled"],
        ["Option history per expiry", f"at most {missing['expiry_history_depth_days']['max']} calendar days (about 3 weeks)", "GEX horizon set to 21 days"],
    ], [1.8, 2.6, 2.1], size=8.5)

    doc.add_heading("11. Next steps", level=1)
    table(doc, ["When", "Work"], [
        ["Now", "Volatility-state vector for the model head (IV, RV, calendar and total GEX)"],
        ["Early November", "Neural models behind the same harness: DeepSets, single-snapshot self-attention, attention + GRU "
                           "(2 layers × 4 heads, 64-d, about 100k parameters); sanity checks (overfit a tiny batch, permutation and padding invariance)"],
        ["November", "Pre-registered RQ1–RQ3 tests, window and field ablations, attention maps"],
        ["Dec 3 / Dec 10, Dec 17", "Final presentation, report and reproducible code"],
    ], [1.5, 5.0], size=9)

    doc.add_heading("Appendix A. Feature dictionary", level=1)
    share = features_r["missing_share"]
    table(doc, ["Feature", "Group", "Source", "Coverage"],
          [[c, LINEAGE[c][0], LINEAGE[c][1], f"{1 - share[c]:.1%}"] for c in features_r["feature_columns"]], [2.5, 1.3, 2.0, 0.7], size=8)
    footer = section.footer.paragraphs[0]
    run = footer.add_run(f"NiftyGEX · generated by scripts/write_progress_report.py · data config {m['features']['config_hash'][:12]}")
    run.font.size, run.font.color.rgb = Pt(8), MUTED

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    write_report(config, "progress_report", {
        "output": str(OUT.relative_to(REPO_ROOT)), "stages": stages, "lineage": {c: LINEAGE[c] for c in features_r["feature_columns"]},
        "baselines": rows, "tokens": {"prediction": pred, "sequence": seq, "rows": m["tokens"]["counts"]["tokens"]},
        "validation": {"checks": len(leakage), "leakage_tests_passed": tests},
    })
    print(OUT)


if __name__ == "__main__":
    main()
