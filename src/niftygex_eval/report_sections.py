"""Report sections shared by the progress and final Word reports: data sources, pipeline, features, token history,
target, leakage controls, data gaps and the feature dictionary. Every number comes from manifests and reports."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from niftygex.features.lineage import LINEAGE, SOURCES, TOKEN_SOURCES
from niftygex_eval.docx_report import bullets, figure, load, pipeline_diagram, table


@dataclass
class DataContext:
    m: dict[str, Any]
    raw: dict[str, Any]
    cleaning: dict[str, Any]
    features: dict[str, Any]
    tokens: dict[str, Any]
    targets: dict[str, Any]
    validation: dict[str, Any]
    missing: dict[str, Any]
    forward: dict[str, Any]
    figures: Path
    figure_no: int = field(default=0)

    def caption(self, text: str) -> str:
        self.figure_no += 1
        return f"Figure {self.figure_no}. {text}"

    @property
    def window(self) -> dict:
        return self.cleaning["date_range"]

    @property
    def leakage_checks(self) -> int:
        return len(self.validation["timestamp_and_leakage_checks"])

    @property
    def leakage_tests(self) -> int:
        return self.validation["leakage_test_run"]["counts"].get("passed", 0)

    @property
    def sequence(self) -> dict:
        return self.tokens["tokens_per_sequence_snapshot"]["t2"]


def load_data_context(config: dict[str, Any]) -> DataContext:
    paths = config["paths"]
    reports, manifests = Path(paths["reports_root"]), Path(paths["manifests_root"])
    if not (reports / "data_overview.json").exists():
        raise SystemExit("Run scripts/report_data.py first (data_overview.json missing)")
    m = {s: load(manifests / f"{s}.json") for s in ("clean", "forward", "greeks", "gex", "features", "tokens", "targets", "validation")}
    return DataContext(m, load(manifests / "raw_import_manifest.json"), load(reports / "cleaning_report.json"),
                       load(reports / "features_report.json"), load(reports / "tokens_report.json"), load(reports / "targets_t2_report.json"),
                       load(reports / "validation_report.json"), load(reports / "missing_data.json"), load(reports / "forward_report.json"),
                       reports / "figures")


def pipeline_stages(ctx: DataContext) -> list[tuple[str, str]]:
    m, options = ctx.m, ctx.cleaning["row_counts"]["options"]
    return [
        ("Raw import", f"{ctx.raw['file_count']} files\n{ctx.raw['total_size_bytes'] / 1e6:,.0f} MB"),
        ("Clean", f"{options['after_dedup']:,}\noption bars"),
        ("Forward + IV", f"{m['forward']['counts']['iv']:,}\nIV rows"),
        ("Greeks", f"{m['greeks']['counts']['greeks_rows_out']:,}\neligible legs"),
        ("GEX + levels", f"{m['gex']['counts']['snapshots']:,} snapshots\n{m['gex']['counts']['strike_rows']:,} strike rows"),
        ("Features", f"{m['features']['counts']['rows']:,} × {m['features']['counts']['feature_columns']}"),
        ("Tokens", f"{m['tokens']['counts']['tokens']:,}\ntoken rows"),
        ("Target", f"{m['targets']['counts']['rows']:,} rows\n{m['targets']['counts']['labeled']:,} labeled"),
    ]


def data_sections(doc, ctx: DataContext, first: int, appendix: str = "Appendix A") -> int:
    """Sections first..first+5: sources, pipeline, features, tokens, target, leakage. Returns the next section number."""
    n_sources, n_pipeline, n_features, n_tokens, n_target, n_leakage = range(first, first + 6)
    m, window, options = ctx.m, ctx.window, ctx.cleaning["row_counts"]["options"]

    doc.add_heading(f"{n_sources}. Data sources", level=1)
    doc.add_paragraph("A private archive of NSE market data. The pipeline adapts a prototype co-developed with Sunil Kandakur "
                      "(see PROVENANCE.md). Raw files are copied once with checksums and never modified.")
    table(doc, ["Dataset", "Files", "Size (MB)", "Coverage", "Granularity", "Used for"],
          [[SOURCES[k][0], v["file_count"], f"{v['total_size_bytes'] / 1e6:,.1f}", f"{v['date_min']} to {v['date_max']}", SOURCES[k][1], SOURCES[k][2]]
           for k, v in ctx.raw["sources"].items()] + [["Regime calendar", "config", "—", "lot sizes, expiry weekdays, rule changes",
                                                       "event dates", "Lot size, regime flag"]],
          [1.3, 0.5, 0.65, 1.3, 0.9, 1.85], size=8.5)
    doc.add_paragraph(f"Project window: {window['start']} to {window['end']}. Timestamps are IST; bars are start-labeled "
                      "(a bar labeled t covers t to t + 5 minutes and is usable at t + 5 minutes).")

    doc.add_heading(f"{n_pipeline}. Pipeline", level=1)
    figure(doc, pipeline_diagram(pipeline_stages(ctx), ctx.figures), ctx.caption("Stages and their main outputs (counts from the stage manifests)."))
    table(doc, ["Stage", "What it does", "Output"], [
        ["Clean", f"Session filter (09:15–15:25), holidays, deduplication ({options['exact_duplicate_rows_removed'] + options['conflicting_extra_rows_removed']:,} "
                  "duplicate rows removed), one futures contract per day, spot relabel for the end-labeled window, missing-OI and missing-spot flags",
         f"{options['raw_rows']:,} → {options['after_dedup']:,} option bars"],
        ["Forward + IV", f"Put–call parity forward per snapshot; Black-76 IV from the out-of-the-money leg ({ctx.forward['iv_ok_rate']:.1%} of legs solve)",
         f"{m['forward']['counts']['iv']:,} IV rows"],
        ["Greeks", "Black-76 gamma and delta for eligible legs (traded, solved, positive time to expiry)", f"{m['greeks']['counts']['greeks_rows_out']:,} legs"],
        ["GEX + levels", "GEX = gamma × OI × spot² × 0.01 under naive, absolute and participant-inferred signs; call/put walls, max pain, "
                         "max-GEX strike, flip level", f"{m['gex']['counts']['snapshots']:,} snapshots"],
        ["Features", f"40 aggregate features per 15:15 snapshot (Section {n_features})", f"{m['features']['counts']['rows']:,} × {m['features']['counts']['feature_columns']}"],
        ["Tokens", f"Per-contract rows of the day's nearest expiry at 72 snapshots per day (Section {n_tokens})", f"{m['tokens']['counts']['tokens']:,} rows"],
        ["Target", "Next-day futures RV over its trailing 20-day mean; stored apart from the features", f"{m['targets']['counts']['rows']:,} rows"],
        ["Validation", "Parity residuals, IV checks, ATM IV vs VIX, lot sizes, timestamp and leakage checks", f"{ctx.leakage_checks} checks, all pass"],
    ], [1.0, 4.0, 1.5], size=8.5)

    doc.add_heading(f"{n_features}. Features and their sources", level=1)
    doc.add_paragraph(f"{m['features']['counts']['rows']:,} snapshots (one per trading day at 15:15) × "
                      f"{m['features']['counts']['feature_columns']} features. Every feature uses only data available by the "
                      f"snapshot; daily sources enter from 09:15 on the next trading day. The full per-feature list is in {appendix}.")
    groups: dict[str, list[str]] = {}
    for c in ctx.features["feature_columns"]:
        groups.setdefault(LINEAGE[c][0], []).append(c)
    table(doc, ["Group", "Features", "Source", "Available at the snapshot"],
          [[g, str(len(cols)), ", ".join(dict.fromkeys(part.strip() for c in cols for part in LINEAGE[c][1].replace(" − ", " + ").split(" + "))),
            "; ".join(dict.fromkeys(LINEAGE[c][2] for c in cols))]
           for g, cols in groups.items()], [1.3, 0.7, 2.2, 2.3], size=8.5)
    figure(doc, ctx.figures / "data_feature_coverage.png", ctx.caption("Share of snapshots with a value, per feature and year. The 2022 "
           "gaps are the source's missing option OI (Jan–Apr 2022); the 2026 VIX gap is the daily VIX file ending in April."), width=4.4)
    figure(doc, ctx.figures / "data_key_features.png", ctx.caption("Four representative features over the sample."))

    pred, seq = ctx.tokens["tokens_per_prediction_snapshot"]["t2"], ctx.sequence
    doc.add_heading(f"{n_tokens}. Token history (the deep-learning input)", level=1)
    doc.add_paragraph("Each token is one liquid option contract of the day's nearest included expiry (same-day expiries excluded). "
                      f"Tokens exist at every 5-minute snapshot from 09:20 to the 15:15 prediction snapshot ({ctx.tokens['sequence_snapshots_per_day']} "
                      "per day), so the GRU's window length is a model setting: the last 12 snapshots (one hour) by default, and 24 or 72 "
                      "for the window ablation. Each day uses one expiry, so a history never switches contracts.")
    table(doc, ["", "Snapshots", "With tokens", "Median", "5th pct", "95th pct", "Max"], [
        ["Prediction snapshot (15:15)", f"{pred['snapshots']:,}", f"{pred['with_tokens']:,}", f"{pred['median']:.0f}", f"{pred['p5']:.0f}", f"{pred['p95']:.0f}", pred["max"]],
        ["All sequence snapshots", f"{seq['snapshots']:,}", f"{seq['with_tokens']:,}", f"{seq['median']:.0f}", f"{seq['p5']:.0f}", f"{seq['p95']:.0f}", seq["max"]],
    ], [1.9, 0.85, 0.85, 0.65, 0.65, 0.7, 0.5], size=8.5)
    doc.add_paragraph(f"{ctx.tokens['days_with_full_sequence']:,} days have tokens at all {ctx.tokens['sequence_snapshots_per_day']} snapshots; the "
                      "rest are days without option data, a truncated session, or days whose bars end early.")
    table(doc, ["Token field", "Source"], [[a, b] for a, b in TOKEN_SOURCES], [2.6, 3.9], size=8.5)
    figure(doc, ctx.figures / "data_tokens_by_time.png", ctx.caption("Contracts per snapshot through the day."))
    figure(doc, ctx.figures / "data_token_day.png", ctx.caption("One day of token history (12 June 2026): strike GEX by strike and time. "
           "This strike × time picture is what the sequence model reads."))

    doc.add_heading(f"{n_target}. Target", level=1)
    reasons = ctx.targets["reason_counts"]
    doc.add_paragraph(f"{ctx.targets['rows']:,} target rows; {reasons.get('ok', 0):,} are usable for tercile labels, and the regression "
                      "outcome is available on every row with a next-day RV and a trailing mean. The target becomes known at 15:30 on "
                      "the next trading day, always after its snapshot.")
    table(doc, ["Reason", "Rows"], [[k, f"{v:,}"] for k, v in sorted(reasons.items())], [3.0, 1.0], size=8.5)

    iv_vix, parity = ctx.validation["atm_iv_vs_india_vix"]["overall"], ctx.validation["parity_residuals"]["overall"]
    doc.add_heading(f"{n_leakage}. Leakage controls and validation", level=1)
    bullets(doc, [
        "A feature at snapshot T uses only rows with available_at ≤ T; checks compare availability times, never bar labels.",
        "Daily sources (VIX, participant OI, end-of-day OI) enter from 09:15 on the next trading day and are never forward-filled.",
        "Targets are stored apart from features and become known after the snapshot.",
        "The walk-forward harness splits by expiry cycle with a one-cycle embargo and purges training rows whose label resolves "
        "inside the test window; rows are never shuffled.",
        f"Validation: {ctx.leakage_checks} timestamp checks and {ctx.leakage_tests} leakage tests pass. Parity residual median {parity['p50']:.2f} index points; "
        f"ATM IV tracks India VIX with correlation {iv_vix['correlation']:.2f} over {iv_vix['days']:,} days.",
    ])
    return first + 6


def data_gaps_section(doc, ctx: DataContext, number: int) -> None:
    missing = ctx.missing
    doc.add_heading(f"{number}. Known data gaps", level=1)
    table(doc, ["Gap", "Extent", "Handling"], [
        ["Option OI recorded as 0", f"{missing['missing_option_oi']['days']} days (2019-07-30; 2022-01-03 to 2022-04-11)", "OI set to missing; GEX and OI features missing"],
        ["Spot missing", f"{len(missing['missing_spot']['days'])} days", "Not imputed; excluded where spot is needed"],
        ["Option / futures days missing", f"{missing['missing_options_days']} option days, {missing['missing_futures_days']} futures day", "Flagged"],
        ["Daily India VIX", f"ends {missing['daily_vix']['last_available']}", "Missing, never forward-filled"],
        ["Option history per expiry", f"at most {missing['expiry_history_depth_days']['max']} calendar days (about 3 weeks)", "GEX horizon set to 21 days"],
    ], [1.8, 2.6, 2.1], size=8.5)


def feature_appendix(doc, ctx: DataContext, title: str = "Appendix A. Feature dictionary") -> None:
    doc.add_heading(title, level=1)
    share = ctx.features["missing_share"]
    table(doc, ["Feature", "Group", "Source", "Coverage"],
          [[c, LINEAGE[c][0], LINEAGE[c][1], f"{1 - share[c]:.1%}"] for c in ctx.features["feature_columns"]], [2.5, 1.3, 2.0, 0.7], size=8)
