"""Final report (Word): data, pipeline, models, evaluation design, the full ladder, RQ1-RQ3, ablations, attention maps.

Every number comes from the manifests, the reports in outputs/reports/ and the results store; verdict sentences are
generated from the pre-registered test results. Run first: scripts/report_data.py, run_forecast.py, run_neural.py,
run_rq_tests.py, report_results.py and report_attention.py (ablation and attention-map sections appear once their outputs exist).
Output: docs/deliverables/NiftyGEX_Final_Report.docx (gitignored). --tag dev uses the development-run files and
writes outputs/reports/NiftyGEX_Final_Report_dev.docx instead.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from niftygex.config import REPO_ROOT
from niftygex.io.parquet import write_report
from niftygex_eval.config import load_eval_config
from niftygex_eval.rq1_summary import METRICS_PLAIN, ladder_improvements, qlike_explained, rq1_summary
from niftygex_eval.docx_report import bullets, figure, footer, load, new_document, table
from niftygex_eval.report_sections import data_gaps_section, data_sections, feature_appendix, load_data_context

LABELS = {"historical_mean": "Historical mean", "persistence": "Persistence", "har": "HAR-RV", "har_gex": "HAR + GEX",
          "lightgbm": "Tuned LightGBM", "deepsets": "DeepSets", "attention": "Self-attention", "attention_gru": "Attention + GRU"}
RQ2_NAMES = {"attention_vs_deepsets (interactions)": "Self-attention vs DeepSets (interactions)",
             "attention_gru_vs_attention (history)": "Attention + GRU vs self-attention (history)"}
STEPS = [("0", "historical_mean", "Training-window mean", "—"), ("0", "persistence", "Yesterday's RV relative to its 20-day level", "2 RV terms"),
         ("1", "har", "HAR-RV: OLS on daily, weekly and monthly RV plus today's RV so far", "4 RV terms"),
         ("2", "har_gex", "HAR + GEX: ridge adding normalized GEX (3 sign conventions) and the inferred sign", "HAR + 4 GEX terms"),
         ("3", "lightgbm", "Tuned LightGBM on the summary features", "~40 summary features"),
         ("4", "deepsets", "Per-contract MLP, masked mean pooling: reads every contract, no interactions", "tokens (15:15) + state"),
         ("5", "attention", "Self-attention across contracts, attention pooling: one snapshot", "tokens (15:15) + state"),
         ("6", "attention_gru", "Self-attention per snapshot, then a GRU over the last hour: our model", "tokens (12 snapshots) + state")]


def fmt(value, spec: str = ".3f") -> str:
    return "—" if value is None else format(value, spec)


def interval(ic: dict) -> str:
    ci = ic.get("ci95")
    return f"{ci[0]:+.3f} to {ci[1]:+.3f}" if ci else "—"


def parameter_counts(neural: dict) -> dict[str, int]:
    from niftygex_eval.nn.models import build_model

    shape = (len(neural["token_features"]), len(neural["state_columns"]))
    return {name: sum(p.numel() for p in build_model(name, *shape, dict(neural["model"])).parameters()) for name in neural["models"]}


def comparison_sentence(name: str, c: dict, alpha: float) -> str:
    ic = c["ic"]
    if ic.get("p_two_sided") is None:
        return f"{name}: not enough walks to test."
    verdict = ("significantly better" if ic["mean_diff"] > 0 else "significantly worse") if ic["p_two_sided"] < alpha else "not significantly different"
    return (f"{name}: IC difference {ic['mean_diff']:+.3f} (95% interval {interval(ic)}, p {ic['p_two_sided']:.3f}, wins {ic['a_wins']} of "
            f"{ic['walks']} walks), {verdict}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="main")
    tag = parser.parse_args().tag
    suffix = "" if tag == "main" else f"_{tag}"
    cfg = load_eval_config()
    config, evaluation, root = cfg["data_config"], cfg["eval"], Path(cfg["results_root"])
    reports = Path(config["paths"]["reports_root"])
    needed = [root / "forecast_comparison.json", root / f"neural_comparison{suffix}.json", root / f"rq_tests{suffix}.json", reports / f"results{suffix}.json"]
    if missing := [str(p.name) for p in needed if not p.exists()]:
        raise SystemExit(f"Missing {missing}: run run_forecast.py, run_neural.py, run_rq_tests.py and report_results.py first")
    ctx = load_data_context(config)
    forecast, neural_run, tests, results = (load(p) for p in needed)
    neural = evaluation["neural"]
    ladder = {r["model"]: r for r in results["ladder"]}
    alpha, rq1, rq2, rq3 = tests["alpha"], tests["RQ1"], tests["RQ2"], tests["RQ3"]
    params = parameter_counts(neural)
    model_rows = {r["model"]: r for r in neural_run["comparison_table"]}
    ours, har = ladder.get("attention_gru", {}), ladder.get("har", {})
    best_classical = max((r for k, r in ladder.items() if k in ("persistence", "har", "har_gex", "lightgbm") and r.get("ic_ci95")),
                         key=lambda r: r["ic_mean_per_walk"])
    walks = neural_run["walks"]

    doc = new_document("Long Attention, Short Gamma", "Final report: can attention over the full NIFTY option chain forecast volatility better than GEX?")
    if tag != "main":
        doc.add_paragraph(f"DEVELOPMENT RUN ({walks} walks, not a result): this document checks the report layout only.")

    summary = rq1_summary(rq1["comparisons"], ladder, alpha, rq1["beats_every_rung"])
    doc.add_heading("Abstract", level=1)
    doc.add_paragraph(
        "Traders compress the NIFTY option chain into one number, gamma exposure (GEX). We built an auditable point-in-time pipeline "
        f"over {ctx.m['clean']['counts']['trading_days_observed']:,} trading days ({ctx.window['start']} to {ctx.window['end']}), a token history "
        f"of every liquid contract at 72 snapshots per day ({ctx.m['tokens']['counts']['tokens']:,} rows), and compared eight models on the same "
        f"{walks} walk-forward test windows for next-day volatility relative to its recent normal. Our attention + GRU model reaches a mean "
        f"per-walk IC of {fmt(ours.get('ic_mean_per_walk'))}, against {fmt(har.get('ic_mean_per_walk'))} for HAR-RV and "
        f"{fmt(best_classical['ic_mean_per_walk'])} for the best classical model ({LABELS[best_classical['model']]}). "
        f"RQ1: {summary['headline']} {summary['loss']} {summary['rule']}")

    doc.add_heading("1. Introduction and research questions", level=1)
    doc.add_paragraph("Option dealers hedge, and a common belief is that their aggregate gamma decides whether markets are calm (dealers long "
                      "gamma) or jumpy (dealers short gamma). GEX squeezes about 70 contracts into one score. We ask whether a model that "
                      "reads every contract, with self-attention across contracts and a GRU over the recent snapshots, forecasts next-day "
                      "volatility better than GEX-based, classical and tree-based models, and if so, why.")
    table(doc, ["Question", "Test (pre-registered in configs/eval.yaml)"], [
        ["RQ1: does reading every contract predict better?", "Attention + GRU vs every classical rung: mean per-walk IC gain, 95% interval, paired t; a win needs every comparison to reject"],
        ["RQ2: if it wins, why?", "Self-attention vs DeepSets (interactions); attention + GRU vs self-attention (history); Holm within the family"],
        ["RQ3: does it survive rule changes?", "IC gain over tuned LightGBM, walks after vs before the November 2024 rule change (Welch t)"],
    ], [2.4, 4.1], size=9)

    n = data_sections(doc, ctx, 2)

    doc.add_heading(f"{n}. Models", level=1)
    doc.add_paragraph("Eight models form a ladder; each step adds one ability, so the comparison shows what every part contributes. "
                      "All predict log(next-day RV / trailing 20-day mean) on the same rows.")
    table(doc, ["Step", "Model", "What it does", "Inputs", "Parameters"],
          [[s, LABELS[k], d, i, f"{params[k]:,}" if k in params else "—"] for s, k, d, i in STEPS], [0.4, 1.2, 2.9, 1.3, 0.8], size=8.5)
    t, m_cfg = neural["training"], neural["model"]
    bullets(doc, [
        f"Neural architecture: per-contract embedding ({len(neural['token_features'])} → {m_cfg['d_model']}), {m_cfg['layers']} self-attention layers × "
        f"{m_cfg['heads']} heads (feed-forward {m_cfg['ff_dim']}, no positional encoding; position comes from log-moneyness), attention pooling "
        f"with a log-count channel, GRU (hidden {m_cfg['gru_hidden']}) over the last {neural['window']} snapshots, head on the chain summary plus "
        f"{len(neural['state_columns'])} volatility-state features.",
        f"Training: Huber loss (δ = {t['huber_delta']}), AdamW (learning rate {t['lr']}, weight decay {t['weight_decay']}), batch {t['batch_size']}, "
        f"early stopping on the inner validation split (patience {t['patience']}, at most {t['max_epochs']} epochs), "
        f"{len(t['seeds'])} seeds averaged; the early-stopped models predict the test block (no refit). Inputs are scaled per walk on training days only.",
        "Training record: " + "; ".join(f"{LABELS[k]} stopped at epoch {r['stop_epoch_mean']:.0f} on average ({r['seconds'] / 60:.1f} min in total)"
                                         for k, r in model_rows.items()) + ".",
    ])

    doc.add_heading(f"{n + 1}. Evaluation design", level=1)
    harness = evaluation["harness"]
    bullets(doc, [
        f"Walk-forward on expiry cycles: expanding training window (at least {harness['min_train_cycles']} cycles), a {harness['embargo_cycles']}-cycle "
        f"embargo, test blocks of {harness['test_cycles']} cycles ({walks} walks), and a purge of training rows whose label resolves inside the test block.",
        "Tuning and early stopping use an inner split of each training window only; test blocks are never used for choices.",
        "Primary score: IC, the rank correlation of prediction and outcome within each test block, averaged over walks. Supporting: RMSE, QLIKE "
        "(the standard volatility loss) and Diebold–Mariano tests on daily losses.",
        f"Significance: the RQ1–RQ3 tests were fixed in advance; RQ2 uses Holm within its family; everything else is exploratory. α = {alpha}.",
    ])

    doc.add_heading(f"{n + 2}. Results: the full ladder", level=1)
    doc.add_paragraph("Three scores, all on the same 26 test periods:")
    bullets(doc, [f"{name}: {text}." for name, text in METRICS_PLAIN])
    doc.add_paragraph("Why QLIKE?")
    bullets(doc, qlike_explained())
    table(doc, ["Step", "Model", "RMSE", "QLIKE", "Mean IC", "IC 95% interval"],
          [[s, LABELS[k], fmt(ladder[k]["rmse_mean_per_walk"]), fmt(ladder[k]["qlike_mean_per_walk"]), fmt(ladder[k].get("ic_mean_per_walk")),
            f"{ladder[k]['ic_low']:.3f} to {ladder[k]['ic_high']:.3f}" if ladder[k].get("ic_ci95") else "—"] for s, k, _, _ in STEPS if k in ladder],
          [0.5, 1.6, 0.9, 0.9, 0.9, 1.5], size=9)
    if (ctx.figures / f"results_errors{suffix}.png").exists():
        figure(doc, ctx.figures / f"results_errors{suffix}.png", ctx.caption("Forecast error per model (lower is better): RMSE and QLIKE."), width=6.3)
    doc.add_paragraph("What each step adds: change against HAR-RV for the errors (negative = lower error, better) and IC gain against "
                      "persistence, HAR-RV and the previous step (differences of means, descriptive; the paired tests follow in RQ1 and RQ2).")
    pct = lambda v: "—" if v is None else f"{v:+.1f}%"
    sgn = lambda v: "—" if v is None else f"{v:+.3f}"
    table(doc, ["Model", "RMSE vs HAR", "QLIKE vs HAR", "IC", "IC vs persistence", "IC vs HAR", "IC vs previous step"],
          [[r["name"], pct(r["rmse_vs_har_pct"]), pct(r["qlike_vs_har_pct"]), fmt(r["ic"]), sgn(r["ic_vs_persistence"]), sgn(r["ic_vs_har"]),
            sgn(r["ic_vs_previous"])] for r in ladder_improvements(ladder)], [1.4, 0.9, 0.9, 0.6, 0.9, 0.8, 1.0], size=8.5)
    figure(doc, ctx.figures / f"results_ladder_ic{suffix}.png", ctx.caption("Mean per-walk IC with 95% intervals: classical (blue) and neural (orange)."), width=5.8)
    figure(doc, ctx.figures / f"results_ic_by_walk{suffix}.png", ctx.caption("IC in each test block for HAR-RV, tuned LightGBM and attention + GRU."))

    doc.add_heading(f"{n + 3}. RQ1: does reading every contract predict better?", level=1)
    doc.add_paragraph(f"{summary['headline']} {summary['loss']}")
    if summary["gex"]:
        doc.add_paragraph(summary["gex"])
    doc.add_paragraph(summary["rule"])
    table(doc, ["Attention + GRU vs", "RMSE", "QLIKE", "IC gain (95% interval)", "Walks won", "Ranking (IC)", "Error (RMSE / QLIKE)"],
          [[r["name"], f"{r['rmse_lower_pct']:+.1f}% lower" if r["rmse_lower_pct"] >= 0 else f"{-r['rmse_lower_pct']:.1f}% higher",
            f"{r['qlike_lower_pct']:+.1f}% lower" if r["qlike_lower_pct"] >= 0 else f"{-r['qlike_lower_pct']:.1f}% higher",
            f"{r['ic_diff']:+.3f} ({r['ci95'][0]:+.3f} to {r['ci95'][1]:+.3f})", f"{r['wins']}/{r['walks']}", r["ic_status"],
            f"{r['rmse_status']} (p {fmt(r['dm_squared_p'])}) / {r['qlike_status']} (p {fmt(r['dm_qlike_p'])})"] for r in summary["rungs"]],
          [1.2, 0.8, 0.8, 1.3, 0.6, 0.6, 1.6], size=8)
    doc.add_paragraph("Ranking: ahead = 95% interval of the per-walk IC gain above 0, on par = interval includes 0. Error: Diebold–Mariano test on "
                      f"daily losses at α = {alpha} (squared error for RMSE, QLIKE loss for QLIKE), unadjusted and supporting; the pre-registered "
                      "rule is the IC test.")
    bullets(doc, [comparison_sentence(f"vs {LABELS[k.split('_vs_')[1]]}", c, alpha) for k, c in rq1["comparisons"].items()])

    doc.add_heading(f"{n + 4}. RQ2: what does each part contribute?", level=1)
    holm = {mbr["test"]: mbr for mbr in rq2["holm"]["members"]}
    table(doc, ["Comparison", "IC difference", "95% interval", "p", "Holm p", "Significant"],
          [[RQ2_NAMES.get(name, name), fmt(c["ic"].get("mean_diff"), "+.3f"), interval(c["ic"]), fmt(c["ic"].get("p_two_sided")), fmt(holm[name]["p_holm"]),
            "yes" if holm[name]["reject_holm"] else "no"] for name, c in rq2["comparisons"].items()], [2.4, 0.9, 1.2, 0.6, 0.7, 0.7], size=8.5)

    doc.add_heading(f"{n + 5}. RQ3: does it survive the rule change?", level=1)
    doc.add_paragraph(f"IC gain of attention + GRU over tuned LightGBM: {fmt(rq3['mean_gain_before'], '+.3f')} on {rq3['walks_before']} walks before the "
                      f"November 2024 rule change and {fmt(rq3['mean_gain_after'], '+.3f')} on {rq3['walks_after']} walks after "
                      f"(Welch t {fmt(rq3['welch_t'], '.2f')}, p {fmt(rq3['p_two_sided'])}). The post-change period is short, so this test has little power.")
    figure(doc, ctx.figures / f"results_regime{suffix}.png", ctx.caption("Per-walk IC gain over tuned LightGBM; orange walks follow the rule change."))

    section = n + 6
    # Mac ablations only (same device as the main run); the cluster's window runs are in the exploratory section (device rule).
    ablations = sorted(p for p in root.glob("neural_comparison_*.json")
                       if not p.name.endswith("_dev.json") and "hpc" not in p.name and "dev_" not in p.name) if tag == "main" else []
    doc.add_heading(f"{section}. Ablations (exploratory)", level=1)
    if ablations:
        base = model_rows.get("attention_gru", {}).get("ic_mean_per_walk")
        rows = []
        for path in ablations:
            run = load(path)
            for r in run["comparison_table"]:
                rows.append([run["tag"], LABELS[r["model"]], fmt(r["ic_mean_per_walk"]),
                             fmt(r["ic_mean_per_walk"] - base, "+.3f") if base is not None and r["model"] == "attention_gru" else "—"])
        table(doc, ["Variant", "Model", "Mean IC", "vs main attention + GRU"], rows, [2.2, 1.6, 1.0, 1.7], size=9)
        doc.add_paragraph("History-length ablations (L = 24, L = 72) ran on Explorer V100s against a V100 reference and are reported with the "
                          "exploratory experiments (E2) in the next sections.")
    else:
        doc.add_paragraph("Pending: window lengths 24 and 72 and the OI-change field drop (scripts/run_neural.py --window / --drop).")

    attention_path = reports / "attention.json"
    doc.add_heading(f"{section + 1}. What the model attends to", level=1)
    if attention_path.exists() and tag == "main":
        attention = load(attention_path)
        doc.add_paragraph("Exploratory, from the pooling weights of self-attention and attention + GRU on every test day (scripts/report_attention.py). "
                          "A contract's relative weight is its pooling weight times the number of contracts in the snapshot; 1 is a plain average.")
        bullets(doc, attention["findings"])
        for item in attention["figures"]:
            figure(doc, REPO_ROOT / item["path"], ctx.caption(item["caption"]))
    else:
        doc.add_paragraph("Pending: pooling attention weights are stored per test day (run_neural.py --save-attention); "
                          "scripts/report_attention.py draws the maps.")

    summary_path = reports / "exploratory_summary.json"
    doc.add_heading(f"{section + 2}. Exploratory experiments E1–E3 (not pre-registered)", level=1)
    if summary_path.exists() and tag == "main":
        exploratory = load(summary_path)
        doc.add_paragraph("After the pre-registered tests, three exploratory experiments were registered in configs/eval_exploratory_v2.yaml before any "
                          "of their runs, with their own config hash and Holm families: E1 diagnostics on the next-day target (HAR residuals, a "
                          "state-only control, a hybrid HAR + attention + GRU correction, a forecast combination, conditional IC), E2 more history "
                          "for the next-day model (longer intraday windows, previous days' chains), and E3 a new target, the next one and two hours "
                          "of realized volatility. Neural models compared with each other ran on the same GPU type (Explorer V100s, three seeds each); "
                          "neural-vs-classical comparisons may mix devices. None of this changes the RQ1–RQ3 answers above.")
        table(doc, ["Exp.", "Question", "Evidence", "Answer"],
              [[r["experiment"], r["question"], r["evidence"], r["verdict"]] for r in exploratory["rows"]], [0.5, 2.0, 2.9, 1.1], size=8)
        doc.add_paragraph(exploratory["overall"])
        for name, caption in (("e1_residual_ic", "E1: IC of the predicted vs actual HAR residual, per walk, for three residual models."),
                              ("e2_history_ic_gain", "E2: per-walk IC gain over attention + GRU at L = 12 for longer windows and previous days' chains."),
                              ("e3_ic_by_time_of_day", "E3: IC within each snapshot time across test days, 1 and 2 hours ahead.")):
            if (ctx.figures / f"{name}.png").exists():
                figure(doc, ctx.figures / f"{name}.png", ctx.caption(caption))
        doc.add_paragraph("Details: docs/findings_exploratory_diagnostics.md (E1), docs/findings_exploratory_history.md (E2), docs/findings_intraday.md (E3), "
                          "docs/findings_exploratory_summary.md.")
    else:
        doc.add_paragraph("Pending: scripts/report_exploratory_summary.py summarizes E1–E3 once their tests are in.")

    data_gaps_section(doc, ctx, section + 3)

    doc.add_heading(f"{section + 4}. Limitations", level=1)
    bullets(doc, [
        f"Fewer than 2,000 labeled days and {walks} test blocks: intervals are wide, and effects of a few hundredths of IC are hard to detect.",
        "Who the dealers are is not observed; the inferred GEX sign comes from participant OI covering all index options, not NIFTY alone.",
        "The source records some option OI as zero (2022) and the daily VIX file ends in April 2026; these rows are flagged, never imputed.",
        "The pre-registered tests use one horizon (next day) and one snapshot time (15:15); the 1–2 hour horizons (E3) are exploratory.",
        "The main neural runs used the Mac (Apple GPU); the exploratory neural runs used Explorer V100s, so neural models are compared only within "
        "one device. The V100 references reproduce the Mac's mean IC closely.",
    ])

    doc.add_heading(f"{section + 5}. Conclusion", level=1)
    conclusion = f"{summary['headline']} {summary['loss']} " + (f"{summary['gex']} " if summary["gex"] else "") + summary["rule"]
    if summary_path.exists() and tag == "main":
        conclusion += " Exploratory follow-up (E1–E3): " + load(summary_path)["overall"]
    doc.add_paragraph(conclusion)

    feature_appendix(doc, ctx)
    doc.add_heading("Appendix B. Reproducibility", level=1)
    bullets(doc, [
        "Environment: conda env create -f env/environment.yml (Python 3.11, PyTorch 2.10 from conda-forge).",
        "Data stages: clean_chain → compute_basis → compute_forward → compute_greeks → compute_gex → compute_features → compute_tokens → "
        "compute_targets → validate_data (scripts/, about 3 minutes).",
        "Models and tests: run_forecast → run_neural --save-attention → run_rq_tests → report_results → report_attention → write_final_report.",
        "Exploratory E1–E3: configs/eval_exploratory_v2.yaml and configs/intraday.yaml; Mac CPU runs (compute_intraday_rv, the --grid intraday stages, "
        "run_intraday classical rungs, run_exploratory_e1 residual LightGBM), Explorer GPU runs (hpc/, docs/hpc.md; seeds merged by scripts/merge_seeds.py), "
        "then the --tests modes and the report_exploratory_* scripts.",
        f"Data config hash {ctx.m['features']['config_hash'][:12]}; eval config hash {tests['eval_config_hash'][:12]}; every results-store run records "
        "its git commit and whether the working tree was clean.",
    ])
    footer(doc, f"NiftyGEX · generated by scripts/write_final_report.py · data {ctx.m['features']['config_hash'][:12]} · eval {tests['eval_config_hash'][:12]}")

    out = REPO_ROOT / "docs" / "deliverables" / "NiftyGEX_Final_Report.docx" if tag == "main" else reports / "NiftyGEX_Final_Report_dev.docx"
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)
    write_report(config, f"final_report{suffix}", {"output": str(out), "tag": tag, "rq1_beats_every_rung": rq1["beats_every_rung"],
                                                    "parameters": params, "ablation_files": [p.name for p in ablations]})
    print(out)


if __name__ == "__main__":
    main()
