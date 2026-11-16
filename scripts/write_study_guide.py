"""Generated parts of the study guide: existing work, every model with its metrics, and the limitations.

Writes docs/study_guide_generated.md and outputs/reports/figures/study_models_progress.png. Every number is read from
the results and report files (or computed here from the curated futures bars, for the realized-volatility noise floor),
so the guide's model tables and limitations never carry hand-typed numbers. Each section has an "Explain it" box:
a short script for presenting it, to be edited into the presenter's own words.

    PYTHONPATH=src python scripts/write_study_guide.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from niftygex.config import REPO_ROOT
from niftygex.io.parquet import write_report
from niftygex_eval.config import load_exploratory_v3_config
from niftygex_eval.figures import SERIES, TEXT_2, save
from niftygex_eval.rq1_summary import ladder_improvements, rq1_summary


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def p_text(p) -> str:
    return "—" if p is None else ("< 0.001" if p < 0.001 else f"{p:.2g}")


def pct(new: float, ref: float) -> str:
    return f"{100 * (new - ref) / ref:+.1f}%"


def box(text: str) -> list[str]:
    return [f"> **Explain it:** {text}", ""]


def verdict(member: dict | None, better: str = "yes, significant") -> str:
    if not member:
        return "descriptive"
    if member.get("reject_holm"):
        return better if member["mean_diff"] > 0 else "significantly worse"
    return "not significant"


def noise_floor(futures: Path) -> dict:
    """Measurement noise of daily realized volatility from 5-minute futures returns (realized quarticity, delta method)."""
    bars = pl.read_parquet(futures).sort("datetime")
    bars = bars.with_columns(pl.col("close").log().diff().over("trade_date").alias("r")).drop_nulls("r")
    days = bars.group_by("trade_date").agg(pl.len().alias("n"), (pl.col("r") ** 2).sum().alias("rv2"), (pl.col("r") ** 4).sum().alias("rq"))
    days = days.filter(pl.col("n") >= 70)
    sd = np.sqrt((2 / 3) * days["rq"].to_numpy() / days["rv2"].to_numpy() ** 2 / 4)   # sd of log(volatility estimate)
    return {"days": days.height, "median_bars": float(days["n"].median()), "median_sd": float(np.median(sd)), "rms_sd": float(np.sqrt(np.mean(sd ** 2)))}


def figure_progress(rows: list[tuple[str, float, float, str]], directory: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 0.42 * len(rows) + 1.5), sharey=True)
    colours = {"classical": SERIES[0], "neural": SERIES[1], "correction": SERIES[2]}
    for ax, idx, title in ((axes[0], 1, "IC (ranking, higher is better)"), (axes[1], 2, "RMSE (error, lower is better)")):
        for i, row in enumerate(rows):
            ax.barh(i, row[idx], color=colours[row[3]])
            ax.annotate(f"{row[idx]:.3f}", (row[idx], i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8.5, color=TEXT_2)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    axes[0].set_xlim(0, max(r[1] for r in rows) * 1.2)
    axes[1].set_xlim(0, max(r[2] for r in rows) * 1.2)
    axes[0].set_yticks(range(len(rows)), [r[0] for r in rows])
    axes[0].invert_yaxis()
    fig.suptitle("Next-day models on the same 26 test periods: classical (blue), neural (orange), corrections and combinations (green)",
                 x=0.01, ha="left", fontsize=9.5)
    return save(fig, directory, "study_models_progress")


def main() -> None:
    cfg = load_exploratory_v3_config()
    root, reports = Path(cfg["results_root"]), Path(cfg["data_config"]["paths"]["reports_root"])
    results, tests = load(reports / "results.json"), load(root / "rq_tests.json")
    e1, e2, v3 = load(root / "exploratory_e1.json"), load(root / "exploratory_e2.json"), load(root / "exploratory_v3.json")
    e3 = {h: load(root / f"exploratory_e3_{h}.json") for h in ("1h", "2h")}
    e3_tests = {m["test"]: m for m in load(root / "exploratory_e3.json")["tests"]["members"]}
    missing, attention = load(reports / "missing_data.json"), load(reports / "attention.json")
    ladder = {r["model"]: r for r in results["ladder"]}
    q1 = rq1_summary(tests["RQ1"]["comparisons"], ladder, tests["alpha"], tests["RQ1"]["beats_every_rung"])
    t1 = {m["test"]: m for m in e1["tests"]["members"]}
    t2 = {m["test"]: m for m in e2["tests"]["members"]}
    t3v = {m["test"]: m for m in v3["tests"]["members"]}
    s1, sv = e1["summary"], v3["summary"]
    rq2 = {k.split(" ")[0]: c for k, c in tests["RQ2"]["comparisons"].items()}
    rq2_holm = {m["test"].split(" ")[0]: m for m in tests["RQ2"]["holm"]["members"]}
    har = ladder["har"]
    lines: list[str] = []

    # ------------------------------------------------------------------ existing work
    lines += ["## Existing work we built on", "",
              "Four lines of work shaped the project: they set the question, the baselines, the metrics and the tests.", "",
              *box("Traders and researchers believe dealer hedging moves markets; volatility forecasters know simple HAR models are hard to "
                   "beat; and set models like DeepSets and attention can read a variable-size option chain. We put the three together "
                   "and tested them with the evaluation rules the forecasting literature recommends."),
              "| Area | Work | What we took from it |", "|---|---|---|",
              "| Dealer hedging moves prices | Ni, Pearson, Poteshman & White (2021), *Review of Financial Studies* 34(4) | Option market-maker hedging raises the volatility and the chance of large moves of the underlying: the reason GEX could matter |",
              "| | Baltussen, Da, Lammers & Martens (2021), *Journal of Financial Economics* 142(1) | Hedging short gamma trades with the price move and creates intraday momentum; stronger when negative gamma exposure is larger |",
              "| | Barbon & Buraschi, \"Gamma Fragility\" (working paper, 2020–21) | Large dealer gamma imbalances go with intraday momentum, reversals and fragility |",
              "| | Practitioner GEX (dealer gamma exposure) summaries | The one-number summary we test against: gamma × open interest × spot² × 1%, summed over the chain |",
              "| Volatility forecasting | Andersen, Bollerslev, Diebold & Labys (2003), *Econometrica* | Realized volatility from intraday returns as the measure of what happened |",
              "| | Corsi (2009), *Journal of Financial Econometrics* | HAR-RV: daily, weekly and monthly volatility; our main classical baseline |",
              "| | Christensen, Siggaard & Veliyev (2023), *Journal of Financial Econometrics*; Audrino & Chassot (2024, \"HARd to Beat\") | Machine learning can improve on HAR, but the gains are modest and a well-fitted HAR is hard to beat: our expectation for the size of any gain |",
              "| Models for sets and sequences | Zaheer et al. (2017), Deep Sets; Vaswani et al. (2017), attention; Lee et al. (2019), Set Transformer | Reading a variable-size, unordered set of contracts: DeepSets, self-attention and attention pooling |",
              "| | Cho et al. (2014), GRU; Ke et al. (2017), LightGBM | The recurrent layer over snapshots and the tuned tree baseline |",
              "| Fair evaluation | Patton (2011), *Journal of Econometrics* | QLIKE and MSE rank volatility forecasts correctly even with a noisy volatility proxy |",
              "| | Diebold & Mariano (1995); Holm (1979); López de Prado (2018) | Comparing forecast errors, correcting for several tests, and walk-forward splits with embargo and purge |",
              "", "**What was new here:** an auditable point-in-time NIFTY option-chain pipeline, every contract as a token, and a test of "
              "GEX against attention models under rules fixed in advance. To our knowledge this comparison has not been published for NIFTY "
              "(say \"to our knowledge\": we did not run a systematic review).", ""]

    # ------------------------------------------------------------------ every model
    steps = {r["model"]: r for r in ladder_improvements(ladder)}
    lines += ["## Every model we ran, and how the metrics changed", "",
              f"{q1['headline']} {q1['loss']} The best setup overall is the exploratory HAR-IV + chain correction: IC "
              f"{sv['har_iv_plus_chain']['ic']:.3f}, RMSE {sv['har_iv_plus_chain']['rmse']:.3f}, QLIKE {sv['har_iv_plus_chain']['qlike']:.3f}.", "",
              *box("We built the models as a ladder, each adding one ability, so every step shows what it adds. RMSE and QLIKE measure the "
                   "size of the misses (lower is better); IC measures whether calm and wild days are put in the right order (higher is better)."),
              "![Every next-day model: IC and RMSE](../outputs/reports/figures/study_models_progress.png)", "",
              "### The main ladder (pre-registered, next day)", "",
              "| Step | Model | What it does | IC | RMSE | QLIKE | IC vs HAR | RMSE vs HAR | QLIKE vs HAR |", "|---|---|---|---|---|---|---|---|---|"]
    what = {"historical_mean": "Always forecasts the training average", "persistence": "Tomorrow is like today",
            "har": "Linear model on daily, weekly and monthly past volatility", "har_gex": "HAR plus total GEX under three sign conventions",
            "lightgbm": "Decision trees on about 40 summary features", "deepsets": "Reads every contract, then a plain average",
            "attention": "Contracts compare notes (self-attention), then a learned average", "attention_gru": "Self-attention on 12 snapshots, then a GRU over the hour (ours)"}
    for step, model in enumerate(["historical_mean", "persistence", "har", "har_gex", "lightgbm", "deepsets", "attention", "attention_gru"]):
        r, st = ladder[model], steps.get(model, {})
        ic = r.get("ic_mean_per_walk")
        ic_text = "—" if ic is None else f"{ic:.3f}"
        gain = st.get("ic_vs_har")
        gain_text = "—" if gain is None else f"{gain:+.3f}"
        label = r["label"].split(" ", 1)[1]
        lines.append(f"| {max(step - 1, 0)} | {label} | {what[model]} | {ic_text} | "
                     f"{r['rmse_mean_per_walk']:.3f} | {r['qlike_mean_per_walk']:.3f} | {gain_text} | "
                     f"{'—' if model == 'har' else pct(r['rmse_mean_per_walk'], har['rmse_mean_per_walk'])} | "
                     f"{'—' if model == 'har' else pct(r['qlike_mean_per_walk'], har['qlike_mean_per_walk'])} |")
    gex = next(r for r in q1["rungs"] if r["rung"] == "har_gex")
    inter, hist = rq2["attention_vs_deepsets"]["ic"], rq2["attention_gru_vs_attention"]["ic"]
    lines += ["", f"- **RQ1 (attention + GRU vs the classical rungs):** {q1['rule']}",
              f"- **GEX:** {q1['gex']}",
              f"- **RQ2:** contracts comparing notes (self-attention vs DeepSets) {inter['mean_diff']:+.3f} IC (Holm p "
              f"{p_text(rq2_holm['attention_vs_deepsets']['p_holm'])}); memory of the hour (GRU) {hist['mean_diff']:+.3f} IC (Holm p "
              f"{p_text(rq2_holm['attention_gru_vs_attention']['p_holm'])}). Neither is significant.", ""]
    lines += box(f"HAR is the bar to clear. GEX adds almost nothing to it ({steps['har_gex']['ic_vs_har']:+.3f} IC). Our model ties the best on "
                 f"ranking and has the smallest errors: QLIKE {abs(steps['attention_gru']['qlike_vs_har_pct']):.0f}% below HAR and "
                 f"{gex['qlike_lower_pct']:.0f}% below HAR + GEX.")

    lines += ["### E1: using the chain as a correction (exploratory, next day)", "",
              "| Model | What it does | IC | RMSE | QLIKE | Test (Holm within E1) | Result |", "|---|---|---|---|---|---|---|"]
    e1_rows = [("residual_lightgbm", "HAR + LightGBM correction", "LightGBM on chain-only features predicts HAR's error", "har_plus_residual_lightgbm_vs_har"),
               ("residual_deepsets", "HAR + DeepSets correction", "DeepSets on the tokens predicts HAR's error", "har_plus_residual_deepsets_vs_har"),
               ("residual_attention", "HAR + attention correction", "Self-attention on the tokens predicts HAR's error", "har_plus_residual_attention_vs_har"),
               ("hybrid_attention_gru", "Hybrid", "Attention + GRU that sees HAR's forecast and outputs a correction", "hybrid_vs_har"),
               ("combination", "Combination", "Plain average of HAR, LightGBM and attention + GRU", "combination_vs_har"),
               ("state_mlp", "State-only control", "Same network, volatility numbers only, no contracts", None)]
    for key, name, desc, test in e1_rows:
        m = t1.get(test) if test else None
        res = (f"{m['mean_diff']:+.3f} IC vs HAR, Holm p {p_text(m['p_holm'])}" if m else
               f"{e1['secondary']['state_mlp_vs_attention_gru_hpc']['ic']['mean_diff']:+.3f} IC vs attention + GRU (secondary)")
        lines.append(f"| {name} | {desc} | {s1[key]['ic']:.3f} | {s1[key]['rmse']:.3f} | {s1[key]['qlike']:.3f} | {res} | {verdict(m) if m else 'not significant'} |")
    lines += [""] + box(f"Instead of asking the network to forecast volatility from scratch, we let HAR forecast and asked the chain what HAR "
                        f"gets wrong. That works: corrections and combinations lift IC from {har['ic_mean_per_walk']:.3f} to about "
                        f"{s1['residual_lightgbm']['ic']:.2f}–{s1['combination']['ic']:.2f}, all significant.")

    lines += ["### E2: more history for the next-day model (exploratory)", "",
              "| Variant | IC | IC vs L = 12 | Holm p | Result |", "|---|---|---|---|---|"]
    for key, name in (("window_24_vs_l12", "Last 24 snapshots (2 hours)"), ("window_72_vs_l12", "Whole day (72 snapshots)"),
                      ("multiday_5_vs_l12", "Previous 5 days' chains + last hour"), ("multiday_10_vs_l12", "Previous 10 days' chains + last hour")):
        m = t2[key]
        lines.append(f"| {name} | {m['mean_a']:.3f} | {m['mean_diff']:+.3f} | {p_text(m['p_holm'])} | {verdict(m)} |")
    lines += [""] + box("Giving the model more of the past did not help: longer windows moved IC by about a hundredth, within noise.")

    lines += ["### E3: a new target, volatility over the next 1 and 2 hours (exploratory)", "",
              "| Model | IC 1 h | RMSE 1 h | IC 2 h | RMSE 2 h |", "|---|---|---|---|---|"]
    for key, name in (("persistence", "Persistence"), ("intraday_har", "Intraday HAR"), ("har_gex", "HAR + GEX"), ("lightgbm", "Tuned LightGBM"),
                      ("deepsets", "DeepSets"), ("attention", "Self-attention"), ("attention_gru", "Attention + GRU")):
        a, b = e3["1h"]["scores"][key], e3["2h"]["scores"][key]
        lines.append(f"| {name} | {a['ic_mean_per_walk']:.3f} | {a['rmse_mean_per_walk']:.3f} | {b['ic_mean_per_walk']:.3f} | {b['rmse_mean_per_walk']:.3f} |")
    m1, m2 = e3_tests["attention_gru_vs_best_classical_1h"], e3_tests["attention_gru_vs_best_classical_2h"]
    lines += ["", f"Attention + GRU vs the best classical model: 1 h {m1['mean_diff']:+.3f} IC (Holm p {p_text(m1['p_holm'])}), "
                  f"2 h {m2['mean_diff']:+.3f} IC (Holm p {p_text(m2['p_holm'])}): significantly worse. Retraining with the main run's patience "
                  f"(v3 A4) changed nothing.", ""]
    lines += box("Over the next hour or two, how volatile the market has just been is the best guide, and HAR uses it directly; our "
                 "reading (not tested) is that the chain mostly adds noise at that horizon.")

    lines += ["### v3 checks: separating implied volatility from positioning (exploratory)", "",
              "| Model | What it does | IC | RMSE | QLIKE | Test (Holm across eight) | Result |", "|---|---|---|---|---|---|---|"]
    v3_rows = [("har_iv", "HAR-IV", "HAR plus the option-implied volatility (ATM IV)", "har_iv_vs_har", "vs HAR"),
               ("har_iv_plus_chain", "HAR-IV + chain correction", "Chain-only LightGBM corrects HAR-IV", "har_iv_plus_chain_vs_har_iv", "vs HAR-IV"),
               ("residual_deepsets_no_iv", "DeepSets correction, no IV", "E1 correction with the IV field removed", "residual_deepsets_no_iv_vs_har", "vs HAR"),
               ("residual_attention_no_iv", "Attention correction, no IV", "E1 correction with the IV field removed", "residual_attention_no_iv_vs_har", "vs HAR"),
               ("combination", "Combination", "Average of HAR, LightGBM, attention + GRU", "combination_vs_lightgbm", "vs LightGBM")]
    for key, name, desc, test, against in v3_rows:
        m = t3v[test]
        lines.append(f"| {name} | {desc} | {sv[key]['ic']:.3f} | {sv[key]['rmse']:.3f} | {sv[key]['qlike']:.3f} | "
                     f"{m['mean_diff']:+.3f} IC {against}, Holm p {p_text(m['p_holm'])} | {verdict(m)} |")
    lines += [""] + box("Implied volatility is a strong input on its own, but the chain still adds information on top of it: the correction "
                        f"improves HAR-IV by {t3v['har_iv_plus_chain_vs_har_iv']['mean_diff']:+.3f} IC. That is the best result of the project.")

    progress = [("HAR-RV", har["ic_mean_per_walk"], har["rmse_mean_per_walk"], "classical"),
                ("HAR + GEX", ladder["har_gex"]["ic_mean_per_walk"], ladder["har_gex"]["rmse_mean_per_walk"], "classical"),
                ("Tuned LightGBM", ladder["lightgbm"]["ic_mean_per_walk"], ladder["lightgbm"]["rmse_mean_per_walk"], "classical"),
                ("HAR-IV", sv["har_iv"]["ic"], sv["har_iv"]["rmse"], "classical"),
                ("DeepSets", ladder["deepsets"]["ic_mean_per_walk"], ladder["deepsets"]["rmse_mean_per_walk"], "neural"),
                ("Self-attention", ladder["attention"]["ic_mean_per_walk"], ladder["attention"]["rmse_mean_per_walk"], "neural"),
                ("Attention + GRU (ours)", ladder["attention_gru"]["ic_mean_per_walk"], ladder["attention_gru"]["rmse_mean_per_walk"], "neural"),
                ("HAR + LightGBM correction", s1["residual_lightgbm"]["ic"], s1["residual_lightgbm"]["rmse"], "correction"),
                ("Hybrid (E1)", s1["hybrid_attention_gru"]["ic"], s1["hybrid_attention_gru"]["rmse"], "correction"),
                ("Combination", s1["combination"]["ic"], s1["combination"]["rmse"], "correction"),
                ("HAR-IV + chain correction", sv["har_iv_plus_chain"]["ic"], sv["har_iv_plus_chain"]["rmse"], "correction")]
    figure = figure_progress(progress, reports / "figures")

    # ------------------------------------------------------------------ limitations
    noise = noise_floor(Path(cfg["data_config"]["paths"]["curated_root"]) / "futures" / "nifty_front_month.parquet")
    b8 = v3["reported"]["b8"]
    detect = max(v["detectable_at_power"] for k, v in b8.items() if k != "persistence")
    ruled = max(v["ruled_out_above"] for k, v in b8.items() if k != "persistence")
    oi = missing["missing_option_oi"]
    oi_ranges = ", ".join(r["start"] if r["start"] == r["end"] else f"{r['start']} to {r['end']}" for r in oi["ranges"])
    vix = missing["daily_vix"]
    ag = attention["models"]["attention_gru"]
    lines += ["## Limitations", "",
              "What limits these results, grouped so you can answer \"what would you do differently?\" quickly.", "",
              *box("Our data access, the size of the sample and what we can observe limit how strong our claims can be. None of these breaks "
                   "the comparison: every model sees the same data under the same rules."),
              "**Data access**", "",
              "- **Fixed history window.** The data come from a private archive pulled through a retail broker API (ICICI Direct). "
              "The API serves only recent history (about the last six years, per the team), so earlier years cannot be fetched; full "
              "historical NIFTY option-chain data is sold to enterprise clients only, not to individual users.",
              f"- **Short history per contract.** Each expiry has at most {missing['expiry_history_depth_days']['max']} days of option "
              "history in the archive, so GEX and the tokens see contracts only near their expiry.",
              f"- **Gaps we could not fill.** Option open interest is missing (recorded as 0 by the source) on {oi['days']} trading days "
              f"({oi_ranges}); daily VIX ends "
              f"{vix['last_available']} ({vix['trading_days_without']} trading days without it); spot is missing on "
              f"{len(missing['missing_spot']['days'])} days. All are flagged and never imputed.", "",
              "**What we can observe**", "",
              "- **Who the dealers are is unknown.** Nobody publishes which side dealers hold, so GEX needs an assumed sign; we tested three "
              "conventions (naive, absolute, inferred from previous-day participant data, which covers all index options, not NIFTY alone).",
              "- **Implied volatility cannot be separated fully from the chain.** Gamma, delta and strike GEX are computed from IV, so "
              "removing the IV field (v3 A2) still leaves IV inside; v3 A1 is the test that separates them.",
              "- **Tokens come from the nearest expiry only,** and one snapshot time (15:15) is used for the next-day target.", "",
              "**Sample size and statistics**", "",
              f"- **Few independent test periods.** About {ladder['har']['walks']} walk-forward test periods (under 2,000 labelled days): "
              f"the paired test detects IC gains of about {detect:.2f} with 80% power, and gains above about {ruled:+.2f} are ruled out; "
              "smaller real gains cannot be confirmed.",
              f"- **The rule-change test is weak.** Only {results['rq3']['walks_after']} test periods fall after November 2024.",
              f"- **Realized volatility is itself noisy.** With about {noise['median_bars']:.0f} five-minute bars a day it misses true volatility "
              f"by about {noise['rms_sd']:.2f} (RMS, on the log scale), so even a perfect forecaster would show an RMSE near that; much lower "
              "RMSE is not reachable on this target.",
              "- **Exploratory results are exploratory.** E1–E3 and the v3 checks were registered before their runs but after the main "
              "results were known; they explain the main result and do not change RQ1–RQ3.", "",
              "**Scope**", "",
              "- **One market and one main horizon:** NIFTY only, next-day volatility for the pre-registered tests (1–2 hours exploratory).",
              "- **No test of economic value:** we measure forecast accuracy, not trading or hedging profit after costs.",
              f"- **What attention shows is limited:** we stored only the pooling weights (a near-plain average on all but "
              f"{ag['concentrated_days_any_head']} of {ag['prediction_snapshots']:,} days); interactions inside the encoder layers were not measured.",
              "- **Devices:** the main neural runs used the Mac GPU, the exploratory ones Explorer V100s; neural models are only compared "
              "with each other on the same device.", ""]

    out = REPO_ROOT / "docs" / "study_guide_generated.md"
    out.write_text("\n".join(["<!-- Generated by scripts/write_study_guide.py from the results files; do not edit by hand. -->", "", *lines]))
    write_report(cfg["data_config"], "study_guide", {"output": str(out.relative_to(REPO_ROOT)), "figure": str(figure.relative_to(REPO_ROOT)),
                                                     "noise_floor": noise, "detectable_gain": detect, "ruled_out_above": ruled})
    print(out)


if __name__ == "__main__":
    main()
