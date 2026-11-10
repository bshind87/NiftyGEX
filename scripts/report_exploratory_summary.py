"""Cross-experiment summary of the exploratory E1-E3 (docs/findings_exploratory_summary.md, outputs/reports/exploratory_summary.json).

Reads the test results written by scripts/run_exploratory_e1.py, run_exploratory_e2.py and run_intraday.py --tests.
Each row states, in plain words, whether the option chain adds predictive information beyond recent volatility for
that question; the verdicts follow from the registered Holm results by rule, and every number comes from the files.
"""
from __future__ import annotations

import json
from pathlib import Path

from niftygex.config import REPO_ROOT
from niftygex.io.parquet import write_report
from niftygex_eval.config import load_exploratory_config


NAMES = {"lightgbm": "tuned LightGBM", "intraday_har": "intraday HAR", "har_gex": "HAR + GEX", "persistence": "persistence",
         "window_24": "L = 24", "window_72": "L = 72", "multiday_5": "multi-day K = 5", "multiday_10": "multi-day K = 10"}


def ci(values) -> str:
    return f"{values[0]:+.3f} to {values[1]:+.3f}"


def verdict(member: dict, better: str, worse: str, neutral: str) -> str:
    if not member.get("reject_holm"):
        return neutral
    return better if member["mean_diff"] > 0 else worse


def main() -> None:
    cfg = load_exploratory_config()
    root = Path(cfg["results_root"])
    e1 = json.loads((root / "exploratory_e1.json").read_text())
    e2 = json.loads((root / "exploratory_e2.json").read_text())
    e3 = json.loads((root / "exploratory_e3.json").read_text())
    e3_h = {h: json.loads((root / f"exploratory_e3_{h}.json").read_text()) for h in ("1h", "2h")}
    t1 = {m["test"]: m for m in e1["tests"]["members"]}
    t2 = {m["test"]: m for m in e2["tests"]["members"]}
    t3 = {m["test"]: m for m in e3["tests"]["members"]}
    s1 = e1["summary"]

    best_plus = max((t1[n] for n in ("har_plus_residual_lightgbm_vs_har", "har_plus_residual_deepsets_vs_har", "har_plus_residual_attention_vs_har")),
                    key=lambda m: m["mean_diff"])
    state = e1["secondary"]["state_mlp_vs_attention_gru_hpc"]["ic"]
    best_e2 = max(t2.values(), key=lambda m: m["mean_diff"])
    rows = [
        {"experiment": "E1", "question": "Next day: does the chain add information beyond HAR, as a correction?",
         "evidence": f"HAR + residual (best) {best_plus['mean_diff']:+.3f} IC over HAR (95% CI {ci(best_plus['ci95'])}, Holm p {best_plus['p_holm']:.2g}); "
                     f"hybrid {t1['hybrid_vs_har']['mean_diff']:+.3f} over HAR",
         "verdict": verdict(best_plus, "Yes: as a correction to HAR", "No: it hurts", "Not detectably")},
        {"experiment": "E1", "question": "Next day: does a forecast combination beat HAR?",
         "evidence": f"equal-weight HAR, LightGBM, attention + GRU: IC {s1['combination']['ic']:.3f}, {t1['combination_vs_har']['mean_diff']:+.3f} over HAR "
                     f"(Holm p {t1['combination_vs_har']['p_holm']:.2g})",
         "verdict": verdict(t1["combination_vs_har"], "Yes", "No: it is worse", "Not detectably")},
        {"experiment": "E1", "question": "Next day: do the tokens add to the neural model beyond the volatility state?",
         "evidence": f"state-only MLP {s1['state_mlp']['ic']:.3f} vs attention + GRU {s1['attention_gru_hpc']['ic']:.3f} "
                     f"(state-only {state['mean_diff']:+.3f}, 95% CI {ci(state['ci95'])}; secondary, not Holm-tested)",
         "verdict": "No detectable gain" if state["ci95"][0] < 0 < state["ci95"][1] else ("No: the state alone does better" if state["mean_diff"] > 0 else "Yes")},
        {"experiment": "E2", "question": "Next day: does more history (longer intraday window, previous days' chains) help?",
         "evidence": f"best variant {NAMES[best_e2['test'].removesuffix('_vs_l12')]} {best_e2['mean_diff']:+.3f} IC over L = 12 (Holm p {best_e2['p_holm']:.2g}); "
                     f"{sum(m['reject_holm'] for m in t2.values())} of {len(t2)} tests reject",
         "verdict": "Not detectably" if not any(m["reject_holm"] for m in t2.values()) else "Yes, for some variants"},
    ]
    for h, block in e3_h.items():
        member = t3[f"attention_gru_vs_best_classical_{h}"]
        best = block["best_classical"]
        rows.append({"experiment": "E3", "question": f"Next {h[0]} hour{'s' if h != '1h' else ''}: does attention + GRU beat the best classical rung?",
                     "evidence": f"attention + GRU {block['scores']['attention_gru']['ic_mean_per_walk']:.3f} vs {NAMES[best]} "
                                 f"{block['scores'][best]['ic_mean_per_walk']:.3f} ({member['mean_diff']:+.3f}, 95% CI {ci(member['ci95'])}, Holm p {member['p_holm']:.2g})",
                     "verdict": verdict(member, "Yes", "No: significantly worse", "Not detectably")})
    overall = ("Across E1-E3 the option chain adds predictive information beyond recent volatility only when it corrects or is combined with "
               "a volatility model (E1); read on its own by attention models it adds nothing detectable for the next day (E1 state control, E2) "
               "and does worse than simple volatility models for the next one to two hours (E3).")
    checks = [f"{a == b}" for a, b in ((rows[0]["verdict"], "Yes: as a correction to HAR"), (rows[3]["verdict"], "Not detectably"),
                                         (rows[4]["verdict"], "No: significantly worse"), (rows[5]["verdict"], "No: significantly worse"))]
    if not all(c == "True" for c in checks):
        overall = "The verdicts differ from the pattern the overall sentence describes; read the table row by row."

    lines = ["# Exploratory E1–E3: where does the option chain add information?", "",
             "> **Exploratory.** Registered in `configs/eval_exploratory_v2.yaml` before any run; separate from the pre-registered RQ1–RQ3.", "",
             "Generated by `scripts/report_exploratory_summary.py` from the E1–E3 test results. Details: `docs/findings_exploratory_diagnostics.md` (E1), "
             "`docs/findings_exploratory_history.md` (E2), `docs/findings_intraday.md` (E3).", "",
             "| Experiment | Question | Evidence | Answer |", "|---|---|---|---|",
             *[f"| {r['experiment']} | {r['question']} | {r['evidence']} | **{r['verdict']}** |" for r in rows], "",
             f"**In one sentence.** {overall}", ""]
    out = REPO_ROOT / "docs" / "findings_exploratory_summary.md"
    out.write_text("\n".join(lines))
    write_report(cfg["data_config"], "exploratory_summary", {"rows": rows, "overall": overall, "exploratory_config_hash": cfg["exploratory_hash"]})
    print(out)


if __name__ == "__main__":
    main()
