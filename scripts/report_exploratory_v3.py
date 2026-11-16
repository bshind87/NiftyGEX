"""Exploratory v3 report: checks A1-A4 and reported results B5-B8 (docs/findings_exploratory_v3.md, with a figure).

Reads outputs/results/exploratory_v3.json (scripts/run_exploratory_v3.py --parts tests, after the Explorer runs are
merged). Writes outputs/reports/exploratory_v3.json and outputs/reports/figures/v3_*.png. Every verdict follows from
the registered Holm results by rule, and the A1 sentence is the one fixed in configs/eval_exploratory_v3.yaml.
Everything here is exploratory.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from niftygex.config import REPO_ROOT
from niftygex.io.parquet import write_report
from niftygex_eval.config import load_exploratory_v3_config
from niftygex_eval.figures import SERIES, TEXT_2, save

SECONDARY_LABELS = {
    "har_iv_plus_chain_vs_har": "HAR-IV + chain correction vs HAR",
    "residual_deepsets_no_iv_vs_residual_deepsets": "DeepSets correction: no IV vs with IV (E1)",
    "residual_attention_no_iv_vs_residual_attention": "Attention correction: no IV vs with IV (E1)",
    "e3_patience8_vs_patience3_1h": "E3 1 h attention + GRU: patience 8 vs patience 3",
    "e3_patience8_vs_patience3_2h": "E3 2 h attention + GRU: patience 8 vs patience 3",
}
TEST_LABELS = {
    "har_iv_vs_har": "A1 HAR-IV vs HAR",
    "har_iv_plus_chain_vs_har_iv": "A1 HAR-IV + chain correction vs HAR-IV",
    "residual_deepsets_no_iv_vs_har": "A2 HAR + DeepSets correction, no IV vs HAR",
    "residual_attention_no_iv_vs_har": "A2 HAR + attention correction, no IV vs HAR",
    "combination_vs_lightgbm": "A3 Combination vs tuned LightGBM",
    "combination_vs_har_iv": "A3 Combination vs HAR-IV",
    "e3_attention_gru_patience8_vs_best_classical_1h": "A4 E3 1 h: attention + GRU (patience 8) vs best classical",
    "e3_attention_gru_patience8_vs_best_classical_2h": "A4 E3 2 h: attention + GRU (patience 8) vs best classical",
}
NAMES = {"har": "HAR-RV", "har_gex": "HAR + GEX", "lightgbm": "Tuned LightGBM", "persistence": "Persistence", "har_iv": "HAR-IV",
         "har_iv_plus_chain": "HAR-IV + chain correction", "har_gex_naive": "HAR + GEX (naive only)", "har_gex_absolute": "HAR + GEX (absolute only)",
         "har_gex_inferred": "HAR + GEX (inferred + sign)", "residual_deepsets": "HAR + DeepSets correction (E1, with IV)",
         "residual_attention": "HAR + attention correction (E1, with IV)", "residual_deepsets_no_iv": "HAR + DeepSets correction, no IV",
         "residual_attention_no_iv": "HAR + attention correction, no IV", "combination": "Combination (HAR, LightGBM, attention + GRU)",
         "attention_gru": "Attention + GRU (Mac)", "deepsets": "DeepSets (Mac)", "attention": "Self-attention (Mac)", "state_mlp": "State-only MLP (V100)", "deepsets_hpc": "DeepSets (V100)",
         "attention_hpc": "Self-attention (V100)", "attention_gru_hpc": "Attention + GRU (V100)", "intraday_har": "intraday HAR",
         "residual_lightgbm": "HAR + LightGBM correction (E1)",
         "e3_1h_attention_gru_patience8": "E3 1 h attention + GRU, patience 8", "e3_2h_attention_gru_patience8": "E3 2 h attention + GRU, patience 8", "hybrid_attention_gru": "Hybrid attention + GRU (E1)"}
SUBSETS = {"expiry_day": "Expiry day", "not_expiry_day": "Not expiry day", "near_flip": "Within 0.5% of flip", "not_near_flip": "Not near flip",
           "gex_tercile_low": "GEX low", "gex_tercile_mid": "GEX mid", "gex_tercile_high": "GEX high",
           "dealer_long_gamma": "Dealers long gamma", "dealer_short_gamma": "Dealers short gamma"}


def f(value, spec: str = ".3f") -> str:
    return "—" if value is None or (isinstance(value, float) and not np.isfinite(value)) else format(value, spec)


def interval(ci) -> str:
    return "—" if not ci else f"{ci[0]:+.3f} to {ci[1]:+.3f}"


def p_text(p) -> str:
    return "—" if p is None else ("< 0.001" if p < 0.001 else f"{p:.2g}")


def figure_tests(members: list[dict], directory: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for i, m in enumerate(members):
        colour = SERIES[0] if m["reject_holm"] else TEXT_2
        ax.plot(m["ci95"], [i, i], color=colour, linewidth=2)
        ax.plot(m["mean_diff"], i, "D", color=colour, markersize=8, markeredgecolor="white", markeredgewidth=1.2)
    ax.axvline(0, color=TEXT_2, linewidth=0.8)
    ax.set_yticks(range(len(members)), [TEST_LABELS[m["test"]] for m in members], fontsize=8)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean per-walk IC difference (A minus B), 95% interval; blue = rejects after Holm")
    ax.set_title("Exploratory v3 primary tests (Holm across eight)")
    return save(fig, directory, "v3_primary_tests")


def provenance_rows(runs: dict, commit_expected: str | None) -> tuple[list[str], dict]:
    lines = ["| Run | Run id | Commit | git_dirty | v3 config hash | Device / GPU |", "|---|---|---|---|---|---|"]
    v3_runs = {k: r for k, r in runs.items() if r.get("exploratory_v3_config_hash")}
    for name, r in v3_runs.items():
        lines.append(f"| {NAMES.get(name, name)} | `{r['run_id']}` | `{(r['git_commit'] or '')[:7]}` | {r['git_dirty']} | "
                     f"`{r['exploratory_v3_config_hash'][:12]}` | {r.get('gpu_type') or r.get('device')} |")
    commits = sorted({(r["git_commit"] or "")[:7] for r in v3_runs.values()})
    check = {"v3_runs": len(v3_runs), "commits": commits, "all_clean": all(r["git_dirty"] is False for r in v3_runs.values()),
             "v3_hashes": sorted({r["exploratory_v3_config_hash"][:12] for r in v3_runs.values()})}
    return lines, check


def main() -> None:
    cfg = load_exploratory_v3_config()
    v3 = cfg["exploratory_v3"]
    root = Path(cfg["results_root"])
    figures = Path(cfg["data_config"]["paths"]["reports_root"]) / "figures"
    res = json.loads((root / "exploratory_v3.json").read_text())
    if not res["tests"].get("members"):
        raise SystemExit(f"v3 tests are not complete: pending {res['tests'].get('pending')}")
    members = res["tests"]["members"]
    by = {m["test"]: m for m in members}
    s, sec, rep = res["summary"], res["secondary"], res["reported"]
    a1 = v3["a1_har_iv"]["interpretation"]
    a1_sentence = a1["reject"] if by["har_iv_plus_chain_vs_har_iv"]["reject_holm"] else a1["not_reject"]
    fallback = res["har_iv_fallback"]

    def gain(name: str) -> str:
        m = by[name]
        return (f"{m['mean_diff']:+.3f} IC (95% CI {interval(m['ci95'])}, Holm p {p_text(m['p_holm'])}, "
                f"{'rejects' if m['reject_holm'] else 'does not reject'}; {m['a_wins']}/{m['walks']} walks)")

    no_iv = {p: sec.get(f"{p}_vs_{p.removesuffix('_no_iv')}", {}).get("ic", {}) for p in ("residual_deepsets_no_iv", "residual_attention_no_iv")}
    budget = {h: sec.get(f"e3_patience8_vs_patience3_{h}", {}).get("ic", {}) for h in ("1h", "2h")}
    rejected = [m["test"] for m in members if m["reject_holm"]]
    summary = [
        f"{len(rejected)} of {len(members)} registered v3 tests reject at 5% after Holm.",
        f"A1. HAR-IV (HAR plus log ATM IV) vs HAR: {gain('har_iv_vs_har')}. {fallback['rows_without_extra_columns']} of {fallback['test_rows']} "
        "test rows have no ATM IV and use plain HAR.",
        f"A1. HAR-IV + chain correction vs HAR-IV: {gain('har_iv_plus_chain_vs_har_iv')}. **{a1_sentence}**",
        f"A2. Token corrections without the IV field, vs HAR: DeepSets {gain('residual_deepsets_no_iv_vs_har')}; attention "
        f"{gain('residual_attention_no_iv_vs_har')}. Change from dropping IV (vs the E1 versions with IV, unadjusted): DeepSets "
        f"{f(no_iv['residual_deepsets_no_iv'].get('mean_diff'), '+.3f')} (95% CI {interval(no_iv['residual_deepsets_no_iv'].get('ci95'))}), "
        f"attention {f(no_iv['residual_attention_no_iv'].get('mean_diff'), '+.3f')} (95% CI {interval(no_iv['residual_attention_no_iv'].get('ci95'))}). "
        "Gamma, delta and strike GEX still depend on IV, so IV's influence is not fully removed; A1 is the separating test.",
        f"A3. Combination vs tuned LightGBM: {gain('combination_vs_lightgbm')}; vs HAR-IV: {gain('combination_vs_har_iv')}.",
        f"A4. E3 attention + GRU with patience 8 and at most 60 epochs vs the best classical rung: 1 h ({NAMES.get(res['best_classical']['1h'])}) "
        f"{gain('e3_attention_gru_patience8_vs_best_classical_1h')}; 2 h ({NAMES.get(res['best_classical']['2h'])}) "
        f"{gain('e3_attention_gru_patience8_vs_best_classical_2h')}. Against the patience-3 runs (unadjusted): 1 h "
        f"{f(budget['1h'].get('mean_diff'), '+.3f')} (95% CI {interval(budget['1h'].get('ci95'))}), 2 h {f(budget['2h'].get('mean_diff'), '+.3f')} "
        f"(95% CI {interval(budget['2h'].get('ci95'))}); mean stopping epoch "
        + ", ".join(f"{h}: patience 3 {e['attention_gru_patience3']['stop_epoch_mean']:.1f}, patience 8 {e['attention_gru_patience8']['stop_epoch_mean']:.1f}"
                    for h, e in res["stop_epochs"].items() if "attention_gru_patience8" in e) + ".",
    ]
    primary = ["| Test | Mean A | Mean B | IC diff | 95% interval | p | Holm p | Holm reject | Walks won |", "|---|---|---|---|---|---|---|---|---|"]
    for m in members:
        primary.append(f"| {TEST_LABELS[m['test']]} | {f(m['mean_a'])} | {f(m['mean_b'])} | {f(m['mean_diff'], '+.3f')} | {interval(m['ci95'])} | "
                       f"{p_text(m['p'])} | {p_text(m['p_holm'])} | {'yes' if m['reject_holm'] else 'no'} | {m['a_wins']}/{m['walks']} |")
    losses = ["| Comparison | RMSE diff | QLIKE diff | DM p (squared) | DM p (QLIKE) |", "|---|---|---|---|---|"]
    for name, v in sec.items():
        label = TEST_LABELS.get(name) or SECONDARY_LABELS.get(name, name)
        losses.append(f"| {label} | {f(v['rmse'].get('mean_diff'), '+.4f')} | {f(v['qlike'].get('mean_diff'), '+.4f')} | "
                      f"{p_text(v['dm_squared'].get('p_two_sided'))} | {p_text(v['dm_qlike'].get('p_two_sided'))} |")

    b7 = rep["b7"]
    gex = ["| Model | IC | IC diff vs HAR | 95% interval | p (unadjusted) | Walks won |", "|---|---|---|---|---|---|"]
    for key, v in b7.items():
        model = key.removesuffix("_vs_har")
        gex.append(f"| {NAMES[model]} | {f(v['ic'].get('mean_a'))} | {f(v['ic'].get('mean_diff'), '+.3f')} | {interval(v['ic'].get('ci95'))} | "
                   f"{p_text(v['ic'].get('p_two_sided'))} | {v['ic'].get('a_wins')}/{v['ic'].get('walks')} |")
    main_gex = b7["har_gex_vs_har"]["ic"]
    gex_answer = ("Yes" if main_gex["ci95"][0] > 0 else "No" if main_gex["ci95"][1] < 0 else "Not detectably")

    b8 = rep["b8"]
    serious = {k: v for k, v in b8.items() if k != "persistence"}
    lo_rule, hi_rule = min(v["ruled_out_above"] for v in serious.values()), max(v["ruled_out_above"] for v in serious.values())
    lo_mde, hi_mde = min(v["detectable_at_power"] for v in serious.values()), max(v["detectable_at_power"] for v in serious.values())
    detect = ["| Attention + GRU vs | IC gain | 95% interval | Ruled out above | Detectable with 80% power |", "|---|---|---|---|---|"]
    detect += [f"| {NAMES[k]} | {v['mean_diff']:+.3f} | {interval(v['ci95'])} | {v['ruled_out_above']:+.3f} | {v['detectable_at_power']:.3f} |"
               for k, v in b8.items()]
    detect_sentence = (f"Against HAR, HAR + GEX and tuned LightGBM, IC gains larger than about {hi_rule:+.2f} can be ruled out (upper ends of the "
                       f"95% intervals, {lo_rule:+.3f} to {hi_rule:+.3f}); smaller gains cannot. With {b8['har']['walks']} walks the paired test detects gains of "
                       f"about {lo_mde:.3f}–{hi_mde:.3f} with 80% power.")

    b6 = rep["b6"]
    state = ["| Model | IC | RMSE | QLIKE |", "|---|---|---|---|"]
    state += [f"| {NAMES[m]} | {f(v['ic'])} | {f(v['rmse'], '.4f')} | {f(v['qlike'], '.4f')} |" for m, v in b6.items()]
    state_tests = ["| State-only MLP vs | QLIKE diff | 95% interval | p | RMSE diff | p |", "|---|---|---|---|---|---|"]
    for k, v in rep["b6_tests"].items():
        other = k.removeprefix("state_mlp_vs_")
        state_tests.append(f"| {NAMES[other]} | {f(v['qlike'].get('mean_diff'), '+.4f')} | {interval(v['qlike'].get('ci95'))} | "
                           f"{p_text(v['qlike'].get('p_two_sided'))} | {f(v['rmse'].get('mean_diff'), '+.4f')} | {p_text(v['rmse'].get('p_two_sided'))} |")
    neural_q = [b6[m]["qlike"] for m in ("deepsets_hpc", "attention_hpc", "attention_gru_hpc")]
    q_state, q_classical = b6["state_mlp"]["qlike"], min(b6["har"]["qlike"], b6["lightgbm"]["qlike"])
    token_better = [k for k, v in rep["b6_tests"].items() if v["qlike"].get("ci95") and v["qlike"]["ci95"][0] > 0]   # state worse, significantly
    reading = ("the state-only model shares the neural QLIKE advantage and no token model has a significantly lower QLIKE than it (paired "
               "per-walk tests below), so the advantage comes mainly from the volatility state and the training setup (Huber loss, network), "
               "not detectably from the tokens"
               if q_state < q_classical and not token_better else
               "the tokens add a significant part of the QLIKE advantage for " + ", ".join(NAMES[k.removeprefix('state_mlp_vs_')] for k in token_better)
               if q_state < q_classical else "the state-only model does not share the neural QLIKE advantage")
    state_sentence = (f"The state-only MLP, which sees no tokens, has QLIKE {q_state:.3f}; the neural models with tokens {min(neural_q):.3f}–{max(neural_q):.3f}; "
                      f"HAR {b6['har']['qlike']:.3f} and LightGBM {b6['lightgbm']['qlike']:.3f} (lower is better): {reading}.")

    cond = rep["b5"]
    models = [m for m in cond if m in NAMES]
    conditional = ["| Model | " + " | ".join(SUBSETS[s_] for s_ in SUBSETS) + " |", "|---|" + "---|" * len(SUBSETS)]
    for m in models:
        cells = []
        for key in SUBSETS:
            c = cond[m].get(key, {})
            cells.append(f"{f(c.get('ic'), '.2f')} ({f((c.get('ci95') or [None, None])[0], '.2f')}–{f((c.get('ci95') or [None, None])[1], '.2f')})")
        conditional.append(f"| {NAMES[m]} | " + " | ".join(cells) + " |")
    rows_line = "Rows per subset (test days): " + ", ".join(f"{SUBSETS[k]} {cond['har'][k]['rows']}" for k in SUBSETS) + "."

    prov, check = provenance_rows(res["runs"], None)
    path = figure_tests(members, figures)
    lines = ["# Exploratory v3: checks A1–A4 and reported results B5–B8", "",
             "> **Exploratory.** Registered in `configs/eval_exploratory_v3.yaml` before any of its runs; separate from the pre-registered "
             "RQ1–RQ3 (`configs/eval.yaml`) and from E1–E3 (`configs/eval_exploratory_v2.yaml`), neither of which changed.",
             f"> v3 config hash `{res['exploratory_v3_config_hash'][:12]}`; main eval hash `{res['eval_config_hash'][:12]}`; v2 hash "
             f"`{res['exploratory_config_hash'][:12]}`. B5–B8 use runs that existed (or numbers seen) before registration, so they are descriptive: "
             "no Holm family. The A3 test against LightGBM uses existing runs only and was first computed during a code check after "
             "registration, before the other v3 runs.", "",
             "Generated by `scripts/report_exploratory_v3.py`.", "",
             "## Summary", "", *[f"- {line}" for line in summary], "",
             "## Primary tests (Holm across the eight)", "", *primary, "", f"![{path.stem}](../{path.relative_to(REPO_ROOT)})", "",
             "## Losses and Diebold–Mariano (A minus B; negative is better; unadjusted)", "", *losses, "",
             "## B7. Does GEX predict next-day volatility beyond recent volatility?", "",
             f"**{gex_answer}** (HAR + GEX vs HAR, {main_gex['mean_diff']:+.3f} IC, 95% CI {interval(main_gex['ci95'])}). "
             "By sign convention (ridge, alpha chosen on the inner split, exploratory):", "", *gex, "",
             "## B8. Detectable effect for RQ1", "", detect_sentence, "", *detect, "",
             "## B6. Is the neural QLIKE advantage from the tokens?", "", state_sentence, "", *state, "", *state_tests, "",
             "## B5. Conditional IC (pooled IC per subset, bootstrap 95% interval over walks)", "",
             "As registered in v2 `e1.conditional` (2,000 resamples of walks). Each cell: IC (95% interval).", "", *conditional, "", rows_line, "",
             "## Runs", "", f"v3 runs: {check['v3_runs']}; commits {', '.join(f'`{c}`' for c in check['commits'])}; all `git_dirty = false`: "
             f"{check['all_clean']}; v3 config hashes {', '.join(f'`{h}`' for h in check['v3_hashes'])}.", "", *prov, ""]
    out = REPO_ROOT / "docs" / "findings_exploratory_v3.md"
    out.write_text("\n".join(lines))
    write_report(cfg["data_config"], "exploratory_v3", {"tests": members, "a1_interpretation": a1_sentence, "har_iv_fallback": fallback,
                                                        "gex_answer": gex_answer, "detectable": detect_sentence, "state_only": state_sentence,
                                                        "provenance_check": check, "exploratory_v3_config_hash": res["exploratory_v3_config_hash"],
                                                        "figure": str(path.relative_to(REPO_ROOT))})
    print(out)


if __name__ == "__main__":
    main()
