"""Render docs/findings_t3.md from the results store (all numbers come from stored runs)."""
from __future__ import annotations

import json
from pathlib import Path

from niftygex.config import REPO_ROOT, load_config
from niftygex_eval.config import load_eval_config


def f(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def table(rows: list[dict], columns: list[tuple[str, str]], digits: int = 3) -> str:
    out = ["| " + " | ".join(label for _, label in columns) + " |", "|" + "---|" * len(columns)]
    out += ["| " + " | ".join(f(row.get(key), digits) for key, _ in columns) + " |" for row in rows]
    return "\n".join(out)


def comparison_rows(listed: dict, placebo: dict, slices: list[str]) -> list[dict]:
    rows = []
    for name in slices:
        mirror = placebo[name]
        rows.append({"slice": name, "comparator": "matched-distance mirror", "days": mirror["days"], "max_gex": mirror["hit_rate_actual"],
                     "comparator_rate": mirror["hit_rate_mirror"], "p": mirror["actual_vs_mirror_sign_test"]["p_value_two_sided"]})
        for key, label in (("snapped_mirror", "snapped mirror (listed)"), ("matched_rank", "distance-matched GEX rank 2–5"),
                           ("matched_rank_within_gap", "rank 2–5 within distance gap")):
            block = listed[name][key]
            if not block.get("days"):
                continue
            rows.append({"slice": name, "comparator": label, "days": block["days"], "max_gex": block["hit_rate_max_gex"],
                         "comparator_rate": block["hit_rate_comparator"], "p": block["binary_sign_test"]["p_two_sided"],
                         "diff_bps": block["continuous"]["mean_difference_bps"], "p_cont": block["continuous"]["paired_t_p"]})
    return rows


def reading_lines(placebo: dict, listed: dict, f3: dict) -> list[str]:
    """Narrative bullets with every number taken from the stored results."""
    def member(text: str) -> dict:
        return next(m for m in f3["members"] if m["test"] == text)

    o_m, e_m = member("overall: max-GEX vs matched-distance mirror"), member("expiry_days: max-GEX vs matched-distance mirror")
    o_s, e_s = member("overall: max-GEX vs snapped mirror"), member("expiry_days: max-GEX vs snapped mirror")
    o_r = listed["overall"]["matched_rank"]
    o_g, e_g = listed["overall"]["matched_rank_within_gap"], listed["expiry_days"]["matched_rank_within_gap"]
    close = listed["by_start_distance_pct"][0]
    close_placebo = placebo["by_start_distance_pct"][0]
    return [
        f"- **No attraction to the max-GEX strike against mirror comparators.** Overall and on expiry days, the max-GEX strike is not hit more often "
        f"than the matched-distance mirror (Holm p {f(o_m['p_holm'])} and {f(e_m['p_holm'])}) or the snapped listed mirror "
        f"(Holm p {f(o_s['p_holm'])} and {f(e_s['p_holm'])}).",
        f"- **The rank 2–5 rejection is not evidence for or against pinning.** The distance-matched GEX rank 2–5 comparison rejects under Holm, "
        f"but in the *opposite* direction (max-GEX hit less often). The match is loose: median comparator distance is "
        f"{f(o_r['median_comparator_distance_pct'])}% of spot against {f(o_r['median_max_gex_distance_pct'])}% for the max-GEX strike, and "
        "farther strikes are overshot less.",
        f"- **When the match is tight, the difference disappears.** Restricted to matches within {listed['max_gap_pct']}% of spot, hit rates are "
        f"{f(o_g['hit_rate_max_gex'])} vs {f(o_g['hit_rate_comparator'])} overall (p {f(o_g['binary_sign_test']['p_two_sided'])}, {o_g['days']} days) "
        f"and {f(e_g['hit_rate_max_gex'])} vs {f(e_g['hit_rate_comparator'])} on expiry days (p {f(e_g['binary_sign_test']['p_two_sided'])}). "
        "As defined, the loose-match comparator in F3 is distance-confounded; the tight-match variant is the valid listed-strike test.",
        f"- **The close-strike effect is round-number attraction, not gamma.** For strikes within {close['bucket']}% of spot, the max-GEX strike beat "
        f"the unsnapped mirror ({f(close_placebo['hit_rate_actual'])} vs {f(close_placebo['hit_rate_mirror'])}, "
        f"p {f(close_placebo['actual_vs_mirror_sign_test']['p_value_two_sided'])}). Against the snapped listed mirror it does not "
        f"(p {f(close['snapped_mirror']['binary_sign_test']['p_two_sided'])}, {close['snapped_mirror']['days']} days). The mirror only looked "
        "worse because it is not a round strike.",
    ]


def main() -> None:
    cfg = load_eval_config()
    root = Path(cfg["results_root"])
    reports = Path(load_config()["paths"]["reports_root"])
    placebo = json.loads((root / "t3_placebo.json").read_text())
    listed = json.loads((root / "t3_listed_placebo.json").read_text())
    holm = json.loads((root / "multiple_testing.json").read_text())
    target = json.loads((reports / "targets_t3_report.json").read_text())
    f3 = next(fam for fam in holm["families"] if fam["family"] == "F3")
    f3c = next(fam for fam in holm["families"] if fam["family"] == "F3-secondary-continuous")
    columns = [("slice", "slice"), ("comparator", "comparator"), ("days", "days"), ("max_gex", "hit rate max-GEX"),
               ("comparator_rate", "hit rate comparator"), ("p", "sign-test p"), ("diff_bps", "Δ reduction (bps)"), ("p_cont", "paired-t p")]
    lines = [
        "# T3 findings: does the index drift toward the max-GEX strike late in the session?",
        "",
        "Generated by `scripts/write_findings_t3.py` from the results store; every number comes from a stored run "
        f"(placebo `{placebo['run']}`, listed-strike placebo `{listed['run']}`).",
        "",
        "**Target.** T3 asks whether the index at 15:30 (the close of the 15:25 bar) is closer to the 14:00 snapshot's max-GEX strike than the "
        "14:00 spot was. Truncated sessions are excluded.",
        "",
        f"- **Labeled days:** {target['rows'] - sum(v for k, v in target['reason_counts'].items() if k != 'ok')}.",
        f"- **Share that moved toward the strike:** {f(target['label_share_toward_overall'])}.",
        f"- **Median start distance:** {f(target['start_distance_pct_median'])}% of spot.",
        "",
        "## Why a raw hit rate is not evidence",
        "",
        "The max-GEX strike usually sits very close to spot, so a 90-minute move often overshoots it. The no-effect hit rate is therefore well "
        "below 50% and depends on distance. Every test below compares the max-GEX strike with a comparator on the same days and the same price "
        "path. Only the strike differs.",
        "",
        "The comparators:",
        "",
        "- **Matched-distance mirror:** a strike at the same distance on the other side of spot. It is not a listed strike.",
        "- **Snapped mirror:** the mirror moved to the nearest listed strike. Days where it lands on the max-GEX strike are dropped.",
        "- **Distance-matched GEX rank 2–5:** the listed strike, among the 2nd- to 5th-largest GEX strikes, whose distance from spot best matches "
        "the max-GEX strike's.",
        "",
        "The listed comparators remove a round-number explanation, because every strike they use is also a listed round strike.",
        "",
        "## Primary family F3 (Holm-adjusted)",
        "",
        table(f3["members"], [("test", "test"), ("estimate", "hit-rate difference"), ("p", "p"), ("p_holm", "Holm p"), ("reject_holm", "reject at 5%")]),
        "",
        "Secondary: the continuous version (distance reduction in bps, paired t-test):",
        "",
        table(f3c["members"], [("test", "test"), ("estimate", "mean difference (bps)"), ("p", "p"), ("p_holm", "Holm p"), ("reject_holm", "reject at 5%")]),
        "",
        "## All comparisons for the primary slices",
        "",
        table(comparison_rows(listed, placebo, ["overall", "expiry_days"]), columns),
        "",
        "## Exploratory slices (unadjusted)",
        "",
        "By start distance (listed comparators):",
        "",
        table([{"bucket": b["bucket"], "days": b["days"], "max_gex": b["matched_rank"].get("hit_rate_max_gex"),
                "rank_rate": b["matched_rank"].get("hit_rate_comparator"), "rank_p": b["matched_rank"].get("binary_sign_test", {}).get("p_two_sided"),
                "snap_days": b["snapped_mirror"].get("days"), "snap_rate": b["snapped_mirror"].get("hit_rate_comparator"),
                "snap_p": b["snapped_mirror"].get("binary_sign_test", {}).get("p_two_sided")} for b in listed["by_start_distance_pct"]],
              [("bucket", "start distance % of spot"), ("days", "days"), ("max_gex", "max-GEX"), ("rank_rate", "rank 2–5"), ("rank_p", "p"),
               ("snap_days", "snapped days"), ("snap_rate", "snapped mirror"), ("snap_p", "p")]),
        "",
        "By start distance (matched-distance placebo, unsnapped mirror):",
        "",
        table([{"bucket": b["bucket"], "days": b["days"], "actual": b["hit_rate_actual"], "mirror": b["hit_rate_mirror"],
                "p": b["actual_vs_mirror_sign_test"]["p_value_two_sided"]} for b in placebo["by_start_distance_pct"]],
              [("bucket", "start distance % of spot"), ("days", "days"), ("actual", "max-GEX"), ("mirror", "mirror"), ("p", "p")]),
        "",
        "Hit rate by GEX rank (overall; ranks 2–5 sit at different distances, so these rates are not directly comparable):",
        "",
        table(listed["overall"]["per_rank"], [("gex_rank", "GEX rank"), ("days", "days"), ("hit_rate", "hit rate"),
                                               ("mean_reduction_bps", "mean reduction (bps)"), ("median_distance_pct", "median distance %")]),
        "",
        "## Reading",
        "",
        *reading_lines(placebo, listed, f3),
        "",
    ]
    out = REPO_ROOT / "docs" / "findings_t3.md"
    out.write_text("\n".join(lines))
    print(out)


if __name__ == "__main__":
    main()
