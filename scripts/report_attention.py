"""Attention maps (exploratory): what the pooling attention of self-attention and attention + GRU looks at.

Reads the pooling weights stored by ``run_neural.py --save-attention`` (table ``attention`` in each run of
outputs/results/neural_comparison.json) and joins them to the token history on (prediction snapshot, lag, token_index).
A contract's relative weight is its weight times the number of real contracts in that snapshot: 1 means a plain
average. Only the pooling step is stored, not the contract-to-contract attention inside the encoder layers.

Writes outputs/reports/attention.json, figures outputs/reports/figures/attention_*.png and docs/findings_attention.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from niftygex.config import REPO_ROOT
from niftygex.io.parquet import write_report
from niftygex_eval.config import load_eval_config
from niftygex_eval.figures import GRID, SERIES, TEXT_2, label_line_end, save

MODELS = {"attention": "Self-attention", "attention_gru": "Attention + GRU"}
COLORS = {"attention": SERIES[0], "attention_gru": SERIES[1]}
# Token fields compared with the relative weight inside each prediction snapshot (Spearman, then averaged over days).
FIELDS = {"log_moneyness": "log-moneyness", "abs_log_moneyness": "|log-moneyness|", "is_call": "call (vs put)", "iv": "IV",
          "gamma": "gamma", "abs_strike_gex": "|strike GEX|", "open_interest_contracts": "open interest", "volume_today": "volume today",
          "oi_change_since_previous_close_pct": "OI change since previous close, %"}
CONCENTRATED = 0.9   # a snapshot is concentrated when effective contracts / real contracts falls below this
MONEYNESS_BIN, MONEYNESS_RANGE = 0.01, 0.06


def load_weights(root: Path, runs: dict[str, str], tokens_root: Path) -> dict[str, pl.DataFrame]:
    """Lag-0 and history weights per model, joined to the token fields."""
    fields = ["log_moneyness", "is_call", "iv", "gamma", "strike_gex_absolute_crore", "open_interest_contracts", "volume_today", "strike_price",
              "oi_change_since_previous_close_pct"]
    tokens = (pl.scan_parquet(tokens_root / "*.parquet").select("prediction_at", "lag", "token_index", *fields).collect()
              .with_columns(pl.col("log_moneyness").abs().alias("abs_log_moneyness"), pl.col("strike_gex_absolute_crore").abs().alias("abs_strike_gex")))
    out = {}
    for name in MODELS:
        weights = pl.read_parquet(root / "runs" / runs[name] / "attention.parquet")
        joined = weights.join(tokens, left_on=["snapshot_at", "lag", "token_index"], right_on=["prediction_at", "lag", "token_index"], how="inner")
        if joined.height != weights.height:
            raise SystemExit(f"{name}: {weights.height - joined.height} attention rows have no matching token")
        out[name] = joined.with_columns((pl.col("weight") * pl.len().over("snapshot_at", "lag")).alias("relative"))
    return out


def concentration(frame: pl.DataFrame) -> pl.DataFrame:
    """Per (prediction snapshot, lag): real contracts, effective contracts share and the largest relative weight."""
    heads = [c for c in frame.columns if c.startswith("weight_head_")]
    return frame.group_by("snapshot_at", "lag").agg(
        pl.len().alias("contracts"), ((1 / (pl.col("weight") ** 2).sum()) / pl.len()).alias("effective_share"),
        pl.col("relative").max().alias("max_relative"),
        *[((1 / (pl.col(h) ** 2).sum()) / pl.len()).alias(f"effective_share_{h.removeprefix('weight_')}") for h in heads]).sort("snapshot_at", "lag")


def top_contracts(lag0: pl.DataFrame, days: pl.Series) -> pl.DataFrame:
    """The most-weighted contract on each given prediction snapshot, with its OI-change rank within that day."""
    ranked = lag0.filter(pl.col("snapshot_at").is_in(days.to_list())).with_columns(
        pl.col("oi_change_since_previous_close_pct").rank("min", descending=True).over("snapshot_at").alias("oi_change_rank"))
    return (ranked.sort("relative", descending=True).group_by("snapshot_at", maintain_order=True).first()
            .select("snapshot_at", "relative", "strike_price", "is_call", "log_moneyness", "oi_change_since_previous_close_pct", "oi_change_rank",
                    "volume_today", "open_interest_contracts").sort("snapshot_at"))


def field_correlations(lag0: pl.DataFrame) -> dict[str, float]:
    """Mean within-snapshot Spearman correlation of the relative weight with each token field."""
    out = {}
    for column in FIELDS:
        per_day = (lag0.filter(pl.col(column).is_not_null() & pl.col(column).is_finite()).group_by("snapshot_at")
                   .agg(pl.corr(pl.col(column).cast(pl.Float64), "relative", method="spearman").alias("rho"), pl.len().alias("n"))
                   .filter((pl.col("n") > 5) & pl.col("rho").is_not_nan()))
        out[column] = float(per_day["rho"].mean())
    return out


def moneyness_profile(lag0: pl.DataFrame) -> pl.DataFrame:
    """Mean relative weight by log-moneyness bin, calls and puts separately, as a % deviation from a plain average."""
    return (lag0.filter(pl.col("log_moneyness").abs() <= MONEYNESS_RANGE + MONEYNESS_BIN / 2)
            .with_columns(((pl.col("log_moneyness") / MONEYNESS_BIN).round() * MONEYNESS_BIN).round(4).alias("bin"))
            .group_by("bin", "is_call").agg(((pl.col("relative").mean() - 1) * 100).alias("deviation_pct"), pl.len().alias("tokens"))
            .sort("is_call", "bin"))


def figure_concentration(stats: dict[str, pl.DataFrame], directory: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for name, frame in stats.items():
        day = frame.filter(pl.col("lag") == 0)
        ax.plot(day["snapshot_at"].dt.date().to_list(), day["max_relative"].to_list(), "o", markersize=3.5, color=COLORS[name], alpha=0.8, label=MODELS[name])
    ax.axhline(1, color=TEXT_2, linewidth=0.8)
    ax.annotate("1 = plain average", (0.005, 1), xycoords=("axes fraction", "data"), xytext=(0, 4), textcoords="offset points", fontsize=8.5, color=TEXT_2)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylabel("largest relative weight (log)")
    ax.set_title("Pooling attention is a plain average on almost every test day (15:15 snapshot)")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.92))
    return save(fig, directory, "attention_concentration")


def figure_moneyness(profiles: dict[str, pl.DataFrame], directory: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, (name, profile) in zip(axes, profiles.items()):
        for color, is_call, label in ((SERIES[0], 1, "calls"), (SERIES[1], 0, "puts")):
            side = profile.filter(pl.col("is_call") == is_call)
            ax.plot(side["bin"], side["deviation_pct"], "-o", markersize=5, color=color, label=label)
            label_line_end(ax, side["bin"][-1], side["deviation_pct"][-1], label)
        ax.axhline(0, color=TEXT_2, linewidth=0.8)
        ax.axvline(0, color=GRID, linewidth=0.8)
        ax.set_title(MODELS[name])
        ax.set_xlabel("log-moneyness, log(K / F)")
    axes[0].set_ylabel("weight vs plain average, %")
    axes[0].legend(loc="lower left")
    largest = max(float(p["deviation_pct"].abs().max()) for p in profiles.values())
    fig.suptitle(f"Mean pooling weight across the chain: deviations stay within {largest:.1f}%", x=0.01, ha="left", fontsize=11)
    return save(fig, directory, "attention_moneyness")


def figure_fields(correlations: dict[str, dict[str, float]], directory: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    rows = list(FIELDS)
    for offset, name in zip((-0.15, 0.15), correlations):
        values = [correlations[name][c] for c in rows]
        ax.plot(values, np.arange(len(rows)) + offset, "o", markersize=8, color=COLORS[name], markeredgecolor="white", markeredgewidth=2, label=MODELS[name])
    ax.axvline(0, color=TEXT_2, linewidth=0.8)
    ax.set_yticks(range(len(rows)), [FIELDS[c] for c in rows])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("mean within-day rank correlation with the relative weight")
    ax.set_title("Which contracts get slightly more weight (15:15 snapshot)")
    ax.legend(loc="lower right")
    return save(fig, directory, "attention_fields")


def figure_case_days(lag0: pl.DataFrame, cases: dict[str, dict], directory: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, (title, case) in zip(axes, cases.items()):
        day = lag0.filter(pl.col("snapshot_at") == case["snapshot_at_dt"])
        for color, is_call, label in ((SERIES[0], 1, "calls"), (SERIES[1], 0, "puts")):
            side = day.filter(pl.col("is_call") == is_call)
            ax.plot(side["log_moneyness"], side["relative"], "o", markersize=5, color=color, label=label)
        ax.axhline(1, color=TEXT_2, linewidth=0.8)
        ax.set_yscale("log")
        ax.set_title(f"{title}: {case['date']} ({case['contracts']} contracts)")
        ax.set_xlabel("log-moneyness, log(K / F)")
    axes[0].set_ylabel("relative weight (log; 1 = average)")
    axes[0].legend(loc="upper left")
    fig.suptitle("Attention + GRU at 15:15: a typical day and the most concentrated day", x=0.01, ha="left", fontsize=11)
    return save(fig, directory, "attention_case_days")


def main() -> None:
    cfg = load_eval_config()
    root = Path(cfg["results_root"])
    data_config = cfg["data_config"]
    neural = json.loads((root / "neural_comparison.json").read_text())
    missing = [m for m in MODELS if m not in neural["runs"]]
    if missing:
        raise SystemExit(f"neural_comparison.json has no run for {missing}; run scripts/run_neural.py --save-attention first")
    figures = Path(data_config["paths"]["reports_root"]) / "figures"
    weights = load_weights(root, neural["runs"], Path(data_config["paths"]["features_root"]) / "tokens")
    stats = {name: concentration(frame) for name, frame in weights.items()}
    lag0 = {name: frame.filter(pl.col("lag") == 0) for name, frame in weights.items()}

    summary = {}
    for name, frame in stats.items():
        day = frame.filter(pl.col("lag") == 0)
        flagged = day.filter(pl.min_horizontal(pl.col("^effective_share_head_.*$")) < CONCENTRATED)["snapshot_at"]
        tops = top_contracts(lag0[name], flagged)
        summary[name] = {
            "run_id": neural["runs"][name], "prediction_snapshots": day.height, "snapshots_all_lags": frame.height,
            "median_contracts": float(day["contracts"].median()),
            "effective_share": {"p10": float(day["effective_share"].quantile(0.1)), "median": float(day["effective_share"].median())},
            "max_relative": {"median": float(day["max_relative"].median()), "max": float(day["max_relative"].max())},
            "concentrated_days": int(day.filter(pl.col("effective_share") < CONCENTRATED).height),
            "concentrated_days_any_head": flagged.len(),
            "concentrated_top_contracts": tops.with_columns(pl.col("snapshot_at").dt.date().cast(pl.String)).to_dicts(),
            "concentrated_top_is_largest_oi_change": int((tops["oi_change_rank"] == 1).sum()),
            "max_abs_moneyness_deviation_pct": None,
            "field_correlations": field_correlations(lag0[name]),
        }

    # Case days for attention + GRU: the most concentrated 15:15 snapshot and the one closest to the median concentration.
    gru = stats["attention_gru"].filter(pl.col("lag") == 0)
    median = gru["max_relative"].median()
    picks = {"Typical day": gru.with_columns((pl.col("max_relative") - median).abs().alias("gap")).sort("gap", "snapshot_at").row(0, named=True),
             "Most concentrated day": gru.sort("max_relative", descending=True).row(0, named=True)}
    cases = {}
    for title, row in picks.items():
        day = lag0["attention_gru"].filter(pl.col("snapshot_at") == row["snapshot_at"])
        top = day.sort("relative", descending=True).row(0, named=True)
        cases[title] = {"snapshot_at_dt": row["snapshot_at"], "date": str(row["snapshot_at"].date()), "contracts": row["contracts"],
                        "max_relative": row["max_relative"], "effective_share": row["effective_share"],
                        "top_contract": {"strike_price": top["strike_price"], "is_call": top["is_call"], "log_moneyness": top["log_moneyness"],
                                         "volume_today": top["volume_today"], "open_interest_contracts": top["open_interest_contracts"],
                                         "oi_change_since_previous_close_pct": top["oi_change_since_previous_close_pct"],
                                         "oi_change_median_in_day": float(day["oi_change_since_previous_close_pct"].median())}}

    profiles = {name: moneyness_profile(frame) for name, frame in lag0.items()}
    for name, profile in profiles.items():
        summary[name]["max_abs_moneyness_deviation_pct"] = float(profile["deviation_pct"].abs().max())
    paths = {"attention_concentration": figure_concentration(stats, figures),
             "attention_moneyness": figure_moneyness(profiles, figures),
             "attention_fields": figure_fields({n: s["field_correlations"] for n, s in summary.items()}, figures),
             "attention_case_days": figure_case_days(lag0["attention_gru"], cases, figures)}

    gru_s, att_s, concentrated = summary["attention_gru"], summary["attention"], cases["Most concentrated day"]
    top = concentrated["top_contract"]
    side = "call" if top["is_call"] else "put"
    findings = [
        f"Pooling attention is close to a plain average. At 15:15 the median effective share of contracts is {gru_s['effective_share']['median']:.4f} "
        f"for attention + GRU and {att_s['effective_share']['median']:.4f} for self-attention (1 = every contract weighted equally); the median largest "
        f"relative weight is {gru_s['max_relative']['median']:.3f} and {att_s['max_relative']['median']:.3f}.",
        f"Concentrated snapshots (effective share below {CONCENTRATED} in at least one head) are rare: {gru_s['concentrated_days_any_head']} of "
        f"{gru_s['prediction_snapshots']} test days for attention + GRU and {att_s['concentrated_days_any_head']} for self-attention.",
        f"The most concentrated day is {concentrated['date']} ({concentrated['contracts']} contracts): one {side} at log-moneyness "
        f"{top['log_moneyness']:+.3f} gets {concentrated['max_relative']:.1f} times the average weight. Its OI changed by "
        f"{top['oi_change_since_previous_close_pct']:+,.0f}% since the previous close (day median {top['oi_change_median_in_day']:+.1f}%).",
        f"On {gru_s['concentrated_top_is_largest_oi_change']} of the {gru_s['concentrated_days_any_head']} concentrated days for attention + GRU "
        f"(and {att_s['concentrated_top_is_largest_oi_change']} of {att_s['concentrated_days_any_head']} for self-attention), the most-weighted contract "
        "is the day's largest OI change. This field is a percentage and is not log-transformed in the inputs, so a newly opened strike gives an "
        "extreme value that dominates the pooling; the planned field-drop ablation (run_neural.py --drop oi_change_since_previous_close_pct) tests whether it matters.",
        f"Within a day, attention + GRU gives slightly more weight to calls (rank correlation {gru_s['field_correlations']['is_call']:+.2f}) and less to "
        f"high-volume, high-OI and high-|GEX| strikes ({gru_s['field_correlations']['volume_today']:+.2f}, {gru_s['field_correlations']['open_interest_contracts']:+.2f}, "
        f"{gru_s['field_correlations']['abs_strike_gex']:+.2f}), but these tilts are small: binned by log-moneyness, the mean weight differs from "
        f"a plain average by at most {gru_s['max_abs_moneyness_deviation_pct']:.1f}% ({att_s['max_abs_moneyness_deviation_pct']:.1f}% for self-attention).",
        "Reading: the pooling step adds little beyond averaging, which matches the RQ2 result that self-attention does not beat DeepSets. "
        "Interactions inside the encoder layers are not measured here; only the pooling weights are stored.",
    ]
    captions = {
        "attention_concentration": "Largest relative pooling weight on each test day at 15:15 (log scale); 1 is a plain average.",
        "attention_moneyness": "Mean pooling weight by log-moneyness, calls and puts, as a % deviation from a plain average (15:15, all test days).",
        "attention_fields": "Mean within-day rank correlation between a contract's pooling weight and its token fields (15:15, all test days).",
        "attention_case_days": "Pooling weights of attention + GRU across the chain on a typical day and on the most concentrated day.",
    }
    for case in cases.values():
        case.pop("snapshot_at_dt")
    write_report(data_config, "attention", {
        "eval_config_hash": neural["eval_config_hash"], "window": neural["window"], "exploratory": True, "concentrated_threshold": CONCENTRATED,
        "models": summary, "cases": cases, "moneyness_profile": {n: p.to_dicts() for n, p in profiles.items()}, "findings": findings,
        "figures": [{"path": str(p.relative_to(REPO_ROOT)), "caption": captions[k]} for k, p in paths.items()],
    })

    rel = "../" + str(figures.relative_to(REPO_ROOT))
    lines = ["# Attention maps (exploratory)", "",
             "Generated by `scripts/report_attention.py` from the pooling weights stored by `scripts/run_neural.py --save-attention` "
             f"(runs `{att_s['run_id']}`, `{gru_s['run_id']}`; eval config hash `{neural['eval_config_hash'][:12]}`). A contract's relative weight is its "
             "pooling weight times the number of real contracts in that snapshot, so 1 means a plain average. Not part of the pre-registered tests.", "",
             "## Findings", "", *[f"- {line}" for line in findings], ""]
    for key, path in paths.items():
        lines += [f"![{captions[key]}]({rel}/{path.name})", "", f"*{captions[key]}*", ""]
    out = REPO_ROOT / "docs" / "findings_attention.md"
    out.write_text("\n".join(lines))
    print(out)


if __name__ == "__main__":
    main()
