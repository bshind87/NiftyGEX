"""SHAP figures for the T2 ablation set C: mean |SHAP| ranking and dependence plots split by days to expiry and ATM IV."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from niftygex_eval.config import load_eval_config  # noqa: E402

# Reference palette (dataviz skill): light surface, text tokens, ordinal blue ramp steps 250/450/650 (validated --ordinal).
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ORDINAL = ["#86b6ef", "#2a78d6", "#104281"]
SERIES = "#2a78d6"
DTE_BINS = [(1, 1, "1 day"), (2, 3, "2–3 days"), (4, 99, "4+ days")]


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9)


def latest_run(root: Path, stage: str, model: str) -> str:
    run = None
    for line in (root / "index.jsonl").read_text().splitlines():
        record = json.loads(line)
        if record["stage"] == stage and record["model"] == model:
            run = record["run_id"]
    return run


def binned(x: np.ndarray, y: np.ndarray, bins: int = 8) -> tuple[np.ndarray, np.ndarray]:
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    centers, means = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (x >= lo) & (x <= hi)
        if mask.sum() >= 10:
            centers.append(np.median(x[mask]))
            means.append(y[mask].mean())
    return np.array(centers), np.array(means)


def dependence_figure(shap: pl.DataFrame, feature: str, output: str, title: str, run: str, path: Path) -> None:
    rows = shap.filter(pl.col("output") == output)
    target = rows.filter(pl.col("feature") == feature).select("walk_id", "snapshot_at", pl.col("value").alias("x"), "shap")
    context = rows.filter(pl.col("feature").is_in(["days_to_expiry", "atm_iv"])).pivot(on="feature", index=["walk_id", "snapshot_at"], values="value")
    # Missing feature values reach SHAP as NaN; treat them as null so they never fall into a bucket.
    data = (target.join(context, on=["walk_id", "snapshot_at"])
            .with_columns(*[pl.col(c).fill_nan(None) for c in ("x", "days_to_expiry", "atm_iv")])
            .drop_nulls(["x", "shap"]))
    iv_cuts = np.nanquantile(data["atm_iv"].drop_nulls().to_numpy(), [1 / 3, 2 / 3])
    splits = {
        "Days to expiry": [(label, (pl.col("days_to_expiry") >= lo) & (pl.col("days_to_expiry") <= hi)) for lo, hi, label in DTE_BINS],
        "ATM IV tercile": [("low IV", pl.col("atm_iv") <= iv_cuts[0]),
                           ("mid IV", (pl.col("atm_iv") > iv_cuts[0]) & (pl.col("atm_iv") <= iv_cuts[1])),
                           ("high IV", pl.col("atm_iv") > iv_cuts[1])],
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, facecolor=SURFACE)
    for ax, (split_name, groups) in zip(axes, splits.items()):
        style(ax)
        for color, (label, mask) in zip(ORDINAL, groups):
            part = data.filter(mask)
            if part.height < 20:
                continue
            x, y = part["x"].to_numpy(), part["shap"].to_numpy()
            ax.scatter(x, y, s=9, color=color, alpha=0.25, linewidths=0, edgecolors="none")
            cx, cy = binned(x, y)
            ax.plot(cx, cy, color=color, linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5,
                    label=f"{label} (n={part.height})")
        ax.axhline(0, color=TEXT_2, linewidth=0.8)
        ax.set_title(f"by {split_name.lower()}", color=TEXT, fontsize=10, loc="left")
        ax.set_xlabel(feature, color=TEXT_2, fontsize=9)
        legend = ax.legend(frameon=False, fontsize=8, loc="best")
        for text in legend.get_texts():
            text.set_color(TEXT)
    axes[0].set_ylabel(f"SHAP ({output}, raw margin)", color=TEXT_2, fontsize=9)
    fig.suptitle(title, color=TEXT, fontsize=11, x=0.01, ha="left")
    fig.text(0.01, 0.005, f"run {run}; lines = binned mean SHAP; dots = test rows", color=TEXT_2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def ranking_figure(summary: pl.DataFrame, output: str, highlight: set[str], title: str, run: str, path: Path) -> None:
    rows = summary.filter(pl.col("output") == output).sort("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(7, 0.28 * rows.height + 1.2), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="y", visible=False)
    labels = [f"{f} *" if f in highlight else f for f in rows["feature"].to_list()]
    ax.barh(labels, rows["mean_abs_shap"].to_numpy(), color=SERIES, height=0.62, edgecolor=SURFACE, linewidth=2)
    ax.set_xlabel("mean |SHAP| (per walk, averaged across walks)", color=TEXT_2, fontsize=9)
    ax.set_title(title, color=TEXT, fontsize=11, loc="left")
    ax.text(0.99, 0.01, "* GEX / levels feature", transform=ax.transAxes, ha="right", va="bottom", color=TEXT_2, fontsize=8)
    fig.text(0.01, 0.005, f"run {run}", color=TEXT_2, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    cfg = load_eval_config()
    root = Path(cfg["results_root"])
    stage = cfg["eval"]["stage_b"]
    shap_set = stage["shap"]["feature_set"]
    gex_features = set(stage["feature_sets"]["C_adds"])
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for task, output in (("classification", "high"), ("regression", "value")):
        run = latest_run(root, "t2_ablation", f"{task}_{shap_set}")
        shap = pl.read_parquet(root / "runs" / run / "shap.parquet")
        summary = pl.read_parquet(root / "runs" / run / "shap_summary.parquet")
        label = "high-vol tercile" if task == "classification" else "log RV ratio"
        path = out / f"shap_ranking_{task}.png"
        ranking_figure(summary, "all_outputs" if task == "classification" else "value", gex_features,
                       f"T2 {task}, set {shap_set}: mean |SHAP|", run, path)
        written.append(path)
        top = (summary.filter((pl.col("output") == ("all_outputs" if task == "classification" else "value")) & pl.col("feature").is_in(list(gex_features)))
               .sort("mean_abs_shap", descending=True).head(3)["feature"].to_list())
        for feature in top:
            path = out / f"shap_dependence_{task}_{feature}.png"
            dependence_figure(shap, feature, output, f"T2 {task} ({label}): SHAP of {feature} vs its value", run, path)
            written.append(path)
    (root / "figures" / "index.json").write_text(json.dumps([str(p.relative_to(root)) for p in written], indent=2) + "\n")
    print("\n".join(str(p) for p in written))


if __name__ == "__main__":
    main()
