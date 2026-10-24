"""T3 listed-strike placebo and continuous T3: the max-GEX strike vs other listed (round) strikes."""
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from scipy.stats import binomtest, ttest_rel, wilcoxon

RANKS = (2, 3, 4, 5)


def strike_ranks(chain: pl.DataFrame) -> pl.DataFrame:
    """Per (snapshot_at, strike): absolute GEX summed over legs and included expiries, ranked descending (1 = max).

    Ties break toward the strike closer to spot, then the lower strike, as in the levels stage.
    """
    per_strike = chain.group_by("snapshot_at", "strike_price").agg(pl.col("gex_absolute").sum(), pl.col("spot").first())
    return per_strike.sort(["snapshot_at", "gex_absolute", "strike_price"], descending=[False, True, False]).with_columns(
        ((pl.col("strike_price") - pl.col("spot")).abs()).alias("_dist")
    ).sort(["snapshot_at", "gex_absolute", "_dist", "strike_price"], descending=[False, True, False, False]).with_columns(
        pl.int_range(1, pl.len() + 1).over("snapshot_at").alias("gex_rank")
    ).drop("_dist")


def comparators(days: pl.DataFrame, ranks: pl.DataFrame) -> pl.DataFrame:
    """Attach rank-2..5 strikes, the distance-matched rank-2..5 strike, and the mirror snapped to the listed grid."""
    listed = ranks.group_by("snapshot_at").agg(pl.col("strike_price").sort().alias("listed"))
    wide = ranks.filter(pl.col("gex_rank").is_in(RANKS)).pivot(on="gex_rank", index="snapshot_at", values="strike_price")
    wide = wide.rename({str(r): f"rank{r}_strike" for r in RANKS if str(r) in wide.columns})
    rank1 = ranks.filter(pl.col("gex_rank") == 1).select("snapshot_at", pl.col("strike_price").alias("rank1_strike"))
    frame = days.join(rank1, on="snapshot_at", how="left").join(wide, on="snapshot_at", how="left").join(listed, on="snapshot_at", how="left")
    start, strike = frame["t3_start_price"].to_numpy(), frame["t3_strike"].to_numpy()
    actual_distance = np.abs(strike - start)
    matched, gap, snapped = [], [], []
    for i, row in enumerate(frame.iter_rows(named=True)):
        candidates = [(abs(abs(row[f"rank{r}_strike"] - start[i]) - actual_distance[i]), r, row[f"rank{r}_strike"])
                      for r in RANKS if row.get(f"rank{r}_strike") is not None]
        best = min(candidates) if candidates else None
        matched.append(best[2] if best else None)
        gap.append(best[0] if best else None)
        grid = np.asarray(row["listed"] or [], dtype=float)
        mirror = start[i] - (strike[i] - start[i])
        snapped.append(float(grid[np.argmin(np.abs(grid - mirror))]) if grid.size else None)
    return frame.with_columns(
        pl.Series("matched_rank_strike", matched, dtype=pl.Float64),
        pl.Series("matched_rank_distance_gap", gap, dtype=pl.Float64),
        pl.Series("mirror_strike", start - (strike - start)),
        pl.Series("snapped_mirror_strike", snapped, dtype=pl.Float64),
    ).drop("listed")


def hit(start: np.ndarray, end: np.ndarray, strike: np.ndarray) -> np.ndarray:
    return np.abs(end - strike) < np.abs(start - strike)


def reduction_bps(start: np.ndarray, end: np.ndarray, strike: np.ndarray) -> np.ndarray:
    """Distance reduction toward the strike, in basis points of the start price (positive = moved toward)."""
    return (np.abs(start - strike) - np.abs(end - strike)) / start * 1e4


def compare(frame: pl.DataFrame, comparator: str) -> dict[str, Any]:
    """Binary sign test and continuous paired tests: max-GEX strike vs a comparator strike, same days."""
    usable = frame.filter(pl.col(comparator).is_not_null() & (pl.col(comparator) != pl.col("t3_strike")))
    excluded = frame.height - usable.height
    if not usable.height:
        return {"days": 0, "excluded_days": excluded}
    start, end = usable["t3_start_price"].to_numpy(), usable["t3_end_price"].to_numpy()
    actual_k, other_k = usable["t3_strike"].to_numpy(), usable[comparator].to_numpy()
    a, b = hit(start, end, actual_k), hit(start, end, other_k)
    only_a, only_b = int(np.sum(a & ~b)), int(np.sum(b & ~a))
    ra, rb = reduction_bps(start, end, actual_k), reduction_bps(start, end, other_k)
    diff = ra - rb
    return {
        "days": int(usable.height), "excluded_days": excluded,
        "hit_rate_max_gex": float(a.mean()), "hit_rate_comparator": float(b.mean()),
        "binary_sign_test": {"max_gex_only": only_a, "comparator_only": only_b,
                             "p_two_sided": float(binomtest(only_a, only_a + only_b, 0.5).pvalue) if only_a + only_b else None},
        "median_comparator_distance_pct": float(np.median(np.abs(other_k - start) / start * 100)),
        "median_max_gex_distance_pct": float(np.median(np.abs(actual_k - start) / start * 100)),
        "continuous": {
            "mean_reduction_bps_max_gex": float(ra.mean()), "mean_reduction_bps_comparator": float(rb.mean()),
            "mean_difference_bps": float(diff.mean()),
            "paired_t_p": float(ttest_rel(ra, rb).pvalue) if usable.height > 2 else None,
            "wilcoxon_p": float(wilcoxon(diff).pvalue) if np.any(diff != 0) else None,
        },
    }


def per_rank(frame: pl.DataFrame) -> list[dict[str, Any]]:
    out = []
    for r in RANKS:
        column = f"rank{r}_strike"
        part = frame.filter(pl.col(column).is_not_null())
        if not part.height:
            continue
        start, end, k = part["t3_start_price"].to_numpy(), part["t3_end_price"].to_numpy(), part[column].to_numpy()
        out.append({"gex_rank": r, "days": part.height, "hit_rate": float(hit(start, end, k).mean()),
                    "mean_reduction_bps": float(reduction_bps(start, end, k).mean()),
                    "median_distance_pct": float(np.median(np.abs(k - start) / start * 100))})
    return out


def run_listed_placebo(frame: pl.DataFrame, groups: dict[str, pl.Expr], buckets: int, max_gap_pct: float) -> dict[str, Any]:
    """Comparisons per slice: matched rank 2-5 (all days, and within ``max_gap_pct`` distance), and snapped mirror."""
    frame = frame.with_columns(
        (pl.col("matched_rank_distance_gap") / pl.col("t3_start_price") * 100).alias("matched_gap_pct"),
        ((pl.col("t3_strike") - pl.col("t3_start_price")).abs() / pl.col("t3_start_price") * 100).alias("start_distance_pct"),
    ).with_columns(
        pl.when(pl.col("matched_gap_pct") <= max_gap_pct).then(pl.col("matched_rank_strike")).otherwise(None).alias("matched_rank_strike_close"),
    )

    def block(part: pl.DataFrame) -> dict[str, Any]:
        return {
            "days": part.height,
            "rank1_matches_level_strike_share": float((part["rank1_strike"] == part["t3_strike"]).mean()) if part.height else None,
            "matched_rank": compare(part, "matched_rank_strike"),
            "matched_rank_within_gap": compare(part, "matched_rank_strike_close"),
            "snapped_mirror": compare(part, "snapped_mirror_strike"),
            "per_rank": per_rank(part),
        }

    out: dict[str, Any] = {"max_gap_pct": max_gap_pct, "overall": block(frame)}
    for name, mask in groups.items():
        out[name] = block(frame.filter(mask))
    edges = np.quantile(frame["start_distance_pct"].to_numpy(), np.linspace(0, 1, buckets + 1))
    out["by_start_distance_pct"] = [
        {"bucket": f"[{edges[i]:.3f}, {edges[i + 1]:.3f}]",
         **block(frame.filter(pl.col("start_distance_pct").is_between(edges[i], edges[i + 1], closed="both" if i == buckets - 1 else "left")))}
        for i in range(buckets)
    ]
    return out
