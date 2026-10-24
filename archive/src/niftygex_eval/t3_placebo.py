"""T3 placebo test: is moving toward the max-GEX strike more common than toward a matched-distance pseudo-strike?"""
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from scipy.stats import binomtest


def toward(start: np.ndarray, end: np.ndarray, strike: np.ndarray) -> np.ndarray:
    return np.abs(end - strike) < np.abs(start - strike)


def placebo_summary(start: np.ndarray, end: np.ndarray, strike: np.ndarray, draws: int, rng: np.random.Generator) -> dict[str, Any]:
    """Actual hit rate vs (a) random-side pseudo-strikes at matched distance and (b) the mirror strike."""
    n = start.size
    if n == 0:
        return {"days": 0}
    offset = strike - start
    actual = toward(start, end, strike)
    mirror = toward(start, end, start - offset)
    signs = rng.choice([-1.0, 1.0], size=(draws, n))
    placebo_rates = toward(start[None, :], end[None, :], start[None, :] + signs * np.abs(offset)[None, :]).mean(axis=1)
    actual_rate = float(actual.mean())
    only_actual, only_mirror = int(np.sum(actual & ~mirror)), int(np.sum(mirror & ~actual))
    discordant = only_actual + only_mirror
    return {
        "days": int(n),
        "hit_rate_actual": actual_rate,
        "hit_rate_mirror": float(mirror.mean()),
        "placebo_random_side": {
            "mean": float(placebo_rates.mean()), "p2_5": float(np.quantile(placebo_rates, 0.025)), "p97_5": float(np.quantile(placebo_rates, 0.975)),
            "p_value_actual_ge_placebo": float((np.sum(placebo_rates >= actual_rate) + 1) / (draws + 1)),
        },
        "actual_vs_mirror_sign_test": {
            "actual_only": only_actual, "mirror_only": only_mirror,
            "p_value_two_sided": float(binomtest(only_actual, discordant, 0.5).pvalue) if discordant else None,
        },
        "both_missed": int(np.sum(~actual & ~mirror)),
    }


def run_placebo(frame: pl.DataFrame, draws: int, seed: int, buckets: int, groups: dict[str, pl.Expr]) -> dict[str, Any]:
    """Overall, by distance bucket, by year, and by the given groups (one RNG stream per slice for reproducibility)."""
    frame = frame.with_columns(((pl.col("t3_strike") - pl.col("t3_start_price")).abs() / pl.col("t3_start_price") * 100).alias("start_distance_pct"))
    edges = np.quantile(frame["start_distance_pct"].to_numpy(), np.linspace(0, 1, buckets + 1))

    def run(part: pl.DataFrame, offset: int) -> dict[str, Any]:
        return placebo_summary(part["t3_start_price"].to_numpy(), part["t3_end_price"].to_numpy(), part["t3_strike"].to_numpy(),
                               draws, np.random.default_rng(seed + offset))

    out: dict[str, Any] = {"overall": run(frame, 0)}
    out["by_start_distance_pct"] = []
    for i in range(buckets):
        low, high = edges[i], edges[i + 1]
        part = frame.filter(pl.col("start_distance_pct").is_between(low, high, closed="both" if i == buckets - 1 else "left"))
        out["by_start_distance_pct"].append({"bucket": f"[{low:.3f}, {high:.3f}]", **run(part, 100 + i)})
    out["by_year"] = [{"year": y, **run(part, 1000 + y)} for (y,), part in sorted(frame.group_by(pl.col("trade_date").dt.year()), key=lambda kv: kv[0])]
    for offset, (name, mask) in enumerate(groups.items()):
        out[name] = run(frame.filter(mask), 5000 + offset)
    return out
