"""Greeks stage: forward-stage IV rows -> Black-76 gamma and delta per eligible leg, with sanity report."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from niftygex.config import load_config
from niftygex.io.parquet import describe_files, require_upstream, staged_outputs, write_manifest, write_parquet, write_report
from niftygex.pricing.greeks import add_greeks

STAGE = "greeks"
DTE_BUCKETS = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 7), (8, 14), (15, 30), (31, 60), (61, 10_000)]


def strike_step(strikes: pl.Expr) -> pl.Expr:
    """Smallest positive spacing between distinct strikes in a group."""
    return strikes.unique().sort().diff().filter(strikes.unique().sort().diff() > 0).min()


def snapshot_gamma_summary(rows: pl.DataFrame) -> pl.DataFrame:
    """Per (bar, expiry): max-gamma strike, ATM strike gamma, strike step, and forward."""
    per_strike = rows.group_by(["datetime", "trade_date", "expiry_date", "strike_price"]).agg(
        pl.col("gamma").first(), pl.col("forward").first(), pl.col("time_to_expiry").first(),
    )
    return (
        per_strike.sort(["datetime", "gamma"], descending=[False, True])
        .group_by(["datetime", "trade_date", "expiry_date"], maintain_order=True)
        .agg(
            pl.col("strike_price").first().alias("max_gamma_strike"),
            pl.col("forward").first(),
            pl.col("time_to_expiry").first(),
            pl.col("gamma").sort_by((pl.col("strike_price") - pl.col("forward")).abs()).first().alias("atm_gamma"),
            pl.col("strike_price").sort_by((pl.col("strike_price") - pl.col("forward")).abs()).first().alias("atm_strike"),
            strike_step(pl.col("strike_price")).alias("strike_step"),
            pl.len().alias("strikes"),
        )
    )


def main() -> None:
    config = load_config()
    forward_manifest = require_upstream(config, "forward")
    rate = float(config["regimes"].get("risk_free_rate", {}).get("default", 0.0))
    greeks_root = Path(config["paths"]["greeks_root"])
    iv_files = sorted((greeks_root / "iv").rglob("*.parquet"))
    summaries: list[pl.DataFrame] = []
    rows_in = rows_out = 0

    with staged_outputs(greeks_root, ["greeks"], config) as stage:
        for path in iv_files:
            legs = pl.read_parquet(path)
            rows_in += legs.height
            greeks = add_greeks(legs, rate)
            rows_out += greeks.height
            relative = path.relative_to(greeks_root / "iv")
            write_parquet(greeks, stage / "greeks" / relative)
            if greeks.height:
                summaries.append(snapshot_gamma_summary(greeks))

    summary = pl.concat(summaries).with_columns(
        ((pl.col("max_gamma_strike") - pl.col("forward")).abs() <= pl.col("strike_step")).alias("max_gamma_within_one_strike"),
        (pl.col("expiry_date") - pl.col("trade_date")).dt.total_days().alias("dte"),
        (pl.col("atm_gamma") * pl.col("forward")).alias("atm_gamma_x_forward"),
    )
    nearest = summary.filter(pl.col("expiry_date") == pl.col("expiry_date").min().over("datetime"))

    def pct(frame: pl.DataFrame) -> float:
        valid = frame.filter(pl.col("strike_step").is_not_null())
        return float(valid["max_gamma_within_one_strike"].mean()) if valid.height else float("nan")

    buckets = []
    for low, high in DTE_BUCKETS:
        part = nearest.filter(pl.col("dte").is_between(low, high))
        if part.height:
            buckets.append({
                "dte_days": f"{low}-{high}" if low != high else str(low), "snapshots": part.height,
                "median_atm_gamma": float(part["atm_gamma"].median()),
                "median_atm_gamma_x_forward": float(part["atm_gamma_x_forward"].median()),
            })
    medians = [b["median_atm_gamma_x_forward"] for b in buckets]
    per_expiry = (
        summary.group_by("expiry_date")
        .agg(pl.corr("time_to_expiry", "atm_gamma_x_forward", method="spearman").alias("rho"), pl.len().alias("n"))
        .filter(pl.col("n") >= 20)
    )
    rho = per_expiry["rho"].drop_nans().drop_nulls()

    write_report(config, "greeks_report", {
        "upstream_forward_config_hash": forward_manifest["config_hash"],
        "rule": "Black-76 gamma/delta on the parity forward with OTM-sourced IV; rows require eligible and time_to_expiry > 0",
        "rows": {"iv_rows_in": rows_in, "greeks_rows_out": rows_out, "excluded": rows_in - rows_out},
        "max_gamma_near_forward": {
            "definition": "share of (bar, expiry) snapshots whose max-gamma strike is within one strike step of the forward",
            "all_expiries": {"snapshots": summary.height, "share": pct(summary)},
            "nearest_expiry": {"snapshots": nearest.height, "share": pct(nearest)},
            "nearest_expiry_by_dte": [
                {"dte_days": f"{low}-{high}" if low != high else str(low), "share": pct(part)}
                for low, high in DTE_BUCKETS
                if (part := nearest.filter(pl.col("dte").is_between(low, high))).height
            ],
        },
        "atm_gamma_vs_time_to_expiry": {
            "definition": "nearest-expiry ATM-strike gamma by calendar days to expiry; gamma x forward removes the index-level effect",
            "by_dte": buckets,
            "median_gamma_x_forward_monotone_decreasing_in_dte": all(a >= b for a, b in zip(medians, medians[1:])),
            "per_expiry_spearman_time_to_expiry_vs_atm_gamma_x_forward": {
                "expiries": int(rho.len()),
                "share_negative": float((rho < 0).mean()) if rho.len() else None,
                "median_rho": float(np.median(rho.to_numpy())) if rho.len() else None,
            },
        },
    })
    write_manifest(
        config, STAGE,
        inputs=describe_files(iv_files, config, checksums=False),
        outputs=describe_files(sorted((greeks_root / "greeks").rglob("*.parquet")), config, checksums=False),
        counts={"iv_rows_in": rows_in, "greeks_rows_out": rows_out},
        quality_flags={"max_gamma_within_one_strike_nearest": pct(nearest)},
        upstream={"forward": forward_manifest["config_hash"]},
    )


if __name__ == "__main__":
    main()
