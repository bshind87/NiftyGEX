"""Tokens stage: nearest-expiry per-leg token rows per (snapshot_at, target)."""
from __future__ import annotations

from pathlib import Path

import polars as pl

from niftygex.config import load_config
from niftygex.exposure.selection import total_gex_expiries
from niftygex.features.tokens import KEY, TOKEN_FIELDS, build_tokens, nearest_expiry_legs
from niftygex.io.parquet import describe_files, require_upstream, staged_outputs, write_manifest, write_parquet, write_report
from niftygex.io.regimes import lot_sizes_by_expiry
from niftygex.validation.leakage import assert_available_by

STAGE = "tokens"


def main() -> None:
    config = load_config()
    gex_manifest = require_upstream(config, "gex")
    data, paths = config["data"], config["paths"]
    exposure, curated, greeks = Path(paths["exposure_root"]), Path(paths["curated_root"]), Path(paths["greeks_root"])
    grid = pl.read_parquet(exposure / "snapshots" / "snapshot_grid.parquet")
    strikes = pl.read_parquet(exposure / "gex_strikes" / "*.parquet")
    joined = grid.select(*KEY).join(strikes, on="snapshot_at")
    selected = pl.concat([joined.filter((pl.col("target") == t) & total_gex_expiries(t, data["gex"])) for t in ("t1", "t2", "t3")])
    legs = nearest_expiry_legs(selected)

    # Cumulative volume today per contract over bars available by the snapshot (all legs in the IV band).
    contract = ["expiry_date", "strike_price", "right"]
    needed = legs.select("snapshot_at", "trade_date", "expiry_date").unique()
    volume_parts = []
    for (expiry,), part in needed.group_by(["expiry_date"]):
        path = greeks / "iv" / f"year={expiry.year}" / f"expiry={expiry:%Y%m%d}.parquet"
        rows = pl.read_parquet(path, columns=["trade_date", "available_at", *contract, "volume"])
        volume_parts.append(
            rows.join(part.select("snapshot_at", "trade_date"), on="trade_date")
            .filter(pl.col("available_at") <= pl.col("snapshot_at"))
            .group_by(["snapshot_at", *contract]).agg(pl.col("volume").sum().alias("volume_today"))
        )
    volume_today = pl.concat(volume_parts)
    eod = pl.read_parquet(curated / "oi" / "option_eod_oi_1525.parquet").rename({"available_date": "trade_date"})
    lots = lot_sizes_by_expiry(config["regimes"], legs["expiry_date"].unique().to_list()).select("expiry_date", "units_per_lot")
    tokens = build_tokens(legs, eod, volume_today, lots, float(data["features"]["tokens"]["gex_unit_rupees"]))

    assert_available_by(tokens, "snapshot_at")
    assert_available_by(tokens.filter(pl.col("eod_oi_available_at").is_not_null()), "snapshot_at", "eod_oi_available_at")
    if tokens.group_by(KEY).agg(pl.col("expiry_date").n_unique())["expiry_date"].max() != 1:
        raise SystemExit("Tokens must come from a single expiry per snapshot")

    features_root = Path(paths["features_root"])
    with staged_outputs(features_root, ["tokens"], config) as stage:
        for (year,), part in tokens.with_columns(pl.col("trade_date").dt.year().alias("year")).partition_by(["year"], as_dict=True).items():
            write_parquet(part.drop("year"), stage / "tokens" / f"year={year}.parquet")

    per_snapshot = tokens.group_by(KEY).agg(pl.len().alias("tokens"), pl.col("trade_date").first())
    coverage = grid.join(per_snapshot.select(*KEY, "tokens"), on=KEY, how="left").with_columns(pl.col("tokens").fill_null(0))
    write_report(config, "tokens_report", {
        "upstream_gex_config_hash": gex_manifest["config_hash"],
        "definition": "one token per GEX-chain leg (eligible, available within staleness) of the nearest included expiry; ordered by strike, calls first",
        "fields": TOKEN_FIELDS,
        "rows": tokens.height,
        "tokens_per_snapshot_by_target": {
            t: {"snapshots": part.height, "with_tokens": int((part["tokens"] > 0).sum()),
                "median": float(part["tokens"].median()), "p5": float(part["tokens"].quantile(0.05)), "p95": float(part["tokens"].quantile(0.95)),
                "max": int(part["tokens"].max())}
            for (t,), part in coverage.group_by(["target"])
        },
        "snapshots_with_tokens_share_by_year": coverage.group_by(pl.col("trade_date").dt.year().alias("year")).agg(
            (pl.col("tokens") > 0).mean().alias("share"), pl.col("tokens").median().alias("median_tokens")).sort("year").to_dicts(),
        "field_missing_share": {c: float(tokens[c].is_null().mean()) for c in TOKEN_FIELDS},
    })
    write_manifest(
        config, STAGE,
        inputs=describe_files(sorted((exposure / "gex_strikes").glob("*.parquet")), config, checksums=False),
        outputs=describe_files(sorted((features_root / "tokens").glob("*.parquet")), config),
        counts={"tokens": tokens.height, "snapshots_with_tokens": per_snapshot.height},
        quality_flags={"snapshots_without_tokens": int((coverage["tokens"] == 0).sum())},
        upstream={"gex": gex_manifest["config_hash"]},
    )


if __name__ == "__main__":
    main()
