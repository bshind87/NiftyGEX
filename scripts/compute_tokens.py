"""Tokens stage: per-leg token rows of each day's nearest included expiry at every sequence snapshot.

The sequence grid runs every 5 minutes from 09:20 to each target's prediction snapshot (15:15 for T2), so a
sequence model can slice its last L snapshots. The chain at each snapshot is built with the same rules as the
GEX stage (latest eligible Greeks row within the staleness window, spot as of the snapshot, previous-day
participant signs); work runs one expiry at a time and is written one year at a time.
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import polars as pl

from niftygex.config import load_config
from niftygex.exposure.gex import dealer_signs, strike_gex
from niftygex.exposure.selection import TARGETS, nearest_expiry_by_day
from niftygex.exposure.snapshots import as_of, sequence_grid
from niftygex.features.tokens import KEY, TOKEN_FIELDS, build_tokens
from niftygex.io.parquet import describe_files, require_upstream, staged_outputs, write_manifest, write_parquet, write_report
from niftygex.io.regimes import lot_sizes_by_expiry
from niftygex.validation.leakage import assert_available_by

STAGE = "tokens"
CONTRACT = ["expiry_date", "strike_price", "right"]
GREEK_COLUMNS = ["available_at", "trade_date", *CONTRACT, "open_interest", "forward", "time_to_expiry", "iv", "gamma", "delta"]
SEQUENCE = ["prediction_at", "lag"]


def expiry_of(path: Path) -> date:
    digits = re.search(r"expiry=(\d{8})", path.name).group(1)
    return date(int(digits[:4]), int(digits[4:6]), int(digits[6:]))


def expiry_tokens(path: Path, iv_path: Path, snaps: pl.DataFrame, spot: pl.DataFrame, signs: pl.DataFrame, eod: pl.DataFrame,
                  lots: pl.DataFrame, *, staleness: int, scale: float, crore: float) -> pl.DataFrame:
    """Tokens of one expiry at the sequence snapshots of the days it is the nearest included expiry."""
    days = snaps["trade_date"].unique()
    rows = pl.read_parquet(path, columns=GREEK_COLUMNS).filter(pl.col("trade_date").is_in(days))
    legs = as_of(rows, snaps, CONTRACT, max_staleness_minutes=staleness)
    legs = strike_gex(legs.join(spot, on="snapshot_at", how="left").join(signs, on="trade_date", how="left"), scale).with_columns(
        (pl.col("expiry_date") - pl.col("trade_date")).dt.total_days().alias("days_to_expiry"))
    legs = legs.join(snaps.select("snapshot_at", "target"), on="snapshot_at")
    # Cumulative volume today per contract over bars available by the snapshot (all legs in the IV band).
    bars = (pl.read_parquet(iv_path, columns=["trade_date", "available_at", *CONTRACT, "volume"]).filter(pl.col("trade_date").is_in(days))
            .sort("available_at").with_columns(pl.col("volume").fill_null(0).cum_sum().over(["trade_date", *CONTRACT]).alias("volume_today")))
    volume_today = as_of(bars, snaps, CONTRACT, max_staleness_minutes=None).select("snapshot_at", *CONTRACT, "volume_today")
    tokens = build_tokens(legs, eod, volume_today, lots, crore)
    return tokens.join(snaps.select(*KEY, *SEQUENCE), on=KEY, how="left")


def main() -> None:
    config = load_config()
    gex_manifest = require_upstream(config, "gex")
    data, paths = config["data"], config["paths"]
    exposure, curated, greeks = Path(paths["exposure_root"]), Path(paths["curated_root"]), Path(paths["greeks_root"])
    gex_cfg = data["gex"]
    staleness, scale = int(data["snapshot"]["max_staleness_minutes"]), float(gex_cfg["scale_percent_move"])
    crore = float(data["features"]["tokens"]["gex_unit_rupees"])

    calendar = pl.read_parquet(curated / "calendar" / "trading_calendar.parquet").filter(pl.col("observed"))
    days = calendar["trading_date"].to_list()
    grid = sequence_grid(days, data["snapshot_grid"])
    files = {expiry_of(p): p for p in sorted((greeks / "greeks").rglob("*.parquet"))}
    pairs = pl.concat([pl.read_parquet(p, columns=["trade_date"]).unique().with_columns(pl.lit(e).alias("expiry_date")) for e, p in files.items()])
    nearest = pl.concat([nearest_expiry_by_day(pairs, t, gex_cfg) for t in TARGETS])
    spot_rows = pl.read_parquet(curated / "market" / "nifty_spot.parquet").filter(pl.col("close").is_not_null()).select("trade_date", "available_at", "close")
    spot = as_of(spot_rows, grid.select("trade_date", "snapshot_at").unique(), [], max_staleness_minutes=staleness).select(
        "snapshot_at", pl.col("close").alias("spot"), pl.col("available_at").alias("spot_available_at"))
    signs = dealer_signs(pl.read_parquet(curated / "oi" / "participant_oi.parquet"), gex_cfg["inferred"]["dealer_client_type"])
    eod = pl.read_parquet(curated / "oi" / "option_eod_oi_1525.parquet").rename({"available_date": "trade_date"})
    lots = lot_sizes_by_expiry(config["regimes"], list(files)).select("expiry_date", "units_per_lot")

    features_root = Path(paths["features_root"])
    pending: dict[int, list[pl.DataFrame]] = {}
    per_snapshot: list[pl.DataFrame] = []
    expiries_per_day: list[pl.DataFrame] = []
    missing: dict[str, float] = {c: 0.0 for c in TOKEN_FIELDS}
    total = 0
    with staged_outputs(features_root, ["tokens"], config) as stage:
        def flush(before_year: int | None) -> None:
            for year in sorted(y for y in pending if before_year is None or y < before_year):
                write_parquet(pl.concat(pending.pop(year)).sort(["snapshot_at", "token_index"]), stage / "tokens" / f"year={year}.parquet")

        for expiry, path in sorted(files.items()):
            snaps = grid.join(nearest.filter(pl.col("expiry_date") == expiry).select("trade_date", "target"), on=["trade_date", "target"])
            if not snaps.height:
                continue
            flush(int(snaps["trade_date"].min().year))
            tokens = expiry_tokens(path, greeks / "iv" / path.relative_to(greeks / "greeks"), snaps, spot, signs, eod, lots,
                                   staleness=staleness, scale=scale, crore=crore)
            assert_available_by(tokens, "snapshot_at")
            assert_available_by(tokens.filter(pl.col("eod_oi_available_at").is_not_null()), "snapshot_at", "eod_oi_available_at")
            if tokens.filter(pl.col("snapshot_at") > pl.col("prediction_at")).height:
                raise SystemExit("A sequence snapshot falls after its prediction snapshot")
            total += tokens.height
            for c in TOKEN_FIELDS:
                missing[c] += float(tokens[c].null_count())
            per_snapshot.append(tokens.group_by(KEY).agg(pl.len().alias("tokens")))
            expiries_per_day.append(tokens.group_by("trade_date", "target").agg(pl.col("expiry_date").n_unique().alias("n")))
            for (year,), part in tokens.with_columns(pl.col("trade_date").dt.year().alias("year")).partition_by(["year"], as_dict=True).items():
                pending.setdefault(int(year), []).append(part.drop("year"))
        flush(None)

    if pl.concat(expiries_per_day)["n"].max() != 1:
        raise SystemExit("Tokens must come from a single expiry per trading day")
    coverage = grid.join(pl.concat(per_snapshot), on=KEY, how="left").with_columns(pl.col("tokens").fill_null(0))
    prediction = coverage.filter(pl.col("lag") == 0)

    def stats(part: pl.DataFrame) -> dict:
        return {"snapshots": part.height, "with_tokens": int((part["tokens"] > 0).sum()), "median": float(part["tokens"].median()),
                "p5": float(part["tokens"].quantile(0.05)), "p95": float(part["tokens"].quantile(0.95)), "max": int(part["tokens"].max())}

    write_report(config, "tokens_report", {
        "upstream_gex_config_hash": gex_manifest["config_hash"],
        "definition": "one token per GEX-chain leg (eligible, available within staleness) of the day's nearest included expiry, at every "
                      f"sequence snapshot from {data['snapshot_grid']['sequence_start']} to the prediction snapshot every "
                      f"{data['snapshot_grid']['sequence_step_minutes']} minutes; ordered by strike, calls first; lag 0 is the prediction snapshot",
        "fields": TOKEN_FIELDS + SEQUENCE,
        "rows": total,
        "sequence_snapshots_per_day": int(grid.group_by("trade_date").len()["len"].max()),
        "tokens_per_prediction_snapshot": {t: stats(part) for (t,), part in prediction.group_by(["target"])},
        "tokens_per_sequence_snapshot": {t: stats(part) for (t,), part in coverage.group_by(["target"])},
        "median_tokens_by_lag": coverage.group_by("lag").agg(pl.col("tokens").median().alias("median_tokens"),
                                                             (pl.col("tokens") > 0).mean().alias("share_with_tokens")).sort("lag").to_dicts(),
        "sequence_coverage_by_year": coverage.group_by(pl.col("trade_date").dt.year().alias("year")).agg(
            (pl.col("tokens") > 0).mean().alias("share_with_tokens"), pl.col("tokens").median().alias("median_tokens")).sort("year").to_dicts(),
        "days_with_full_sequence": int(coverage.group_by("trade_date").agg((pl.col("tokens") > 0).all().alias("full"))["full"].sum()),
        "field_missing_share": {c: missing[c] / total for c in TOKEN_FIELDS},
    })
    write_manifest(
        config, STAGE,
        inputs=describe_files(sorted((greeks / "greeks").rglob("*.parquet")), config, checksums=False),
        outputs=describe_files(sorted((features_root / "tokens").glob("*.parquet")), config),
        counts={"tokens": total, "sequence_snapshots": grid.height, "sequence_snapshots_with_tokens": int((coverage["tokens"] > 0).sum())},
        quality_flags={"sequence_snapshots_without_tokens": int((coverage["tokens"] == 0).sum()),
                       "prediction_snapshots_without_tokens": int((prediction["tokens"] == 0).sum())},
        upstream={"gex": gex_manifest["config_hash"]},
    )


if __name__ == "__main__":
    main()
