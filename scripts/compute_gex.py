"""GEX stage: Greeks + spot + participant OI -> per-strike/per-expiry/per-target GEX and key levels on the snapshot grid."""
from __future__ import annotations

import math
import re
from datetime import date
from functools import reduce
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from niftygex.config import load_config
from niftygex.exposure.gex import dealer_signs, strike_gex
from niftygex.exposure.levels import call_wall, flip_level, max_gex_strike, max_pain, put_wall
from niftygex.exposure.selection import TARGETS, total_gex_expiries
from niftygex.exposure.snapshots import as_of, snapshot_grid
from niftygex.io.parquet import describe_files, require_upstream, staged_outputs, write_manifest, write_parquet, write_report
from niftygex.validation.leakage import assert_available_by

STAGE = "gex"
OWNED = ["snapshots", "gex_strikes", "oi_chain", "gex_expiry", "gex_totals", "levels"]
CONTRACT = ["expiry_date", "strike_price", "right"]
GREEK_COLUMNS = ["datetime", "available_at", "trade_date", *CONTRACT, "open_interest", "volume", "forward", "time_to_expiry", "iv", "gamma", "delta"]
CONVENTIONS = ("naive", "absolute", "inferred")
CRORE = 1e7


def expiry_of(path: Path) -> date:
    digits = re.search(r"expiry=(\d{8})", path.name).group(1)
    return date(int(digits[:4]), int(digits[4:6]), int(digits[6:]))


def oi_gcd(values: pl.Series) -> int:
    integers = values.drop_nulls().cast(pl.Int64).unique()
    integers = integers.filter(integers > 0).to_list()
    return reduce(math.gcd, integers) if integers else 0


def gcd_runs(rows: list[tuple[date, int]]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for expiry, value in sorted(rows):
        if runs and runs[-1]["oi_gcd"] == value:
            runs[-1]["to_expiry"] = expiry
            runs[-1]["expiries"] += 1
        else:
            runs.append({"oi_gcd": value, "from_expiry": expiry, "to_expiry": expiry, "expiries": 1})
    return runs


def _quantiles(values: pl.Series, levels=(0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)) -> dict[str, float | None]:
    values = values.drop_nulls()
    if not values.len():
        return {f"p{int(level * 100)}": None for level in levels}
    return {f"p{int(level * 100)}": float(values.quantile(level, interpolation="linear")) for level in levels}


def compute_levels(target_chain: pl.DataFrame, target_oi: pl.DataFrame, spots: dict, signs: dict, scale: float, grid_points: int, rate: float) -> pl.DataFrame:
    gex_groups = target_chain.partition_by(["snapshot_at", "target"], as_dict=True)
    oi_groups = target_oi.partition_by(["snapshot_at", "target"], as_dict=True)
    rows = []
    for key in sorted(set(gex_groups) | set(oi_groups)):
        snapshot_at, target = key
        spot = spots.get(snapshot_at)
        record: dict[str, Any] = {"snapshot_at": snapshot_at, "target": target, "spot": spot}
        chain, oi = gex_groups.get(key), oi_groups.get(key)
        if spot is None or not np.isfinite(spot):
            rows.append(record)
            continue
        if oi is not None and oi.height:
            by_strike = oi.group_by("strike_price").agg(
                pl.col("open_interest").filter(pl.col("right") == "Call").sum().alias("call_oi"),
                pl.col("open_interest").filter(pl.col("right") == "Put").sum().alias("put_oi"),
            ).sort("strike_price")
            k = by_strike["strike_price"].to_numpy()
            for name, level in (("call_wall", call_wall(k, by_strike["call_oi"].to_numpy(), spot)),
                                ("put_wall", put_wall(k, by_strike["put_oi"].to_numpy(), spot))):
                record[name], record[f"{name}_tied"] = level.value, level.tied
            nearest = oi.filter(pl.col("expiry_date") == pl.col("expiry_date").min())
            pain_strikes = nearest.group_by("strike_price").agg(
                pl.col("open_interest").filter(pl.col("right") == "Call").sum().alias("call_oi"),
                pl.col("open_interest").filter(pl.col("right") == "Put").sum().alias("put_oi"),
            ).sort("strike_price")
            pain = max_pain(pain_strikes["strike_price"].to_numpy(), pain_strikes["call_oi"].to_numpy(), pain_strikes["put_oi"].to_numpy(), spot)
            record["max_pain"], record["max_pain_tied"] = pain.value, pain.tied
            record["max_pain_expiry"] = nearest["expiry_date"][0]
        if chain is not None and chain.height:
            per_strike = chain.group_by("strike_price").agg(pl.col("gex_absolute").sum()).sort("strike_price")
            top = max_gex_strike(per_strike["strike_price"].to_numpy(), per_strike["gex_absolute"].to_numpy(), spot)
            record["max_gex_strike"], record["max_gex_strike_tied"] = top.value, top.tied
            record["strike_band_low"], record["strike_band_high"] = float(chain["strike_price"].min()), float(chain["strike_price"].max())
            arrays = {name: chain[column].to_numpy() for name, column in (
                ("strike", "strike_price"), ("forward", "forward"), ("time_to_expiry", "time_to_expiry"), ("iv", "iv"), ("open_interest", "open_interest"))}
            is_call = (chain["right"] == "Call").to_numpy()
            naive = flip_level(**arrays, sign=np.where(is_call, 1.0, -1.0), spot=spot, scale=scale, grid_points=grid_points, rate=rate)
            record["flip_naive"], record["flip_naive_crossings"] = naive.value, naive.crossings
            sign = signs.get(chain["trade_date"][0])
            if sign is not None:
                inferred = flip_level(**arrays, sign=np.where(is_call, sign[0], sign[1]), spot=spot, scale=scale, grid_points=grid_points, rate=rate)
                record["flip_inferred"], record["flip_inferred_crossings"] = inferred.value, inferred.crossings
        rows.append(record)
    schema = {"snapshot_at": pl.Datetime("us", "Asia/Kolkata"), "target": pl.String, "spot": pl.Float64,
              "call_wall": pl.Float64, "call_wall_tied": pl.Boolean, "put_wall": pl.Float64, "put_wall_tied": pl.Boolean,
              "max_pain": pl.Float64, "max_pain_tied": pl.Boolean, "max_pain_expiry": pl.Date,
              "max_gex_strike": pl.Float64, "max_gex_strike_tied": pl.Boolean, "strike_band_low": pl.Float64, "strike_band_high": pl.Float64,
              "flip_naive": pl.Float64, "flip_naive_crossings": pl.Int64, "flip_inferred": pl.Float64, "flip_inferred_crossings": pl.Int64}
    return pl.DataFrame(rows, schema=schema).sort(["snapshot_at", "target"])


def main() -> None:
    config = load_config()
    greeks_manifest = require_upstream(config, "greeks")
    data = config["data"]
    gex_cfg, rate = data["gex"], float(config["regimes"].get("risk_free_rate", {}).get("default", 0.0))
    if gex_cfg["open_interest_unit"] != "index_units":
        raise SystemExit("Only open_interest_unit: index_units is implemented")
    scale, max_days = float(gex_cfg["scale_percent_move"]), int(gex_cfg["max_days_to_expiry"])
    staleness = int(data["snapshot"]["max_staleness_minutes"])
    grid_points = int(data["levels"]["flip_grid_points"])
    curated, greeks_root = config["paths"]["curated_root"], Path(config["paths"]["greeks_root"])
    exposure_root = Path(config["paths"]["exposure_root"])

    calendar = pl.read_parquet(curated / "calendar" / "trading_calendar.parquet").filter(pl.col("observed"))
    greek_files = sorted((greeks_root / "greeks").rglob("*.parquet"))
    expiries = {expiry_of(path): path for path in greek_files}
    grid = snapshot_grid(calendar["trading_date"].to_list(), data["snapshot_grid"]).join(
        calendar.select(pl.col("trading_date").alias("trade_date"), "missing_spot", "truncated_session", "data_quality"), on="trade_date", how="left",
    )
    snaps = grid.select("trade_date", "snapshot_at").unique()

    spot_rows = pl.read_parquet(curated / "market" / "nifty_spot.parquet").filter(pl.col("close").is_not_null()).select("trade_date", "available_at", "close")
    spot = as_of(spot_rows, snaps, [], max_staleness_minutes=staleness).select(
        "snapshot_at", pl.col("close").alias("spot"), pl.col("available_at").alias("spot_available_at"))
    signs = dealer_signs(pl.read_parquet(curated / "oi" / "participant_oi.parquet"), gex_cfg["inferred"]["dealer_client_type"])

    chains, oi_chains, gcds = [], [], []
    for expiry, path in sorted(expiries.items()):
        relevant = snaps.filter((pl.col("trade_date") <= expiry) & ((pl.lit(expiry) - pl.col("trade_date")).dt.total_days() <= max_days))
        rows = pl.read_parquet(path, columns=GREEK_COLUMNS)
        gcds.append((expiry, oi_gcd(rows["open_interest"])))
        if not relevant.height:
            continue
        days = relevant["trade_date"].unique()
        chains.append(as_of(rows.filter(pl.col("trade_date").is_in(days)), relevant, CONTRACT, max_staleness_minutes=staleness))
        iv_rows = pl.read_parquet(greeks_root / "iv" / path.relative_to(greeks_root / "greeks"),
                                  columns=["available_at", "trade_date", *CONTRACT, "open_interest"])
        oi_chains.append(as_of(iv_rows.filter(pl.col("trade_date").is_in(days)), relevant, CONTRACT, max_staleness_minutes=None))

    chain = strike_gex(
        pl.concat(chains).join(spot, on="snapshot_at", how="left").join(signs, on="trade_date", how="left"), scale,
    ).with_columns((pl.col("expiry_date") - pl.col("trade_date")).dt.total_days().alias("days_to_expiry"))
    oi_chain = pl.concat(oi_chains)

    # Every input joined into a snapshot must have been available by then.
    assert_available_by(chain, "snapshot_at")
    assert_available_by(oi_chain, "snapshot_at")
    assert_available_by(chain.filter(pl.col("spot").is_not_null()), "snapshot_at", "spot_available_at")
    assert_available_by(chain.filter(pl.col("dealer_sign_call").is_not_null()), "snapshot_at", "participant_available_at")

    sums = [pl.col(f"gex_{name}").sum().alias(f"gex_{name}") for name in CONVENTIONS]
    flags = [
        pl.col("spot").is_not_null().all().alias("spot_available"),
        pl.col("open_interest").is_not_null().all().alias("oi_available"),
        pl.col("dealer_sign_call").is_not_null().all().alias("inferred_available"),
    ]

    def nullify(frame: pl.DataFrame) -> pl.DataFrame:
        # Polars sums of all-null groups are 0; missing inputs must stay missing.
        base = pl.col("spot_available") & pl.col("oi_available")
        return frame.with_columns(
            *[pl.when(base).then(pl.col(f"gex_{n}")).otherwise(None).alias(f"gex_{n}") for n in ("naive", "absolute")],
            pl.when(base & pl.col("inferred_available")).then(pl.col("gex_inferred")).otherwise(None).alias("gex_inferred"),
        )

    expiry_totals = nullify(chain.group_by(["snapshot_at", "trade_date", "expiry_date"]).agg(*sums, *flags, pl.len().alias("legs"))).sort(["snapshot_at", "expiry_date"])
    target_chain = grid.select("snapshot_at", "target").join(chain, on="snapshot_at")
    target_chain = pl.concat([target_chain.filter((pl.col("target") == t) & total_gex_expiries(t, gex_cfg)) for t in TARGETS])
    target_oi = grid.select("snapshot_at", "target", "trade_date").join(oi_chain, on=["snapshot_at", "trade_date"])
    target_oi = pl.concat([target_oi.filter((pl.col("target") == t) & total_gex_expiries(t, gex_cfg)) for t in TARGETS])
    totals = nullify(target_chain.group_by(["snapshot_at", "target"]).agg(
        *sums, *flags, pl.col("expiry_date").n_unique().alias("expiries"), pl.len().alias("legs"),
        (pl.col("days_to_expiry") == 0).any().alias("includes_same_day_expiry"),
        pl.col("days_to_expiry").min().alias("min_days_to_expiry"), pl.col("days_to_expiry").max().alias("max_days_to_expiry"),
        pl.col("spot").first(),
    ))
    totals = grid.join(totals, on=["snapshot_at", "target"], how="left").with_columns(
        *[pl.col(flag).fill_null(False) for flag in ("spot_available", "oi_available", "inferred_available")]).sort(["snapshot_at", "target"])

    spots = dict(spot.select("snapshot_at", "spot").iter_rows())
    sign_map = {day: (c, p) for day, c, p in signs.select("trade_date", "dealer_sign_call", "dealer_sign_put").iter_rows() if c is not None and p is not None}
    levels = grid.select("snapshot_at", "target", "trade_date").join(
        compute_levels(target_chain.filter(pl.col("open_interest").is_not_null()), target_oi.filter(pl.col("open_interest").is_not_null()),
                       spots, sign_map, scale, grid_points, rate), on=["snapshot_at", "target"], how="left")

    with staged_outputs(exposure_root, OWNED, config) as stage:
        write_parquet(grid, stage / "snapshots" / "snapshot_grid.parquet")
        for (year,), part in chain.with_columns(pl.col("trade_date").dt.year().alias("year")).partition_by(["year"], as_dict=True).items():
            write_parquet(part.drop("year"), stage / "gex_strikes" / f"year={year}.parquet")
        for (year,), part in oi_chain.with_columns(pl.col("trade_date").dt.year().alias("year")).partition_by(["year"], as_dict=True).items():
            write_parquet(part.drop("year"), stage / "oi_chain" / f"year={year}.parquet")
        write_parquet(expiry_totals, stage / "gex_expiry" / "gex_expiry.parquet")
        write_parquet(totals, stage / "gex_totals" / "gex_totals.parquet")
        write_parquet(levels, stage / "levels" / "levels.parquet")

    write_reports(config, gex_cfg=gex_cfg, totals=totals, levels=levels, chain=chain, oi_chain=oi_chain, grid=grid, gcds=gcds,
                  signs=signs, staleness=staleness, upstream=greeks_manifest["config_hash"])
    write_manifest(
        config, STAGE,
        inputs=describe_files(greek_files + [curated / "market" / "nifty_spot.parquet", curated / "oi" / "participant_oi.parquet"], config, checksums=False),
        outputs=describe_files([p for name in OWNED for p in (exposure_root / name).rglob("*.parquet")], config, checksums=False),
        counts={"snapshots": grid.height, "strike_rows": chain.height, "oi_rows": oi_chain.height},
        quality_flags={"spot_unavailable_snapshots": int((~totals["spot_available"]).sum()),
                       "inferred_unavailable_snapshots": int((~totals["inferred_available"]).sum())},
        upstream={"greeks": greeks_manifest["config_hash"]},
    )


def write_reports(config: dict[str, Any], **ctx: Any) -> None:
    totals, levels, gex_cfg = ctx["totals"], ctx["levels"], ctx["gex_cfg"]
    per_target: dict[str, Any] = {}
    selection: dict[str, Any] = {}
    for target in TARGETS:
        part = totals.filter(pl.col("target") == target)
        with_chain = part.filter(pl.col("legs").is_not_null())
        available = part.filter(pl.col("gex_naive").is_not_null() & pl.col("gex_inferred").is_not_null())
        both_signed = available.filter((pl.col("gex_naive") != 0) & (pl.col("gex_inferred") != 0))
        per_target[target] = {
            "snapshots": part.height,
            "with_chain": with_chain.height,
            "spot_available_share": float(part["spot_available"].mean()),
            "option_oi_available_share": float(part["oi_available"].mean()),
            "gex_available_snapshots": int(part["gex_naive"].is_not_null().sum()),
            "inferred_sign_available_share": float(part["inferred_available"].mean()),
            "naive_inferred_total_sign_agreement_share": float((both_signed["gex_naive"].sign() == both_signed["gex_inferred"].sign()).mean()) if both_signed.height else None,
            "total_gex_crore": {name: _quantiles(part[f"gex_{name}"] / CRORE) for name in CONVENTIONS},
            "share_positive": {name: float((part[f"gex_{name}"].drop_nulls() > 0).mean()) for name in ("naive", "inferred")},
        }
        expiry_days = ctx["chain"].filter(pl.col("expiry_date") == pl.col("trade_date")).select("trade_date").unique()
        on_expiry_day = with_chain.join(expiry_days, on="trade_date", how="semi")
        selection[target] = {
            "same_day_expiry_rule": "kept" if gex_cfg["same_day_expiry_in_total_gex"][target] else "excluded",
            "snapshots_including_same_day_expiry": int(with_chain["includes_same_day_expiry"].sum()),
            "expiry_day_snapshots": on_expiry_day.height,
            "expiry_day_snapshots_including_same_day_expiry": int(on_expiry_day["includes_same_day_expiry"].sum()),
            "min_days_to_expiry_included": int(with_chain["min_days_to_expiry"].min()),
            "max_days_to_expiry_included": int(with_chain["max_days_to_expiry"].max()),
            "snapshots_on_non_expiry_days": with_chain.height - on_expiry_day.height,
            **({"expiry_days_without_same_day_contract_in_chain": sorted(set(
                on_expiry_day.filter(~pl.col("includes_same_day_expiry"))["trade_date"].to_list()))}
               if gex_cfg["same_day_expiry_in_total_gex"][target] else {}),
        }
    signs = ctx["signs"]
    write_report(config, "gex_report", {
        "upstream_greeks_config_hash": ctx["upstream"],
        "formula": "GEX_leg = gamma (Black-76, per forward point) x open_interest (index units) x spot^2 x "
                   f"{gex_cfg['scale_percent_move']}; rupees; reported in crore",
        "conventions": {
            "naive": "calls +, puts -",
            "absolute": "all +",
            "inferred": f"call legs x sign(net index-call position of {gex_cfg['inferred']['dealer_client_type']}), put legs x sign(net index-put position); "
                        "previous trading day's participant OI; missing days stay NaN. Participant OI covers all index options "
                        "(not NIFTY only); its mix changes at the weekly-expiry rationalisation regime marker",
        },
        "chain_rule": f"latest eligible row per contract with snapshot - {ctx['staleness']} min < available_at <= snapshot; "
                      "spot is the latest non-null spot close in the same window",
        "per_target": per_target,
        "expiry_selection_check": selection,
        "inferred_flip_structure": {
            "definition": "a flip can exist under the inferred convention only when the call and put dealer signs differ",
            "trading_days_with_differing_call_put_signs_share": float((signs["dealer_sign_call"] != signs["dealer_sign_put"]).mean()),
        },
        "dealer_sign_days": {
            "call": dict(signs.group_by("dealer_sign_call").len().sort("dealer_sign_call").iter_rows()),
            "put": dict(signs.group_by("dealer_sign_put").len().sort("dealer_sign_put").iter_rows()),
        },
        "chain_coverage": {
            "gex_legs_per_snapshot_expiry_median": float(ctx["chain"].group_by(["snapshot_at", *CONTRACT[:1]]).len()["len"].median()),
            "oi_legs_per_snapshot_expiry_median": float(ctx["oi_chain"].group_by(["snapshot_at", *CONTRACT[:1]]).len()["len"].median()),
        },
        "expiry_history_depth": {
            "note": "each expiry has at most ~21 calendar days of option history in the source, so the configured "
                    f"{gex_cfg['max_days_to_expiry']}-day horizon is effectively bounded by the data",
            "max_days_to_expiry_in_any_total": int(totals["max_days_to_expiry"].max()),
        },
        "lot_size_inference": {
            "method": "gcd of positive open_interest values per expiry; equals the lot size when OI is in index units",
            "configured_lot_sizes": config["regimes"].get("lot_sizes", []),
            "runs": gcd_runs(ctx["gcds"]),
        },
    })

    level_summary: dict[str, Any] = {}
    for target in TARGETS:
        part = levels.filter(pl.col("target") == target)
        has_spot = part.filter(pl.col("spot").is_not_null())
        entry: dict[str, Any] = {"snapshots": part.height, "with_spot_and_option_oi": has_spot.height}
        for name in ("call_wall", "put_wall", "max_pain", "max_gex_strike", "flip_naive", "flip_inferred"):
            distance = (has_spot[name] - has_spot["spot"]) / has_spot["spot"] * 100
            entry[name] = {
                "missing": int(has_spot[name].null_count()),
                "tied": int(has_spot[f"{name}_tied"].sum()) if f"{name}_tied" in has_spot.columns else None,
                "median_distance_pct": float(distance.median()) if distance.drop_nulls().len() else None,
                "median_abs_distance_pct": float(distance.abs().median()) if distance.drop_nulls().len() else None,
                "median_distance_points": float((has_spot[name] - has_spot["spot"]).median()) if distance.drop_nulls().len() else None,
            }
        for convention in ("naive", "inferred"):
            crossings = has_spot[f"flip_{convention}_crossings"]
            entry[f"flip_{convention}"]["inside_strike_band_share"] = float(has_spot[f"flip_{convention}"].is_not_null().mean())
            entry[f"flip_{convention}"]["crossings_distribution"] = dict(
                has_spot.group_by(f"flip_{convention}_crossings").len().sort(f"flip_{convention}_crossings").iter_rows())
            entry[f"flip_{convention}"]["multiple_crossings"] = int((crossings > 1).sum())
        differing = has_spot.join(ctx["signs"].filter(pl.col("dealer_sign_call") != pl.col("dealer_sign_put")).select("trade_date"), on="trade_date", how="semi")
        entry["flip_inferred"]["inside_strike_band_share_when_call_put_signs_differ"] = (
            float(differing["flip_inferred"].is_not_null().mean()) if differing.height else None)
        level_summary[target] = entry
    write_report(config, "levels_report", {
        "definitions": {
            "call_wall": "strike with the largest call OI strictly above spot, summed over the target's included expiries",
            "put_wall": "strike with the largest put OI strictly below spot, summed over included expiries",
            "max_pain": "nearest included expiry: settlement strike minimizing total intrinsic payout",
            "max_gex_strike": "strike with the largest absolute-convention GEX",
            "flip": "total GEX recomputed over hypothetical spot levels across the strike band (forward moves with spot, "
                    "sticky-strike IV); crossing closest to spot; missing when no sign change inside the band",
            "ties": "closest strike to spot wins, then the lower strike",
            "oi_source": "walls and max pain use the latest row of the day per contract (no volume requirement)",
        },
        "per_target": level_summary,
    })


if __name__ == "__main__":
    main()
