"""Forward stage: curated options -> per-expiry parity forwards, parity residuals, and close-based IV over the configured band."""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from niftygex.clean.session import parse_time
from niftygex.config import load_config
from niftygex.io.availability import at_ist_time
from niftygex.io.parquet import describe_files, require_upstream, staged_outputs, write_manifest, write_parquet, write_report
from niftygex.pricing.forward import estimate_forwards, parity_forward
from niftygex.pricing.implied_vol import IV_STATUSES, assign_otm_iv, invert_with_status

STAGE = "forward"
OWNED = ["forward", "parity", "iv"]
LEG_COLUMNS = ["datetime", "available_at", "trade_date", "expiry_date", "strike_price", "right", "close", "volume",
               "open_interest", "truncated_session", "missing_spot", "data_quality"]


def expiry_files(options_root: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(options_root.glob("*.parquet")):
        match = re.search(r"(\d{8})\.parquet$", path.name)
        if match:
            grouped[match.group(1)].append(path)
    return grouped


def _summary(values: np.ndarray) -> dict[str, float | int | None]:
    values = values[np.isfinite(values)]
    if not values.size:
        return {"count": 0, "median": None, "p95": None, "max": None}
    return {"count": int(values.size), "median": float(np.median(values)), "p95": float(np.quantile(values, 0.95)), "max": float(np.max(values))}


def main() -> None:
    config = load_config()
    clean_manifest = require_upstream(config, "clean")
    data = config["data"]
    pricing, band, settings = data["pricing"], data["liquidity"], data["forward"]
    rate = float(config["regimes"].get("risk_free_rate", {}).get("default", data.get("risk_free_rate", 0.0)))
    seconds_per_year = float(pricing["days_per_year"]) * 86400.0
    expiry_clock = parse_time(pricing["expiry_time"])
    low, high = float(band["moneyness_min"]), float(band["moneyness_max"])
    min_volume = float(settings["min_volume"])
    options_root = config["paths"]["curated_root"] / "options"
    grouped = expiry_files(options_root)

    quality = Counter()
    statuses = Counter()
    own_statuses = Counter()
    source_counts = Counter()
    below_intrinsic = Counter()
    pair_residuals: list[np.ndarray] = []
    snapshot_residuals: list[np.ndarray] = []
    agreement: list[np.ndarray] = []
    rows = {"forward": 0, "parity": 0, "iv": 0, "iv_eligible": 0, "legs_outside_band": 0, "legs_without_forward": 0}
    skipped: list[str] = []

    with staged_outputs(config["paths"]["greeks_root"], OWNED, config) as stage:
        for expiry, paths in sorted(grouped.items()):
            legs = pl.concat([pl.read_parquet(path, columns=LEG_COLUMNS) for path in paths], how="vertical_relaxed")
            expiries = legs["expiry_date"].unique().to_list()
            if expiries != [date(int(expiry[:4]), int(expiry[4:6]), int(expiry[6:]))]:
                raise SystemExit(f"File expiry {expiry} disagrees with expiry_date column {expiries}")
            if set(legs["right"].unique().to_list()) != {"Call", "Put"}:
                skipped.append(expiry)
                continue
            # Time to expiry runs from when the bar's close is observed (available_at) to the expiry clock.
            legs = legs.with_columns(
                ((at_ist_time(pl.col("expiry_date"), expiry_clock) - pl.col("available_at")).dt.total_seconds() / seconds_per_year)
                .alias("time_to_expiry")
            )
            calls = legs.filter((pl.col("right") == "Call") & (pl.col("volume") >= min_volume))
            puts = legs.filter((pl.col("right") == "Put") & (pl.col("volume") >= min_volume))
            pairs = calls.select("datetime", "strike_price", "time_to_expiry", pl.col("close").alias("call_close"), pl.col("volume").alias("call_volume")).join(
                puts.select("datetime", "strike_price", pl.col("close").alias("put_close"), pl.col("volume").alias("put_volume")),
                on=["datetime", "strike_price"],
            )
            pairs = pairs.with_columns(
                pl.Series("parity_forward", parity_forward(pairs["strike_price"].to_numpy(), pairs["call_close"].to_numpy(), pairs["put_close"].to_numpy(), pairs["time_to_expiry"].to_numpy(), rate))
            )
            forwards, near = estimate_forwards(
                pairs, near_atm_width=float(settings["near_atm_width"]), min_pairs=int(settings["min_pairs"]), max_residual=float(settings["max_residual"]),
            )
            forwards = forwards.join(legs.select("datetime", "available_at", "time_to_expiry").unique("datetime"), on="datetime", how="left").with_columns(
                pl.lit(legs["expiry_date"][0]).alias("expiry_date")
            )

            legs = legs.join(forwards.select("datetime", "forward", "forward_quality"), on="datetime", how="left")
            rows["legs_without_forward"] += legs.filter(pl.col("forward").is_null()).height
            legs = legs.with_columns((pl.col("strike_price") / pl.col("forward")).alias("moneyness"))
            in_band = legs.filter(pl.col("moneyness").is_between(low, high, closed="both"))
            rows["legs_outside_band"] += legs.filter(pl.col("forward").is_not_null()).height - in_band.height
            iv, status = invert_with_status(
                in_band["close"].to_numpy(), in_band["forward"].to_numpy(), in_band["strike_price"].to_numpy(),
                in_band["time_to_expiry"].to_numpy(), rate, (in_band["right"] == "Call").to_numpy(),
                iterations=int(pricing["iv_iterations"]), bounds=tuple(pricing["iv_bounds"]),
                max_error_fraction=float(pricing["iv_max_price_error_fraction"]),
            )
            # Each leg's own solve is kept as a diagnostic; both legs of a strike use the OTM leg's IV.
            in_band = in_band.with_columns(
                pl.Series("iv_own", iv), pl.Series("iv_own_status", status.tolist(), dtype=pl.String),
                pl.col("moneyness").log().alias("log_moneyness"),
            ).pipe(assign_otm_iv).with_columns(
                (
                    (pl.col("volume") > 0) & (pl.col("iv_source_volume") > 0) & (pl.col("iv_status") == "ok")
                    & (pl.col("forward_quality") == "good") & (pl.col("time_to_expiry") > 0)
                ).alias("eligible")
            )

            year = expiry[:4]
            write_parquet(forwards, stage / "forward" / f"year={year}" / f"expiry={expiry}.parquet")
            write_parquet(near, stage / "parity" / f"year={year}" / f"expiry={expiry}.parquet")
            write_parquet(in_band, stage / "iv" / f"year={year}" / f"expiry={expiry}.parquet")

            quality.update(dict(forwards.group_by("forward_quality").len().iter_rows()))
            statuses.update(dict(in_band.group_by("iv_status").len().iter_rows()))
            own_statuses.update(dict(in_band.group_by("iv_own_status").len().iter_rows()))
            below = in_band.filter(pl.col("iv_own_status") == "below_intrinsic")
            below_intrinsic["own_leg_below_intrinsic"] += below.height
            below_intrinsic["own_leg_below_intrinsic_now_iv_ok"] += below.filter(pl.col("iv_status") == "ok").height
            below_intrinsic["own_leg_below_intrinsic_now_eligible"] += int(below["eligible"].sum())
            below_intrinsic["legs_with_iv_status_below_intrinsic"] += in_band.filter(pl.col("iv_status") == "below_intrinsic").height
            source_counts.update(dict(
                in_band.with_columns((pl.col("right") == pl.col("iv_source_right")).alias("is_source"))
                .group_by("is_source").len().iter_rows()
            ))
            pair_residuals.append(near["parity_residual"].abs().to_numpy())
            snapshot_residuals.append(forwards["forward_residual_median_abs"].drop_nulls().to_numpy())
            both = in_band.filter((pl.col("iv_own_status") == "ok") & (pl.col("volume") > 0)).pivot(on="right", index=["datetime", "strike_price"], values="iv_own")
            if {"Call", "Put"} <= set(both.columns):
                agreement.append((both["Call"] - both["Put"]).abs().drop_nulls().to_numpy())
            rows["forward"] += forwards.height
            rows["parity"] += near.height
            rows["iv"] += in_band.height
            rows["iv_eligible"] += int(in_band["eligible"].sum())

    attempted = sum(statuses.values())
    write_report(config, "forward_report", {
        "upstream_clean_config_hash": clean_manifest["config_hash"],
        "risk_free_rate": rate,
        "time_to_expiry": f"calendar years from available_at to {pricing['expiry_time']} IST on expiry_date ({pricing['days_per_year']} days/year)",
        "forward_rule": f"median parity forward over pairs with both legs volume >= {min_volume} within {settings['near_atm_width']} of the provisional median",
        "iv_band": {"moneyness_min": low, "moneyness_max": high, "moneyness": "strike / parity forward"},
        "expiries": len(grouped) - len(skipped),
        "expiries_skipped_missing_side": skipped,
        "rows": rows,
        "forward_quality_snapshots": dict(sorted(quality.items())),
        "parity_residual_abs_per_pair": _summary(np.concatenate(pair_residuals) if pair_residuals else np.array([])),
        "parity_residual_median_abs_per_snapshot": _summary(np.concatenate(snapshot_residuals) if snapshot_residuals else np.array([])),
        "iv_rule": "both legs of a strike take the IV solved from the OTM leg (call if strike >= forward, put otherwise); "
                   "iv_own keeps each leg's own solve; eligible = own volume > 0, OTM-leg volume > 0, iv ok, forward good, time_to_expiry > 0",
        "iv_status_counts": {status: statuses.get(status, 0) for status in IV_STATUSES},
        "iv_ok_rate": statuses.get("ok", 0) / attempted if attempted else None,
        "iv_own_status_counts": {status: own_statuses.get(status, 0) for status in IV_STATUSES},
        "legs_by_role": {"otm_source_leg": source_counts.get(True, 0), "itm_leg_assigned_otm_iv": source_counts.get(False, 0)},
        "below_intrinsic": dict(below_intrinsic),
        "time_to_expiry_nonpositive_legs_excluded": statuses.get("nonpositive_time", 0),
        "call_put_own_iv_abs_diff_traded_pairs": _summary(np.concatenate(agreement) if agreement else np.array([])),
    })
    greeks_root = config["paths"]["greeks_root"]
    write_manifest(
        config, STAGE,
        inputs=describe_files(sorted(options_root.glob("*.parquet")), config),
        outputs=describe_files([p for name in OWNED for p in (greeks_root / name).rglob("*.parquet")], config, checksums=False),
        counts=rows,
        quality_flags={"forward_quality": dict(quality), "iv_status": dict(statuses), "below_intrinsic": dict(below_intrinsic)},
        upstream={"clean": clean_manifest["config_hash"]},
    )


if __name__ == "__main__":
    main()
