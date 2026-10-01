"""Basis stage: curated futures + spot -> per-bar basis with jump classification and basis_quality."""
from __future__ import annotations

from typing import Any

import polars as pl

from niftygex.clean.basis import (
    alignment_segments,
    atm_synthetic,
    basis_frame,
    classify_basis,
    label_alignment,
    roll_transitions,
)
from niftygex.clean.market import end_labeled_windows
from niftygex.config import load_config
from niftygex.io.parquet import describe_files, require_upstream, staged_outputs, write_manifest, write_parquet, write_report

STAGE = "basis"


def _quantiles(series: pl.Series, levels: list[float]) -> dict[str, float]:
    series = series.drop_nulls()
    return {f"p{round(level * 100, 2):g}": float(series.quantile(level, interpolation="linear")) for level in levels}


def _segments_summary(daily: pl.DataFrame, min_days: int) -> dict[str, Any]:
    segments = alignment_segments(daily)
    return {
        "segments_at_least_min_days": [s for s in segments if s["days"] >= min_days],
        "short_segments": len([s for s in segments if s["days"] < min_days]),
        "days_by_best_lag": dict(daily.group_by("best_lag").len().sort("best_lag").iter_rows()),
        "mean_daily_corr_by_lag": {
            column.removeprefix("corr_lag"): float(daily[column].fill_nan(None).mean())
            for column in daily.columns if column.startswith("corr_lag")
        },
        "days_not_lag0": daily.filter(pl.col("best_lag") != 0).select("day", "best_lag", "best_corr").to_dicts(),
    }


def alignment_report(config: dict[str, Any], futures_path, spot_path) -> dict[str, Any]:
    """Infer per-day label alignment of spot and intraday VIX against futures and the option-implied ATM synthetic."""
    bar_minutes = int(config["data"]["session"]["bar_minutes"])
    curated = config["paths"]["curated_root"]
    futures = pl.read_parquet(futures_path).select("datetime", pl.col("close").alias("ref"))
    spot = pl.read_parquet(spot_path).select("datetime", pl.col("close").alias("spot"))
    synthetic = atm_synthetic(pl.scan_parquet(curated / "options" / "*.parquet"))
    vix = pl.read_parquet(curated / "vix" / "intraday.parquet").select("datetime", (1.0 / pl.col("close")).alias("inverse_vix"))
    pairs = {
        "spot_vs_futures": label_alignment(futures, spot, "spot", bar_minutes),
        "futures_vs_option_synthetic": label_alignment(synthetic.rename({"synthetic": "ref"}), futures.rename({"ref": "futures"}), "futures", bar_minutes),
        "spot_vs_option_synthetic": label_alignment(synthetic.rename({"synthetic": "ref"}), spot, "spot", bar_minutes),
        "inverse_vix_vs_spot": label_alignment(spot.rename({"spot": "ref"}), vix, "inverse_vix", bar_minutes),
        "inverse_vix_vs_futures": label_alignment(futures, vix, "inverse_vix", bar_minutes),
    }
    min_days = 5
    payload = {
        "definition": "per-day correlation of same-day 5-minute returns; best_lag -1 means the second series' label t+5min "
                      "matches the reference label t (second series end-labeled relative to the reference)",
        "segment_min_days": min_days,
        **{name: _segments_summary(daily, min_days) for name, daily in pairs.items()},
    }
    return {"daily": pairs["spot_vs_futures"], "payload": payload}


def spot_field_sync(futures: pl.DataFrame, spot: pl.DataFrame, in_window: pl.Expr) -> dict[str, Any]:
    """Median within-day |change| in log(futures close / spot field) for each spot field, inside vs outside the window.

    The field with the smallest value is the spot price sampled closest to the futures close.
    """
    joined = (
        futures.select("datetime", "trade_date", pl.col("close").alias("future"))
        .join(spot.select("datetime", "open", "high", "low", "close"), on="datetime").drop_nulls().sort("datetime")
        .with_columns(pl.col("open").shift(-1).over("trade_date").alias("next_open"), in_window.alias("in_window"))
    )
    result: dict[str, Any] = {}
    for label, part in (("inside_window", joined.filter(pl.col("in_window"))), ("outside_window", joined.filter(~pl.col("in_window")))):
        result[label] = {
            field: float(
                part.drop_nulls(field).select(((pl.col("future") / pl.col(field)).log() * 1e4).diff().over("trade_date").abs().median()).item()
            )
            for field in ("close", "open", "high", "low", "next_open")
        }
    return result


def boundary_check(steps: pl.DataFrame, windows: list, trading_days: int, ratio_max: float) -> dict[str, Any]:
    """Median within-day |basis step| over ``trading_days`` days before and from each window boundary."""
    days = steps["trade_date"].unique().sort().to_list()
    boundaries = [("window_start", first) for first, _ in windows] + [
        ("first_day_after_window", next((d for d in days if d > last), None)) for _, last in windows
    ]
    rows = []
    for name, boundary in boundaries:
        if boundary is None:
            continue
        before = [d for d in days if d < boundary][-trading_days:]
        after = [d for d in days if d >= boundary][:trading_days]
        sides = {}
        for side, chosen in (("before", before), ("after", after)):
            part = steps.filter(pl.col("trade_date").is_in(chosen))
            step, vol = float(part["step"].median()), float(part["future_step"].median())
            sides[side] = {"first": chosen[0], "last": chosen[-1], "days": len(chosen), "median_abs_step_bps": step,
                           "median_abs_future_return_bps": vol, "step_per_unit_future_return": step / vol}
        raw = [sides["before"]["median_abs_step_bps"], sides["after"]["median_abs_step_bps"]]
        normalized = [sides["before"]["step_per_unit_future_return"], sides["after"]["step_per_unit_future_return"]]
        ratio, ratio_normalized = max(raw) / min(raw), max(normalized) / min(normalized)
        rows.append({
            "boundary": name, "date": boundary, **sides,
            "ratio_larger_to_smaller": ratio, "similar": ratio <= ratio_max,
            "volatility_normalized_ratio": ratio_normalized, "similar_after_volatility_normalization": ratio_normalized <= ratio_max,
        })
    return {"trading_days_each_side": trading_days, "similar_ratio_max": ratio_max, "boundaries": rows,
            "all_similar": bool(rows) and all(row["similar"] for row in rows)}


def main() -> None:
    config = load_config()
    settings = config["data"]["basis"]
    clean_manifest = require_upstream(config, "clean")
    curated = config["paths"]["curated_root"]
    futures_path = curated / "futures" / "nifty_front_month.parquet"
    spot_path = curated / "market" / "nifty_spot.parquet"
    threshold = float(settings["jump_threshold_bps"])

    frame = basis_frame(pl.read_parquet(futures_path), pl.read_parquet(spot_path))
    frame = classify_basis(frame, threshold, int(settings["revert_max_bars"]), float(settings["revert_tolerance_fraction"]))
    rolls = roll_transitions(frame)

    alignment = alignment_report(config, futures_path, spot_path)
    # Jumps split by the spot label convention inferred for each day.
    frame = frame.join(
        alignment["daily"].select(pl.col("day").alias("trade_date"), pl.col("best_lag").alias("spot_vs_futures_best_lag")),
        on="trade_date", how="left",
    )

    with staged_outputs(curated, ["basis"], config) as stage:
        write_parquet(frame, stage / "basis" / "nifty_basis.parquet")
        write_parquet(alignment["daily"], stage / "basis" / "spot_label_alignment.parquet")

    within_day = (
        frame.filter(pl.col("basis_bps").is_not_null()).sort("datetime")
        .with_columns(pl.col("basis_bps").diff().over("trade_date").abs().alias("step"))["step"]
    )
    windows = end_labeled_windows(config["regimes"], "spot")
    in_window = pl.any_horizontal([pl.col("trade_date").is_between(a, b, closed="both") for a, b in windows]) if windows else pl.lit(False)
    steps = (
        frame.filter(pl.col("basis_bps").is_not_null()).sort("datetime")
        .with_columns(
            pl.col("basis_bps").diff().over("trade_date").abs().alias("step"),
            (pl.col("future_close").log().diff().over("trade_date").abs() * 1e4).alias("future_step"),
            in_window.alias("in_window"),
        )
        .filter(pl.col("step").is_not_null())
    )
    check = settings["boundary_check"]
    boundaries = boundary_check(steps, windows, int(check["trading_days"]), float(check["similar_ratio_max"]))
    inside = _quantiles(steps.filter(pl.col("in_window"))["step"], [0.5])["p50"]
    outside = _quantiles(steps.filter(~pl.col("in_window"))["step"], [0.5])["p50"]
    boundaries["explanation"] = (
        f"In-window median |basis step| {inside:.2f} bps vs out-of-window {outside:.2f} bps. "
        + ("Noise is continuous across every relabel-window boundary, so the gap reflects differences between "
           "periods (the out-of-window sample is mostly 2019-2020), not residual label misalignment."
           if boundaries["all_similar"] else
           "At least one boundary shows a noise step; the in/out gap is not explained by period differences alone.")
    )
    boundaries["monthly_median_abs_step_bps"] = [
        {"month": month, "in_window": bool(flag), "median_abs_step_bps": float(step), "step_per_unit_future_return": float(norm)}
        for month, flag, step, norm in steps.group_by(pl.col("trade_date").dt.strftime("%Y-%m").alias("month"))
        .agg(pl.col("in_window").any(), pl.col("step").median(), (pl.col("step").median() / pl.col("future_step").median()).alias("normalized"))
        .sort("month").iter_rows()
    ]
    boundaries["spot_field_sync"] = spot_field_sync(pl.read_parquet(futures_path), pl.read_parquet(spot_path), in_window)
    alignment["payload"]["relabel_window_boundary_check"] = boundaries
    write_report(config, "spot_label_alignment", alignment["payload"])
    jumps = frame.filter(pl.col("jump_class").is_not_null())
    per_day = jumps.group_by("trade_date").agg(
        (pl.col("jump_class") == "spike").sum().alias("spikes"),
        (pl.col("jump_class") == "event").sum().alias("events"),
        (pl.col("jump_class") == "unresolved").sum().alias("unresolved"),
    )
    roll_days = set(rolls["trade_date"].to_list())
    jump_days = set(per_day["trade_date"].to_list())
    event_days = per_day.filter(pl.col("events") > 0).sort("trade_date")
    spike_days = per_day.filter(pl.col("spikes") > 0).sort("trade_date")
    event_rows = (
        jumps.filter(pl.col("jump_class") == "event")
        .join(frame.select("datetime", pl.col("basis_bps").shift(1).alias("prior_bps")), on="datetime", how="left")
        .select("datetime", "expiry_date", "future_close", "spot_close", "basis", "basis_bps", "prior_bps", "data_quality")
    )
    spike_rows = frame.filter(pl.col("basis_quality") == "bad").select(
        "datetime", "expiry_date", "future_close", "spot_close", "basis", "basis_bps", "jump_class",
    )

    payload: dict[str, Any] = {
        "definition": {
            "basis": "front-month futures close minus spot close on the same bar label; bps of spot",
            "jump": f"|change in basis_bps| > {threshold} between consecutive valid bars of the same trading day",
            "reverting_spike": f"basis returns within {settings['revert_tolerance_fraction']} x threshold of the pre-jump level "
                               f"within {settings['revert_max_bars']} bars; spike bars get basis_quality=bad",
            "event": "jump that does not revert within the window",
            "unresolved": "jump too close to the session end to test reversion",
            "roll_transition": "day boundary where the once-per-day front-month contract changes",
        },
        "upstream_clean_config_hash": clean_manifest["config_hash"],
        "bars": frame.height,
        "basis_quality_counts": dict(frame.group_by("basis_quality").len().sort("basis_quality").iter_rows()),
        "basis_bps_distribution": _quantiles(frame["basis_bps"], [0.0, 0.001, 0.01, 0.5, 0.99, 0.999, 1.0]),
        "within_day_abs_step_bps_distribution": _quantiles(within_day, [0.5, 0.9, 0.99, 0.999, 0.9999, 1.0]),
        "jump_counts": dict(jumps.group_by("jump_class").len().sort("jump_class").iter_rows()),
        "jump_counts_by_spot_vs_futures_best_lag": [
            {"best_lag": lag, "jump_class": cls, "bars": n}
            for lag, cls, n in jumps.group_by("spot_vs_futures_best_lag", "jump_class").len().sort("spot_vs_futures_best_lag", "jump_class").iter_rows()
        ],
        "within_day_abs_step_bps_by_best_lag": {
            str(lag): _quantiles(group["step"], [0.5, 0.99, 0.999])
            for (lag,), group in steps.group_by(["spot_vs_futures_best_lag"])
        },
        "within_day_abs_step_bps_by_spot_label_window": {
            "inside_configured_end_labeled_window": _quantiles(steps.filter(pl.col("in_window"))["step"], [0.5, 0.99, 0.999]),
            "outside_window": _quantiles(steps.filter(~pl.col("in_window"))["step"], [0.5, 0.99, 0.999]),
        },
        "threshold_source": "jump_threshold_bps in configs/data.yaml, set from the 99.9th percentile of the aligned "
                            "(best_lag 0) within-day |basis step| distribution",
        "dates": {
            "roll_transition_dates": len(roll_days),
            "dates_with_any_within_day_jump": len(jump_days),
            "event_days": event_days.height,
            "reverting_spike_days": spike_days.height,
            "jump_days_that_are_also_roll_days": len(jump_days & roll_days),
        },
        "roll_transitions": {
            "basis_step_points": _quantiles(rolls["basis_step"], [0.0, 0.5, 1.0]),
            "rows": rolls.to_dicts(),
            "negative_steps": rolls.filter(pl.col("basis_step") < 0).select(
                "trade_date", pl.col("old_expiry").alias("from_contract"), pl.col("new_expiry").alias("to_contract"),
                "last_basis", "first_basis", "basis_step",
            ).to_dicts(),
        },
        "event_days": event_days.to_dicts(),
        "event_jump_bars": event_rows.to_dicts(),
        "reverting_spike_days": spike_days.to_dicts(),
        "spike_bars_flagged_bad": spike_rows.to_dicts(),
    }
    write_report(config, "basis_jumps", payload)
    write_manifest(
        config, STAGE,
        inputs=describe_files([futures_path, spot_path], config),
        outputs=describe_files(sorted((curated / "basis").glob("*.parquet")), config),
        counts={"bars": frame.height, "roll_days": len(roll_days), "jumps": jumps.height},
        quality_flags={"basis_quality": payload["basis_quality_counts"], "event_days": event_days.height, "spike_days": spike_days.height},
        upstream={"clean": clean_manifest["config_hash"]},
    )


if __name__ == "__main__":
    main()
