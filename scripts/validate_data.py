"""Validation stage: data-quality and leakage checks over all stage outputs -> JSON report and docs/validation_report.md."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import polars as pl

from niftygex.clean.session import session_settings
from niftygex.config import REPO_ROOT, load_config
from niftygex.exposure.selection import TARGETS
from niftygex.io.availability import observation_close
from niftygex.io.parquet import describe_files, require_upstream, write_manifest, write_report
from niftygex.validation.checks import lot_size_check, summary
from niftygex.validation.report import render_markdown

STAGE = "validation"
MONEYNESS_BUCKETS = [(0.0, 0.01), (0.01, 0.03), (0.03, 0.05), (0.05, 0.09)]
# Snapshot-time values (known at the snapshot) that target tables may carry for context.
SNAPSHOT_CONTEXT = {"spot"}
LEAKAGE_TESTS = ["tests/test_leakage.py", "tests/test_calendar.py", "tests/test_stage_contracts.py"]


def year_summary(frame: pl.LazyFrame, value: str) -> list[dict[str, Any]]:
    return (
        frame.group_by(pl.col("datetime").dt.year().alias("year"))
        .agg(pl.len().alias("count"), pl.col(value).quantile(0.5, interpolation="linear").alias("p50"),
             pl.col(value).quantile(0.95, interpolation="linear").alias("p95"), pl.col(value).quantile(0.99, interpolation="linear").alias("p99"),
             pl.col(value).max().alias("max"))
        .sort("year").collect().to_dicts()
    )


def parity_check(greeks_root: Path) -> dict[str, Any]:
    pairs = pl.scan_parquet(greeks_root / "parity" / "**" / "*.parquet").select("datetime", pl.col("parity_residual").abs().alias("abs_residual"))
    forwards = pl.scan_parquet(greeks_root / "forward" / "**" / "*.parquet")
    quality = forwards.group_by(pl.col("datetime").dt.year().alias("year"), "forward_quality").len().collect().pivot(
        on="forward_quality", index="year", values="len").sort("year").fill_null(0)
    return {
        "definition": "abs(parity forward - snapshot median forward) per near-ATM pair, index points",
        "overall": summary(pairs.select("abs_residual").collect()["abs_residual"]),
        "by_year": year_summary(pairs, "abs_residual"),
        "forward_quality_by_year": quality.to_dicts(),
    }


def iv_agreement_check(greeks_root: Path) -> dict[str, Any]:
    legs = pl.scan_parquet(greeks_root / "iv" / "**" / "*.parquet").filter((pl.col("iv_own_status") == "ok") & (pl.col("volume") > 0))
    keys = ["datetime", "expiry_date", "strike_price"]
    calls = legs.filter(pl.col("right") == "Call").select(*keys, "log_moneyness", pl.col("iv_own").alias("call_iv"))
    puts = legs.filter(pl.col("right") == "Put").select(*keys, pl.col("iv_own").alias("put_iv"))
    pairs = calls.join(puts, on=keys).with_columns((pl.col("call_iv") - pl.col("put_iv")).abs().alias("abs_diff"), pl.col("log_moneyness").abs().alias("abs_log_m"))
    buckets = []
    for low, high in MONEYNESS_BUCKETS:
        part = pairs.filter(pl.col("abs_log_m").is_between(low, high, closed="left")).select("abs_diff").collect()["abs_diff"]
        buckets.append({"abs_log_moneyness": f"[{low}, {high})", **summary(part)})
    return {
        "definition": "abs(call own-leg IV - put own-leg IV) at the same bar, expiry and strike; both legs traded and solved",
        "overall": summary(pairs.select("abs_diff").collect()["abs_diff"]),
        "by_year": year_summary(pairs, "abs_diff"),
        "by_moneyness": buckets,
    }


def atm_iv_vs_vix(config: dict[str, Any]) -> dict[str, Any]:
    exposure, curated = Path(config["paths"]["exposure_root"]), Path(config["paths"]["curated_root"])
    target = config["data"]["validation"]["atm_iv_vs_vix_target"]
    grid = pl.read_parquet(exposure / "snapshots" / "snapshot_grid.parquet").filter(pl.col("target") == target).select("snapshot_at")
    strikes = pl.read_parquet(exposure / "gex_strikes" / "*.parquet").join(grid, on="snapshot_at")
    longest = strikes.filter(pl.col("days_to_expiry") == pl.col("days_to_expiry").max().over("snapshot_at"))
    atm = longest.sort((pl.col("strike_price") - pl.col("forward")).abs()).group_by("snapshot_at", maintain_order=True).agg(
        pl.col("trade_date").first(), pl.col("iv").first().alias("atm_iv"), pl.col("days_to_expiry").first())
    vix = pl.read_parquet(curated / "vix" / "daily.parquet").select(pl.col("observation_date").alias("trade_date"), (pl.col("close") / 100).alias("vix"))
    joined = atm.join(vix, on="trade_date").drop_nulls(["atm_iv", "vix"]).with_columns(
        (pl.col("atm_iv") / pl.col("vix")).alias("ratio"), ((pl.col("atm_iv") - pl.col("vix")) * 100).alias("diff_vol_points"))

    def stats(frame: pl.DataFrame) -> dict[str, Any]:
        return {"days": frame.height, "correlation": float(frame.select(pl.corr("atm_iv", "vix")).item()) if frame.height > 2 else None,
                "median_ratio": float(frame["ratio"].median()), "median_diff_vol_points": float(frame["diff_vol_points"].median())}

    return {
        "definition": f"ATM IV (strike nearest the forward) of the longest available expiry at the {target} snapshot vs the same day's "
                      "India VIX close / 100. Validation only: same-day VIX is never a feature. The longest expiry is at most ~21 days, "
                      "so this is not a 30-day IV",
        "overall": stats(joined),
        "by_year": [{"year": y, **stats(part)} for (y,), part in sorted(joined.group_by(pl.col("trade_date").dt.year()), key=lambda kv: kv[0])],
        "by_days_to_longest_expiry": [
            {"days": f"{low}-{high}", **stats(part)}
            for low, high in ((0, 6), (7, 13), (14, 21)) if (part := joined.filter(pl.col("days_to_expiry").is_between(low, high))).height
        ],
    }


def leakage_checks(config: dict[str, Any]) -> list[dict[str, Any]]:
    paths = config["paths"]
    curated, exposure = Path(paths["curated_root"]), Path(paths["exposure_root"])
    session = session_settings(config["data"])
    checks: list[dict[str, Any]] = []

    features = pl.read_parquet(Path(paths["features_root"]) / "aggregate" / "aggregate.parquet")
    for column in sorted(c for c in features.columns if c.endswith("_available_at")):
        late = features.filter(pl.col(column) > pl.col("snapshot_at")).height
        checks.append({"check": f"features.{column} <= snapshot_at", "passed": late == 0, "violations": late,
                       "rows_with_source": int(features[column].is_not_null().sum())})
    for name in TARGETS:
        targets = pl.read_parquet(Path(paths["targets_root"]) / name / f"{name}.parquet")
        target_only = sorted(set(targets.columns) - {"snapshot_at", "trade_date", "target_date"} - SNAPSHOT_CONTEXT)
        leaked = sorted(set(target_only) & set(features.columns))
        shared = sorted(SNAPSHOT_CONTEXT & set(targets.columns) & set(features.columns))
        if shared:
            # Snapshot-time context copied into a target table must equal the feature values exactly.
            compared = targets.select("snapshot_at", *shared).join(
                features.select("snapshot_at", *[pl.col(c).alias(f"feature_{c}") for c in shared]).unique("snapshot_at"), on="snapshot_at")
            mismatch = compared.filter(pl.any_horizontal([~pl.col(c).eq_missing(pl.col(f"feature_{c}")) for c in shared])).height
            checks.append({"check": f"{name.upper()} snapshot-context columns equal feature values", "passed": mismatch == 0, "violations": mismatch, "columns": shared})
        checks.append({"check": f"feature table contains no {name.upper()} target columns", "passed": not leaked, "violations": len(leaked), "columns": leaked})
        early = targets.filter(pl.col("target_available_at").is_not_null() & (pl.col("target_available_at") <= pl.col("snapshot_at"))).height
        gap = targets.drop_nulls("target_available_at").select((pl.col("target_available_at") - pl.col("snapshot_at")).dt.total_minutes().min()).item()
        checks.append({"check": f"{name.upper()} target_available_at > snapshot_at", "passed": early == 0, "violations": early, "min_gap_minutes": gap})
    for name, path in (("option_eod_oi", curated / "oi" / "option_eod_oi_1525.parquet"), ("participant_oi", curated / "oi" / "participant_oi.parquet"),
                       ("vix_daily", curated / "vix" / "daily.parquet")):
        frame = pl.read_parquet(path).with_columns(observation_close(pl.col("observation_date"), session["end"]).alias("close"))
        bad = frame.filter(pl.col("available_at").is_null() | (pl.col("available_at") <= pl.col("close")) | (pl.col("available_date") <= pl.col("observation_date"))).height
        checks.append({"check": f"{name} available strictly after the observation close (next trading day)", "passed": bad == 0, "violations": bad, "rows": frame.height})
    tokens = pl.scan_parquet(Path(paths["features_root"]) / "tokens" / "*.parquet")
    late_tokens = tokens.select(((pl.col("available_at") > pl.col("snapshot_at")) | (pl.col("eod_oi_available_at") > pl.col("snapshot_at"))).sum()).collect().item()
    multi_expiry = tokens.group_by("trade_date", "target").agg(pl.col("expiry_date").n_unique().alias("n")).filter(pl.col("n") > 1).select(pl.len()).collect().item()
    after_prediction = tokens.select((pl.col("snapshot_at") > pl.col("prediction_at")).sum()).collect().item()
    checks.append({"check": "token rows available by snapshot (bar and previous EOD OI)", "passed": late_tokens == 0, "violations": late_tokens})
    checks.append({"check": "tokens come from one (nearest included) expiry per trading day", "passed": multi_expiry == 0, "violations": multi_expiry})
    checks.append({"check": "token sequence snapshots at or before their prediction snapshot", "passed": after_prediction == 0, "violations": after_prediction})
    bad_bars = pl.scan_parquet(curated / "options" / "*.parquet").select(
        (pl.col("available_at") != pl.col("datetime") + pl.duration(minutes=session["bar_minutes"])).sum()).collect().item()
    checks.append({"check": "curated option bars available_at = label + bar", "passed": bad_bars == 0, "violations": bad_bars})
    late_strikes = pl.scan_parquet(exposure / "gex_strikes" / "*.parquet").select(
        (pl.col("available_at") > pl.col("snapshot_at")).sum() + (pl.col("spot_available_at") > pl.col("snapshot_at")).sum()
        + (pl.col("participant_available_at") > pl.col("snapshot_at")).sum()).collect().item()
    checks.append({"check": "GEX chain, spot and participant inputs available by snapshot", "passed": late_strikes == 0, "violations": late_strikes})
    return checks


def run_leakage_tests() -> dict[str, Any]:
    result = subprocess.run([sys.executable, "-m", "pytest", "-o", "addopts=", "-q", "-p", "no:cacheprovider", *LEAKAGE_TESTS],
                            cwd=REPO_ROOT, capture_output=True, text=True)
    lines = [line for line in result.stdout.strip().splitlines() if line.strip()]
    tail = lines[-1] if lines else ""
    counts = {key: int(value) for value, key in re.findall(r"(\d+) (passed|failed|skipped|errors?)", tail)}
    return {"modules": LEAKAGE_TESTS, "exit_code": result.returncode, "passed": result.returncode == 0, "counts": counts}


def main() -> None:
    config = load_config()
    upstream = {stage: require_upstream(config, stage)["config_hash"] for stage in ("clean", "forward", "greeks", "gex", "features", "tokens", "targets")}
    greeks_root = Path(config["paths"]["greeks_root"])
    curated = Path(config["paths"]["curated_root"])
    greeks_report = json.loads((Path(config["paths"]["reports_root"]) / "greeks_report.json").read_text())

    lots = lot_size_check(pl.scan_parquet(curated / "options" / "*.parquet"), config["regimes"])
    judged = lots.filter(pl.col("positive_oi_rows").is_not_null())
    bhavcopy = Path(config["paths"]["raw_root"]) / "bhavcopy"
    checks = leakage_checks(config)
    tests = run_leakage_tests()
    payload = {
        "parity_residuals": parity_check(greeks_root),
        "call_put_iv_agreement": iv_agreement_check(greeks_root),
        "atm_iv_vs_india_vix": atm_iv_vs_vix(config),
        "gamma_sanity": {key: greeks_report[key] for key in ("max_gamma_near_forward", "atm_gamma_vs_time_to_expiry")},
        "lot_sizes": {
            "expiries_judged": judged.height, "expiries_failed": judged.filter(~pl.col("passed")).height,
            "expiries_without_positive_oi": lots.height - judged.height,
            "exceptions": judged.filter(pl.col("lot_exception")).select("expiry_date", "units_per_lot", "oi_divisor", "share_divisible_by_lot").to_dicts(),
        },
        "bhavcopy_reconciliation": {
            "status": "skipped" if not bhavcopy.exists() or not any(bhavcopy.iterdir()) else "not_implemented",
            "reason": f"no bhavcopy files under {bhavcopy.relative_to(REPO_ROOT)}; the 15:25 option OI is the EOD proxy",
        },
        "timestamp_and_leakage_checks": checks,
        "leakage_test_run": tests,
        "all_leakage_checks_passed": all(check["passed"] for check in checks) and tests["passed"],
        "upstream_config_hashes": upstream,
    }
    write_report(config, "validation_report", payload)
    markdown = REPO_ROOT / "docs" / "validation_report.md"
    markdown.write_text(render_markdown(payload))
    write_manifest(
        config, STAGE,
        inputs=describe_files([p for p in (Path(config["paths"]["reports_root"])).glob("*.json") if p.name != "validation_report.json"], config, checksums=False),
        outputs=describe_files([Path(config["paths"]["reports_root"]) / "validation_report.json", markdown], config, checksums=False),
        counts={"checks": len(checks)},
        quality_flags={"all_leakage_checks_passed": payload["all_leakage_checks_passed"]},
        upstream=upstream,
    )
    if not payload["all_leakage_checks_passed"]:
        raise SystemExit("Validation found leakage or timestamp-ordering violations; see outputs/reports/validation_report.json")


if __name__ == "__main__":
    main()
