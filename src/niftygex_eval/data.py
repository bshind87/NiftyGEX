"""Load the modeling table: features joined to targets for one target, with expiry cycles attached."""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from niftygex.io.parquet import read_manifest
from niftygex_eval.harness import assign_cycles


def option_expiry_dates(curated_root: Path) -> list[date]:
    dates = set()
    for path in (curated_root / "options").glob("*.parquet"):
        match = re.search(r"(\d{8})\.parquet$", path.name)
        if match:
            digits = match.group(1)
            dates.add(date(int(digits[:4]), int(digits[4:6]), int(digits[6:])))
    return sorted(dates)


def modeling_table(data_config: dict[str, Any], target: str) -> tuple[pl.DataFrame, list[str], dict[str, str]]:
    """Feature rows of ``target`` joined to that target's labels; returns (table, feature columns, upstream hashes).

    Feature and target files must come from the same data configuration (manifest hashes agree).
    """
    paths = data_config["paths"]
    hashes = {stage: read_manifest(data_config, stage)["config_hash"] for stage in ("features", "targets")}
    if len(set(hashes.values())) != 1:
        raise SystemExit(f"Feature and target manifests disagree on config_hash: {hashes}")
    features = pl.read_parquet(Path(paths["features_root"]) / "aggregate" / "aggregate.parquet").filter(pl.col("target") == target)
    targets = pl.read_parquet(Path(paths["targets_root"]) / target / f"{target}.parquet").drop("trade_date", strict=False)
    report = json.loads((Path(paths["reports_root"]) / "features_report.json").read_text())
    table = features.join(targets, on="snapshot_at", how="inner").sort("snapshot_at")
    table = assign_cycles(table, option_expiry_dates(Path(paths["curated_root"])))
    return table, list(report["feature_columns"]), hashes
