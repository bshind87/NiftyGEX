"""Stage output handling: staged directory replacement, manifests, and reports."""
from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from niftygex import __version__
from niftygex.config import config_hash
from niftygex.io.provenance import git_provenance


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _guard_not_raw(path: Path, config: dict[str, Any]) -> None:
    raw_root = Path(config["paths"]["raw_root"]).resolve()
    resolved = path.resolve()
    if resolved == raw_root or raw_root in resolved.parents:
        raise PermissionError(f"Stages never write under {raw_root}: {path}")


@contextmanager
def staged_outputs(final_root: Path, names: list[str], config: dict[str, Any]) -> Iterator[Path]:
    """Write a stage's owned child directories into a staging root, then swap them in on success.

    A failed stage leaves the previous outputs untouched.
    """
    _guard_not_raw(final_root, config)
    staging = final_root.parent / f".{final_root.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    for name in names:
        (staging / name).mkdir(parents=True)
    yield staging
    final_root.mkdir(parents=True, exist_ok=True)
    for name in names:
        target = final_root / name
        if target.exists():
            shutil.rmtree(target)
        (staging / name).rename(target)
    shutil.rmtree(staging)


def write_parquet(frame: pl.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path, compression="zstd", statistics=True)
    return path


def _relative(path: Path, config: dict[str, Any]) -> str:
    for root_key in ("data_root", "outputs_root"):
        root = Path(config["paths"][root_key]).resolve()
        try:
            return f"{root_key}/{path.resolve().relative_to(root)}"
        except ValueError:
            continue
    return str(path)


def describe_files(paths: list[Path], config: dict[str, Any], *, checksums: bool = True) -> list[dict[str, Any]]:
    records = []
    for path in sorted(paths):
        record: dict[str, Any] = {"path": _relative(path, config), "size_bytes": path.stat().st_size}
        if checksums:
            record["sha256"] = file_sha256(path)
        if path.suffix == ".parquet":
            record["rows"] = pl.scan_parquet(path).select(pl.len()).collect().item()
        records.append(record)
    return records


def manifest_path(config: dict[str, Any], stage: str) -> Path:
    return Path(config["paths"]["outputs_root"]) / "manifests" / f"{stage}.json"


def read_manifest(config: dict[str, Any], stage: str) -> dict[str, Any]:
    path = manifest_path(config, stage)
    if not path.exists():
        raise FileNotFoundError(f"Missing upstream manifest {path}; run the {stage} stage first")
    return json.loads(path.read_text())


def require_upstream(config: dict[str, Any], stage: str) -> dict[str, Any]:
    """Fail when an upstream stage ran under a different configuration."""
    manifest = read_manifest(config, stage)
    current = config_hash(config)
    if manifest.get("config_hash") != current:
        raise SystemExit(
            f"Upstream stage '{stage}' ran with config_hash {manifest.get('config_hash')}, "
            f"current is {current}; rerun {stage} and downstream stages"
        )
    return manifest


def write_manifest(
    config: dict[str, Any],
    stage: str,
    *,
    inputs: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    counts: dict[str, Any],
    quality_flags: dict[str, Any],
    upstream: dict[str, str] | None = None,
) -> Path:
    payload = {
        "stage": stage,
        "package_version": __version__,
        "profile": config["profile"],
        "config_hash": config_hash(config),
        **git_provenance(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "upstream_config_hashes": upstream or {},
        "inputs": inputs,
        "outputs": outputs,
        "counts": counts,
        "quality_flags": quality_flags,
    }
    path = manifest_path(config, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return path


def write_report(config: dict[str, Any], name: str, payload: dict[str, Any]) -> Path:
    """Write a machine-readable report to outputs/reports/<name>.json, tagged with the config hash."""
    path = Path(config["paths"]["outputs_root"]) / "reports" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"report": name, "config_hash": config_hash(config), **payload}
    path.write_text(json.dumps(body, indent=2, default=str) + "\n")
    return path
