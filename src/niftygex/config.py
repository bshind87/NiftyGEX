"""Configuration loading with profile and environment overrides."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "configs"


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_config(profile: str | None = None) -> dict[str, Any]:
    selected_profile = profile or os.environ.get("NIFTYGEX_PROFILE", "local")
    profile_path = CONFIG_ROOT / f"paths.{selected_profile}.yaml"
    if not profile_path.exists():
        raise FileNotFoundError(f"Missing config profile: {profile_path}")
    paths = _read_yaml(profile_path)
    data = _read_yaml(CONFIG_ROOT / "data.yaml")
    regimes = _read_yaml(CONFIG_ROOT / "regimes.yaml")

    data_root_value = os.environ.get("NIFTYGEX_DATA_ROOT", paths.get("data_root", "./data"))
    data_root = _resolve_repo_path(data_root_value)
    outputs_root = _resolve_repo_path(paths.get("outputs_root", "./outputs"))
    legacy_value = os.environ.get("NIFTYGEX_LEGACY_DATA_ROOT", paths.get("legacy_source_root"))
    resolved_paths = {
        "data_root": data_root,
        "outputs_root": outputs_root,
        "raw_root": data_root / "raw",
        "curated_root": data_root / "curated",
        "greeks_root": data_root / "greeks",
        "features_root": data_root / "features",
        "targets_root": data_root / "targets",
        "exposure_root": data_root / "exposure",
        "reports_root": outputs_root / "reports",
        "manifests_root": outputs_root / "manifests",
        "legacy_source_root": _resolve_repo_path(legacy_value) if legacy_value else None,
    }
    return {"profile": selected_profile, "paths": resolved_paths, "data": data, "regimes": regimes}


def load_intraday_config(profile: str | None = None) -> dict[str, Any]:
    """The data config plus configs/intraday.yaml (exploratory E3); its config_hash differs from the T2 hash."""
    config = load_config(profile)
    return {**config, "intraday": _read_yaml(CONFIG_ROOT / "intraday.yaml")}


def config_hash(config: dict[str, Any]) -> str:
    """Return a stable hash for serializable configuration values."""
    serializable = {
        key: {name: str(value) if isinstance(value, Path) else value for name, value in value.items()}
        if isinstance(value, dict)
        else value
        for key, value in config.items()
        if key != "paths"
    }
    # The legacy import source does not affect any stage output, so it is excluded from the hash.
    serializable["paths"] = {
        key: str(value) for key, value in config["paths"].items() if key != "legacy_source_root"
    }
    payload = json.dumps(serializable, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()
