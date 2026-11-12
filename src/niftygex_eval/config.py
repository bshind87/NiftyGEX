"""Evaluation configuration: configs/eval.yaml plus the data-layer config for paths."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from niftygex.config import CONFIG_ROOT, load_config


def load_eval_config(profile: str | None = None) -> dict[str, Any]:
    data_config = load_config(profile)
    with (CONFIG_ROOT / "eval.yaml").open() as stream:
        evaluation = yaml.safe_load(stream)
    # The store lives under the profile's outputs root (outputs/results locally, scratch on the cluster).
    store = Path(data_config["paths"]["outputs_root"]) / Path(evaluation["results_store"]).relative_to("outputs")
    return {"data_config": data_config, "eval": evaluation, "results_root": store}


def eval_hash(evaluation: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(evaluation, sort_keys=True, default=str).encode()).hexdigest()


def load_exploratory_config(profile: str | None = None) -> dict[str, Any]:
    """The main evaluation config plus configs/eval_exploratory_v2.yaml (exploratory experiments E1-E3).

    ``exploratory_hash`` covers the exploratory file only; runs also record the main ``eval_config_hash``.
    """
    cfg = load_eval_config(profile)
    with (CONFIG_ROOT / "eval_exploratory_v2.yaml").open() as stream:
        exploratory = yaml.safe_load(stream)
    return {**cfg, "exploratory": exploratory, "exploratory_hash": eval_hash(exploratory)}


def load_exploratory_v3_config(profile: str | None = None) -> dict[str, Any]:
    """configs/eval_exploratory_v2.yaml (E1-E3, unchanged) plus configs/eval_exploratory_v3.yaml (checks A1-A4, B5-B8).

    ``exploratory_v3_hash`` covers the v3 file only; v3 runs record it next to the main and v2 hashes.
    """
    cfg = load_exploratory_config(profile)
    with (CONFIG_ROOT / "eval_exploratory_v3.yaml").open() as stream:
        v3 = yaml.safe_load(stream)
    return {**cfg, "exploratory_v3": v3, "exploratory_v3_hash": eval_hash(v3)}
