"""Evaluation configuration: configs/eval.yaml plus the data-layer config for paths."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import yaml

from niftygex.config import CONFIG_ROOT, REPO_ROOT, load_config


def load_eval_config(profile: str | None = None) -> dict[str, Any]:
    data_config = load_config(profile)
    with (CONFIG_ROOT / "eval.yaml").open() as stream:
        evaluation = yaml.safe_load(stream)
    store = REPO_ROOT / evaluation["results_store"]
    return {"data_config": data_config, "eval": evaluation, "results_root": store}


def eval_hash(evaluation: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(evaluation, sort_keys=True, default=str).encode()).hexdigest()
