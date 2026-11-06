"""Results store: one directory per run with run.json, predictions and metrics, plus an append-only index."""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from niftygex.io.provenance import git_provenance


class ResultsStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    @staticmethod
    def run_id(spec: dict[str, Any]) -> str:
        """Deterministic id from the run specification (stage, model, hashes, parameters)."""
        return hashlib.sha256(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def write_run(self, spec: dict[str, Any], tables: dict[str, pl.DataFrame], summary: dict[str, Any]) -> Path:
        """Write a run; rerunning an identical spec replaces the previous copy of that run."""
        for key in ("stage", "data_config_hash", "eval_config_hash"):
            if key not in spec:
                raise ValueError(f"run spec needs {key!r}")
        spec = {**git_provenance(), **spec}
        run_id = self.run_id(spec)
        directory = self.root / "runs" / run_id
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
        record = {"run_id": run_id, "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "spec": spec, "summary": summary}
        (directory / "run.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
        for name, table in tables.items():
            table.write_parquet(directory / f"{name}.parquet")
        index = self.root / "index.jsonl"
        index.parent.mkdir(parents=True, exist_ok=True)
        with index.open("a") as stream:
            stream.write(json.dumps({"run_id": run_id, "created_at_utc": record["created_at_utc"], "stage": spec["stage"],
                                     "model": spec.get("model"), "target": spec.get("target"),
                                     "data_config_hash": spec["data_config_hash"], "eval_config_hash": spec["eval_config_hash"],
                                     "git_commit": spec.get("git_commit"), "git_dirty": spec.get("git_dirty")}) + "\n")
        return directory

    def checkpoint_dir(self, spec: dict[str, Any]) -> Path:
        """Per-walk checkpoints of the run that ``spec`` will become (same id, so a restart on the same commit resumes)."""
        return self.root / "checkpoints" / self.run_id({**git_provenance(), **spec})

    def clear_checkpoint(self, spec: dict[str, Any]) -> None:
        shutil.rmtree(self.checkpoint_dir(spec), ignore_errors=True)

    def read_run(self, run_id: str) -> dict[str, Any]:
        return json.loads((self.root / "runs" / run_id / "run.json").read_text())
