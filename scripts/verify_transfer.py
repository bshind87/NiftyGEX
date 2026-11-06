"""Check copied data against the stage manifests: every recorded output exists with the same size, rows and checksum.

  NIFTYGEX_PROFILE=hpc PYTHONPATH=src python scripts/verify_transfer.py [--no-checksums]

Run on the cluster after hpc/push_data.sh (hpc/verify_transfer.sbatch), or locally as a self-check. Reads every
outputs/manifests/*.json written on the Mac: the stages' own output records (paths relative to data_root or
outputs_root) and the raw import's file list. Writes outputs/reports/transfer_check.json; exits 1 on any problem.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

from niftygex.config import load_config
from niftygex.io.parquet import file_sha256


def resolve(record_path: str, paths: dict) -> Path | None:
    """A manifest path on this machine; absolute paths outside data/ and outputs/ (docs) are not data and are skipped."""
    for root_key in ("data_root", "outputs_root"):
        prefix = f"{root_key}/"
        if record_path.startswith(prefix):
            return Path(paths[root_key]) / record_path[len(prefix):]
    marker = "/data/raw/"
    if marker in record_path:
        return Path(paths["data_root"]) / "raw" / record_path.split(marker, 1)[1]
    return None


def check(record: dict, path: Path, checksums: bool) -> str | None:
    if not path.exists():
        return "missing"
    if record.get("size_bytes") is not None and path.stat().st_size != record["size_bytes"]:
        return f"size {path.stat().st_size} != {record['size_bytes']}"
    if record.get("rows") is not None and path.suffix == ".parquet":
        rows = pl.scan_parquet(path).select(pl.len()).collect().item()
        if rows != record["rows"]:
            return f"rows {rows} != {record['rows']}"
    if checksums and record.get("sha256") and file_sha256(path) != record["sha256"]:
        return "sha256 differs"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-checksums", action="store_true", help="sizes and row counts only")
    args = parser.parse_args()
    config = load_config()
    paths = config["paths"]
    stages, problems, checked, skipped = {}, [], 0, 0
    for manifest_path in sorted(Path(paths["manifests_root"]).glob("*.json")):
        manifest = json.loads(manifest_path.read_text())
        records = manifest.get("outputs") or [{**f, "path": f["destination"]} for f in manifest.get("files", []) if "destination" in f]
        ok = 0
        for record in records:
            path = resolve(record["path"], paths)
            if path is None:
                skipped += 1
                continue
            checked += 1
            issue = check(record, path, not args.no_checksums)
            if issue:
                problems.append({"stage": manifest.get("stage"), "path": record["path"], "issue": issue})
            else:
                ok += 1
        stages[manifest_path.stem] = {"config_hash": manifest.get("config_hash"), "files_ok": ok, "rows": manifest.get("counts")}
    report = {"profile": config["profile"], "data_root": str(paths["data_root"]), "checksums": not args.no_checksums,
              "files_checked": checked, "files_skipped_not_data": skipped, "problems": problems, "stages": stages}
    out = Path(paths["reports_root"]) / "transfer_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"checked {checked} files ({skipped} non-data skipped): {len(problems)} problems -> {out}")
    for problem in problems[:20]:
        print("  ", problem)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
