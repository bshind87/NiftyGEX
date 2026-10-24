"""Copy legacy raw sources into the configured, gitignored data/raw tree.

Paths come from the active profile (``NIFTYGEX_PROFILE``, ``NIFTYGEX_DATA_ROOT``). The legacy source root
comes from ``--source-root``, ``NIFTYGEX_LEGACY_DATA_ROOT``, or the profile's ``legacy_source_root``.
A dry run writes ``raw_import_preview.json`` and never touches the copy manifest. Copying never
overwrites an existing raw file with different contents.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from niftygex.config import config_hash, load_config
from niftygex.io.parquet import file_sha256
from niftygex.io.provenance import git_provenance

SOURCE_SUBDIRECTORIES = ("options", "futures/NIFTY", "equity", "india_vix", "fiidii")


@dataclass(frozen=True)
class FileRecord:
    source_group: str
    source: str
    destination: str
    size_bytes: int
    sha256: str | None
    date_min: str | None
    date_max: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", type=Path, default=None, help="Legacy data root; overrides env and profile.")
    parser.add_argument("--copy", action="store_true", help="Copy files after showing the inventory.")
    return parser.parse_args()


def resolve_source_root(value: Path | None, config: dict) -> Path:
    if value is not None:
        return value.expanduser().resolve()
    configured = config["paths"].get("legacy_source_root")
    if configured is None:
        raise SystemExit("Provide --source-root, set NIFTYGEX_LEGACY_DATA_ROOT, or set legacy_source_root in the profile")
    return Path(configured)


def is_uncompressed_duplicate(path: Path) -> bool:
    return path.suffix == ".csv" and path.with_name(path.name + ".gz").exists()


def discover_files(source_root: Path) -> list[Path]:
    files: list[Path] = []
    for subdirectory in SOURCE_SUBDIRECTORIES:
        root = source_root / subdirectory
        if not root.exists():
            continue
        files.extend(
            path for path in root.rglob("*")
            if path.is_file() and path.name != ".DS_Store" and not is_uncompressed_duplicate(path)
        )
    return sorted(files)


def date_coverage(path: Path) -> tuple[str | None, str | None]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as parquet

        parquet_file = parquet.ParquetFile(path)
        columns = set(parquet_file.schema.names)
        date_column = "datetime" if "datetime" in columns else "date" if "date" in columns else None
        if date_column is None:
            return None, None
        values = parquet_file.read(columns=[date_column]).column(date_column).to_pylist()
        dates = [value.isoformat()[:10] for value in values if value is not None]
        return (min(dates), max(dates)) if dates else (None, None)

    stream = gzip.open(path, "rt", newline="") if path.suffix == ".gz" else path.open("rt", newline="")
    with stream:
        date_min: str | None = None
        date_max: str | None = None
        for row in csv.DictReader(stream):
            value = row.get("datetime") or row.get("date")
            if not value:
                continue
            day = value[:10]
            date_min = day if date_min is None else min(date_min, day)
            date_max = day if date_max is None else max(date_max, day)
    return date_min, date_max


def build_records(files: list[Path], source_root: Path, raw_root: Path, include_checksums: bool) -> list[FileRecord]:
    records = []
    for source in files:
        date_min, date_max = date_coverage(source)
        records.append(
            FileRecord(
                source_group=source.relative_to(source_root).parts[0],
                source=str(source),
                destination=str(raw_root / source.relative_to(source_root)),
                size_bytes=source.stat().st_size,
                sha256=file_sha256(source) if include_checksums else None,
                date_min=date_min,
                date_max=date_max,
            )
        )
    return records


def write_manifest(path: Path, records: list[FileRecord], config: dict, *, copied: bool, skipped_identical: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    source_summary: dict[str, dict] = {}
    for record in records:
        summary = source_summary.setdefault(record.source_group, {"file_count": 0, "total_size_bytes": 0, "date_min": None, "date_max": None})
        summary["file_count"] += 1
        summary["total_size_bytes"] += record.size_bytes
        if record.date_min is not None:
            summary["date_min"] = record.date_min if summary["date_min"] is None else min(summary["date_min"], record.date_min)
        if record.date_max is not None:
            summary["date_max"] = record.date_max if summary["date_max"] is None else max(summary["date_max"], record.date_max)
    payload = {
        "stage": "import_raw",
        "profile": config["profile"],
        "config_hash": config_hash(config),
        **git_provenance(),
        "source_mode": "copy-only",
        "copied": copied,
        "skipped_identical_existing": skipped_identical,
        "file_count": len(records),
        "total_size_bytes": sum(record.size_bytes for record in records),
        "sources": source_summary,
        "files": [asdict(record) for record in records],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def copy_records(records: list[FileRecord]) -> int:
    """Copy new files; skip identical existing files; refuse to overwrite a different raw file."""
    conflicts = [
        record.destination for record in records
        if Path(record.destination).exists() and file_sha256(Path(record.destination)) != record.sha256
    ]
    if conflicts:
        raise SystemExit(f"data/raw is immutable; {len(conflicts)} destination files differ from the source, e.g. {conflicts[0]}")
    skipped = 0
    for record in records:
        destination = Path(record.destination)
        if destination.exists():
            skipped += 1
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record.source, destination)
    return skipped


def main() -> None:
    args = parse_args()
    config = load_config()
    source_root = resolve_source_root(args.source_root, config)
    raw_root = Path(config["paths"]["raw_root"])
    manifests = Path(config["paths"]["manifests_root"])
    files = discover_files(source_root)
    total_size = sum(path.stat().st_size for path in files)
    print(f"source_root={source_root}")
    print(f"raw_root={raw_root}")
    print(f"files={len(files)}")
    print(f"total_size_bytes={total_size}")
    print(f"total_size_gib={total_size / 1024**3:.3f}")
    if not args.copy:
        preview = manifests / "raw_import_preview.json"
        write_manifest(preview, build_records(files, source_root, raw_root, include_checksums=False), config, copied=False)
        print(f"dry-run only; no files copied (pass --copy to copy); preview={preview}")
        return
    records = build_records(files, source_root, raw_root, include_checksums=True)
    skipped = copy_records(records)
    manifest = manifests / "raw_import_manifest.json"
    write_manifest(manifest, records, config, copied=True, skipped_identical=skipped)
    print(f"copied={len(records) - skipped} skipped_identical={skipped}")
    print(f"manifest={manifest}")


if __name__ == "__main__":
    main()
