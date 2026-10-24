# Runtime data layout

This directory is local runtime storage and is fully gitignored except for this file. Do not commit raw, curated, Greek, feature, target, or generated files.

```text
data/
  raw/                 immutable copies of legacy sources (scripts/import_raw.py)
    options/ futures/ equity/ india_vix/ fiidii/ bhavcopy/
  curated/             stage clean (options, futures, market, vix, oi, calendar) and stage basis (basis)
  greeks/              stage forward (forward, parity, iv)
  features/
  targets/
```

Stage manifests are written to `outputs/manifests/` and reports to `outputs/reports/`.

Raw data is imported by `scripts/import_raw.py`. The importer copies files and never moves or deletes the legacy source. It skips an uncompressed file when the corresponding `.csv.gz` file exists, skips files already present with identical checksums, and refuses to overwrite a raw file with different contents.
