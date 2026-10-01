# NiftyGEX

NiftyGEX is an auditable data-preparation and feature-extraction pipeline for NIFTY options. It turns raw option, futures, spot, VIX, and participant-OI sources into curated IST-aware Parquet data, then computes Black-76 forwards, implied volatility, Greeks, GEX, aggregate features, token features, and forward-only targets for the CS 7150 term project.

Modeling and walk-forward evaluation live in the separate evaluation layer `src/niftygex_eval/`; the dashboard is not built yet.

## Setup

```bash
conda env create -f env/environment.yml
conda activate niftygex
```

All stage runs and tests use this pinned environment (Python 3.11, Polars 1.22).

The default profile is `local`, loaded from `configs/paths.local.yaml`. Select another profile with:

```bash
export NIFTYGEX_PROFILE=local
# Optional path override:
export NIFTYGEX_DATA_ROOT=/path/to/data
```

The local profile resolves `data_root: ./data` relative to the repository root. HPC paths are selected with `NIFTYGEX_PROFILE=hpc`; raw data stays outside any source-control operation through the profile and environment override. The legacy import source is given by `--source-root`, `NIFTYGEX_LEGACY_DATA_ROOT`, or `legacy_source_root` in the profile.

## Data layout

```text
data/
  raw/          copied source files; never written by pipeline stages
  curated/      cleaned options, futures, spot, VIX, OI, trading calendar, and basis
  greeks/       per-expiry forwards, parity residuals, IV, and Greeks
  exposure/     snapshot grid, GEX (strike, expiry, target), and key levels
  features/     aggregate and token features
  targets/      separate forward-looking targets
outputs/
  manifests/    one manifest per stage with config hash, inputs, outputs, counts, and quality flags
  reports/      machine-readable reports, each written by a script under scripts/
```

`data/`, `outputs/`, Parquet, and CSV files are gitignored. `data/README.md` is the tracked layout note.

## Run order

Run commands from the repository root with `PYTHONPATH=src`.

1. Preview and import raw data. The first command does not copy; the second explicitly copies.

```bash
PYTHONPATH=src python scripts/import_raw.py \
  --source-root /path/to/nifty-options-feature/data
PYTHONPATH=src python scripts/import_raw.py \
  --source-root /path/to/nifty-options-feature/data --copy
```

The dry run writes `outputs/manifests/raw_import_preview.json`. `--copy` skips files already present with identical checksums and refuses to overwrite a raw file whose contents differ.

2. Clean sessions, chains, futures, spot, VIX, and OI; build the trading calendar and day-level quality flags:

```bash
PYTHONPATH=src python scripts/clean_chain.py
```

3. Compute the futures-spot basis, classify basis jumps, and check spot/VIX label alignment:

```bash
PYTHONPATH=src python scripts/compute_basis.py
```

4. Compute per-expiry parity forwards, parity residuals, and close-based IV over the configured moneyness band:

```bash
PYTHONPATH=src python scripts/compute_forward.py
```

5. Compute Black-76 gamma and delta for eligible legs:

```bash
PYTHONPATH=src python scripts/compute_greeks.py
```

6. Build the snapshot grid, per-strike/per-expiry/per-target GEX under the naive, absolute, and inferred conventions, and key levels:

```bash
PYTHONPATH=src python scripts/compute_gex.py
```

7. Build aggregate snapshot features, then T2 targets (stored separately), then validation:

```bash
PYTHONPATH=src python scripts/compute_features.py
PYTHONPATH=src python scripts/compute_tokens.py
PYTHONPATH=src python scripts/compute_targets.py
PYTHONPATH=src python scripts/validate_data.py
PYTHONPATH=src python scripts/report_missing_data.py   # outputs/reports/missing_data.md
```

`validate_data.py` writes `docs/validation_report.md` and exits non-zero on any timestamp/leakage violation.

8. Evaluation layer (`src/niftygex_eval`, `configs/eval.yaml`; see `docs/evaluation.md`):

```bash
PYTHONPATH=src python scripts/run_stage_a.py   # T2 HAC regressions on total GEX
PYTHONPATH=src python scripts/run_stage_b.py   # T2 tercile classifiers in the walk-forward harness
PYTHONPATH=src python scripts/run_t2_ablation.py  # A/B/C feature-set ablation (classification + regression)
PYTHONPATH=src python scripts/write_findings_t2.py  # docs/findings_t2.md from the results store
PYTHONPATH=src python scripts/run_t3_placebo.py   # T3 matched-distance placebo
```

Runs are written to the results store `outputs/results/` (`runs/<run_id>/`, `index.jsonl`) with the data and eval config hashes.

Each stage after cleaning refuses to run if its upstream manifest was produced under a different configuration hash; after any config change, rerun from step 2.

The next stage is T1 (code and fixtures now; runs when official closes are available). Each stage will read/write Parquet and record its configuration and output manifest.

## Tests

```bash
PYTHONPATH=src pytest -q
```

Focused checks can be run before the full suite:

```bash
PYTHONPATH=src pytest tests/test_config.py tests/test_pricing.py -q
PYTHONPATH=src python -m py_compile src/niftygex/**/*.py scripts/*.py
```

`tests/test_stage_contracts.py` checks invariants on the generated outputs (session bounds, `available_at`, next-day availability, one futures contract per day, missing-spot flags, manifest hashes) and is skipped when `data/` has not been built.

Reports under `outputs/reports/`: `cleaning_report`, `duplicate_conflicts`, `bar_completeness`, `spot_outliers`, `basis_jumps`, `spot_label_alignment`, `forward_report`, `greeks_report`, `gex_report`, `levels_report`, `features_report`, `targets_t2_report`, `targets_t3_report`, and `validation_report`. Project status is tracked in `docs/PROJECT_STATUS.md`. Schemas and conventions are in `docs/schema.md` and `docs/pricing.md`; adapted-code provenance is in `PROVENANCE.md`.

## Acknowledgements

The original prototype that this pipeline adapts from (`nifty-options-feature`) was developed by Sunil Kandakur together with Bhalchandra; its repository is authored under Sunil Kandakur's name. We thank Sunil Kandakur for the original prototype and for permission to adapt it for this course project.
