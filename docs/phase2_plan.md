# Phase 2 plan: NiftyGEX data preparation and feature extraction

Date: 2026-09-30

This plan incorporates the approved audit corrections. Phase 2 in this document is design-only: no files will be copied from the legacy project until approval.

## Design decisions

- Common data coverage: `2019-02-01` through `2026-07-03`, configured in `configs/data.yaml`.
- Raw data is stored inside the repository at `data/`, but the complete tree is gitignored. Runtime paths resolve from `configs/paths.<profile>.yaml`, selected by `NIFTYGEX_PROFILE` (`local` or `hpc`); `NIFTYGEX_DATA_ROOT` overrides the profile's data root. The local profile uses `./data` relative to the repository root.
- No raw data, generated outputs, Parquet, or CSV files are committed. `configs/paths.*.yaml` is ignored; `configs/paths.example.yaml` is committed; `data/README.md` is the sole tracked exception under `data/`.
- No bhavcopy is currently available. The 15:25 option-bar OI is the EOD proxy, usable from the next trading day. Optional bhavcopy input and reconciliation skip cleanly when absent.
- Daily India VIX lagged to the previous close is the main VIX feature. Intraday VIX is optional and may remain NaN.
- IV is computed from bar close because bid/ask is absent. A snapshot option row requires `volume > 0` and the configured forward-relative moneyness band.
- The synthetic forward is the median of parity-implied forwards across configured near-ATM strikes. Store each parity residual plus an aggregate forward-quality flag.
- GEX uses spot in the `S^2` scaling. Lot sizes and expiry weekdays come from manually maintained `configs/regimes.yaml`.
- Inferred GEX sign is NaN, with an availability flag, when participant data is missing; never forward fill it.
- T1 uses an optional daily official NSE closing-index input. Its absence is reported and T1 rows remain unavailable rather than silently substituting an unofficial close.
- Every stage consumes and produces configured Parquet datasets, records its config/profile, and can run independently as a local process or a SLURM array task.
- All timestamps are timezone-aware `Asia/Kolkata` (IST), including source normalization, partition keys, availability timestamps, and target horizons.
- Features are built only on the configured snapshot grid: T2 at 15:15 once per trading day, T3 at 14:00, and T1 at 10:00, 12:00, and 14:00 on expiry days.

## Target repository structure

```text
NiftyGEX/
  CLAUDE.md
  README.md
  PROVENANCE.md
  .gitignore
  pyproject.toml
  configs/
    data.yaml
    regimes.yaml
    paths.example.yaml
    paths.local.yaml             # committed, data_root: ./data
    paths.hpc.yaml               # ignored, user-created
  env/
    environment.yml
  hpc/
    sync_to_hpc.sh
    cleanup_hpc.sh
    slurm/                       # intentionally empty in Phase 2/3
  scripts/
    import_raw.py
    clean_chain.py
    compute_greeks.py
    compute_features.py
    compute_targets.py
    validate_data.py
  src/niftygex/
    __init__.py
    config.py
    pipeline.py
    io/
      __init__.py
      sources.py
      parquet.py
      schemas.py
      availability.py
    clean/
      __init__.py
      chain.py
      oi.py
      session.py
      filters.py
    pricing/
      __init__.py
      black76.py
      forward.py
      implied_vol.py
      greeks.py
    exposure/
      __init__.py
      gex.py
      levels.py
      signs.py
    features/
      __init__.py
      aggregate.py
      tokens.py
    targets/
      __init__.py
      pinning.py
      vol_regime.py
      intraday_drift.py
    validation/
      __init__.py
      checks.py
      leakage.py
      report.py
  tests/
    test_black76.py
    test_forward.py
    test_gex.py
    test_cleaning.py
    test_features.py
    test_targets.py
    test_leakage.py
    test_validation.py
  docs/
    audit.md
    phase2_plan.md
    schema.md
    validation_report.md
```

`data/` and `outputs/` are runtime directories and are ignored with re-inclusion-safe child patterns; only the tracked `data/README.md` is allowed under `data/`. `data/` is the default local runtime root; HPC and environment overrides can point to another root.

## Config and job contract

`src/niftygex/config.py` will resolve configuration in this order:

1. Read `NIFTYGEX_PROFILE`, defaulting to the committed local profile.
2. Load `configs/paths.<profile>.yaml`.
3. If `NIFTYGEX_DATA_ROOT` is set, replace only the profile's external raw-data root while retaining profile-relative curated/output roots.
4. Load `data.yaml` and `regimes.yaml`.
5. Attach a stable config hash to stage metadata and output manifests.

Each CLI will accept a date range and optional partition/unit key, but all defaults come from config. Stages will use deterministic partition keys such as `year=YYYY/month=MM` or `date=YYYY-MM-DD`; a SLURM array task maps to one configured partition without code changes. A stage will fail if its declared input manifest/config hash is incompatible, rather than silently mixing profiles or parameter versions.

`configs/data.yaml` will define `snapshot_grid`, `lookback`, `max_gex_days_to_expiry: 30`, and all cleaning/liquidity/moneyness filters. GEX is computed and stored per expiry; the aggregate exposure view sums expiries with days-to-expiry at or below the configured threshold. Per-strike tokens use the nearest expiry only.

Partitioned stages load the configured lookback window from earlier partitions before computing rolling features or causal cutoffs. A regression test will assert that a single-month run equals the corresponding month extracted from a full-range run.

The output contract will separate features from targets. Feature Parquet contains only information available by snapshot time; target Parquet contains forward outcomes and target availability metadata. No target columns will be joined into the feature-stage input.

## Files to create or rewrite

### Project/configuration files

- `CLAUDE.md`: project conventions, stage/job contract, no-hard-coded-path rule, and validation commands.
- `README.md`: setup, profile selection, independent stage commands, local execution, and HPC execution.
- `.gitignore`: use `data/*` and `outputs/*` with the tracked `data/README.md` exception; block `*.parquet`, `*.csv`, and machine-specific path profiles while allowing committed local/example profiles.
- `pyproject.toml`: Python 3.11+, minimal runtime dependencies, package configuration, pytest, and lint settings.
- `configs/data.yaml`: `2019-02-01` to `2026-07-03`, timezone, session rules, snapshot grid, configured rolling lookback, maximum expiry aggregation horizon, moneyness/liquidity filters, near-ATM forward settings, risk-free default, optional-source policy, and partitioning.
- `configs/regimes.yaml`: lot-size history, expiry weekdays, holidays/session exceptions, STT/rule-change dates, and optional by-date risk-free-rate series as manually maintained regime inputs.
- `configs/paths.example.yaml`: documented external path keys for raw options, futures, spot, VIX, participant OI, optional bhavcopy, official closes, curated data, targets, reports, and logs.
- `configs/paths.local.yaml`: committed local profile containing only `data_root: ./data`, resolved relative to the repository root. `paths.hpc.yaml` remains ignored.
- `env/environment.yml`: pinned Python 3.11, Polars, PyArrow, NumPy, SciPy, PyYAML, pytest, and the selected lint/tool versions.
- `PROVENANCE.md`: one row per adapted legacy file with original path, original author, change summary, and license/course-disclosure note. Every adapted file's author field will initially be `TO CONFIRM` for the user to fill in.

### IO and cleaning

- `src/niftygex/config.py`: new profile/env resolver and config-hash contract; rewrite of legacy `nofe/config.py`.
- `src/niftygex/io/sources.py`: external raw CSV/CSV.GZ/Parquet readers and normalized source names; adapt selected logic from `nofe/io/loaders.py`.
- `src/niftygex/io/parquet.py`: independent partitioned Parquet reads/writes, manifests, and atomic stage metadata; adapt persistence patterns from `nofe/panel/persist.py` and `nofe/labels/persist.py`.
- `src/niftygex/io/schemas.py`: normalized raw/curated schemas and schema validation; adapt concepts from `nofe/io/schema.py`.
- `src/niftygex/io/availability.py`: source publication/availability timestamps, including next-day EOD OI and participant-OI lag.
- `scripts/import_raw.py`: copy-only legacy importer; skips uncompressed duplicates of `.csv.gz`, prints the total size before copying, and writes file counts, sizes, and SHA-256 checksums to an output manifest.
- `src/niftygex/clean/session.py`: 09:15-15:30 IST session filtering, special/truncated-session handling, and timestamp normalization.
- `src/niftygex/clean/chain.py`: deduplication by snapshot/expiry/strike/right, normalized option fields, and curated-chain output.
- `src/niftygex/clean/oi.py`: 15:25 EOD OI proxy, optional bhavcopy join, next-day availability, and no-bhavcopy skip behavior.
- `src/niftygex/clean/filters.py`: `volume > 0`, forward-relative moneyness band, liquidity flags, and curated-row eligibility.
- `src/niftygex/clean/futures.py`: configured front-month selection and roll rule for futures basis; new implementation.
- `scripts/clean_chain.py`: thin config-keyed cleaning-stage entry point, rewritten from the useful shape of `scripts/build_dataset.py`/`build_panel.py`.

### Pricing and exposure

- `src/niftygex/pricing/black76.py`: pure Black-76 pricing and Greek formulas, adapted from `nofe/greeks/black76.py`; remove unused model-specific Greeks unless required.
- `src/niftygex/pricing/black76.py`: use a config constant risk-free rate by default, with an optional by-date series from `regimes.yaml`; compute time to expiry as calendar years to 15:30 IST on expiry day.
- `src/niftygex/pricing/forward.py`: median near-ATM parity-implied forward, requiring volume > 0 for both call and put in the same bar; store per-strike residuals, count/dispersion metrics, and quality flag; rewrite `nofe.greeks.pricing.implied_forward`.
- `src/niftygex/pricing/implied_vol.py`: close-price IV inversion with validity/convergence/volume/moneyness flags; adapt `invert_iv` from `nofe/greeks/pricing.py`.
- `src/niftygex/pricing/greeks.py`: snapshot-level IV and Greek orchestration with no disk I/O; adapt only the pure core from `nofe/greeks/engine.py`.
- `src/niftygex/exposure/gex.py`: per-expiry GEX under naive, absolute, and inferred conventions using spot `S^2`, configured lot size, and explicit missing-sign behavior; aggregate only expiries within the configured 30-day horizon.
- `src/niftygex/exposure/signs.py`: participant/OI-change inference and availability flags with no forward fill; new implementation.
- `src/niftygex/exposure/levels.py`: flip, walls, max-GEX, and max-pain calculations; new implementation, with explicit tie/missing policies.
- `scripts/compute_greeks.py`: config-keyed Parquet stage for forward, IV, and Greeks.

### Features and targets

- `src/niftygex/features/aggregate.py`: one row per snapshot with the approved approximately 20 features across GEX profile, positioning, volatility surface, and market state; rewrite selected primitives from `nofe/features/chain.py`, `option.py`, and `market.py`.
- `src/niftygex/features/tokens.py`: one vector per liquid strike/snapshot from the nearest expiry only, stable field order, call/put encoding, missingness, and source availability fields; new contract.
- `scripts/compute_features.py`: independent aggregate/token stage entry point.
- `src/niftygex/targets/pinning.py`: T1 from official NSE closing index, nearest strike, max-GEX/max-pain distance, and eligibility/availability reason; new implementation.
- `src/niftygex/targets/vol_regime.py`: primary T2 next-day RV versus trailing 20-day RV terciles, with RV/ATM-IV ratio retained as a secondary column and cutoffs computed causally; new implementation.
- `src/niftygex/targets/intraday_drift.py`: T3 14:00-15:30 movement toward max-GEX, using only post-snapshot bars; new implementation.
- `scripts/compute_targets.py`: independent target-stage entry point writing separate target Parquet.

### Validation and documentation

- `src/niftygex/validation/checks.py`: parity residuals, call/put IV agreement, approximately 30-day ATM IV versus India VIX, optional bhavcopy reconciliation, and gamma sanity checks.
- `src/niftygex/validation/leakage.py`: feature-source timestamp `<=` snapshot timestamp assertions, participant/EOD lag assertions, and target separation checks.
- `src/niftygex/validation/report.py`: deterministic Markdown/JSON report writer, including skipped optional checks and reasons.
- `scripts/validate_data.py`: config-keyed validation stage producing `docs/validation_report.md` and machine-readable output.
- `docs/schema.md`: every curated, Greek, GEX, aggregate, token, and target column with type, unit, source, timestamp, availability time, and missingness semantics. It will explicitly document close-based IV and the `volume > 0` plus moneyness eligibility rule.

### Tests

- `tests/test_black76.py`: known Black-76 values, call/put gamma equality, and ATM gamma behavior.
- `tests/test_forward.py`: parity recovery, median near-ATM aggregation, residuals, and quality flags.
- `tests/test_forward.py`: also rejects a parity pair unless both legs have positive volume in the same snapshot bar.
- `tests/test_gex.py`: all three sign conventions, spot `S^2` scaling, lot-size changes, and missing inferred signs.
- `tests/test_cleaning.py`: session boundaries, deduplication, volume/moneyness filters, 15:25 EOD proxy availability, timezone-aware IST timestamps, and configured futures front-month roll selection.
- `tests/test_features.py`: aggregate/token shapes, daily-VIX lag behavior, per-expiry GEX aggregation horizon, nearest-expiry tokens, snapshot-grid enforcement, and partition lookback equivalence.
- `tests/test_targets.py`: T1 official-close eligibility, primary T2 next-day RV versus trailing 20-day RV terciles, secondary RV/ATM-IV ratio, and T3 post-snapshot horizon.
- `tests/test_leakage.py`: deliberately future-dated feature source that must fail the timestamp assertion.
- `tests/test_validation.py`: optional bhavcopy skip, IV agreement, VIX comparison, parity residual, and gamma sanity reports.

### HPC helpers

- `hpc/sync_to_hpc.sh`: `rsync` only named curated feature and target Parquet partitions plus manifests/config metadata; never raw data. The initial script will require explicit source/destination/profile arguments and a list of partitions.
- `hpc/cleanup_hpc.sh`: guarded deletion of configured HPC data/output paths, requiring an explicit profile and confirmation flag; never touches repository source.
- `hpc/slurm/`: remains empty in this phase, as requested. The stage CLIs and partition contracts are designed for future array wrappers.

## Legacy adaptations and exclusions

The likely adapted legacy files are:

- `src/nofe/greeks/black76.py`: pure formulas; preserve behavior, add types/tests.
- `src/nofe/greeks/pricing.py`: IV inversion only; replace the forward estimator with the approved median policy.
- `src/nofe/greeks/engine.py`: pure snapshot orchestration only; remove live/disk drivers and hard-coded operational assumptions.
- `src/nofe/io/loaders.py`: source parsing and column normalization; replace repo-root paths with profile paths.
- `src/nofe/io/schema.py`: schema ideas; expand with units and availability times.
- `src/nofe/io/calendar.py`: calendar helpers; move regimes/rule changes into explicit config.
- `src/nofe/features/chain.py`: chain OI aggregation.
- `src/nofe/features/option.py`: only reviewed causal feature primitives.
- `src/nofe/features/market.py`: only selected spot/futures/basis primitives after timestamp review.
- `src/nofe/panel/persist.py` and `src/nofe/labels/persist.py`: Parquet persistence patterns only.

The following will not be copied: all `live/`, `model/`, `dataset/`, dashboard, training, backtest, report, experiment, old-label builders, generated data, notebooks, bytecode, and absolute-path analysis scripts.

Files flagged for user review because they reference another user's home directory:

- `scripts/analysis/adx_ledger_diag.py`
- `scripts/analysis/adx_monthly_table.py`
- `scripts/analysis/adx_val_pick.py`
- `scripts/analysis/flow_features.py`

These are analysis/backtest artifacts and are not proposed adaptations.

## Build order and approval gate

After approval, implementation will proceed in this order, running focused tests after each group and reporting results before moving on:

1. Skeleton and configuration, including timezone, snapshot grid, lookback, rate, and partition contracts.
2. `import_raw.py` and manifest/checksum handling.
3. Cleaning: session, chain, futures, and OI.
4. Pricing and forward estimation.
5. Per-expiry GEX and levels.
6. Aggregate features.
7. T2 targets.
8. Leakage and validation checks.
9. Token features.
10. T1 and T3 targets.

No new build work or legacy adaptation will begin until this updated plan is confirmed.
