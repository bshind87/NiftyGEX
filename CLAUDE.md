# NiftyGEX conventions

## Scope

This repository owns auditable NIFTY data preparation and feature extraction (`src/niftygex/`, the data layer) and, separately, the evaluation layer (`src/niftygex_eval/`: walk-forward harness, results store, the forecasting ladder and neural models; configured by `configs/eval.yaml`). Never add modeling, dashboard, or evaluation-harness code to the data layer; the evaluation layer reads only feature/target Parquet and manifests and never writes under `data/`. Results go to `outputs/results/` with both the data and eval config hashes.

## Structure

- `configs/`: all tunable dates, session rules, regimes, profiles, and filters.
- `data/`: gitignored runtime inputs and Parquet outputs; `data/raw/` is immutable after import.
- `outputs/`: gitignored runtime metadata. Stage manifests go in `outputs/manifests/<stage>.json`; every report is written by a script under `scripts/` to `outputs/reports/<name>.json`. Never hand-write report numbers.
- `src/niftygex/`: pure domain logic and stage helpers.
- `scripts/`: thin config-keyed stage entry points.
- `tests/`: focused unit and stage-contract tests.
- `docs/`: schemas, provenance, audit, and validation reports.
- `archive/`: frozen exploratory analyses outside the proposal plan (Stage A/B, T1/T3 targets and placebos, Holm F1–F4); not imported, tested or maintained.
- `notebooks/`: presentation and exploration notebooks (kernel `Python 3.11 (niftygex)`); they read data and reports through `load_config()` and never write under `data/` or `outputs/`.

## Path and stage rules

- Never hard-code machine paths, dates, or regime assumptions in Python.
- Resolve paths through `configs/paths.<profile>.yaml`, `NIFTYGEX_PROFILE`, and `NIFTYGEX_DATA_ROOT`.
- Every stage is Parquet-in/Parquet-out and can run independently by configured partition.
- Stages never write to `data/raw/`.
- Preserve manifests with configuration hashes, input provenance, counts, and quality flags. Manifests and results-store run specs also record `git_commit` and `git_dirty`. Never add Claude attribution to commits; the owner makes commits manually.
- Keep stage processing deterministic and sequential on macOS.
- Downstream stages refuse to run when an upstream manifest's `config_hash` differs from the current config.
- Stages write to a staging directory and swap their owned output directories in only on success.

## Time and market conventions

- All timestamps are timezone-aware `Asia/Kolkata`.
- Bars are start-labeled: a bar labeled `t` covers `[t, t+5min)` and is available at `t+5min`. Regular session labels are 09:15 through 15:25 inclusive (75 bars); 15:30 labels are dropped from every source.
- Every curated row carries `available_at`. Daily sources (EOD OI proxy, participant OI, daily VIX) are available at 09:15 IST on the next trading day strictly after the observation date.
- The trading calendar is the union of option and futures dates minus configured holidays, extended by weekday rule past the last observed date. A missing next trading day is an error, never a fallback to the same day.
- Spot and intraday VIX are end-labeled in the raw data from 2020-12-08 to 2026-04-23; `end_labeled_windows` in `configs/regimes.yaml` relabels them before the session filter. Verify alignment against futures and the option-implied forward (`spot_label_alignment` report) after any source change.
- IV for both legs of a strike comes from the OTM leg (call if K >= F, put if K < F); each leg's own IV is diagnostic only.
- Greeks use only `eligible` rows with `time_to_expiry > 0`. Total GEX excludes same-day expiries for T2 (`gex.same_day_expiry_in_total_gex`).
- Option `open_interest` is in index units (lot-size multiples): GEX = gamma x OI x spot^2 x 0.01 with no lot multiplier. Lot sizes in `regimes.yaml` are keyed by contract expiry and are tested against OI divisibility; the GEX horizon is 21 days (data maximum).
- T2 realized volatility uses front-month futures returns (one contract per day); spot RV is secondary. Tercile cutoffs use only ratios whose target day is strictly before the snapshot day.
- Daily sources in features match exactly the previous trading day (`available_date == trade_date`); never as-of a stale observation.
- Participant OI covers all index options; split results at the `weekly_expiry_rationalisation` regime marker.
- Significance claims use Holm within the pre-registered families in `configs/eval.yaml` (`multiple_testing`); everything else is labeled exploratory.
- Walk-forward evaluation is expiry-cycle aligned with >= 1 cycle of embargo, a purge on `target_available_at`, and no shuffling; `check_split` must pass for every walk.
- The source records missing option OI as 0 (2022-01-03..2022-04-11, 2019-07-30); those days are `missing_option_oi`, OI is nulled, and GEX/OI levels are missing. Never treat zero OI as real exposure.
- Time to expiry is calendar-year time from `available_at` to 15:30 IST on expiry day.
- Risk-free rate comes from configuration, not code constants.
- Duplicate resolution is deterministic and documented in `docs/schema.md`: options keep higher volume; futures keep the later-dated source file, then higher volume; then stable tie-breakers. Participant-OI keys with conflicting duplicates are dropped, not resolved by row order.
- Non-regular sessions are dropped. Truncated sessions remain flagged. Data-quality dates remain available with day-level `data_quality` codes.
- Days with option/futures data but no spot are flagged `missing_spot`, never imputed, and excluded from anything needing spot (GEX `S^2` scaling, RV, targets).
- Spot bad prints (|spot close / front-month futures close - 1| above the configured threshold) are nulled and logged.
- Futures front month is selected once per trading day; never switch contracts intraday.

## Leakage rules

- A feature at snapshot time `T` may use only rows with `available_at <= T`. Leakage tests compare `available_at`, never bar labels.
- Same-day EOD OI is unavailable intraday; the 15:25 proxy is available the next trading day.
- Participant OI is lagged one full trading day; never forward-fill missing participant signs.
- Targets use only data after the snapshot and are stored separately from features.
- Daily VIX gaps remain NaN with an availability flag; never forward-fill.
- Add or update timestamp-ordering tests whenever a stage adds a source.

## Checkpoint workflow

- Follow the approved build order and stop at requested checkpoints; stop and summarize after each stage group.
- After each group of edits, run focused tests before widening the scope.
- Report row counts before/after cleaning rules, duplicate splits, special-session actions, missing-source behavior, and output manifests.
- Do not proceed to downstream pricing/features/targets when a focused invariant fails.
- At the end of every checkpoint, update `docs/PROJECT_STATUS.md`: task statuses, summaries, outputs and run ids, the decisions log, and the "Last updated" date, hashes and test count. Mark anything not backed by code, tests, reports or results-store runs as ❓ Unverified, not ✅. Keep it a short status page.
- Regenerate `outputs/reports/missing_data.md` (`scripts/report_missing_data.py`) after any data change.

## Environment and test commands

Use the pinned `niftygex` conda env (Python 3.11, `env/environment.yml`) for every stage run and test; do not use the base interpreter.

```bash
conda activate niftygex
PYTHONPATH=src pytest -q
PYTHONPATH=src pytest tests/test_config.py tests/test_pricing.py -q
PYTHONPATH=src python -m py_compile src/niftygex/**/*.py scripts/*.py
```

When changing configuration or a builder, rerun the affected stage and all downstream stages, then re-run the focused tests and inspect the generated report.
