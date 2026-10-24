# NiftyGEX: Long Attention, Short Gamma

CS 7150 Deep Learning term project (Bhalchandra Shinde and Asad Mulani). Traders compress the NIFTY option chain into one number, gamma exposure (GEX). We ask whether a model that reads every contract (self-attention over about 70 option contracts per snapshot, plus a GRU over the last hour of snapshots) forecasts next-day volatility better than GEX-based, classical and tree-based models.

The proposal deck is `project_docs/Long_Attention_Short_Gamma_CS7150_Proposal_v10.pptx` (v9 as presented, plus wording fixes). Its three research questions:

| | Question | How it is tested |
|---|---|---|
| RQ1 | Does reading every contract predict better? | Attention + GRU against every classical rung, on the same walks and rows |
| RQ2 | Why does it win, if it does? | Switch off cross-contract attention (DeepSets) and history (single snapshot) |
| RQ3 | Does it survive rule changes? | Scores before and after the November 2024 weekly-expiry rationalisation |

The tests are pre-registered in `configs/eval.yaml` (`multiple_testing`).

## Status (2026-10-14)

| Proposal step | State | Where |
|---|---|---|
| Data pipeline: cleaning, pricing, GEX, features, tokens, the T2 target, leakage checks | Done | `src/niftygex/`, `scripts/compute_*.py`, `docs/validation_report.md` |
| Walk-forward harness (expiry cycles, embargo, purge) | Done | `src/niftygex_eval/harness.py` |
| Forecasting path and classical ladder: steps 0–3 (historical mean, persistence, HAR-RV, HAR + GEX, tuned LightGBM) | Done | `src/niftygex_eval/forecast/`, `docs/findings_forecast.md` |
| Token history for sequence models: 72 snapshots per day, 09:20–15:15 (the GRU slices the last L) | Done | `scripts/compute_tokens.py`, `docs/findings_data.md` |
| Volatility-state vector for the model head | Next | — |
| Neural models: steps 4–6 (DeepSets, self-attention, attention + GRU) | Planned | `src/niftygex_eval/nn/` |
| RQ2 switch-off tests, RQ3 rule-change tests, attention maps | Planned | — |

**Tracking:** `docs/PROJECT_STATUS.md` is the single page for what is done and what is pending, mapped to the proposal's steps and RQs. Earlier exploratory analyses are archived in `archive/` (see `archive/README.md`).

## Layout

```text
src/niftygex/        data layer: pure domain logic and stage helpers (no modeling)
src/niftygex_eval/   evaluation layer: harness, forecasters, comparisons, figures, results store
scripts/             thin config-keyed entry points (one per stage, run or report)
configs/             data.yaml, regimes.yaml, eval.yaml, path profiles
tests/               unit, leakage and stage-contract tests
docs/                schema, pricing, evaluation, validation, findings (findings files are script-generated), PROJECT_STATUS
archive/             exploratory analyses outside the proposal plan (frozen; see archive/README.md)
notebooks/           presentation notebooks (read only)
data/                gitignored: raw/ (immutable), curated/, greeks/, exposure/, features/, targets/, sample_data/
outputs/             gitignored: manifests/, reports/ (+ reports/figures/), results/ (runs, comparisons)
```

The evaluation layer reads only feature and target Parquet plus manifests, and never writes under `data/`. Every manifest and results-store run records the data and eval config hashes, `git_commit` and `git_dirty`.

## Setup

```bash
conda env create -f env/environment.yml
conda activate niftygex
```

All stage runs and tests use this pinned environment (Python 3.11, Polars 1.22). The default path profile is `local` (`configs/paths.local.yaml`, `data_root: ./data`). Override with `NIFTYGEX_PROFILE` and `NIFTYGEX_DATA_ROOT`.

Raw data is private and not in the repository. `data/sample_data/` holds a three-month raw extract (May–July 2022) in the same layout, for sharing.

## Run order

Run from the repository root with `PYTHONPATH=src`. Each stage refuses to run when an upstream manifest was produced under a different config hash, so after any change to `configs/data.yaml` rerun from step 2. Steps 2–4 take about three minutes on a laptop.

1. **Import raw data** (dry run, then copy; refuses to overwrite differing files):

   ```bash
   python scripts/import_raw.py --source-root /path/to/nifty-options-feature/data
   python scripts/import_raw.py --source-root /path/to/nifty-options-feature/data --copy
   ```

2. **Clean and price**: sessions, calendar, futures roll, spot relabel; basis; parity forwards and OTM-leg IV; Black-76 gamma and delta; GEX under three sign conventions, with key levels.

   ```bash
   python scripts/clean_chain.py
   python scripts/compute_basis.py
   python scripts/compute_forward.py
   python scripts/compute_greeks.py
   python scripts/compute_gex.py
   ```

3. **Features, tokens, targets and validation** (targets are stored separately from features):

   ```bash
   python scripts/compute_features.py
   python scripts/compute_tokens.py
   python scripts/compute_targets.py
   python scripts/validate_data.py        # exits non-zero on any timestamp or leakage violation
   python scripts/report_missing_data.py  # outputs/reports/missing_data.md
   python scripts/report_data.py          # features and token history: data_overview.json, figures, docs/findings_data.md
   ```

4. **Forecasting ladder**: every rung on the same walks and rows, then the report with figures.

   ```bash
   python scripts/run_forecast.py         # results store + outputs/results/forecast_comparison.json
   python scripts/report_forecast.py      # outputs/reports/forecast_baselines.json, figures, docs/findings_forecast.md
   ```


## Reports

Every number in a report is written by a script; none are entered by hand.
- **Data stages** write to `outputs/reports/`: cleaning, duplicates, bar completeness, spot outliers, basis jumps, spot label alignment, forward, Greeks, GEX, levels, features, targets, validation and missing data.
- **Each major step** writes a JSON report, figures under `outputs/reports/figures/`, and a readable page under `docs/`: `docs/findings_data.md` (features and token history) and `docs/findings_forecast.md` (classical baselines).

## Key conventions

- **Timestamps** are IST. Bars are start-labeled (09:15–15:25) and every row carries `available_at`.
- **Daily sources** become available at 09:15 on the next trading day.
- **Target:** log of next-day front-month futures RV over its trailing 20-day mean, at the 15:15 snapshot. This is the only target the data layer builds (`selection.TARGETS`).
- **GEX** = Γ × OI (index units) × S² × 0.01 over expiries within 21 days. Tokens cover each day's nearest included expiry at every 5-minute snapshot from 09:20 to 15:15.
- **Missing data is never imputed:** missing-OI and missing-spot days are flagged.

Full details are in `CLAUDE.md`, `docs/schema.md` and `docs/pricing.md`.

## Notebooks

`notebooks/01_input_datasets.ipynb` is a presentation overview of the input datasets. Register the kernel once:

```bash
python -m ipykernel install --user --name niftygex --display-name "Python 3.11 (niftygex)"
```

## Tests

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src python -m py_compile src/niftygex/**/*.py scripts/*.py
```

`tests/test_stage_contracts.py` checks invariants on the generated outputs and is skipped when `data/` has not been built. `tests/test_forecast.py` checks the forecasting harness:
- an oracle scores an IC of 1 and a random model scores near 0;
- tuning and fitting never see test days.

## Acknowledgements

The original prototype that this pipeline adapts from (`nifty-options-feature`) was developed by Sunil Kandakur together with Bhalchandra; its repository is authored under Sunil Kandakur's name. We thank Sunil Kandakur for the original prototype and for permission to adapt it for this course project. Adapted-code provenance is in `PROVENANCE.md`.
