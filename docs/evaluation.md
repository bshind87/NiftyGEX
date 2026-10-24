# Evaluation layer

The evaluation code lives in `src/niftygex_eval/` and is configured by `configs/eval.yaml`. It is separate from the data layer (`src/niftygex/`):

- It reads only the feature and target Parquet files and the stage manifests. It never writes under `data/`.
- Its config hash (`eval_config_hash`) is separate from the data config hash, so changing evaluation settings never invalidates a data stage.

## Walk-forward harness (`niftygex_eval/harness.py`)

**Unit: the expiry cycle**
- Each trading day belongs to the cycle of the first option expiry on or after it.
- An expiry day therefore belongs to its own cycle.
- Days after the last known expiry have no cycle and never enter a walk.

**Walks**
- Walks are consecutive, non-overlapping test blocks of `test_cycles` (13) cycles, about one quarter each.
- Each walk trains on all cycles before its embargo (expanding window; `train_window_cycles: null`).
- Training must cover at least `min_train_cycles` (52).
- `embargo_cycles` (1) cycles sit between training and test and are never used. An embargo of 0 is rejected.

**Purge.** Training rows must also have `target_available_at` earlier than the first test snapshot. A label that resolves inside the test period can never be learned from, whatever the target horizon.

**No shuffling.** Rows stay in time order. LightGBM runs with `deterministic: true`, a fixed seed and one thread.

**Checks (`check_split`).** Every walk is checked when it is split, and the run stops on the first violation. The checks are:
- no test snapshot appears in training;
- the last training snapshot precedes the first test snapshot;
- no training label resolves at or after the first test snapshot;
- no embargo cycle is used for training.

`tests/test_harness.py` exercises all of these on synthetic data and on every walk of the real T2 table.

**Current forecasting layout.** 1,800 rows with a finite outcome, from 2019-02-21. 26 walks; test blocks run from 2020-02-24 to 2026-07-02, 1,556 test rows in total (30–65 per walk; the last block is partial).

## Results store (`niftygex_eval/results.py`)

The store lives at `outputs/results/` (gitignored).

**Run directory.** Each run is written to `runs/<run_id>/`, which holds:
- `run.json`, with the spec (stage, target, model, `data_config_hash`, `eval_config_hash`, harness parameters, feature list, model parameters) and a summary;
- one Parquet file per table, for example `predictions`, `walk_metrics`, `walks`, `regressions`.

**Run id.** `run_id` is a hash of the spec, so rerunning an identical spec replaces that run.

**Index.** `index.jsonl` gets one line per write.

**Inputs.** `modeling_table` refuses to join features and targets whose manifests have different data config hashes.

## Forecasting ladder (`niftygex_eval/forecast/`, `scripts/run_forecast.py`, `scripts/report_forecast.py`)

This is the proposal's main evaluation path. Every model predicts `log(rv_ratio_trailing)` at the T2 15:15 snapshot and is scored on the same walks and rows (`configs/eval.yaml`, section `forecast`).

**Modules:**
- `interface.py`: `Forecaster`. The runner calls `tune(train, valid)` on the inner split (the last 26 training cycles after a 1-cycle inner embargo), then `fit(train)` on the full training window with the chosen settings, then `predict(test)`.
- `runner.py`: `run_walks`, with `check_split` on both the outer and inner splits.
- `metrics.py`: IC (Spearman, primary), RMSE, QLIKE and R² against a reference. QLIKE is `exp(2(y − ŷ)) − 2(y − ŷ) − 1`, because the trailing denominator cancels.
- `compare.py`: paired per-walk tests with a 95% CI and walks won; Diebold–Mariano on daily losses with a Newey–West variance; scores by period.
- `baselines.py`: historical mean, persistence, HAR-RV (OLS on log RV terms), HAR + GEX (ridge, alpha tuned on the inner split) and tuned LightGBM.
- `table.py`: the forecasting table, adding normalized GEX and the outcome.

**Outputs:**
- `run_forecast.py`: one results-store run per model, plus `outputs/results/forecast_comparison.json`.
- `report_forecast.py`: `outputs/reports/forecast_baselines.json`, `outputs/reports/figures/forecast_*.png` and `docs/findings_forecast.md`.

The neural models (steps 4–6) will implement the same interface.

## Multiple testing

Holm step-down (`niftygex_eval/multiple_testing.py`) is applied within the families pre-registered in `configs/eval.yaml` (`multiple_testing`). They were fixed before any neural run. Every test outside them is **exploratory** and reported unadjusted.

| Family | Test | Decision rule |
|---|---|---|
| RQ1 | Attention + GRU vs persistence, HAR, HAR + GEX and tuned LightGBM: mean per-walk IC gain, 95% CI, paired t | Intersection–union: a win over the ladder requires every comparison to reject at 5% (no Holm). QLIKE, RMSE, walks won and Diebold–Mariano are supporting |
| RQ2 | DeepSets vs single-snapshot attention (interactions); single-snapshot attention vs attention + GRU (history) | Holm within the family. Window-length and field-drop ablations are exploratory |
| RQ3 | Per-walk IC gain of attention + GRU over tuned LightGBM, walks after vs before the weekly-expiry marker | Welch t. Lot-size and volatility-regime splits are exploratory |

The earlier exploratory families F1–F4 are archived with their analyses.
