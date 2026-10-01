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

**Current T2 layout.** 22 walks. The first training block starts at the 2020-03-05 cycle; test blocks run from 2021-03-05 to 2026-07-02, 1,303 test rows in total. The last block is partial.

## Results store (`niftygex_eval/results.py`)

The store lives at `outputs/results/` (gitignored).

**Run directory.** Each run is written to `runs/<run_id>/`, which holds:
- `run.json`, with the spec (stage, target, model, `data_config_hash`, `eval_config_hash`, harness parameters, feature list, model parameters) and a summary;
- one Parquet file per table, for example `predictions`, `walk_metrics`, `walks`, `regressions`.

**Run id.** `run_id` is a hash of the spec, so rerunning an identical spec replaces that run.

**Index.** `index.jsonl` gets one line per write.

**Inputs.** `modeling_table` refuses to join features and targets whose manifests have different data config hashes.

## Stage A (`scripts/run_stage_a.py`)

**Model.** OLS of `rv_ratio_trailing` (the T2 next-day RV ratio) on total GEX, plus the previous-close India VIX, with Newey–West (HAC, Bartlett) standard errors and `hac_maxlags` 5.

**Specifications**
- **Primary:** each convention's total GEX divided by the causal trailing mean of absolute total GEX over the last 250 T2 snapshots, with at least 60 present. This removes the many-fold growth in OI over the sample.
- **Robustness:** the level in lakh crore rupees.

**Robustness controls** (`robustness_controls`)
- Previous-close VIX, ATM IV, days to expiry, the expiry-day flag, and today's RV ratio (`rv_ratio_trailing_lagged`).
- `rv_ratio_trailing_lagged` is the previous T2 row's target, used only when that row's target day is the snapshot day.
- It resolves at today's close, 15 minutes after the 15:15 snapshot. That is acceptable for a regression control but not for a predictive feature.

**Joint magnitude + sign specification.** Normalized absolute GEX plus `sign(total GEX)` under the naive or inferred convention, with the robustness controls. It tests whether sign adds information beyond magnitude.

**Samples.** Full sample, each year (VIX-control specification only), and before/after the `weekly_expiry_rationalisation` marker.

## Stage B (`scripts/run_stage_b.py`)

**Models.** T2 tercile classifiers, fit inside the harness:

| Model | Features |
|---|---|
| `prior` | None: predicts the training class frequencies |
| `vix_only` | Previous-close VIX only (logistic regression) |
| `logistic` | All features (median imputation with missing-value indicators, standardization, multinomial logistic, C = 1) |
| `lightgbm` | All features (multiclass, fixed parameters, no tuning) |

**More models**

| Model | What it is |
|---|---|
| `lightgbm_no_gex` | Ablation: LightGBM on calendar and vol-state features only (`no_gex_features`) |
| `lightgbm_tuned` | Regularization (`min_child_samples`, `max_depth`, `reg_lambda`) chosen by validation log loss on a time-ordered inner split of each training window: inner-train, then a 1-cycle inner embargo, then the last 26 cycles. The inner split is purged and checked like an outer walk |
| `lightgbm_tuned_platt` | The tuned model plus multinomial logistic calibration on log-probabilities, fitted on the inner validation predictions |
| `lightgbm_tuned_isotonic` | The tuned model plus per-class isotonic calibration, renormalized, fitted on the same predictions |

The tuned variants all use the model fitted on inner-train only, so a calibrator never sees its model's training rows.

**Features.** The features come from `features_report.json`, minus `exclude_features` (currently `spot`). A target column in the feature list is a hard error.

**Comparison output.** `outputs/results/stage_b_t2_comparison.json` reports, for every model:
- pooled and mean per-walk AUC;
- Brier, log loss and ECE;
- AUC before and after the marker.

It also holds paired per-walk comparisons (mean difference, paired t, wins) and the parameters chosen per walk.

**Metrics.** Computed per walk and on pooled test predictions:
- macro one-vs-rest AUC and AUC per class;
- multiclass Brier score, log loss and accuracy;
- class-wise reliability tables and ECE (10 bins);
- splits before and after the regime marker.

## T2 three-way ablation (`scripts/run_t2_ablation.py`)

The feature sets are nested (`stage_b.feature_sets`):

- **A:** vol state and calendar.
- **B:** A plus participant positioning.
- **C:** B plus normalized GEX (as in Stage A) and levels.

**Tasks**
- **Classification:** tuned LightGBM + Platt on terciles, using the label walks.
- **Regression:** tuned LightGBM regression of log(next-day RV ratio), using walks over all rows with an outcome (26 walks, starting in 2020). Parameters are chosen by inner-validation MSE, then the model is refit on the full training window.

**Outputs.** Per-walk paired tests (B vs A, C vs B, C vs A) on AUC, Brier and Spearman IC are written to `outputs/results/t2_ablation.json`. `scripts/write_findings_t2.py` renders `docs/findings_t2.md` from the store.

## T3 placebo (`scripts/run_t3_placebo.py`)

**Hit.** A day is a hit when `abs(end - K) < abs(start - K)`, with the 14:00 spot as start and the 15:25-bar close as end.

**Placebo comparisons**
- **Random-side pseudo-strike:** `K' = start ± abs(K - start)`, on a random side, `draws` times, giving a null distribution of hit rates. The p-value is the share of placebo hit rates at or above the actual hit rate.
- **Mirror strike:** `start - (K - start)`. Only two positions exist at exactly matched distance, so the random-side placebo averages the actual and mirror outcomes.
- **Sign test:** the actual vs mirror comparison is also run as a sign test on the days where exactly one of the two is hit.

**Slices.** Results are reported overall, by start-distance bucket, by year, before/after the marker, and for expiry vs non-expiry days.
