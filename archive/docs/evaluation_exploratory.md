# Evaluation layer: archived exploratory analyses

Moved from `docs/evaluation.md` on 2026-10-14. These analyses are outside the proposal plan; the code is in `archive/` (see `archive/README.md`) and the settings in `archive/configs/eval_exploratory.yaml`.

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

## C1/C2 split and SHAP

**Feature sets**
- **C1:** B plus normalized GEX (3 conventions) and the inferred sign.
- **C2:** B plus levels (call wall, put wall, naive flip, max pain, max-GEX strike distances) and GEX concentration.
- C keeps its earlier definition so earlier results stay comparable.

**SHAP**
- **Method:** TreeSHAP via LightGBM `pred_contrib` on the tuned model's raw output, before Platt, per walk, for set C (`stage_b.shap`).
- **Outputs:** classification gets one output per tercile in log-odds; regression gets one output.
- **Summary:** mean |SHAP| is averaged within each walk, then across walks.
- **Figures:** `scripts/plot_shap_interactions.py` writes a mean-|SHAP| ranking and dependence plots for the top GEX/levels features, split by days to expiry and ATM-IV tercile, to `outputs/results/figures/`.

## T3 listed-strike placebo and continuous T3 (`scripts/run_t3_listed_placebo.py`)

**Comparators** (all listed strikes):
- **GEX rank 2–5:** the strikes with the 2nd- to 5th-largest absolute GEX, each reported separately.
- **Distance-matched rank 2–5:** the rank-2–5 strike whose distance from spot best matches the max-GEX strike's. A variant keeps only days where the match is within `max_distance_gap_pct` (0.1%) of spot.
- **Snapped mirror:** the mirror through spot, snapped to the nearest listed strike. Days where it lands on the max-GEX strike are excluded and counted.

**Tests**
- **Binary:** a sign test on the days where exactly one of the two strikes is hit.
- **Continuous T3:** distance reduction toward the strike in bps of spot, compared with a paired t-test and a Wilcoxon test.

**Slices:** overall, expiry days, non-expiry days, before/after the marker, and start-distance buckets.

## Multiple testing (`scripts/run_holm.py`)

Holm step-down is applied within pre-registered families (`multiple_testing` in `configs/eval.yaml`). Every test outside these families is **exploratory** and reported unadjusted. Results are written to `outputs/results/multiple_testing.json`.

| Family | Members | Primary |
|---|---|---|
| F1 | Stage A joint magnitude+sign specification: sign terms (naive, inferred) × samples (full, pre, post) — 6 tests | Inferred sign, full sample |
| F2 | T2 ablation per-walk paired t-tests: C vs B, C1 vs B, C2 vs B, B vs A on IC — 4 tests. AUC and Brier are separate secondary families | IC |
| F3 | T3 binary sign tests: slices (overall, expiry days) × comparators (matched-distance mirror, snapped mirror, distance-matched rank 2–5) — 6 tests. Continuous T3 (paired t-test) is a secondary family | Overall, expiry days |
| F4 | T1 pinning tests, added when official closes are available | — |
