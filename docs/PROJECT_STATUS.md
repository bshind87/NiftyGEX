# NiftyGEX project status

**Last updated:** 2026-11-02
**Plan of record:** proposal deck v9 as presented; v10 (`project_docs/Long_Attention_Short_Gamma_CS7150_Proposal_v10.pptx`) has the wording fixes (ladder steps 0–6, RQ1–RQ3)
**Data config hash:** `1470391126a4` (all stage manifests in `outputs/manifests/`)
**Eval config hash:** `ce2e671c1e4c` (forecast runs in `outputs/results/index.jsonl`)
**Code:** git commit `b26d683` plus uncommitted changes, so manifests and runs record `git_dirty = true`. You make commits manually.
**Tests:** 114 passed, 1 skipped (the model acceptance tests, until `nn/models.py` lands), 0 failed (`PYTHONPATH=src pytest -q` in the `niftygex` env)
**Next milestone:** progress update, 2026-10-29

Status key: ✅ Completed · 🔄 In progress · ⏳ Pending · ⛔ Blocked · ❓ Unverified

## Next session

Pick up here, in order:

1. **You: commit and push in GitHub Desktop, then send Asad `docs/handoff_asad.md`** and an invite to the repo. His task (E2b) needs only the repo, not the data.
2. **E2c: `nn/trainer.py`.** The Forecaster wrapper (Huber, AdamW, early stopping, seeds), developed against a stand-in model until Asad's lands.
3. **Runner script and report.** `scripts/run_neural.py` and the model results report with figures, with the same pattern as the baselines. Rerun the baselines then too: the eval config hash changed when the `neural` section was added.
4. **After Asad's merge (target 8 Nov):** full runs (26 walks × 3 models × 3 seeds), then the RQ1–RQ3 tests and the window and field-drop ablations.

## Milestones (proposal slide 16)

| Date | Milestone | Status |
|---|---|---|
| 2026-10-08 | Proposal | ✅ Presented with deck v9 |
| 2026-10-29 | Data pipeline, first baselines | 🔄 The pipeline and ladder steps 0–3 are done; the slides are not yet drafted |
| November | Model, switch-off tests, rule-change tests | ⏳ |
| 2026-12-03 / 12-10, 12-17 | Final presentation, code and report | ⏳ |

## Proposal map

| Proposal item | Status | Evidence |
|---|---|---|
| Step 0: historical mean / persistence | ✅ | runs `92f3da4081b08fc6`, `d19d4490d9b024f4` |
| Step 1: HAR-RV | ✅ | run `c258929b5da6a6e3` |
| Step 2: HAR + GEX | ✅ | run `d1bfee2ee7b640da` |
| Step 3: tuned LightGBM on ~40 summaries | ✅ | run `bd11da0a4d1536ff` |
| Step 4: DeepSets | ⏳ | Needs D1–D3 and E1–E3 |
| Step 5: single-snapshot self-attention | ⏳ | Needs E1–E3 |
| Step 6: attention + GRU (our model) | ⏳ | Needs D1–D3 (token history) and E1–E3 |
| RQ1: does reading every contract predict better? | ⏳ | Test pre-registered in `configs/eval.yaml` (`multiple_testing.RQ1`) |
| RQ2: why does it win? | ⏳ | Pre-registered (`RQ2`) |
| RQ3: does it survive rule changes? | ⏳ | Pre-registered (`RQ3`). The before/after split already runs for the classical ladder |
| Attention maps | ⏳ | After step 6 |

## Tasks

### A. Data pipeline ✅

| ID | Task | Status | Summary | Outputs |
|---|---|---|---|---|
| A1 | Raw import, cleaning, calendar | ✅ | 881 files with checksums; start-labeled 09:15–15:25 bars with `available_at`; spot and VIX relabel (2020-12-08 to 2026-04-23); one futures contract per day | `cleaning_report.json`, `spot_label_alignment.json` |
| A2 | Pricing and GEX | ✅ | Parity forward; OTM-leg IV; Black-76 gamma and delta; GEX under naive, absolute and inferred conventions (OI in index units, 21-day horizon); walls, max pain, flip | `forward_report.json`, `greeks_report.json`, `gex_report.json`, `levels_report.json` |
| A3 | Aggregate features | ✅ | 1,824 snapshot rows (one per trading day) × 40 columns, now including the HAR-RV terms (`rv_trailing_{1,5,20}d_annualized_prev`, all through the previous day) | `features_report.json` |
| A4 | Tokens | ✅ | **Token history:**
- 8,738,126 tokens over 72 snapshots per day (09:20–15:15, with `lag` 0 = 15:15), from each day's nearest included expiry that has data.
- A median of 69 per snapshot and 71 at 15:15; 1,814 of 1,824 prediction snapshots have tokens.
- 1,804 days have a full 72-snapshot history; the gaps are no-option days, a truncated session and bars ending early.
- At lag 0 the tokens match the GEX chain.
- Built in 22 s with 1.4 GB peak memory; 505 MB on disk. | `tokens_report.json`, `docs/findings_data.md` |
| A5 | Target | ✅ | log(next-day futures RV / trailing 20-day mean) at 15:15: 1,824 rows, 1,549 tercile-labeled. The data layer builds only this target (`selection.TARGETS = ("t2",)`); T1 and T3 were removed on 2026-10-14 | `targets_t2_report.json` |
| A6 | Validation and leakage | ✅ | Parity, IV agreement, ATM IV vs VIX (0.95); all timestamp and leakage checks pass | `docs/validation_report.md` |
| A7 | Missing-data report | ✅ | Every known gap and its handling (see Data gaps) | `outputs/reports/missing_data.md` |
| A8 | Raw sample for sharing | ✅ | May–Jul 2022, raw layout, rows unchanged | `data/sample_data/` |

### B. Harness and forecasting path ✅

| ID | Task | Status | Summary | Outputs |
|---|---|---|---|---|
| B1 | Walk-forward harness | ✅ | Expiry cycles, 52-cycle minimum training, 1-cycle embargo, 13-cycle test blocks, label purge, `check_split`; **26 walks** for the forecasting target | `src/niftygex_eval/harness.py` |
| B2 | Forecaster interface and runner | ✅ | `tune(train, valid)` on the inner split, `fit(train)`, `predict(test)`; the same rows for every model | `src/niftygex_eval/forecast/` |
| B3 | Scores and comparisons | ✅ | IC (primary), RMSE, QLIKE, R² vs HAR; paired per-walk test with 95% CI, walks won, Diebold–Mariano, before/after split | `forecast/metrics.py`, `forecast/compare.py` |
| B4 | Harness tests | ✅ | Oracle IC = 1, random IC ≈ 0, tuning and fitting never see test days, QLIKE, DM | `tests/test_forecast.py` |
| B5 | Pre-registration | ✅ | RQ1 (intersection–union vs every rung), RQ2 and RQ3 (Holm) written before any neural run | `configs/eval.yaml` |

### C. Classical ladder, steps 0–3 ✅

| ID | Task | Status | Summary | Outputs |
|---|---|---|---|---|
| C1 | Ladder runs | ✅ | Five models on 26 walks, same rows | `outputs/results/forecast_comparison.json` |
| C2 | Report with figures | ✅ | Inputs, scoreboard, ladder IC, IC per walk, before/after, actual vs predicted | `docs/findings_forecast.md`, `outputs/reports/forecast_baselines.json`, `outputs/reports/figures/forecast_*.png` |

### D. Token history and model inputs 🔄

| ID | Task | Status | Summary |
|---|---|---|---|
| D1 | Sequence snapshot grid | ✅ | 72 snapshots per day, 09:20–15:15 (`snapshot_grid.sequence_start`, `sequence_step_minutes`); the model slices the last L (12 = one hour by default; 24 and 72 for the window ablation). One expiry per day, so a history never switches contract |
| D2 | Volatility-state vector | ✅ | 14 columns from the forecast table (`neural.state_columns`): IV, skew, term slope, intraday and 1/5/20-day RV, previous VIX, days to expiry, expiry-day flag, normalized GEX (3 conventions) and the inferred sign. Logs of positive levels; no data-layer change |
| D3 | Leakage test and report | ✅ | Validation and stage-contract checks: every token is available by its snapshot, each sequence snapshot is ≤ its prediction snapshot, one expiry per day, and lag-0 tokens equal the GEX chain. The data report has 4 figures | `docs/findings_data.md`, `outputs/reports/data_overview.json` |

### E. Neural models, steps 4–6 🔄

| ID | Task | Status | Summary |
|---|---|---|---|
| E1 | PyTorch in the pinned env | ✅ | `pytorch=2.10.0` from conda-forge (the pip wheel's bundled OpenMP aborts next to LightGBM's) |
| E2a | `nn/dataset.py` | ✅ | Tokens to `[days, L, N, 15]` tensors with padding mask (index −1 = 15:15), signed logs, a per-walk scaler fitted on training days, missing values as 0. On real data, window 12: 153 MB, 0.5 s, 1,794 of 1,800 days with 15:15 tokens; window 72: 941 MB Tests: `tests/test_nn_dataset.py` |
| E2b | `nn/models.py` (DeepSets, Attention, AttentionGRU) | 🔄 | **Assigned to Asad** (`docs/handoff_asad.md`): check-in 5 Nov, target merge 8 Nov. Acceptance tests in `tests/test_nn_models.py` (skipped until the file exists). The spec was checked against a private reference implementation, which passes all 25 tests |
| E2c | `nn/trainer.py` | ⏳ | Forecaster wrapper: Huber loss, AdamW, early stopping on the inner split, seeds 11/12/13 (`neural.training`) |
| E3 | Sanity checks | 🔄 | Model-level checks are part of Asad's acceptance tests. The end-to-end check (a model trains on one real walk) comes after E2c |
| E4 | Runs and report | ⏳ | Same runner and report pattern as C |

### F. Research-question tests and write-up ⏳

| ID | Task | Status | Summary |
|---|---|---|---|
| F1 | RQ1 | ⏳ | Attention + GRU against every rung |
| F2 | RQ2 | ⏳ | DeepSets vs attention; attention vs attention + GRU; window and field-drop ablations (exploratory) |
| F3 | RQ3 | ⏳ | Per-walk IC gain over LightGBM, after vs before the marker |
| F4 | Attention maps and case studies | ⏳ | — |
| F5 | Midterm progress report (Oct 29) | ✅ | Word report generated from the pipeline: `scripts/write_progress_report.py` writes `docs/deliverables/NiftyGEX_Midterm_Progress_Report.docx` (gitignored). The backup notebook `notebooks/02_pipeline_and_features.ipynb` follows one day through every stage |
| F6 | Proposal deck v10 fixes | ✅ | `project_docs/Long_Attention_Short_Gamma_CS7150_Proposal_v10.pptx`: the slide 13 "Adjust" label; "several times" with 3–5 seeds; "If it wins, why?" and "Why this could beat"; the win rule matches RQ1 (every competitor, on average); "archive of NSE data"; 26 exams; backup titles and notes. Slide 11's charts are still the 14:00 snapshot from the earlier pipeline (no longer rebuilt; the notebook now shows 15:15) |
| F7 | Final report, code and reproducibility pass | ⏳ | A clean-commit end-to-end rerun is still to do |

### G. Owner and external tasks

| ID | Task | Status | Summary |
|---|---|---|---|
| G1 | Daily India VIX, Apr–Jul 2026 | ⏳ | The daily file ends 2026-04-06; 60 trading days missing |
| G2 | From Sunil: missing OI (69 days), the truncated 2022-01-06 expiry, 8 missing spot days | ⏳ | Listed in `missing_data.md` |
| G3 | Sunil's written permission for code and data | ⏳ | Not on file |
| G4 | Verify lot sizes, the 2024-11-20 marker and special sessions against NSE circulars | ⏳ | Marked VERIFY in `configs/regimes.yaml` |
| G5 | Work split with Asad | ⏳ | Not recorded |

### H. Earlier exploratory work (archived 2026-10-14)

Moved to `archive/` (code, tests, configs, docs) and `outputs/archive/` (summaries, figures, T1/T3 outputs); see `archive/README.md`. It ran under data hash `9fc36c738aa5` and is not maintained.

| ID | Analysis | Archived write-up |
|---|---|---|
| H1 | Stage A HAC regressions on total GEX (run `7dd46fb5e9d61de7`) | `archive/docs/findings_t2.md` |
| H2 | Tercile classifiers, calibration, A/B/C/C1/C2 ablation, SHAP | `archive/docs/findings_t2.md` |
| H3 | T3 max-GEX placebos (matched distance, listed strikes, continuous) | `archive/docs/findings_t3.md` |
| H4 | Holm families F1–F4 | `archive/configs/eval_exploratory.yaml` |
| H5 | T1 pinning and T3 drift targets | Removed from the data layer on 2026-10-14: `archive/src/niftygex/`, `archive/docs/schema_t1_t3.md`, last outputs in `outputs/archive/data_targets/` |

## Key findings so far

**Classical ladder** (`docs/findings_forecast.md`; mean per-walk IC over 26 walks, with 95% CI):

| Model | IC | 95% CI | RMSE | QLIKE |
|---|---|---|---|---|
| Persistence | 0.261 | 0.168 to 0.354 | 0.370 | 0.376 |
| HAR-RV | 0.434 | 0.353 to 0.516 | 0.287 | 0.223 |
| HAR + GEX | 0.445 | 0.365 to 0.525 | 0.289 | 0.231 |
| Tuned LightGBM | 0.448 | 0.379 to 0.516 | 0.299 | 0.225 |

- **HAR-RV is the bar to beat.** Neither HAR + GEX (+0.011, p 0.40) nor LightGBM (+0.013, p 0.61) beats it on IC.
- **On squared loss, HAR is better than LightGBM** (Diebold–Mariano p 0.045). LightGBM's R² against HAR is −0.12.
- **Before vs after the Nov 2024 marker (exploratory, pooled):** HAR's IC falls from 0.501 to 0.448, while LightGBM's rises from 0.442 to 0.465.

**Earlier exploratory work (H):**
- With full controls, the GEX magnitude effect disappears.
- The inferred GEX sign survives Holm (F1).
- The gain from levels is suggestive but not significant after Holm.
- T3 shows no attraction to the max-GEX strike.

## Data gaps

From `outputs/reports/missing_data.md`:

| Gap | Extent | Handling |
|---|---|---|
| Option OI recorded as 0 | 69 days: 2019-07-30, 2022-01-03 to 2022-04-11 | OI nulled; GEX, levels and OI features missing |
| Spot missing | 8 days (2024-09-16 to 09-18, 2024-09-30 to 10-01, 2025-08-14, 2025-11-17, 2025-12-01) | Not imputed |
| Option / futures days missing | 5 option days, 1 futures day | Flagged |
| Truncated expiry file | 2022-01-06 expiry ends 2021-12-31 | No expiry-day chain |
| Option history per expiry | About 3 weeks (at most 21 calendar days) | GEX horizon 21 days |
| Daily India VIX | Ends 2026-04-06 | NaN, never forward-filled |
| Bhavcopy, official closes | Absent | Reconciliation skipped; T1 not used |

## Decisions log

**2026-09-30 to 2026-10-04** (data layer and earlier analyses; details in `CLAUDE.md`, `docs/schema.md`)
- Data handling:
  - start-labeled bars with `available_at`;
  - spot relabel;
  - OTM-leg IV;
  - GEX on OI in index units over 21 days;
  - futures RV for the target;
  - expiry-cycle harness (52/1/13).
- Earlier analyses: tuned LightGBM + Platt as the classifier reference; Holm families F1–F4.
- Tooling: git initialized, and you commit manually with no Claude attribution; `notebooks/` added.

**2026-10-08**
- The proposal deck v9 is the plan of record. There are three RQs, and expiry pinning and transfer learning are dropped.

**2026-10-14**
- **Forecasting target** for every model on the ladder: log(next-day futures RV / trailing 20-day mean) at 15:15.
- **HAR-RV terms** were added to the features stage (`features.har_rv_windows`: 1 and 5 days, plus the existing 20 days).
- **Classical baselines:**
  - persistence is the previous day's RV ratio;
  - HAR adds today's RV so far;
  - HAR + GEX is a ridge with alpha tuned on the inner split;
  - LightGBM uses the summary features with normalized GEX instead of rupee totals.
- **RQ1–RQ3 tests** were pre-registered in `configs/eval.yaml` before any neural run.
- **Token scope:** keep nearest-expiry tokens (about 70, as on slide 12) and add total GEX to the volatility-state vector.
- **Reports:** every major step gets a script-generated report with figures (first: `docs/findings_forecast.md`).
- **Archive:** the exploratory analyses (Stage A/B, T3 placebos, Holm F1–F4) moved to `archive/`. The forecast code now owns `inner_walk`, GEX normalization and its LightGBM settings.
- **Removed:** `CS7150_Proposal.html` (superseded by the deck), `docs/phase2_plan.md`, caches and Finder files (`data/raw/` untouched).
- **One target in the data layer:** T1 and T3 were removed from the snapshot grid, GEX, features, tokens, targets and validation; the full pipeline was rerun (features 4,803 → 1,824 rows, tokens 327,009 → 124,960; T2 unchanged).
- **Notebook:** it now shows the 15:15 snapshot (the model's daily snapshot) instead of 14:00.
- **Deck v10:** the wording fixes, the 26-exam count, and a win rule matching RQ1.

**2026-10-16**
- Token history: 72 snapshots per day, 09:20–15:15, built once.
  - The GRU window is a model setting (default L = 12), so the 1-hour, 2-hour and today-so-far ablations need no data rerun.
  - Each day uses one expiry: the nearest one that has Greek rows that day and that T2's total GEX includes. Choosing from the expiry list alone missed 7 days with no rows (2019-02, early January 2022, November 2025).
- The 80% ATM IV on 2019-05-22 is real, not a data error: it is the eve of the general-election results (23 May), and the expiring option spans the announcement.
- Data-step report added: `scripts/report_data.py` writes `docs/findings_data.md`.

**2026-10-24**
- The midterm deliverable is a Word report, not slides.
  - It is generated by `scripts/write_progress_report.py` into `docs/deliverables/`, which is gitignored.
  - `python-docx==1.1.2` is pinned in the env for it.
- Feature lineage (source and availability per feature and token field) lives in `src/niftygex/features/lineage.py`. A test checks every built feature has an entry.
- Added `notebooks/02_pipeline_and_features.ipynb`.

**2026-11-02**
- **Work split.** Asad owns `nn/models.py` (task E2b) against a fixed interface and acceptance tests (`docs/handoff_asad.md`); everything that touches the pipeline, harness and statistics stays here. You make all commits through GitHub Desktop.
- **PyTorch 2.10 from conda-forge**, not pip: the pip wheel's own OpenMP aborts on import next to LightGBM.
- **Neural inputs** (`configs/eval.yaml` → `neural`): 15 token fields, a 14-column volatility state, window L = 12 by default; model defaults d = 64, 4 heads, 2 layers, GRU 64; training with Huber, AdamW and seeds 11/12/13.

## Open questions

None at the moment. The midterm deliverable is a Word report, so the earlier question about slide 11's charts no longer applies.
