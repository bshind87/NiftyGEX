# NiftyGEX project status

**Last updated:** 2026-11-06 (E1–E3 complete, final report regenerated)
**Plan of record:** proposal deck v9 as presented; v10 (`project_docs/Long_Attention_Short_Gamma_CS7150_Proposal_v10.pptx`) has the wording fixes (ladder steps 0–6, RQ1–RQ3)
**Data config hash:** `1470391126a4` (all stage manifests in `outputs/manifests/`)
**Eval config hash:** `552a3501001a` (forecast and development runs in `outputs/results/index.jsonl`)
**Code:** main runs on commit `12b11a0` (`git_dirty = false`); exploratory Mac runs on `88e4bb3` (clean). Commits are made through GitHub Desktop.
**Tests:** 177 passed, 0 failed (`PYTHONPATH=src pytest -q` in the `niftygex` env)
**Next milestone:** exploratory experiments E1–E3 (Mac CPU + Explorer GPU); final presentation 2026-12-03 / 12-10

Status key: ✅ Completed · 🔄 In progress · ⏳ Pending · ⛔ Blocked · ❓ Unverified

## Next session

Goal: results done by mid-November. Pick up here, in order:

Exploratory E1–E3 are complete (`docs/findings_exploratory_summary.md`) and the final Word report is regenerated with them (section 16, separate from RQ1–RQ3).

1. **Final slides** (Dec 3/10): main results (RQ1–RQ3), then the exploratory summary table.
2. **Clean-commit reproducibility rerun** of the main pipeline (data stages, classical ladder, Mac neural main run on `12b11a0`'s code path).
3. Optional, not registered: an intraday hybrid (intraday HAR forecast as an input) to separate inputs from architecture in E3.

## Known issues

- **Apple-GPU memory on long Mac runs.** For attention models at L = 24, MPS driver memory grows about 1 GB per walk and is not returned while the process lives: the backend caches memory for each new tensor shape, and every walk adds new ones (ragged last training batch, ragged last prediction pass, accumulation pieces). Window 24 failed twice (4 and 6 Nov). Constant shapes keep it flat (4.2 GB over 26 walks); one process per walk (`scripts/run_isolated.py`) releases it and reproduces the 4 Nov main run exactly (walks 0–2, every prediction identical). Exploratory neural runs moved to Explorer; details in `docs/hpc.md`.

- **Explorer setup (6 Nov).** (1) An environment build on a GPU node was cancelled by the system after 69 min (job 11680222, `CANCELLED by 0`; the GPU sat idle, which RC terminates): the build now runs on the CPU `short` partition, with a separate 15-minute CUDA check on gpu-short. (2) The `anaconda3/2024.06` module puts an older `libstdc++` on `LD_LIBRARY_PATH`, so scipy failed to import (`GLIBCXX_3.4.30 not found`, job 11681913): every script now puts the env's `lib/` first. (3) rsync through the login node is cut off within a minute: copies go through RC's transfer node (`explorer-xfer`).

## Milestones (proposal slide 16)

| Date | Milestone | Status |
|---|---|---|
| 2026-10-08 | Proposal | ✅ Presented with deck v9 |
| 2026-10-29 | Data pipeline, first baselines | ✅ The midterm Word report was generated (`docs/deliverables/`); its submission is not recorded here ❓ |
| mid-November | Model results (own target) | ✅ Main run and RQ1–RQ3 done 4 Nov (`docs/findings_models.md`) |
| November | Model, switch-off tests, rule-change tests | 🔄 Pre-registered tests done; ablations and attention maps to do |
| 2026-12-03 / 12-10, 12-17 | Final presentation, code and report | ⏳ |

## Proposal map

| Proposal item | Status | Evidence |
|---|---|---|
| Step 0: historical mean / persistence | ✅ | runs `dec5b422449712f0`, `e648d7637a3743f3` |
| Step 1: HAR-RV | ✅ | run `cc6af41802a38fc8` |
| Step 2: HAR + GEX | ✅ | run `cb4cb0f9374dbeab` |
| Step 3: tuned LightGBM on ~40 summaries | ✅ | run `85c1129f0c7cbb1a` |
| Step 4: DeepSets | ✅ | run `baa5bb9503e69096` |
| Step 5: single-snapshot self-attention | ✅ | run `8cfe1b261e53a1e3` |
| Step 6: attention + GRU (our model) | ✅ | run `e5cacfc5b84de69c` (with attention weights) |
| RQ1: does reading every contract predict better? | ✅ | Not supported on IC; `outputs/results/rq_tests.json` |
| RQ2: why does it win? | ✅ | No Holm rejection; `rq_tests.json` |
| RQ3: does it survive rule changes? | ✅ | No change detected (p 0.74); `rq_tests.json` |
| Attention maps | ✅ | `docs/findings_attention.md` (exploratory) |

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

### E. Neural models, steps 4–6 ✅

| ID | Task | Status | Summary |
|---|---|---|---|
| E1 | PyTorch in the pinned env | ✅ | `pytorch=2.10.0` from conda-forge (the pip wheel's bundled OpenMP aborts next to LightGBM's) |
| E2a | `nn/dataset.py` | ✅ | Tokens to `[days, L, N, 15]` tensors with padding mask (index −1 = 15:15), signed logs, a per-walk scaler fitted on training days, missing values as 0. On real data, window 12: 153 MB, 0.5 s, 1,794 of 1,800 days with 15:15 tokens; window 72: 941 MB Tests: `tests/test_nn_dataset.py` |
| E2b | `nn/models.py` (DeepSets, Attention, AttentionGRU) | ✅ | Merged 3 Nov (PR #1, `cc18a49`), built to the interface in `docs/handoff_asad.md`. All 25 acceptance tests pass; attention + GRU has 112,257 parameters; a 2-walk real-data run trains cleanly (early stopping around epoch 13–15). Acceptance tests in `tests/test_nn_models.py` (skipped until the file exists). The spec was checked against a private reference implementation, which passes all 25 tests |
| E2c | `nn/trainer.py` | ✅ | `NeuralForecaster` behind the Forecaster interface: per-walk scaler, Huber loss, AdamW, early stopping on the inner split, seeds 11/12/13, no refit (`neural.training`). Tests: `tests/test_nn_trainer.py` (stand-in model; learns a real signal through `run_walks`, scaler fitted on training rows only, reproducible seeds) |
| E2d | Runner, RQ tests, results report | ✅ | `scripts/run_neural.py` (`--models`, `--window` and `--drop` for the window and field-drop ablations, `--save-attention` to store pooling weights per test day, `--walks` for development), `scripts/run_rq_tests.py` (RQ1–RQ3 as registered), `scripts/report_results.py` (`docs/findings_models.md` with figures). Checked end to end on a 2-walk development run with a stand-in model (`outputs/reports/findings_models_dev.md`, not a result) |
| E3 | Sanity checks | ✅ | Model-level checks are part of the model acceptance tests. End to end on real data: in the development run, models stop at epochs 14–23 of 60, and the largest walk takes DeepSets 0.2 min, attention 0.7 min and attention + GRU 6.6 min (3 seeds, MPS) |
| E4 | Runs and report | ✅ | 4 Nov main run on commit `12b11a0` (`git_dirty = false`), 26 walks × 3 seeds: DeepSets 1.6 min (CPU, mean stop epoch 16.4), self-attention 9.0 min (12.4), attention + GRU 87.4 min (11.9) on MPS. Outputs: `outputs/results/neural_comparison.json`, `docs/findings_models.md`, figures `results_*.png`. The 3 Nov preview runs (`d701f1052a7113fc`, `a9d1fb305a16a259`, dirty) are superseded. Field-drop ablation: run `778509070ba82c1b` (commit `d239c0e`, clean, MPS). Window 24 on the Mac failed twice on GPU memory (see Known issues) and moved to Explorer (section I) |

### F. Research-question tests and write-up 🔄

| ID | Task | Status | Summary |
|---|---|---|---|
| F1 | RQ1 | ✅ | Beats every rung: no. Significant only vs persistence on IC; supporting DM losses favour it over LightGBM |
| F2 | RQ2 | 🔄 | Pre-registered pair done (no Holm rejection). Field drop done (exploratory): without `oi_change_since_previous_close_pct`, attention + GRU IC 0.459 vs 0.447 (+0.012, 95% CI −0.009 to +0.034, 15 of 26 walks), QLIKE 0.204 vs 0.189 (worse); the concentrated-attention days go from 4 to 0. Windows 24 and 72 run on Explorer with a fresh L = 12 reference (E2, section I) |
| F3 | RQ3 | ✅ | Gain +0.003 (19 walks before) vs −0.014 (7 after); Welch p 0.74 |
| F4 | Attention maps and case studies | ✅ | `scripts/report_attention.py` → `outputs/reports/attention.json`, 4 figures `attention_*.png`, `docs/findings_attention.md`; the final report's attention section reads them. Pooling weights are a near-plain average; the rare concentrated days (4 of 1,552) all lock onto the day's largest OI-change % |
| F5 | Midterm progress report (Oct 29) | ✅ | Word report generated from the pipeline: `scripts/write_progress_report.py` writes `docs/deliverables/NiftyGEX_Midterm_Progress_Report.docx` (gitignored). The backup notebook `notebooks/02_pipeline_and_features.ipynb` follows one day through every stage |
| F6 | Proposal deck v10 fixes | ✅ | `project_docs/Long_Attention_Short_Gamma_CS7150_Proposal_v10.pptx`: the slide 13 "Adjust" label; "several times" with 3–5 seeds; "If it wins, why?" and "Why this could beat"; the win rule matches RQ1 (every competitor, on average); "archive of NSE data"; 26 exams; backup titles and notes. Slide 11's charts are still the 14:00 snapshot from the earlier pipeline (no longer rebuilt; the notebook now shows 15:15) |
| F7 | Final report, code and reproducibility pass | 🔄 | `scripts/write_final_report.py` generates the Word final report from the results files: data, pipeline, models (with parameter counts), evaluation design, full ladder, RQ1–RQ3, ablations, attention maps, limitations, conclusion and reproducibility. Shared sections are in `src/niftygex_eval/report_sections.py` and `docx_report.py`. Regenerated 6 Nov with the attention maps, the Mac field-drop ablation and the exploratory E1–E3 section (the cluster's window runs are reported there, against a V100 reference). The clean-commit rerun is still to do |

### I. Exploratory experiments E1–E3 ✅

All labelled exploratory; registered in `configs/eval_exploratory_v2.yaml` (own hash) before any run. The main config and RQ1–RQ3 results are unchanged.

| ID | Task | Status | Summary |
|---|---|---|---|
| I1 | Registration | ✅ | `configs/eval_exploratory_v2.yaml`: E1 (9 tests, Holm), E2 (4 tests vs L = 12, Holm), E3 (2 horizons, Holm), secondary scores, device rule, E3 training (patience 3, at most 30 epochs). To commit |
| I2 | E1 diagnostics (t2) | ✅ | `docs/findings_exploratory_diagnostics.md`. All 9 registered tests reject after Holm. Residual models (no volatility state) predict HAR's out-of-sample residual with IC 0.204–0.218; HAR + correction +0.030 to +0.046 IC; hybrid attention + GRU 0.492 (+0.047 over attention + GRU on V100); equal-weight combination 0.502 (+0.068 over HAR). State-only MLP 0.459 ≈ attention + GRU 0.445 (not significant): tokens add little inside the neural trainer. Leakage check: 0 of 1,800 rows use a source after 15:15. Runs: residual LightGBM `b825612bdd2a35d3` (Mac); merged V100 runs residual attention `b0e1c7e665609637`, residual DeepSets `dbb47542fe9b923b`, state MLP `fcc7b1437ca09ea6`, hybrid `1bff9652940fcd3c`; V100 L = 12 references DeepSets `42cf8f01b8b2ffe9`, attention `477e52e15478c1bd`, attention + GRU `a8164fd180864bf4` (IC 0.437 / 0.409 / 0.445 vs Mac 0.437 / 0.418 / 0.447). Every cluster run: commit `60b6e18`, `git_dirty = false`, Tesla V100-SXM2-32GB |
| I3 | E2 more history (t2) | ✅ | `docs/findings_exploratory_history.md`. No registered test rejects after Holm (reference attention + GRU L = 12 on V100, IC 0.445): L = 24 0.457 (+0.012, p 0.046, Holm 0.19), L = 72 0.460 (+0.016, Holm 0.23), multi-day K = 5 0.438 (−0.007), K = 10 0.441 (−0.004); K = 5 has worse squared loss (DM p 0.012). Merged V100 runs: L = 24 `c126b21feef70d09`, L = 72 `b0516280c6644080`, K = 5 `2e496e8e6005fc94`, K = 10 `91efb6d0b796cc35` (commit `60b6e18`, clean) |
| I4 | E3 data layer | ✅ | `configs/intraday.yaml` (separate from `data.yaml`, T2 hash unchanged), GEX and features stages `--grid intraday` (same 21-day logic) and `--verify` (T2 unchanged), `compute_intraday_rv.py`, strictly causal expiry choice (`causal_nearest_expiry`). Tests: `test_intraday_targets.py`, `test_intraday_features.py`. Built on `88e4bb3` (clean; intraday config hash `e3988a993d5b`): 23,712 rows, 23,031 with both horizons; 59 (1 h) and 83 (2 h) windows skipped for missing bars, 585 rows before the 60-day normal exists. T2 `--verify`: GEX outputs identical, features identical up to float rounding |
| I5 | E3 ladder | ✅ | `docs/findings_intraday.md`. Registered test (Holm across horizons): attention + GRU is significantly **lower** than the best classical rung, 1 h 0.535 vs LightGBM 0.582 (−0.047, Holm p 0.0003, 4/25 walks), 2 h 0.558 vs intraday HAR 0.600 (−0.042, Holm p 0.018); neural RMSE and QLIKE also worse. Sanity: time of day explains ≈0% of the target (0.8% / 1.5% of raw log RV in 10:15–13:15); within-snapshot-time IC of intraday HAR 0.63–0.70, so the high intraday IC is volatility clustering, not a time-of-day ranking; persistence 0.541 / 0.587. Design note: the registered neural state lacks the backward-window RVs intraday HAR uses. Merged V100 runs 1 h: DeepSets `165a586db33f2ca7`, attention `1ae2825ad5d0d3ea`, attention + GRU `30862d8bb1dcdbd8`; 2 h: `d5a2f2971eeca62e`, `338a68d4ddd9f308`, `6b2ada3c65c38c45` (commit `60b6e18`, clean) |
| I6 | Explorer tooling | ✅ | `hpc/` (settings, env setup, code sync, push/pull, transfer check, 53 job scripts, scheduler within partition limits, status), `scripts/merge_seeds.py` (seeds averaged exactly as one local run: tested), per-seed `--seed` in every runner, per-walk resume. Explorer: environment ready (torch 2.10.0+cu126, V100-SXM2, driver 545.23.08), transfer check 4,155 files 0 problems, calibration done (t2 L = 12: 0.4–0.8 min per walk, 1.0 GB GPU; E3 1 h: 0.9–1.5 min per walk, 3.7 GB), production waves from commit `60b6e18` (scheduler job 11685043; gpu-short + courses-gpu, gpu disabled). E1 and references done |
| I7 | Reports | ✅ | `docs/findings_exploratory_diagnostics.md` (E1), `docs/findings_exploratory_history.md` (E2), `docs/findings_intraday.md` (E3), `docs/findings_exploratory_summary.md` (cross-experiment table, verdicts from the Holm results by rule); exploratory section 16 in the final report (`scripts/write_final_report.py`) |

### G. External tasks

| ID | Task | Status | Summary |
|---|---|---|---|
| G1 | Daily India VIX, Apr–Jul 2026 | ⏳ | The daily file ends 2026-04-06; 60 trading days missing |
| G2 | From Sunil: missing OI (69 days), the truncated 2022-01-06 expiry, 8 missing spot days | ⏳ | Listed in `missing_data.md` |
| G3 | Sunil's written permission for code and data | ⏳ | Not on file |
| G4 | Verify lot sizes, the 2024-11-20 marker and special sessions against NSE circulars | ⏳ | Marked VERIFY in `configs/regimes.yaml` |

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

**Exploratory E2** (`docs/findings_exploratory_history.md`): more intraday history tilts IC up slightly (L = 24 / 72: +0.012 / +0.016) and previous days' chains do not help; nothing survives Holm.

**Exploratory E3** (`docs/findings_intraday.md`): for the next 1–2 hours, the neural rungs are significantly worse than the classical ones (attention + GRU −0.047 / −0.042 IC vs the best classical rung after Holm); intraday HAR and persistence capture most of the predictable part.

**Exploratory E1** (`docs/findings_exploratory_diagnostics.md`; Holm within E1, all 9 tests reject): the chain and positioning features carry information beyond HAR (residual IC ≈ 0.21; HAR + correction up to +0.046 IC); the hybrid HAR + attention + GRU correction (0.492) and the equal-weight combination (0.502) are the best next-day models; inside the neural trainer the tokens add little (state-only MLP 0.459 vs attention + GRU 0.445).

**Neural ladder and pre-registered tests** (`docs/findings_models.md`; 26 walks, same rows as the classical ladder):

| Model | IC | 95% CI | RMSE | QLIKE |
|---|---|---|---|---|
| DeepSets | 0.437 | 0.361 to 0.512 | 0.293 | 0.197 |
| Self-attention | 0.418 | 0.324 to 0.512 | 0.293 | 0.197 |
| Attention + GRU | 0.447 | 0.363 to 0.531 | 0.285 | 0.189 |

- **RQ1: not supported.** Attention + GRU ties the best classical models on IC (vs HAR +0.012, p 0.61; vs LightGBM −0.001, p 0.95); it beats only persistence.
- **On losses it is the best model** (lowest RMSE and QLIKE). Diebold–Mariano favours it over LightGBM on squared loss (p 0.015) and QLIKE (p 0.042), and over HAR + GEX on QLIKE (p 0.033). These are supporting tests, not the pre-registered IC rule.
- **RQ2:** attention vs DeepSets −0.018 (Holm p 0.23); history (attention + GRU vs attention) +0.028 (Holm p 0.20). Neither rejects.
- **RQ3:** no detectable change in the gain over LightGBM after the Nov 2024 marker (p 0.74; only 7 walks after).
- **Attention maps (exploratory, `docs/findings_attention.md`):** pooling weights are almost exactly a plain average (median largest weight 1.008× average). The concentrated days (4 of 1,552 for attention + GRU, all late 2024–2026) put the weight on the day's largest OI change %, a field that is not log-transformed. This fits RQ2 (attention ≈ DeepSets); the field-drop ablation tests it.

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
- **Models built against a fixed interface.** `nn/models.py` (E2b) is specified by an interface and acceptance tests (`docs/handoff_asad.md`), so it can be developed without the data. Commits go through GitHub Desktop. Contributions are not tracked per person.
- **PyTorch 2.10 from conda-forge**, not pip: the pip wheel's own OpenMP aborts on import next to LightGBM.
- **Neural inputs** (`configs/eval.yaml` → `neural`): 15 token fields, a 14-column volatility state, window L = 12 by default; model defaults d = 64, 4 heads, 2 layers, GRU 64; training with Huber, AdamW and seeds 11/12/13.
- **Target:** results by mid-November.
- **Neural training without refit:** each seed's early-stopped model (inner split) predicts the test block. Settings: at most 60 epochs, patience 8, batch 64, DeepSets on CPU and the attention models on MPS. This cuts the main run to about 2 hours.
- **HPC is optional:** only for the 72-snapshot window ablation. The HPC path profile already exists.
- **Baselines rerun** under the current eval config: the numbers are unchanged, and the RQ tests require both sets of runs to use the same eval config hash.

**2026-11-04**
- Main neural run on clean commit `12b11a0`; it replaces the 3 Nov preview runs. RQ1–RQ3 were evaluated exactly as pre-registered.
- `oi_change_since_previous_close_pct` stays as configured (not log-transformed). Changing it now would change the eval config hash and break comparability with the pre-registered runs; the `--drop` ablation measures its effect instead.
- Added `docs/glossary.md`: every market, model, training, evaluation and statistics term used in the project.
- The final report's conclusion now also states the loss result (lowest RMSE and QLIKE, Diebold–Mariano vs LightGBM), labeled as supporting evidence, not the IC rule.

**2026-11-06**
- Window 24 failed twice on Mac GPU memory; cause identified (see Known issues). All exploratory neural training moves to Explorer; the Mac keeps code, CPU work and reports. The pre-registered main results stay the Mac runs.
- Exploratory experiments E1–E3 registered in `configs/eval_exploratory_v2.yaml` before any run, with a device rule: neural models compared with each other share a device and GPU type (fresh L = 12 references on Explorer); neural-vs-classical comparisons may mix devices.
- E3 data layer kept out of `configs/data.yaml` (`configs/intraday.yaml`) so the T2 data hash and stages are unchanged; E3 expiry choice strictly causal at each snapshot; E3 uses T2's same-day-expiry rule, so the stored token history applies.
- Confirmed: the T2 normalized-GEX current-day value is the 15:15 snapshot total (matches the GEX stage on all 1,722 days with GEX; inputs available by 15:15).
- One cluster job per seed; seeds merged on the Mac exactly as a local three-seed run.
- Explorer: one GPU type (v100-sxm2) for every job; short jobs on gpu-short, long jobs on gpu, courses-gpu when `sbatch --test-only` says sooner; jobs stop before a walk they cannot finish (deadline from the SLURM end time) and resume. Data on scratch kept private (`go-rwx`, `umask 077`). `docs/hpc_guide/` (course material) is gitignored.
- Explorer jobs start from a clean environment (the login shell activates an unrelated conda env); `configs/paths.hpc.yaml` is now committed (it was gitignored by `configs/paths.*.yaml`); the data push retries per folder with ssh keepalives after a dropped connection, and always restores `go-rwx`.
- The T2 `--verify` checks pass: GEX stage outputs identical; aggregate features identical except `gex_concentration_hhi` at float rounding (≤ 3e-16, summation order varies between runs).

**2026-11-06 (evening)**
- All 51 exploratory GPU jobs finished on Explorer from commit `60b6e18` (clean, V100-SXM2) in about 3.5 hours on gpu-short and courses-gpu; no production job was SIGKILLed. Seeds merged on the Mac exactly as a local three-seed run.
- Results: E1 all 9 tests reject (the chain helps as a correction to HAR; combination best); E2 none reject; E3 attention + GRU significantly worse than the best classical rung at both horizons.
- The final report's ablation table keeps only Mac-device ablations; the V100 window runs are reported in the exploratory section against the V100 reference (device rule).

## Open questions

None at the moment. The midterm deliverable is a Word report, so the earlier question about slide 11's charts no longer applies.
