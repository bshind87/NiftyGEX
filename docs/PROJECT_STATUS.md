# NiftyGEX project status

**Last updated:** 2026-10-01
**Data config hash:** `ad5fca5e441f` (all stage manifests in `outputs/manifests/`)
**Eval config hash:** `100d11eee4fc` (all latest runs in `outputs/results/index.jsonl`)
**Tests:** 104 passed, 0 failed (`PYTHONPATH=src pytest -q` in the `niftygex` env)
**Next milestone:** Proposal presentation, 2026-10-08

Status key: ✅ Completed · 🔄 In progress · ⏳ Pending · ⛔ Blocked · ❓ Unverified

## Milestones

| Date | Milestone | Status |
|---|---|---|
| 2026-10-08 | Proposal presentation | ⏳ Pending. No slides in the repo; the proposal needs updates first (P8.7) |
| 2026-10-29 | Progress update | ⏳ Pending. Data layer, GEX library, aggregate features, Stage A and LightGBM baseline exist already (P1–P5) |
| 2026-12-03 / 12-10 | Final presentation | ⏳ Pending |
| 2026-12-17 | Code and documentation | ⏳ Pending |

## Tasks

### P0 Setup

| ID | Task | Status | Summary | Outputs | Next step / blocker |
|---|---|---|---|---|---|
| P0.1 | Repo skeleton and config profiles | ✅ | Data layer in `src/niftygex`, evaluation in `src/niftygex_eval`. Paths come from profiles plus `NIFTYGEX_PROFILE` / `NIFTYGEX_DATA_ROOT` | `configs/`, `tests/test_config.py` | — |
| P0.2 | Pinned environment | ✅ | Python 3.11.11 and Polars 1.22, plus eval dependencies (scikit-learn, statsmodels, LightGBM; pandas pinned at 2.2.3) | `env/environment.yml`, `pyproject.toml` | — |
| P0.3 | CLAUDE.md, README, PROVENANCE | ✅ | Conventions, run order, and adapted-file provenance with the co-development note | `CLAUDE.md`, `README.md`, `PROVENANCE.md` | Keep in sync at each checkpoint |
| P0.4 | Manifests with config hash | ✅ | Every stage writes a manifest; downstream stages refuse a mismatched upstream hash | `outputs/manifests/*.json` (11) | — |
| P0.5 | .gitignore and data layout | ✅ | `data/` and `outputs/` are ignored; layout documented | `.gitignore`, `data/README.md` | — |
| P0.6 | Version control | ⏳ | The project folder is **not a git repository**, so nothing is versioned yet | — | Decide: initialize git (see Open questions) |
| P0.7 | HPC profile and sync/cleanup scripts | ⏳ | In the plan but never created; everything runs locally | — | Decide if HPC is still needed |
| P0.8 | Partitioned stage runs and lookback-equivalence test | ⏳ | Stages run on the full range only; the planned "one month = slice of the full run" test does not exist | — | Needed only if HPC or partitioned runs come back |

### P1 Data

| ID | Task | Status | Summary | Outputs | Next step / blocker |
|---|---|---|---|---|---|
| P1.1 | Raw import | ✅ | 881 legacy files copied with checksums, re-verified as identical. The import refuses to overwrite raw files | `outputs/manifests/raw_import_manifest.json` | — |
| P1.2 | Session, calendar, availability | ✅ | Bars are start-labeled, labels 09:15–15:25 (75 bars). Every row has `available_at`. Daily sources become available at 09:15 on the next trading day. The calendar has 1,824 days | `cleaning_report.json`, `tests/test_calendar.py` | — |
| P1.3 | Spot and intraday VIX relabel | ✅ | The end-labeled window (2020-12-08 to 2026-04-23) is shifted to start labels; 1,814 of 1,815 days now align. Some noise near the window boundaries remains unexplained (it affects spot RV only) | `spot_label_alignment.json` | Open question on the boundary noise |
| P1.4 | Futures roll | ✅ | One contract per day (nearest not yet expired), 89 contract changes, never an intraday switch | `cleaning_report.json` | — |
| P1.5 | Dedup rules | ✅ | Options keep the higher-volume row; futures keep the later-dated file, then higher volume; conflicting participant-OI rows are dropped. All option and futures duplicates are within a single file | `duplicate_conflicts.json`, `docs/schema.md` | — |
| P1.6 | Special sessions | ✅ | 14 non-regular sessions dropped; 3 truncated sessions flagged and excluded from T3 only | `configs/regimes.yaml` | Dates and reasons still `TO_CONFIRM` (P8.6) |
| P1.7 | Bad-print filter | ✅ | Close-to-close check against the future at 3%; 1 spot bar nulled (2021-10-18) | `spot_outliers.json` | — |
| P1.8 | Missing-data report | ✅ | Every known gap with how it is handled (see Data gaps) | `outputs/reports/missing_data.md`, `scripts/report_missing_data.py` | Rerun after any data change |
| P1.9 | Basis and jump classification | ✅ | Basis per bar. Jump threshold 24.02 bps (99.9th percentile of the aligned distribution); 115 jumps, 89 roll transitions | `basis_jumps.json` | — |
| P1.10 | Lot-size schedule | ✅ | Lot sizes inferred from OI divisibility, by contract expiry, with 8 mixed-lot exceptions; all 377 expiries with OI pass | `regimes.yaml`, `validation_report.json` | Verify against NSE circulars (P8.6) |

### P2 Pricing and exposure

| ID | Task | Status | Summary | Outputs | Next step / blocker |
|---|---|---|---|---|---|
| P2.1 | Forward | ✅ | Median near-ATM put-call parity forward; 398,962 good, 11,794 with too few pairs, 62 with high residuals; median residual 0.85 points | `forward_report.json` | — |
| P2.2 | OTM-leg IV | ✅ | Both legs of a strike take the IV of the out-of-the-money leg; 0 below-intrinsic legs remain; 99.5% of legs solve | `forward_report.json` | — |
| P2.3 | Greeks | ✅ | Black-76 gamma and delta, only where time to expiry is positive. The max-gamma strike is within one strike of the forward 91.8% of the time (nearest expiry) | `greeks_report.json` | — |
| P2.4 | GEX | ✅ | Naive, absolute and inferred conventions; formula Γ × OI (units) × S² × 0.01; 21-day horizon; same-day expiry excluded for T2 | `gex_report.json`, `data/exposure/` | Strike-level OI-change sign refinement deferred |
| P2.5 | Levels | ✅ | Call and put walls, max pain, max-GEX strike, naive and inferred flip. The naive flip lies inside the strike band in 98.7–99.9% of snapshots | `levels_report.json` | — |

### P3 Features and targets

| ID | Task | Status | Summary | Outputs | Next step / blocker |
|---|---|---|---|---|---|
| P3.1 | Aggregate features | ✅ | 4,803 snapshot rows, about 39 features, each with source timestamps | `features_report.json`, `data/features/aggregate/` | — |
| P3.2 | Token features | ✅ | 327,009 token rows from the nearest expiry; median about 70 per snapshot; 98–100% coverage by year | `tokens_report.json`, `data/features/tokens/` | Proposal says 30–50 tokens (P8.7) |
| P3.3 | T2 targets | ✅ | Next-day futures RV against its trailing 20-day level, as causal terciles; 1,549 of 1,824 labeled (starts 2020) | `targets_t2_report.json` | — |
| P3.4 | T3 targets | ✅ | Did the index move toward the max-GEX strike between 14:00 and 15:30; 1,736 labeled, 35.8% moved toward it | `targets_t3_report.json` | — |
| P3.5 | T1 targets | ⛔ | Code not written (`targets/pinning.py` absent); no official NSE closes | — | Write code with fixtures; needs official closes (P8.1) |

### P4 Validation

| ID | Task | Status | Summary | Outputs | Next step / blocker |
|---|---|---|---|---|---|
| P4.1 | Validation report | ✅ | Parity residuals, call vs put IV agreement, ATM IV vs VIX (correlation 0.95), gamma sanity, lot sizes | `docs/validation_report.md` | — |
| P4.2 | Leakage checks | ✅ | 21 timestamp and leakage checks with 0 violations, plus 36 leakage tests passing; the stage fails on any violation | `validation_report.json` | Add checks whenever a source is added |
| P4.3 | Bhavcopy reconciliation | ⛔ | Skipped because no bhavcopy files exist; the 15:25 OI proxy is used instead | — | Needs bhavcopy files |

### P5 Evaluation (classical)

| ID | Task | Status | Summary | Outputs | Next step / blocker |
|---|---|---|---|---|---|
| P5.1 | Walk-forward harness | ✅ | Expiry-cycle walks: 52-cycle training minimum, 1-cycle embargo, 13-cycle test blocks, label purge, no shuffling; 22 T2 walks | `src/niftygex_eval/harness.py`, `tests/test_harness.py`, `docs/evaluation.md` | — |
| P5.2 | Stage A (T2) | ✅ | HAC regressions per sign convention, with VIX-only and full controls, plus the joint magnitude + sign specification | run `44938c5d50046769` | — |
| P5.3 | Stage B (T2) | ✅ | Prior, VIX-only, logistic and LightGBM baselines inside the harness | runs `b0224519dbf4bb65`, `74cdb29de9ae4a0f`, `a2329c25063db2e0`, `a65e7c31d93b5ffe` | — |
| P5.4 | Calibration | ✅ | Tuned LightGBM + Platt is the reference model: Brier 0.624 against 0.669 for the prior, calibration error 0.042 | run `4b3c1866ae919f77` (also tuned `311c5e0c246bc708`, isotonic `b1d8126d90aa8d35`) | — |
| P5.5 | Three-way ablation (T2) | ✅ | A, then A + participant (B), then B + GEX and levels (C), all tuned + Platt; C beats B on AUC (t 2.30) | runs `ae3ba795796eb04f`, `adc1b37810565c4c`, `a93ea985003fb756`; `t2_ablation.json` | — |
| P5.6 | Continuous T2 | ✅ | Tuned LightGBM regression on log RV ratio; mean IC 0.40 / 0.40 / 0.43 for A / B / C | runs `38de43bed6f83cb0`, `ea4fd0cbfba9fd57`, `ba1f5cc65cdd6155` | — |
| P5.7 | C1/C2 ablation split and SHAP | ⏳ | Not started | — | Define C1/C2 (Open questions) |
| P5.8 | T3 placebo | ✅ | Matched-distance random-side and mirror placebos; no effect overall, one significant distance bucket | run `3d07783cfd0b9705`; `t3_placebo.json` | — |
| P5.9 | T3 listed-strike placebo | ⏳ | Proposed to separate GEX from round-number attraction; not started | — | Awaiting go-ahead |
| P5.10 | Continuous T3 | ⏳ | Not started (`t3_distance_reduction_pct` already exists in the T3 targets) | — | — |
| P5.11 | Holm correction | ⏳ | Not applied yet; current p-values are unadjusted | — | Define test families (Open questions) |
| P5.12 | `findings_t2.md` | ✅ | Generated from the results store | `docs/findings_t2.md`, `scripts/write_findings_t2.py` | Regenerate after reruns |
| P5.13 | `findings_t3.md` | ⏳ | Not written | — | After P5.9–P5.11 |

### P6 Neural models

| ID | Task | Status | Summary | Outputs | Next step / blocker |
|---|---|---|---|---|---|
| P6.1 | MLP / FT-Transformer on aggregates | ⏳ | Not started | — | Work split with Asad (P8.8) |
| P6.2 | Attention + GRU on tokens | ⏳ | Not started; the token table is ready | — | — |
| P6.3 | Ablations (pooling, OI-change removal, strike shuffle, sign conventions) | ⏳ | Not started | — | After P6.2 |
| P6.4 | Distribution-shift evaluation (RQ4) | ⏳ | Not started; the regime marker and lot-size schedule are available | — | — |
| P6.5 | Transfer to T1 (RQ5) | ⛔ | Needs T1 | — | P3.5 |

### P7 Dashboard and write-up

| ID | Task | Status | Summary | Outputs | Next step / blocker |
|---|---|---|---|---|---|
| P7.1 | Evaluation dashboard | ⏳ | Not started; the results store it will read exists | — | — |
| P7.2 | Progress-update slides | ⏳ | Not started | — | Due 2026-10-29 |
| P7.3 | Final report and slides | ⏳ | Not started | — | — |
| P7.4 | Reproducibility pass | ⏳ | The pipeline reruns end to end and is deterministic (ablation results reproduced exactly), but there is no git history | — | P0.6 |

### P8 External and owner tasks

| ID | Task | Status | Summary | Outputs | Next step / blocker |
|---|---|---|---|---|---|
| P8.1 | Download official NSE Nifty closes (2019–2026) | ⏳ | Absent (`data/raw/official_closes`) | — | Blocks T1 |
| P8.2 | Download daily India VIX (Apr–Jul 2026) | ⏳ | Daily VIX ends 2026-04-06; 60 trading days missing | — | — |
| P8.3 | From Sunil: missing OI (69 days), the truncated 2022-01-06 expiry, the 8 missing spot days | ⏳ | Gaps listed in `missing_data.md` | — | — |
| P8.4 | Sunil's written permission for code and data | ⏳ | Not on file in the repo | — | — |
| P8.5 | Fill in PROVENANCE authors | ✅ | Author string set as you specified on 2026-09-30 | `PROVENANCE.md` | — |
| P8.6 | Verify lot sizes, the 2024-11-20 marker, and special-session dates against NSE circulars | ⏳ | Lot sizes are inferred from data; the marker and special sessions are marked VERIFY / TO_CONFIRM | `configs/regimes.yaml` | — |
| P8.7 | Update the proposal | ⏳ | Needs: date range, VIX coverage, OTM IV, GEX formula (OI in units), 21-day horizon, futures RV, about 70 tokens, the RQ3 baseline, the disclosure and acknowledgement | `CS7150_Proposal.html` | Before 2026-10-08 |
| P8.8 | Agree the work split with Asad | ⏳ | Not recorded | — | — |

### P9 Future enhancements (out of scope)

| ID | Task | Status | Summary | Outputs | Next step / blocker |
|---|---|---|---|---|---|
| P9.1 | Replay mode | ⏳ | Out of scope | — | — |
| P9.2 | Live mode | ⏳ | Out of scope | — | — |
| P9.3 | Straddle backtest | ⏳ | Out of scope | — | — |

## Key findings so far

- **Stage A, apparent GEX effect.** With only a VIX control, normalized absolute GEX predicts a lower next-day RV ratio: coefficient −0.095, t −5.72, n 1,615 (run `44938c5d50046769`).
- **Stage A, it was confounding.** With ATM IV, days to expiry, the expiry-day flag and today's RV ratio added, the effect is −0.008 (t −0.39). Gamma is mechanically large when IV is low and expiry is near (same run).
- **Inferred sign adds information beyond magnitude.**
  - Full sample: −0.031 (t −3.09).
  - Before the marker: −0.041 (t −2.97). After: −0.021 (t −1.09).
  - The naive sign adds nothing (t −1.35).
  - It may proxy participant positioning rather than NIFTY gamma itself (same run).
- **GEX and levels add a small gain in Stage B.**
  - C vs B, mean per-walk AUC +0.019 (t 2.30, 15 of 22 walks).
  - Continuous IC +0.034 (t 2.56, 16 of 26 walks).
  - Participant features add nothing (B vs A, AUC t 0.74). Runs `a93ea985003fb756` vs `adc1b37810565c4c`; `ba1f5cc65cdd6155` vs `ea4fd0cbfba9fd57`.
- **Calibration fixes overconfidence.** The reference model (tuned LightGBM + Platt) has pooled AUC 0.639 and Brier 0.624 against the prior's 0.669. It beats the prior on 17 of 22 walks (t −3.81). Run `4b3c1866ae919f77`.
- **No overall T3 attraction to the max-GEX strike.**
  - 35.8% moved toward it, vs the placebo's 35.5% [33.5, 37.5] (p 0.41).
  - Real strike vs mirror sign test: p 0.82 (run `3d07783cfd0b9705`).
- **One T3 exception, a possible round-number effect.** When the strike is within 0.12% of spot, 20.7% moved toward it against 14.3% for the mirror (sign test p 0.011). This is uncorrected for multiple comparisons, and it may be round-strike attraction rather than gamma (same run).

## Data gaps

From `outputs/reports/missing_data.md`:

| Gap | Extent | Handling |
|---|---|---|
| Option OI recorded as 0 | 69 days: 2019-07-30, 2022-01-03 to 2022-04-11 | OI nulled; GEX, levels and OI features missing |
| Spot missing | 8 days (2024-09-16 to 09-18, 2024-09-30 to 10-01, 2025-08-14, 2025-11-17, 2025-12-01) | Not imputed; excluded where spot is needed |
| Option / futures days missing | 5 option days (2019-02-01 to 02-06, 2019-11-06); 1 futures day (2022-01-18) | Flagged |
| Truncated expiry file | 2022-01-06 expiry ends 2021-12-31 | No expiry-day chain |
| Option history per expiry | At most 21 calendar days | GEX horizon set to 21 days |
| Daily India VIX | Ends 2026-04-06; 60 trading days missing | NaN, never forward-filled |
| Intraday India VIX | 2023-01-19 to 2026-04-24 only | Optional source |
| Bhavcopy, official closes | Absent | Reconciliation skipped; T1 blocked |
| Spot bad prints / alignment exceptions | 1 bad print (2021-10-18); 1 noisy futures day (2025-12-05) | Bad print nulled; no relabel |

## Decisions log

**2026-09-30**
- Common coverage 2019-02-01 to 2026-07-03, set in `configs/data.yaml`.
- Bars are start-labeled; session labels 09:15–15:25 (75 bars); 15:30 labels dropped.
- Every row carries `available_at` (bar label + 5 min). Daily sources (EOD OI, participant OI, daily VIX) become available at 09:15 on the next trading day.
- The trading calendar is the union of option and futures dates plus a weekday extension; holidays stay empty for now.
- The futures front month is chosen once per day (nearest not yet expired); never switch intraday.
- Dedup: options keep higher volume; futures keep the later-dated file, then higher volume; conflicting participant-OI rows are dropped and logged.
- Missing-spot days are flagged, not imputed. Spot bad prints use a close-to-close test against the future at 3%.
- Spot and intraday VIX are relabeled from end to start labels for 2020-12-08 to 2026-04-23, before the session filter.
- The basis-jump threshold is 24.02 bps, the 99.9th percentile of the aligned distribution.
- IV for both legs of a strike comes from the OTM leg (call if K ≥ F); each leg's own IV is diagnostic only.
- Greeks exclude T ≤ 0. Total GEX excludes same-day expiries for T2 and keeps them for T1 and T3.
- OI is in index units, so GEX = Γ × OI × S² × 0.01 with no lot multiplier. Lot sizes are rebuilt from OI divisibility.
- Missing-OI days are flagged with GEX missing; no imputation.
- GEX horizon is 21 days (the data maximum).
- The inferred sign uses Pro as the dealer proxy; the strike-level OI-change refinement is deferred.
- Level features use the naive flip, with `has_inferred_flip` as a flag.
- Snapshot defaults: 15-minute staleness, latest-bar OI for walls, max pain on the nearest included expiry, sticky-strike flip.
- Regime marker `weekly_expiry_rationalisation` set at 2024-11-20.
- T2 realized volatility uses futures returns (intraday only) with spot RV secondary, and a 250-day cutoff warmup (labels start 2020).
- OI-change feature default confirmed (within 2% of the forward, against the previous day's 15:25 OI).
- `rv_intraday_returns` becomes a QA column, not a feature.
- The evaluation layer is a separate package (`niftygex_eval`) with its own config hash.
- Harness: expiry-cycle walks, 52-cycle training minimum, 1-cycle embargo, 13-cycle test blocks.

**2026-10-01**
- The reference T2 model is tuned LightGBM + Platt calibration (isotonic rejected).
- Ablation set C uses normalized GEX, not raw rupee totals; chain-OI flow features are in none of A/B/C.
- T3 placebo design: matched-distance random-side pseudo-strikes plus the mirror strike with a sign test.

## Open questions

1. **T3 listed-strike placebo (P5.9):** should I run it, to separate GEX attraction from round-strike attraction?
2. **Holm correction (P5.11):** which test families should be corrected (Stage A conventions × samples, ablation comparisons, T3 slices)?
3. **C1/C2 ablation split and SHAP (P5.7):** how are C1 and C2 defined (GEX totals vs levels)? Which SHAP variant (TreeSHAP on the reference model)?
4. **Version control (P0.6):** should I initialize a git repository? It's needed for the reproducibility pass.
5. **HPC (P0.7, P0.8):** are the HPC profile, sync scripts and partitioned runs still required?
6. **Spot boundary noise:** accept the leftover spot noise near the relabel-window boundaries, or investigate further? It affects spot RV (secondary) only.
7. **Token count:** keep about 70 per snapshot (0.92–1.08 band, both legs), or narrow to the proposal's 30–50?
8. **T1:** go ahead with the code and fixtures now, ahead of the official closes?
