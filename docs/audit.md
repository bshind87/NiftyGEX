# Phase 1 audit: `nifty-options-feature`

Date: 2026-09-30

This is a read-only audit of `/Users/bhalchandra/myPythonCode/nifty-options-feature` against the current proposal. No files were copied or changed in the legacy project during this phase.

## Executive summary

The old project has reusable, pure Black-76 primitives, option/futures/VIX/participant-data loaders, a parquet panel builder, and a substantial intraday feature system. It does **not** implement the requested NiftyGEX contract end to end. In particular, its labels are directional option-price hindsight labels, not the proposal's T1 pinning, T2 volatility-regime, or T3 intraday-drift targets; its GEX implementation was not found in the inspected source package; and there is no focused validation suite for timestamp leakage, IV-vs-NSE IV, OI reconciliation, or the requested GEX conventions.

The old repository is also a mixed historical/production workspace containing model training, live trading, dashboard, report, and experiment code. Only a small curated subset should be considered for Phase 2.

## 1. Legacy project inventory

### Core package: `src/nofe`

| Module/group | What it does | Dependencies/notes | Audit disposition |
|---|---|---|---|
| `config.py` | Loads YAML config and resolves paths relative to the legacy repository root. | `pyyaml`, `pathlib`; currently Python >=3.10. | Reuse pattern with refactor; remove legacy-specific config shape and raise Python floor to 3.11. |
| `io/loaders.py` | Loads compressed CSV option, futures, spot, VIX, and participant OI data into Polars frames; coerces dates/datetimes and expected columns. | `polars`; paths come from `config.py`. | Reuse with refactor. Add explicit timezone/session semantics and source availability timestamps. |
| `io/schema.py` | Defines/helps enforce legacy raw and panel schemas. | Polars. | Reuse with refactor; proposal needs a curated schema document with units and availability time, not only runtime schema checks. |
| `io/calendar.py` | Trading/expiry calendar helpers and holiday handling. | Config JSON/YAML and date logic. | Reuse with refactor; expiry-weekday and rule-change calendar must become explicit config. |
| `greeks/black76.py` | Vectorized Black-76 prices, delta, gamma, vega, theta, vanna, vomma, speed, color, and charm. | `numpy`, `scipy.stats.norm`; pure math. | Reuse as-is conceptually, after tests and API/type cleanup. Gamma formula is appropriate for call/put equality. |
| `greeks/pricing.py` | Synthetic forward from one near-ATM call/put pair, Newton IV inversion, and robust quadratic smile fitting. | `numpy`; calls Black-76. | Reuse with refactor. The one-strike forward estimator and smile fit need explicit policy, quality flags, and parity residuals. |
| `greeks/engine.py` | Per-snapshot forward/IV/smile/greek computation and historic/live drivers. | Polars plus the above modules and legacy loaders. | Reuse with refactor; contains extra live/smile behavior and hard-coded operational constants. Must be reduced to the requested pure library plus separate adapters. |
| `greeks/loader.py` | Reads processed legacy greek parquet files. | Polars and legacy paths. | Reuse with refactor only if needed for migration; not a source of truth for the new pipeline. |
| `features/catalog.py` | Catalog of many per-option features, families, descriptions, and warmup bars. | Dataclasses only. | Reuse ideas with refactor; catalog is much larger and directed at old models, not the proposal's ~20 aggregates. |
| `features/option.py` | Per-option returns, EMA/RSI/MACD/ATR, volume/OI changes, IV and greek changes, moneyness. | Polars. | Partial reuse with refactor; useful primitives but not the required aggregate/token feature contract. |
| `features/chain.py` | Adds total OI by timestamp and option side. | Polars. | Reuse with refactor; useful for normalized token features. |
| `features/market.py` | Spot trend, futures OI/flow, basis, returns, and IV-rank features with cached historic loaders. | Polars, NumPy, glob, legacy config. | Partial reuse with refactor. Several features are outside scope, and cached disk drivers conflict with a clean stage boundary. |
| `features/normalize.py` | Feature normalization/scaling for old model inputs. | NumPy/Polars. | Out of scope; modeling layer. |
| `features/sell_extra.py` | Features specific to old short-premium models. | Polars. | Out of scope; dead for this phase. |
| `panel/builder.py` | Joins raw option bars with processed greeks and adds feature inputs. | Polars, legacy persistence/loaders. | Reuse with refactor; panel shape and joins need to be redesigned around curated Parquet stages. |
| `panel/persist.py` | Reads/writes per-expiry panel parquet. | Polars and legacy paths. | Reuse with refactor; retain the independent Parquet-stage idea. |
| `labels/builder.py` | Hindsight directional CALL/PUT labels using option-price paths, trailing stops, sustained returns, and session-only exits. | Polars, legacy config/calendar/panel. | Not reusable for T1/T2/T3; useful only as an example of forward-only label eligibility. |
| `labels/buy2_builder.py`, `sell_builder.py`, `sideways_builder.py` | Additional old trading labels, including multi-day spread and premium targets. | Legacy panel/config and production rules. | Out of scope; unrelated targets and modeling workflows. |
| `labels/catalog.py`, `labels/persist.py` | Catalog and Parquet persistence for old labels. | Dataclasses/Polars. | Reuse persistence pattern with refactor; replace target definitions. |
| `dataset/*` | Sequence/grid construction and persistence for old models. | Polars/NumPy and model-specific schemas. | Out of scope; modeling. |
| `model/seq.py` | Neural sequence model code. | PyTorch. | Out of scope. |
| `live/*` | Broker/live feed adapters, inference, runtime, notifications, and production grids. | Broker APIs, PyTorch, network/runtime dependencies. | Out of scope; do not copy. |
| `dag.py`, `steps.py` | Legacy pipeline/DAG orchestration. | Legacy config and stage conventions. | Reuse only as architectural reference; rewrite for independent configured stages. |
| `utils/*` | Logging and parquet helpers. | Polars/path utilities. | Reuse small utilities only after reviewing API and stage semantics. |
| `test.py` | Legacy smoke/test code. | Not a focused pytest suite. | Do not copy as the new test suite. |

### Scripts and other top-level material

- `scripts/build_greeks.py`, `build_panel.py`, `build_dataset.py`: useful historical batch-driver references; rewrite as thin stage CLIs.
- `scripts/parity_check.py`: checks legacy disk/live feature parity, not proposal validation; reference only.
- `scripts/build_labels.py`, `build_buy2_labels.py`, `build_sell_labels.py`, `build_sideways_labels.py`: old trading-label drivers; do not copy.
- `scripts/train*.py`, `backtest.py`, `live_check.py`, report generators, and `scripts/analysis/*`: modeling, backtesting, diagnostics, and experiments; exclude.
- `dashboard/`, `models/`, `reports/`, and most `processed/`: generated or user-facing/model artifacts; exclude from the new implementation.
- `__pycache__`, `.DS_Store`, archived reports, and generated logs: dead/generated material; exclude.

### Declared dependencies

The legacy `pyproject.toml` declares Polars, PyArrow, NumPy, SciPy, PyYAML, and Rich, with pytest/Ruff as development dependencies and PyTorch/scikit-learn for modeling. The proposal's minimal set is compatible with the useful core, but Rich, PyTorch, scikit-learn, broker/dashboard packages, and model-specific code are unnecessary for this phase.

## 2. Raw data inventory

All paths below are in the legacy project and are read-only inputs for this audit.

| Source | Location and format | Observed schema | Observed coverage |
|---|---|---|---|
| NIFTY options | `data/options/NIFTY/<year>/*.csv.gz` (also a few uncompressed CSV duplicates); 787 compressed files | `datetime, stock_code, expiry_date, right, strike_price, open, high, low, close, volume, open_interest, date` | `2019-01-10 09:15:00` through `2026-07-31 15:25:00`; 22,878,869 rows scanned |
| NIFTY futures | `data/futures/NIFTY/<year>/*.csv.gz`; 91 compressed files | `datetime, stock_code, expiry_date, open, high, low, close, volume, open_interest, count, date` | `2018-11-02 09:15:00` through `2026-07-06 15:25:00`; 416,964 rows scanned |
| NIFTY spot/index | `data/equity/NIFTY_5minute.csv.gz` | `datetime, stock_code, open, high, low, close, volume, date` | `2019-01-01 09:15:00` through `2026-07-31 15:35:00`; 175,517 rows scanned |
| India VIX intraday | `data/india_vix/INDVIX_5minute.csv.gz` | Same OHLCV-style columns as spot | `2023-01-19 13:35:00` through `2026-04-24 16:00:00`; 80,636 rows scanned |
| India VIX daily | `data/india_vix/INDVIX_daily.csv.gz` | Same OHLCV-style columns as spot | `2013-01-01 12:01:00` through `2026-04-06 12:04:00`; 3,266 rows scanned |
| Participant OI | `data/fiidii/participant_oi_2019_to_today.parquet` | `date`, `Client Type`, futures/options long/short columns, totals, and `Unnamed: 15` | `2019-01-01` through `2026-07-03`; 9,255 rows; date is stored as a naive datetime |
| Metadata | `data/metadata/dhan_scrip_nifty.csv.gz`, `NIFTY_weekly_strikes_atm.csv` | Instrument/strike metadata; not yet audited as a canonical source | Date/meaning require confirmation before use |

Important data caveats and approved decisions:

- The inspected option schema has no NSE-published IV column. The approved validation replacement is call-versus-put IV agreement by strike plus approximately 30-day ATM IV versus India VIX.
- The participant file is daily and ends earlier than the options/spot data. Its publication/availability time is not encoded; the new pipeline must lag it by one full trading day.
- The legacy files include timestamps beyond the requested `09:15-15:30` session in at least the spot/VIX sources. Session filtering must be stage-owned and tested.
- The approved common coverage is `2019-02-01` through `2026-07-03`, configured in `configs/data.yaml`. The July 2026 records are real historical data as of this audit date, not future-dated data.
- No bhavcopy is present. Until an optional bhavcopy source is configured, the 15:25 bar's OI is the EOD proxy and is available only from the next trading day; bhavcopy reconciliation must skip cleanly when absent.
- Daily India VIX, lagged to the previous close, is the main VIX feature. Intraday VIX is optional and may be NaN.
- There are no bid/ask fields. IV is computed from bar close, with `volume > 0` and the configured moneyness band required for snapshot eligibility.
- Official NSE closing index values for T1 are not present and require an optional daily official-close input.

## 3. Proposal requirement mapping

Status meanings: **reuse as-is** means the old implementation is close to the requested contract; **reuse with refactor** means useful logic exists but its API, assumptions, or validation are insufficient; **missing** means no implementation was found; **buggy or questionable** means the implementation has a correctness or auditability concern that must be resolved before reuse.

| Required output | Legacy coverage | Status and reason |
|---|---|---|
| 1. Cleaned curated option-chain Parquet plus schema document | `io/loaders.py`, `io/schema.py`, `panel/persist.py`, plus raw option loaders | **Reuse with refactor.** Loading and Parquet persistence exist, but the old panel is model-oriented; there is no demonstrated curated chain contract covering deduplication, session/special-session policy, OI reconciliation, liquidity/forward bands, source availability, and a complete `docs/schema.md`. |
| 2. Black-76 greeks and GEX library; synthetic forward; naive/absolute/inferred signs | `greeks/black76.py`, `greeks/pricing.py`, `greeks/engine.py`; no clearly identified GEX module in `src/nofe` | **Black-76 reuse as-is conceptually; forward/engine reuse with refactor; GEX missing.** The core gamma is pure and vectorized. The forward uses one selected pair rather than a documented robust per-expiry policy. The three requested GEX conventions, lot-size regime calendar, and participant/OI-change inference were not found as a standalone implementation. |
| 3. Aggregate snapshot features, about 20 proposal groups | `features/catalog.py`, `features/option.py`, `features/chain.py`, `features/market.py`, `panel/builder.py` | **Reuse with refactor.** Many lower-level features exist, but they target old directional models and do not provide the proposal's explicit aggregate groups: GEX profile, positioning, volatility surface, and market state with a stable ~20-column snapshot schema. |
| 4. Per-strike liquid token features | `features/option.py`, `features/chain.py`, panel construction | **Reuse with refactor.** Per-option feature generation and chain OI totals are useful, but liquidity selection, fixed token semantics, missing-strike handling, and snapshot-time availability are not defined as the new contract. |
| 5. T1 pinning, T2 volatility regime, T3 intraday drift | `labels/builder.py` and other old label builders | **Missing for proposal.** Existing labels are option-price path/trailing-stop hindsight labels. They do not compute settlement-to-strike pinning, next-day RV terciles relative to trailing 20-day RV/ATM IV, or 14:00-15:30 movement toward max-GEX. |
| 6. Validation checks and `docs/validation_report.md` | `scripts/parity_check.py`, `scripts/*parity*`, limited smoke code | **Missing/partial.** Legacy parity checks compare old feature paths, not IV-vs-NSE IV, parity residuals, last-snapshot-vs-bhavcopy OI, gamma sanity, or feature timestamp ordering. No requested validation report was found. |

Related proposal evaluation constraint: the proposal requires expiry-aligned walk-forward evaluation with at least one expiry of embargo and no shuffling. The old project contains walk-forward/model artifacts, but that harness is explicitly out of scope for this phase and should not be copied into the data layer.

## 4. Leakage, hard-coded paths, and questionable behavior

### Leakage and availability risks

1. `labels/builder.py` is explicitly forward-looking by design, but it is an old target implementation and must not be joined into feature rows as though it were available at snapshot time.
2. The old feature code uses rolling shifts and cached historical frames, but there is no general assertion that every feature source timestamp is `<= snapshot_timestamp`.
3. Participant OI is daily after-close information. The old loader exposes it, but the audit did not find a project-wide one-full-trading-day lag contract.
4. EOD bhavcopy availability is not represented in the raw option schema. Treating same-day EOD OI as an intraday feature would violate the proposal rule; reconciliation must be separate from feature construction.
5. Some old market features splice/select a front-month future by timestamp and use rolling history. This needs explicit tests around contract selection, expiry boundaries, and warmup rows.
6. The old labels select the best future outcome across a strike band. That is appropriate for a hindsight label only, but it is not evidence that the feature pipeline is leakage-safe.

### Hard-coded paths and dates

- The package configuration mostly resolves paths relative to the legacy repository, which is preferable, but the paths point to the old project layout and must not be copied unchanged.
- Several analysis scripts contain absolute paths such as `/Users/sunilkandakur/AI/nifty-options-feature/...`; these are non-portable and dead for the new project.
- Analysis scripts hard-code years and report periods, especially 2019-2026 and 2025/2026 split windows. They are out of scope and should not inform production configuration.
- The greek engine has operational constants such as session close, minimum price, seconds/year, and bars/year. These require a documented configuration policy or should be passed explicitly where they affect data semantics.
- Legacy config includes observed-through dates and manually maintained holidays. Useful as input, but not authoritative for the new project's rule-change calendar without verification.

### Data and implementation questions to resolve in Phase 2/3

- Confirm the intended historical date range and whether the apparent 2026 future-dated filenames are real data, planned paths, or artifacts of the dataset snapshot.
- Confirm whether options are truly 5-minute bars and how duplicate uncompressed/compressed files are deduplicated.
- Implement the approved median near-ATM parity-forward estimator, retaining per-strike residuals and a quality flag.
- Use spot, not forward, in the $S^2$ GEX scaling; source lot sizes and expiry weekdays from manually maintained `configs/regimes.yaml`.
- Represent inferred signs as NaN with an availability flag when participant data is missing; never forward fill.
- Add an optional official-close input for T1 and flag its absence in validation/output metadata.

## 5. Recommended Phase 2 boundary

Copy/rewrite only the following concepts after approval:

- raw-source loaders and schema normalization from `src/nofe/io/`
- holiday/expiry calendar logic from `src/nofe/io/calendar.py`
- pure Black-76 primitives from `src/nofe/greeks/black76.py`
- selected forward/IV inversion logic from `src/nofe/greeks/pricing.py`, rewritten around an explicit parity policy
- selected chain/OI helpers from `src/nofe/features/chain.py`
- small, reviewed feature primitives from `src/nofe/features/option.py` and `market.py`
- Parquet persistence patterns from `panel/persist.py` and `labels/persist.py`
- batch-driver structure from `scripts/build_greeks.py` and `build_panel.py`, rewritten as thin new CLIs

Do not copy live, model, dashboard, report, experiment, old-label, or generated-data directories. The new GEX, cleaning, targets, validation, provenance, configuration, tests, and documentation should be written against the approved new contracts rather than inherited wholesale.

## Phase 1 conclusion

The old project is a useful source of mathematical and loading primitives, but not a drop-in implementation of the requested data layer. The highest-risk areas are missing GEX/sign-convention logic, mismatched targets, unverified data availability timing, and lack of validation/leakage tests. Phase 2 should therefore be a selective rewrite plan, not a bulk copy.
