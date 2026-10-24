# NiftyGEX data schema

## Conventions

- **Timezone.** Every timestamp is timezone-aware `Asia/Kolkata` (IST), stored as `datetime[us, Asia/Kolkata]`.
- **Bar labels.** Bars are start-labeled. A bar labeled `t` covers `[t, t + 5 min)`. The regular session keeps labels 09:15 through 15:25 inclusive, which is 75 bars per day ending at 15:30. Every other label is dropped, including 15:30. The bar length and session labels are set in `configs/data.yaml`.
- **Availability.** Every curated row carries `available_at`, the earliest time a feature may use it:
  - **Intraday bars** (options, futures, spot, intraday VIX, and the basis and forward/IV rows derived from them): `available_at = datetime + 5 min`, the moment the bar's close is known.
  - **Daily sources** (the 15:25 EOD OI proxy, participant OI, and daily VIX): `available_at` is 09:15 IST on the next trading day strictly after `observation_date`. `available_date` holds that date.
- **Snapshots.** A snapshot at time `T` uses only rows with `available_at <= T`. Leakage checks compare `available_at`, never bar labels. For example, a 15:15 snapshot can use the bar labeled 15:10 but not the one labeled 15:15.
- **Trading calendar.** The calendar is the union of option and futures trading dates, minus configured holidays. It is extended past the last observed date by weekday rule, so every observed date has a next trading date. When an observation date has no later trading date, the code raises an error; it never falls back to the observation date itself.
- **Day-level quality.** Each curated row carries `truncated_session`, `missing_spot`, and `data_quality`, joined from the calendar by trade date. These are explained in [Day-level quality codes](#day-level-quality-codes).

## Day-level quality codes

`data_quality` is either `ok` or a `|`-joined list of codes. These are day-level codes, shared by every source on that date.

| Code | Meaning | Action |
|---|---|---|
| `missing_spot` | Trading day (option and/or futures data present) with no spot bars | Not imputed. Exclude from anything that needs spot: GEX `S^2` scaling, realized volatility, and targets |
| `missing_options` / `missing_futures` | No bars from that source on a calendar trading day | Flag only |
| `incomplete_option_bars` / `incomplete_futures_bars` / `incomplete_spot_bars` | Between 1 and 74 distinct bar labels; for options, labels are counted across all contracts | Flag only; rows kept |
| `missing_option_oi` | Fewer than `quality.min_option_oi_share` (0.5) of traded option rows report `open_interest > 0`. The source records missing OI as 0. This covers 2019-07-30 and 2022-01-03 through 2022-04-11, 69 days | `open_interest` is set to null on these days in curated options and the EOD OI proxy. GEX and OI-based levels are missing |
| `configured_flag` | Date listed under `data_quality_flags` in `configs/regimes.yaml` | Flag only; rows kept |

`truncated_session` marks the configured truncated sessions (2020-03-13, 2020-03-23, 2021-02-24). Their rows are kept and flagged. Non-regular sessions in `regimes.yaml` (`non_regular_sessions`) are dropped from every source, including participant OI.

## Duplicate resolution

Duplicates are resolved by sorting on the key and the keep order below, then keeping the first row per key. Every rule is deterministic: it does not depend on the order in which rows or files are read.

| Source | Key | Keep order |
|---|---|---|
| Options | `datetime, expiry_date, strike_price, right` | higher `volume`, then higher `open_interest`, higher `close`, `source_file` (ascending), `source_row` (ascending) |
| Futures | `datetime, expiry_date` | later `source_file_date`, then higher `volume`, higher `open_interest`, higher `close`, `source_file`, `source_row` |
| Spot, intraday VIX | `datetime` | higher `volume`, then `source_row` |
| Daily VIX | `observation_date` | higher `volume`, then `source_row` |
| Participant OI | `observation_date, Client Type` | Exact duplicates collapse. When a key's duplicates disagree, every row for that key is dropped and logged row by row in `duplicate_conflicts.json` (`participant_oi.rows`), so a participant sign is never chosen by row order |

In the current raw data, every option and futures duplicate is inside a single source file. Each file holds one expiry (options: one expiry and right), so the `source_file_date` rule never decides a conflict. Option deduplication runs per file, and the cleaning stage fails if any option key appears in more than one file. `outputs/reports/duplicate_conflicts.json` reports exact versus conflicting rows, the size of the differences, the column that decided each conflict, and clustering by date, month, and source file.

## Curated outputs (`data/curated/`, stage `clean`)

### `options/<year>_<file>.parquet`

One file per raw option file, which covers one expiry and right.

| Column | Type | Unit | Notes |
|---|---|---|---|
| `datetime` | datetime IST | bar start label | |
| `available_at` | datetime IST | | `datetime + 5 min` |
| `trade_date` | date | | Date of `datetime` |
| `stock_code` | string | | `NIFTY` |
| `expiry_date` | date | | Equals the date in the file name; checked by the forward stage |
| `right` | string | `Call` / `Put` | |
| `strike_price` | float64 | index points | |
| `open`, `high`, `low`, `close` | float64 | rupees per unit | IV uses `close`; there is no bid/ask |
| `volume` | float64 | units | Bar volume |
| `open_interest` | float64 | index units | Intraday OI in index units, always a multiple of the lot size; see the GEX report's `lot_size_inference`. Null on `missing_option_oi` days. The 15:25 value is also published as the EOD proxy, usable the next trading day |
| `source_file`, `source_file_date`, `source_row` | string, date, uint32 | provenance | `source_file_date` is the date in the file name (the contract expiry) |
| `truncated_session`, `missing_spot`, `data_quality` | bool, bool, string | day flags | See [Day-level quality codes](#day-level-quality-codes) |

### `futures/nifty_front_month.parquet` and `futures/front_month_contracts.parquet`

- **Contract selection.** The contract is chosen once per trading day by the configured roll rule. For `nearest_expiry_not_expired`, that is the earliest expiry on or after the trade date among contracts with at least one bar that day. The expiring contract stays front month on its expiry day.
- **No intraday switching.** Bars missing from the chosen contract stay missing; there is no fallback to another contract within the day.
- **Columns.** Front-month bars use the option columns minus `right` and `strike_price`. `front_month_contracts.parquet` has one row per day: `trade_date, expiry_date`.

### `market/nifty_spot.parquet`

Spot OHLCV with the same timing and day-flag columns, plus `spot_quality`:

| `spot_quality` | Meaning |
|---|---|
| `ok` | Checked against the front-month future on the same bar |
| `bad_print` | `abs(spot close / futures close - 1)` exceeds `spot_outliers.max_relative_deviation_from_future` (3%). OHLC is nulled and the bar is logged in `outputs/reports/spot_outliers.json` |
| `unchecked` | No futures bar at that label |

The test is close-to-close. Open, high, and low can legitimately diverge on circuit-halt reopen bars, and futures highs and lows have their own bad prints.

**Label convention.** In the raw data, spot and intraday VIX are end-labeled from 2020-12-08 through 2026-04-23; options and futures are start-labeled throughout. `end_labeled_windows` in `configs/regimes.yaml` relabels rows in that window to start labels (label - 5 min) **before** session filtering:

- the real last regular bar (end label 15:30) becomes 15:25 and is kept;
- the end-labeled 09:15 pre-open bar becomes 09:10 and is dropped.

`outputs/reports/spot_label_alignment.json` verifies the result on every run. The only remaining exception, 2025-12-05 (RBI policy day), shows weak correlation at every lag (erratic futures closes), not a label offset.

**Residual boundary noise (accepted).** Within-day spot–futures basis noise is higher in the 20 trading days next to both relabel-window boundaries, and the gap is not explained by volatility (`relabel_window_boundary_check`). The close remains the best-synchronized spot field, and the alignment stays at lag 0. This was accepted on 2026-10-01. It affects only spot-based realized volatility (`rv_spot_*`, the secondary T2 columns). Primary T2 RV uses futures, and GEX uses spot only as a price level.

### `vix/intraday.parquet` and `vix/daily.parquet`

- **Intraday.** Optional source. It has spot-like columns and may be absent on a day; nothing is filled.
- **Daily.**
  - **Columns:** `observation_date`, `source_datetime` (the vendor stamp, around 12:00), OHLCV, `available_date`, `available_at`.
  - **Gaps:** dates without daily VIX stay absent. They are never forward-filled.

### `oi/option_eod_oi_1525.parquet`

The bhavcopy proxy: OI observed on the 15:25 bar.

- **Columns:** `observation_date, observed_bar, expiry_date, strike_price, right, open_interest, available_date, available_at`.
- **Availability:** next trading day at 09:15 IST.

### `oi/participant_oi.parquet`

- **Columns:** raw participant columns plus `observation_date, available_date, available_at`.
- **Availability:** next trading day at 09:15 IST. Missing participant days are never forward-filled.

### `calendar/trading_calendar.parquet`

- **Columns:**
  - `trading_date`
  - `observed` (false for weekday-rule extension days)
  - `next_trading_date`
  - per-source bar counts (`options_bars`, `futures_bars`, `spot_bars`, `vix_intraday_bars`)
  - `truncated_session`, `missing_spot`, `data_quality`
- **Use:** this is the authoritative day table.

## Basis outputs (`data/curated/basis/`, stage `basis`)

The jump threshold `basis.jump_threshold_bps = 24.02` is the 99.9th percentile of the aligned within-day absolute basis-step distribution, measured after the relabel.

### `nifty_basis.parquet`

One row per bar label present in front-month futures or spot.

| Column | Notes |
|---|---|
| `future_close`, `spot_close` | Null when the source has no bar or the spot bar was nulled as a bad print |
| `basis` | `future_close - spot_close`, in index points |
| `basis_bps` | `basis / spot_close * 1e4` |
| `jump_class` | Set on the bar where a within-day jump starts: `spike`, `event`, or `unresolved` |
| `basis_quality` | `good`, `bad` (bars inside a reverting spike), or `missing` |
| `roll_day` | The day's contract differs from the previous trading day's |
| `spot_vs_futures_best_lag` | Inferred spot label alignment for that day (0 = aligned, -1 = spot end-labeled) |

A jump is an absolute change in `basis_bps` above `basis.jump_threshold_bps` between consecutive valid bars of the same day. It is classed as follows:

- **Reverting spike:** the basis returns to within `revert_tolerance_fraction` × threshold of its pre-jump level within `revert_max_bars` bars.
- **Unresolved:** the reversion window runs past the session end.
- **Event:** any other jump.

Roll transitions are the day-boundary steps on roll days.

### `spot_label_alignment.parquet`

Per-day correlations of 5-minute returns between futures and spot at label lags -1, 0, and +1, with `best_lag`.

## Pricing outputs (`data/greeks/`, stage `forward`)

Files are partitioned as `year=YYYY/expiry=YYYYMMDD.parquet`.

| Dataset | Grain | Columns |
|---|---|---|
| `forward/` | One row per (snapshot, expiry) | `forward`, `forward_pair_count`, `forward_residual_median_abs`, `forward_residual_max_abs`, `forward_quality` (`good`, `insufficient_pairs`, or `residual_high`), `time_to_expiry`, `available_at` |
| `parity/` | Near-ATM call/put pairs used by the forward | `call_close`, `put_close`, volumes, `parity_forward`, `provisional_forward`, `near_atm`, `forward`, `parity_residual` |
| `iv/` | Every option leg with moneyness in the configured band | Timing and day-flag columns, `time_to_expiry`, `forward`, `forward_quality`, `moneyness` (`strike / forward`), `log_moneyness`, `iv`, `iv_status`, `iv_source_right`, `iv_source_volume`, `iv_own`, `iv_own_status`, `eligible` |

**Forward rules**

- **Pair eligibility:** a parity pair needs both legs to have `volume >= forward.min_volume` on the same bar.
- **Forward estimate:** the median `parity_forward` over strikes within `forward.near_atm_width` of the all-pair provisional median.

**IV rules**

- **Band:** the IV band is `liquidity.moneyness_min` to `liquidity.moneyness_max` (currently 0.92 to 1.08).
- **OTM sourcing:** both legs of a strike take the IV solved from the strike's out-of-the-money leg: the call when `strike >= forward`, the put when `strike < forward`.
  - `iv_source_right` and `iv_source_volume` describe that OTM leg.
  - `iv_own` and `iv_own_status` keep each leg's own solve, as a diagnostic only. Stale deep in-the-money closes often price below intrinsic.
  - When the OTM leg has no row on the bar, `iv_status` is `otm_leg_missing`.
- **`iv_status` codes:** `ok`, `no_forward`, `nonpositive_time`, `nonpositive_price`, `below_intrinsic`, `above_upper_bound`, `not_converged`, `otm_leg_missing`. `iv` is null or NaN unless the status is `ok`.
- **`eligible`:** own `volume > 0`, `iv_source_volume > 0`, `iv_status == ok`, `forward_quality == good`, and `time_to_expiry > 0`.

**Time and rate**

- **Time to expiry:** calendar years (365 days per year) from `available_at` to 15:30 IST on `expiry_date`. Expiry-day bars labeled 15:25 therefore have `time_to_expiry = 0` and status `nonpositive_time`.
- **Risk-free rate:** comes from `configs/regimes.yaml` (`risk_free_rate.default`).

## Greeks (`data/greeks/greeks/`, stage `greeks`)

Partitioned like `iv/`. Each file holds the IV rows that pass `greeks_input()`, plus:

- `gamma`: Black-76, per forward index point and per index unit. It is identical for both legs of a strike, because both carry the OTM-sourced IV.
- `delta`: Black-76 forward delta.

`outputs/reports/greeks_report.json` holds two sanity checks: the share of snapshots whose max-gamma strike is within one strike step of the forward, and ATM gamma by days to expiry.

## Exposure (`data/exposure/`, stage `gex`)

**Snapshot grid** (`snapshots/snapshot_grid.parquet`)
- **Rows:** one per (`snapshot_at`, `target`).
- **Schedule:** T2 at 15:15 on every trading day (`snapshot_grid.t2_daily`). The targets built are listed in `niftygex.exposure.selection.TARGETS`.

**GEX chain at snapshot `T`**
- **Rows:** per contract, the latest Greeks row with `T - snapshot.max_staleness_minutes < available_at <= T` on the snapshot's trade date.
- **Spot:** the latest non-null spot close in the same window.
- **Participant OI:** the participant row whose `available_date` is the snapshot's trade date, which is the previous trading day's observation. A missing day stays missing.

**GEX per leg (rupees)**
- **Formula:** `GEX = gamma x open_interest x spot^2 x 0.01`.
- **No lot-size multiplier**, because `open_interest` is already in index units.
- **Sign conventions:**

| Convention | Sign |
|---|---|
| `gex_naive` | Calls +, puts − |
| `gex_absolute` | All + |
| `gex_inferred` | Call legs × the sign of the dealer proxy's (`gex.inferred.dealer_client_type`, Pro) net index-call position; put legs × the sign of its net index-put position. Signs come from the previous trading day's participant OI. The participant data covers all index options, not only NIFTY |

**Outputs**

| File | Grain | Notes |
|---|---|---|
| `gex_strikes/year=YYYY.parquet` | (snapshot, expiry, strike, right) | Chain rows with `spot`, dealer signs, and the three GEX columns |
| `gex_expiry/gex_expiry.parquet` | (snapshot, expiry) | Sums per convention, with `spot_available`, `oi_available`, `inferred_available` |
| `gex_totals/gex_totals.parquet` | (snapshot, target) | Sums over the expiries selected by `total_gex_expiries(target)`, plus `expiries`, `includes_same_day_expiry`, and min/max days to expiry |

**Missing totals.** A total is null when spot or option OI is unavailable. The inferred total is also null when participant signs are missing. Nothing is filled.

**Horizon.** `gex.max_days_to_expiry` is 21, the data maximum: each expiry has at most about 21 calendar days of option history in the source.

**Lot sizes.** Lot size is not part of the GEX formula, because OI is already in index units. `configs/regimes.yaml` `lot_sizes` maps contract expiry to index units per lot, as inferred from OI divisibility:

| First expiry with this lot | Units per lot |
|---|---|
| 2019-02-14 | 75 |
| 2021-07-29 | 50 |
| 2024-05-02 | 25 |
| 2025-01-02 | 75 |
| 2026-01-06 | 65 |

`lot_size_exceptions` lists 8 contracts listed before a change, whose OI mixes the old and new lots. A contract test and the validation report check the schedule against the data.

**Participant OI scope.** Participant OI covers all index options, not only NIFTY. Its mix changes at the regime marker `weekly_expiry_rationalisation` (2024-11-20) in `regimes.yaml`.

**Levels** (`levels/levels.parquet`, one row per (snapshot, target))

| Column | Definition |
|---|---|
| `call_wall` | Strike with the largest call OI strictly above spot, over included expiries |
| `put_wall` | Strike with the largest put OI strictly below spot, over included expiries |
| `max_pain` | For the nearest included expiry (`max_pain_expiry`): the settlement strike minimizing total intrinsic payout |
| `max_gex_strike` | Strike with the largest absolute-convention GEX |
| `flip_naive`, `flip_inferred` | Total GEX recomputed on `levels.flip_grid_points` hypothetical spot levels across the chain's strike range. The forward moves proportionally with spot; IV is held per strike (sticky strike); time to expiry is fixed. The value is the interpolated sign change closest to spot. `*_crossings` gives the count, and the value is null when total GEX does not change sign. Under `inferred`, a flip exists only when the call and put dealer signs differ |
| `strike_band_low`, `strike_band_high` | The chain's strike range |
| `*_tied` | Whether the level's optimum was tied |

- **OI source:** walls and max pain use the latest row of the day per contract, with no volume requirement. Rows with null OI are excluded.
- **Ties:** the strike closest to spot wins, then the lower strike.

## Aggregate features (`data/features/aggregate/aggregate.parquet`, stage `features`)

One row per (`snapshot_at`, `target`). Expiries are selected per target by `total_gex_expiries`. Every source column `*_available_at` and `feature_max_source_available_at` must be `<= snapshot_at`; the validation stage checks this. Daily sources are matched to exactly the previous trading day (`available_date == trade_date`) and are never forward-filled.

| Group | Columns |
|---|---|
| Gamma exposure | `gex_total_naive`, `gex_total_absolute`, `gex_total_inferred`, `gex_concentration_hhi` (Herfindahl index of absolute GEX across strikes), `gex_expiries`, `includes_same_day_expiry` |
| Key levels (percent of spot) | `call_wall_distance_pct`, `put_wall_distance_pct`, `max_pain_distance_pct`, `max_gex_strike_distance_pct`, `flip_naive_distance_pct`, `has_flip`, `has_inferred_flip`. The naive flip is the level feature; inferred is a flag only |
| Positioning flow | `oi_change_near_money_calls`, `oi_change_near_money_puts`: OI at the snapshot vs the previous trading day's 15:25 EOD OI, for strikes within `features.near_money_width` (2%) of the forward, over contracts present on both days. `put_call_oi_ratio` |
| Volatility surface | `atm_iv` (nearest included expiry, strike nearest the forward), `skew_25d` (IV of the put nearest −0.25 delta minus IV of the call nearest +0.25 delta), `atm_iv_next`, `term_slope` (next expiry minus nearest), `days_to_next_expiry`, `time_to_expiry` |
| Market state | `futures_basis_bps` (latest `basis_quality=good` bar within staleness), `vix_previous_close`, `rv_intraday_so_far_annualized` (front-month futures returns so far today). `rv_trailing_{1,5,20}d_annualized_prev`: mean annualized futures RV over the 1, 5 and 20 trading days ending the previous trading day (at least 1, 4 and 15 present; `features.har_rv_windows` and `trailing_rv_days`); the HAR-RV daily, weekly and monthly terms, available from that day's close. `qa_rv_intraday_returns`, the return count, is a QA column and not a model feature |
| Participant positions | `participant_{pro,fii,client}_net_index_{calls,puts}`: previous trading day, contracts, all index options |
| Calendar and regime | `days_to_expiry`, `is_expiry_day`, `nearest_expiry`, `nearest_expiry_weekday`, `lot_size`, `regime_post_weekly_expiry_rationalisation` |
| Flags | `spot_available`, `oi_available`, `inferred_available`, `vix_available`, `missing_spot`, `missing_option_oi`, `truncated_session`, `data_quality` |

`outputs/reports/features_report.json` gives missingness per column (overall and per target) and per-year coverage.

## Token features (`data/features/tokens/year=YYYY.parquet`, stage `tokens`)

One row per liquid option leg of the day's nearest included expiry, per sequence snapshot (`snapshot_at`, `target`).

**Sequence snapshots.** Every `snapshot_grid.sequence_step_minutes` (5) from `sequence_start` (09:20, when the 09:15 bar is known) up to and including the target's prediction snapshot (15:15 for T2): 72 per day. `prediction_at` is that day's prediction snapshot, and `lag` counts snapshots back from it (0 = 15:15, 71 = 09:20). A model with an L-snapshot window uses lags 0 to L − 1 of each day. Each day's history stays within that day.

**Which expiry.** One expiry per day (`selection.nearest_expiry_by_day`): the nearest expiry that has Greek rows that day and that total GEX includes for the target, so T2 tokens never use the same-day contract and a history never switches contracts intraday.

**What counts as liquid.** A liquid leg is a GEX-chain row, built with the GEX stage's rules: eligible (traded, OTM-sourced IV solved, good forward, positive time to expiry), the latest row available within the 15-minute staleness window, spot as of the snapshot, previous-day participant signs. At lag 0 the tokens equal the GEX stage's chain for that expiry (stage-contract test).

**Order.** Tokens are sorted by strike, calls before puts, and `token_index` numbers them from 0 within each snapshot. Snapshots usually carry 40–90 tokens; the median is about 70.

| Field | Definition |
|---|---|
| `log_moneyness` | `ln(K / F)` |
| `iv`, `gamma`, `delta` | Black-76, OTM-sourced IV |
| `strike_gex_absolute_crore`, `strike_gex_naive_crore` | Leg GEX in crore rupees: absolute, and signed calls +/puts − |
| `open_interest`, `open_interest_contracts` | Index units, and contracts using the expiry's lot size |
| `oi_change_since_previous_close`, `oi_change_since_previous_close_pct` | Against the previous trading day's 15:25 EOD OI (`eod_oi_available_at` is 09:15 today) |
| `volume_today`, `volume_to_oi` | Cumulative volume over bars available by the snapshot, and that divided by OI |
| `is_call` | 1 for calls, 0 for puts |
| `days_to_expiry`, `time_to_expiry` | Calendar days, and calendar years from `available_at` |
| `available_at` | Of the leg's bar |
| `oi_missing`, `previous_oi_missing` | Missing-input flags; values are never filled |
| `prediction_at`, `lag` | The day's prediction snapshot, and snapshots back from it |

`outputs/reports/tokens_report.json` reports tokens per prediction and per sequence snapshot, by lag, coverage by year, full-sequence days, and field missingness. `docs/findings_data.md` (`scripts/report_data.py`) shows them as figures.

**Token count.** Token counts vary by snapshot (about 70 by default). Sequence models must pad to a common length and use a padding mask; `token_index` gives the order. A narrower moneyness band is a planned ablation, not the default.

## T2 targets (`data/targets/t2/t2.parquet`, stage `targets`)

Targets are stored separately from features. There is one row per T2 snapshot (15:15).

**Realized volatility**
- **Primary source:** front-month futures, one contract per day.
- **Definition:** `rv = sqrt(sum of squared within-day 5-minute log returns)`. Only intraday returns between consecutive bars of the same day count; the overnight gap (previous close to the 09:15 bar) is excluded. RV is null when the day has fewer than `targets.t2.min_bars` (70) bars.
- **Secondary source:** spot RV (`rv_spot_*`).

**Ratio**
- **Definition:** `rv_ratio_trailing = rv_next / rv_trailing_mean`.
- **`rv_next`:** RV on the next trading day.
- **`rv_trailing_mean`:** the mean RV over the 20 trading days ending on the snapshot day, with at least 15 present.

**Labels**
- **`t2_label`:** `low`, `mid`, or `high`.
- **Cutoffs:** `t2_cutoff_low` and `t2_cutoff_high` are the 1/3 and 2/3 quantiles of the expanding history of ratios whose target day is strictly before the snapshot day, so they are known at the snapshot. Labels start once there are at least `cutoff_min_history` (250) values. Data begins in February 2019, so this warmup means **T2 labels start in 2020**; 2019 snapshots carry `t2_reason = insufficient_cutoff_history`.
- **Secondary labels:** `t2_spot_label` comes from the spot-RV ratio. `t2_iv_label` comes from `rv_over_atm_iv` = annualized next-day futures RV / snapshot ATM IV.

**Bookkeeping**
- **`t2_reason`:** `ok`, `no_next_trading_day`, `next_day_rv_missing`, `trailing_rv_insufficient`, or `insufficient_cutoff_history`.
- **`target_available_at`:** when the next day's last bar is available (15:30 IST). It is always later than `snapshot_at`.

## Validation (`scripts/validate_data.py`)

The validation stage writes `outputs/reports/validation_report.json` and `docs/validation_report.md`. It covers:
- parity residuals and forward quality by year;
- call vs put own-leg IV agreement by year and moneyness;
- ATM IV of the longest available expiry vs the same-day India VIX close (validation only);
- gamma sanity;
- the lot-size check;
- bhavcopy reconciliation (skipped when absent);
- timestamp and leakage checks on every output, plus a run of the leakage test modules.

The stage exits non-zero on any leakage violation.

## Greeks and total-GEX selection (`src/niftygex/exposure/selection.py`)

**Greeks input** (`greeks_input()`)
- Only rows with `eligible` and `time_to_expiry > 0` enter Greeks.
- Rows at or after the expiry clock (for example, expiry-day bars labeled 15:25) are excluded.

**Total GEX** (`total_gex_expiries(target)`)
- **Horizon:** an expiry enters total GEX when `0 <= expiry_date - trade_date <= gex.max_days_to_expiry` (21) days.
- **Same-day expiries:** contracts expiring on the snapshot's trade date are governed by `gex.same_day_expiry_in_total_gex` in `configs/data.yaml`:

| Target | Same-day expiry in total GEX |
|---|---|
| T2 (15:15 daily) | excluded: those contracts expire before the forecast day |

## Manifests and reports

- **Manifests.** Each stage writes `outputs/manifests/<stage>.json` (`clean`, `basis`, `forward`, plus `raw_import_manifest.json`). A manifest records:
  - `config_hash` and the profile
  - upstream config hashes
  - inputs with sizes and SHA-256 checksums
  - outputs with row counts
  - stage counts and quality flags
- **Upstream check.** Downstream stages refuse to run when the upstream manifest's `config_hash` differs from the current configuration.
- **Code provenance.** Every manifest and every results-store run spec records `git_commit` (HEAD) and `git_dirty` (uncommitted or untracked changes present).
- **Reports.** Every report is written by a script under `scripts/` to `outputs/reports/<name>.json`:
  - `cleaning_report`
  - `duplicate_conflicts`
  - `bar_completeness`
  - `spot_outliers`
  - `basis_jumps`
  - `spot_label_alignment`
  - `forward_report`
  - `greeks_report`
  - `gex_report`
  - `levels_report`
  - `features_report`
  - `tokens_report`
  - `targets_t1_report`
  - `targets_t2_report`
  - `targets_t3_report`
  - `validation_report`
  - `missing_data`
