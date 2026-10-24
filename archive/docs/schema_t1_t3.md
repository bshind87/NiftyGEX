# Archived schema: T1 and T3 targets

Removed from `docs/schema.md` on 2026-10-14, when the data layer dropped the T1 and T3 targets (the proposal uses T2 only). The code is in `archive/src/niftygex/targets/` and `archive/src/niftygex/io/official_closes.py`.

## T1 targets (`data/targets/t1/t1.parquet`, stage `targets`)

There is one row per T1 snapshot (10:00, 12:00 and 14:00 on expiry days).

**Settlement source.** The settlement is the official NSE closing index on the expiry date. It comes from an optional source, `data/raw/official_closes/*.csv`, with columns:
- `date`: `YYYY-MM-DD`.
- `close`: index points.

The loader rejects missing columns, non-positive closes, unparseable dates, and conflicting values for the same date. When the source or the date is missing, rows are unavailable (`t1_reason`); an unofficial close is never substituted.

| Column | Definition |
|---|---|
| `strike_step` | Smallest spacing of the expiring contract's listed strikes at the snapshot (default 50) |
| `t1_nearest_strike`, `t1_distance_points` | Nearest strike to the settlement, and the distance to it |
| `t1_distance_fraction` | Distance divided by half a strike step; U(0, 1) under no pinning |
| `t1_pinned` | Distance ≤ `pin_fraction_of_step` (0.1) × step; null rate `t1_null_pin_rate` = 0.2 |
| `t1_distance_to_max_gex_pct`, `t1_distance_to_max_pain_pct` | Signed distance from settlement to the snapshot's levels |
| `t1_pinned_to_max_gex`, `t1_pinned_to_max_pain` | Same pin threshold against those levels |
| `t1_reason` | `ok`, `no_official_close_source`, `no_official_close_for_date`, or `no_max_gex_strike` (grid labels kept, level labels missing) |
| `target_available_at` | Expiry date at `settlement_available_time` (16:00 IST) |

**Independence.** The three snapshots of a day share one settlement, so `targets_t1_report.json` computes grid statistics per expiry day.

## T3 targets (`data/targets/t3/t3.parquet`, stage `targets`)

There is one row per T3 snapshot (14:00).

| Column | Definition |
|---|---|
| `t3_start_price` | The snapshot spot: close of the 13:55-labeled bar, available at 14:00 |
| `t3_end_price` | Close of the 15:25-labeled bar, available at 15:30 = `target_available_at` |
| `t3_strike` | Max-GEX strike (absolute convention) at the T3 snapshot |
| `t3_label` | `abs(end - strike) < abs(start - strike)`, i.e. the index moved toward the strike |
| `t3_distance_reduction_pct` | `(start distance - end distance) / start price * 100` |
| `t3_touched_strike` | The path's low/high spans the strike |
| `t3_path_bars` | Spot bars in the path (labels 14:00 to 15:25) |
| `t3_reason` | `ok`, `truncated_session`, `missing_spot`, `no_max_gex_strike`, `no_snapshot_spot`, `missing_end_bar`, `insufficient_path_bars` (fewer than `targets.t3.min_path_bars`, 16 of 18), or `start_at_strike` |

**Exclusions.** Truncated sessions are excluded from T3 only.

**Null baseline.** The max-GEX strike usually sits very close to spot (median start distance 0.21%), so a random walk often overshoots it. The no-effect null for "moved toward" is therefore not 50%, and T3 evaluation needs an explicit null.

