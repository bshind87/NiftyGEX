"""Where every model input comes from: source dataset and availability rule per feature and per token field.

Documentation metadata for reports and notebooks; tests check that every built feature has an entry.
"""
from __future__ import annotations

# raw_import_manifest source key -> (name, granularity, used for)
SOURCES = {
    "options": ("NIFTY index options", "5-minute bars", "IV, Greeks, GEX, levels, tokens"),
    "futures": ("NIFTY futures", "5-minute bars", "Front-month price, basis, realized-volatility target"),
    "equity": ("NIFTY 50 index (spot)", "5-minute bars", "Spot for GEX scaling and moneyness"),
    "india_vix": ("India VIX", "daily and 5-minute", "Previous-day VIX feature; validation of ATM IV"),
    "fiidii": ("NSE participant-wise open interest", "daily, after the close", "Participant positions; inferred dealer sign"),
}

# Feature lineage: feature -> (group, source dataset, when the value is available at the 15:15 snapshot).
OPT, FUT, SPOT, VIX, PART, CAL = ("NIFTY option chain", "Front-month futures", "NIFTY 50 index", "India VIX (daily)",
                                  "NSE participant OI (daily)", "Regime calendar (configs/regimes.yaml)")
AS_OF = "bars available by the snapshot (15-minute staleness)"
NEXT_DAY = "previous trading day's value, available 09:15 today"
LINEAGE = {
    "spot": ("Market state", SPOT, AS_OF),
    **{c: ("Gamma exposure", f"{OPT} + {SPOT}" + (f" + {PART}" if c == "gex_total_inferred" else ""), AS_OF)
       for c in ("gex_total_naive", "gex_total_absolute", "gex_total_inferred", "gex_concentration_hhi")},
    "gex_expiries": ("Gamma exposure", OPT, AS_OF),
    "includes_same_day_expiry": ("Gamma exposure", OPT, AS_OF),
    **{c: ("Key levels", OPT, AS_OF) for c in ("call_wall_distance_pct", "put_wall_distance_pct", "max_pain_distance_pct",
                                              "max_gex_strike_distance_pct", "flip_naive_distance_pct", "has_flip")},
    "has_inferred_flip": ("Key levels", f"{OPT} + {PART}", AS_OF),
    **{c: ("Volatility surface", OPT, AS_OF) for c in ("atm_iv", "atm_iv_next", "skew_25d", "term_slope")},
    **{c: ("Calendar", OPT, "known in advance (listed expiries)") for c in ("days_to_expiry", "time_to_expiry", "days_to_next_expiry",
                                                                           "is_expiry_day", "nearest_expiry_weekday")},
    "put_call_oi_ratio": ("Positioning flow", OPT, AS_OF),
    "oi_change_near_money_calls": ("Positioning flow", OPT, "snapshot OI vs previous day's 15:25 OI (available 09:15)"),
    "oi_change_near_money_puts": ("Positioning flow", OPT, "snapshot OI vs previous day's 15:25 OI (available 09:15)"),
    "futures_basis_bps": ("Market state", f"{FUT} − {SPOT}", AS_OF),
    "vix_previous_close": ("Market state", VIX, NEXT_DAY),
    "rv_intraday_so_far_annualized": ("Volatility state", FUT, "today's bars available by the snapshot"),
    **{f"rv_trailing_{d}d_annualized_prev": ("Volatility state", FUT, "completed days up to yesterday's close") for d in (1, 5, 20)},
    **{f"participant_{p}_net_index_{r}": ("Participant positions", PART, NEXT_DAY) for p in ("pro", "fii", "client") for r in ("calls", "puts")},
    "lot_size": ("Calendar", CAL, "known in advance"),
    "regime_post_weekly_expiry_rationalisation": ("Calendar", CAL, "known in advance"),
}
TOKEN_SOURCES = [
    ("log_moneyness", "ln(strike / forward); forward from put–call parity on the option chain"),
    ("is_call, strike_price, expiry_date", "option chain contract keys"),
    ("iv, gamma, delta", "Black-76 on the option chain (IV from the out-of-the-money leg)"),
    ("strike_gex_absolute_crore, strike_gex_naive_crore", "gamma × OI × spot² × 0.01 (option chain + NIFTY 50 index)"),
    ("open_interest, open_interest_contracts", "option chain OI (index units; contracts via the expiry's lot size)"),
    ("oi_change_since_previous_close (and %)", "against the previous day's 15:25 OI (available 09:15)"),
    ("volume_today, volume_to_oi", "cumulative option volume over bars available by the snapshot"),
    ("days_to_expiry, time_to_expiry", "contract expiry; time measured to 15:30 on expiry day"),
    ("oi_missing, previous_oi_missing", "flags; missing values are never filled"),
    ("prediction_at, lag", "the day's 15:15 prediction snapshot, and 5-minute steps back from it"),
]
