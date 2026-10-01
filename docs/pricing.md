# Pricing conventions

**Timing**
- **Bars:** bars are start-labeled and kept for labels 09:15 through 15:25 IST. A bar's close is observed at `available_at = label + 5 min`.
- **Time to expiry:** calendar years (365 days per year) from `available_at` to 15:30 IST on the expiry date. Both the expiry clock and the days-per-year value are set in `configs/data.yaml` under `pricing`.

**Rate:** the risk-free rate is the configured constant `risk_free_rate.default` in `configs/regimes.yaml`. A by-date series is reserved for later.

**Forward**
- **Pair eligibility:** a forward pair needs positive call and put volume on the same bar and strike.
- **Estimate:** the median parity-implied forward over the configured near-ATM band.
- **Diagnostics:** per-pair residuals (`data/greeks/parity/`) and a per-snapshot quality flag (`data/greeks/forward/`) are stored.

**Implied volatility**
- **Inputs:** IV is inverted from bar close, because there is no bid/ask, for every leg with `strike / forward` in the configured liquidity band (0.92 to 1.08).
- **Bounds:** no-arbitrage bounds are checked before inversion. Each failure gets an explicit `iv_status`.
- **Eligibility:** `eligible` requires `volume > 0`, a converged IV, and a `good` forward.
