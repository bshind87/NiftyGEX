# Provenance

The data-pipeline code builds on pre-existing code from an earlier personal project co-developed by one of the authors, adapted for this study.

The files below were adapted from that project; legacy paths are relative to its root. Every other file in this repository is a new implementation written for NiftyGEX, including the evaluation layer (`src/niftygex_eval/`) and all tests.

| New path | Legacy path | What changed |
|---|---|---|
| `src/niftygex/config.py` | `src/nofe/config.py` | Reworked configuration loading for profile/env resolution, repository-relative paths, legacy-source and report roots, and configuration hashes. |
| `src/niftygex/clean/session.py` | `src/nofe/io/calendar.py` and `src/nofe/io/loaders.py` | Adapted timestamp/session concepts; added IST normalization, start-label session bounds, session-removal counts, and configured session date lists. |
| `src/niftygex/clean/chain.py` | `src/nofe/io/loaders.py` and `src/nofe/io/schema.py` | Reworked option normalization and cleaning; added source-row provenance and higher-volume deterministic deduplication with conflict diagnostics. |
| `src/niftygex/clean/futures.py` | `src/nofe/features/market.py` | Adapted front-month selection; rewritten to choose one contract per trading day by the configured roll rule, with no intraday fallback. |
| `src/niftygex/clean/market.py` | `src/nofe/io/loaders.py` | Adapted spot/VIX loading; added deterministic dedup, daily-VIX observation dates, and configured end-label relabeling. |
| `src/niftygex/clean/oi.py` | `src/nofe/io/loaders.py` | Reworked participant-OI loading around strict next-trading-day availability at 09:15 IST, excluded non-regular sessions, and conflict-dropping dedup; added the 15:25 EOD OI proxy. |
| `scripts/clean_chain.py` | `scripts/build_panel.py` and `scripts/build_dataset.py` | Replaced legacy model-panel orchestration with an independent curated-source cleaning stage: trading calendar, day-level quality, staged outputs, reports, and manifest. |
| `src/niftygex/pricing/black76.py` | `src/nofe/greeks/black76.py` | Adapted pure Black-76 price/gamma formulas into the new package API. |
| `src/niftygex/pricing/forward.py` | `src/nofe/greeks/pricing.py` | Replaced single-pair forward selection with a median near-ATM parity forward (reference and vectorized versions), per-pair residuals, and quality flags. |
| `src/niftygex/pricing/implied_vol.py` | `src/nofe/greeks/pricing.py` | Adapted close-price IV inversion; replaced Newton-only solving with bracketed vectorized bisection, no-arbitrage bound checks, and explicit status codes. |
| `scripts/compute_forward.py` | `scripts/build_greeks.py` and `src/nofe/greeks/engine.py` | New per-expiry Parquet stage: forwards, parity residuals, and IV over the full configured moneyness band, with time to expiry from `available_at`, upstream config-hash check, report, and manifest. |
| `scripts/import_raw.py` | Legacy `data/` layout | Copy-only importer based on the legacy source layout; paths from config; skips uncompressed duplicates; refuses to overwrite differing raw files; checksummed manifest with config hash. |
