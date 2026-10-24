"""Stage A (T2): OLS of the next-day RV ratio on total GEX with an India VIX control and Newey-West errors."""
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import statsmodels.api as sm

LAKH_CRORE = 1e12


def trailing_mean_abs(values: list[float | None], window: int, min_periods: int) -> list[float | None]:
    """Causal trailing mean over the last ``window`` rows (inclusive), ignoring missing values."""
    out: list[float | None] = []
    for i in range(len(values)):
        present = [v for v in values[max(0, i - window + 1): i + 1] if v is not None and np.isfinite(v)]
        out.append(float(np.mean(present)) if len(present) >= min_periods else None)
    return out


def add_regressors(frame: pl.DataFrame, conventions: list[str], window: int, min_periods: int) -> pl.DataFrame:
    """``gex_norm_<c>`` = total GEX / trailing mean absolute total GEX; ``gex_level_<c>`` in lakh crore rupees."""
    frame = frame.sort("snapshot_at")
    scale = trailing_mean_abs(frame["gex_total_absolute"].to_list(), window, min_periods)
    frame = frame.with_columns(pl.Series("gex_scale", scale, dtype=pl.Float64))
    return frame.with_columns(
        *[(pl.col(f"gex_total_{c}") / pl.col("gex_scale")).alias(f"gex_norm_{c}") for c in conventions],
        *[(pl.col(f"gex_total_{c}") / LAKH_CRORE).alias(f"gex_level_{c}") for c in conventions],
    )


def hac_ols(frame: pl.DataFrame, outcome: str, regressors: list[str], maxlags: int) -> dict[str, Any]:
    """y = a + sum_j b_j x_j + e with HAC (Newey-West, Bartlett kernel) standard errors, rows in time order.

    Returns n, R^2 and per-regressor coef/se/t/p plus coef_per_sd (coefficient x the regressor's SD).
    """
    data = frame.sort("snapshot_at").select(outcome, *regressors).drop_nulls()
    data = data.filter(pl.all_horizontal([pl.col(c).cast(pl.Float64).is_finite() for c in data.columns]))
    n = data.height
    if n < 30 + len(regressors):
        return {"n": n, "terms": {}}
    y = data[outcome].cast(pl.Float64).to_numpy()
    x = data.select([pl.col(c).cast(pl.Float64) for c in regressors]).to_numpy()
    if np.any(np.std(x, axis=0) == 0):
        return {"n": n, "terms": {}, "note": "constant regressor in sample"}
    fit = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    terms = {
        name: {"coef": float(fit.params[i + 1]), "se": float(fit.bse[i + 1]), "t": float(fit.tvalues[i + 1]),
               "p": float(fit.pvalues[i + 1]), "coef_per_sd": float(fit.params[i + 1] * np.std(x[:, i], ddof=1))}
        for i, name in enumerate(regressors)
    }
    return {"n": n, "r2": float(fit.rsquared), "outcome_mean": float(y.mean()), "terms": terms}


def add_controls(frame: pl.DataFrame, outcome: str) -> pl.DataFrame:
    """Lagged target: the previous T2 row's ratio, used only when that row's target day is this row's day."""
    frame = frame.sort("snapshot_at")
    return frame.with_columns(
        pl.when(pl.col("target_date").shift(1) == pl.col("trade_date")).then(pl.col(outcome).shift(1)).otherwise(None).alias(f"{outcome}_lagged"),
        pl.col("is_expiry_day").cast(pl.Float64),
    )


def run_stage_a(frame: pl.DataFrame, cfg: dict[str, Any], marker_column: str) -> pl.DataFrame:
    """Every (spec, controls, convention, sample, term) estimate as one row."""
    conventions = list(cfg["conventions"])
    outcome = cfg["outcome"]
    frame = add_regressors(frame, conventions, int(cfg["normalization_window"]), int(cfg["normalization_min_periods"]))
    frame = add_controls(frame, outcome).with_columns(
        *[pl.col(f"gex_total_{c}").sign().alias(f"gex_sign_{c}") for c in cfg["sign_conventions"]]
    )
    marker = marker_column.removeprefix("regime_post_")
    regime_samples = [("full", frame), (f"pre_{marker}", frame.filter(~pl.col(marker_column))), (f"post_{marker}", frame.filter(pl.col(marker_column)))]
    year_samples = [(f"year={y}", part) for (y,), part in sorted(frame.group_by(pl.col("trade_date").dt.year()), key=lambda kv: kv[0])]
    control_sets = {"vix": list(cfg["controls"]), "full_controls": list(cfg["robustness_controls"])}
    specs: list[tuple[str, str, str, list[str], list[tuple[str, pl.DataFrame]]]] = []
    for convention in conventions:
        specs.append(("normalized", "vix", convention, [f"gex_norm_{convention}"], regime_samples + year_samples))
        specs.append(("level_lakh_crore", "vix", convention, [f"gex_level_{convention}"], regime_samples))
        specs.append(("normalized", "full_controls", convention, [f"gex_norm_{convention}"], regime_samples))
    for convention in cfg["sign_conventions"]:
        specs.append(("joint_magnitude_sign", "full_controls", convention, ["gex_norm_absolute", f"gex_sign_{convention}"], regime_samples))
    rows = []
    for spec, controls, convention, terms, samples in specs:
        for sample, part in samples:
            result = hac_ols(part, outcome, [*terms, *control_sets[controls]], int(cfg["hac_maxlags"]))
            for term, values in result["terms"].items():
                rows.append({"spec": spec, "controls": controls, "convention": convention, "sample": sample, "term": term,
                             "is_gex_term": term in terms, "n": result["n"], "r2": result.get("r2"), **values})
    return pl.DataFrame(rows, infer_schema_length=None)
