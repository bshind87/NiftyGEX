"""Key levels: call/put walls, max pain, max-GEX strike, and the GEX flip level.

Tie policy everywhere: among equal optima, the strike closest to spot wins, then the lower strike.
A level is None when its candidate set is empty.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from niftygex.pricing.black76 import gamma


@dataclass(frozen=True)
class Level:
    value: float | None
    tied: bool = False


def _pick(strikes: np.ndarray, scores: np.ndarray, spot: float, *, maximize: bool) -> Level:
    if strikes.size == 0:
        return Level(None)
    best = scores.max() if maximize else scores.min()
    candidates = strikes[np.isclose(scores, best, rtol=0.0, atol=1e-9 * max(1.0, abs(best)))]
    order = np.lexsort((candidates, np.abs(candidates - spot)))
    return Level(float(candidates[order[0]]), tied=candidates.size > 1)


def call_wall(strikes: np.ndarray, call_oi: np.ndarray, spot: float) -> Level:
    """Strike with the largest call OI strictly above spot."""
    mask = strikes > spot
    return _pick(strikes[mask], call_oi[mask], spot, maximize=True)


def put_wall(strikes: np.ndarray, put_oi: np.ndarray, spot: float) -> Level:
    """Strike with the largest put OI strictly below spot."""
    mask = strikes < spot
    return _pick(strikes[mask], put_oi[mask], spot, maximize=True)


def max_pain(strikes: np.ndarray, call_oi: np.ndarray, put_oi: np.ndarray, spot: float) -> Level:
    """Settlement strike minimizing total intrinsic payout to option holders (one expiry)."""
    if strikes.size == 0:
        return Level(None)
    settle = strikes[:, None]
    payout = (call_oi[None, :] * np.maximum(settle - strikes[None, :], 0.0)).sum(axis=1) + (
        put_oi[None, :] * np.maximum(strikes[None, :] - settle, 0.0)
    ).sum(axis=1)
    return _pick(strikes, payout, spot, maximize=False)


def max_gex_strike(strikes: np.ndarray, gex_absolute: np.ndarray, spot: float) -> Level:
    """Strike with the largest absolute-convention GEX (legs summed per strike beforehand)."""
    return _pick(strikes, gex_absolute, spot, maximize=True)


@dataclass(frozen=True)
class Flip:
    value: float | None
    crossings: int


def flip_level(
    *,
    strike: np.ndarray,
    forward: np.ndarray,
    time_to_expiry: np.ndarray,
    iv: np.ndarray,
    open_interest: np.ndarray,
    sign: np.ndarray,
    spot: float,
    scale: float,
    grid_points: int,
    rate: float = 0.0,
) -> Flip:
    """Index level where total GEX changes sign, recomputed over hypothetical spot levels.

    The grid spans the chain's strike range. Each leg's forward moves proportionally with spot
    (F' = F x S'/S); IV is held per strike (sticky strike) and time to expiry is fixed. Returns the
    linearly interpolated crossing closest to spot and the number of crossings; None when total GEX
    does not change sign inside the strike range.
    """
    valid = np.isfinite(sign) & np.isfinite(iv) & np.isfinite(open_interest)
    if valid.sum() == 0 or not np.isfinite(spot):
        return Flip(None, 0)
    k, f, t, v, oi, s = (array[valid] for array in (strike, forward, time_to_expiry, iv, open_interest, sign))
    levels = np.linspace(k.min(), k.max(), grid_points)
    shifted = f[None, :] * (levels[:, None] / spot)
    total = (s[None, :] * gamma(shifted, k[None, :], t[None, :], v[None, :], rate) * oi[None, :]).sum(axis=1) * levels**2 * scale
    signs = np.sign(total)
    change = np.nonzero(signs[:-1] * signs[1:] < 0)[0]
    if change.size == 0:
        return Flip(None, 0)
    left, right = levels[change], levels[change + 1]
    crossing = left - total[change] * (right - left) / (total[change + 1] - total[change])
    return Flip(float(crossing[np.argmin(np.abs(crossing - spot))]), int(change.size))
