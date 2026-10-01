"""Pure Black-76 pricing primitives."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def d1_d2(forward, strike, time_to_expiry, volatility):
    sqrt_t = np.sqrt(time_to_expiry)
    d1 = (np.log(forward / strike) + 0.5 * volatility**2 * time_to_expiry) / (volatility * sqrt_t)
    return d1, d1 - volatility * sqrt_t


def price(forward, strike, time_to_expiry, volatility, rate, is_call):
    d1, d2 = d1_d2(forward, strike, time_to_expiry, volatility)
    discount = np.exp(-rate * time_to_expiry)
    call = discount * (forward * norm.cdf(d1) - strike * norm.cdf(d2))
    put = discount * (strike * norm.cdf(-d2) - forward * norm.cdf(-d1))
    return np.where(np.asarray(is_call, dtype=bool), call, put)


def gamma(forward, strike, time_to_expiry, volatility, rate):
    d1, _ = d1_d2(forward, strike, time_to_expiry, volatility)
    return np.exp(-rate * time_to_expiry) * norm.pdf(d1) / (forward * volatility * np.sqrt(time_to_expiry))


def delta(forward, strike, time_to_expiry, volatility, rate, is_call):
    """Forward delta: discounted N(d1) for calls, -discounted N(-d1) for puts."""
    d1, _ = d1_d2(forward, strike, time_to_expiry, volatility)
    discount = np.exp(-rate * time_to_expiry)
    return np.where(np.asarray(is_call, dtype=bool), discount * norm.cdf(d1), -discount * norm.cdf(-d1))
