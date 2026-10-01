import numpy as np

from niftygex.pricing.black76 import gamma, price
from niftygex.pricing.forward import parity_forward, summarize
from niftygex.pricing.implied_vol import invert


def test_black76_call_put_gamma_equal_and_atm_peaks() -> None:
    call_gamma = gamma(100, 100, 30 / 365, 0.2, 0.0)
    put_gamma = gamma(100, 100, 30 / 365, 0.2, 0.0)
    wing_gamma = gamma(100, 130, 30 / 365, 0.2, 0.0)
    assert np.isclose(call_gamma, put_gamma)
    assert call_gamma > wing_gamma


def test_forward_median_and_iv_round_trip() -> None:
    forward = 100.0
    strike = np.array([95.0, 100.0, 105.0])
    t = np.full(3, 30 / 365)
    sigma = np.full(3, 0.2)
    calls = price(forward, strike, t, sigma, 0.0, np.ones(3, dtype=bool))
    puts = price(forward, strike, t, sigma, 0.0, np.zeros(3, dtype=bool))
    implied = parity_forward(strike, calls, puts, t, 0.0)
    result, mask, meta = summarize(implied, strike, 0.1, 2, 1.0)
    assert np.isclose(result, 100.0)
    assert mask.all() and meta['quality'] == 'good'
    recovered, valid = invert(calls, forward, strike, t, 0.0, np.ones(3, dtype=bool))
    assert valid.all() and np.allclose(recovered, sigma, atol=1e-5)


def test_vectorized_forward_matches_reference_summary() -> None:
    import polars as pl

    from niftygex.pricing.forward import estimate_forwards

    strikes = np.array([95.0, 98.0, 100.0, 102.0, 105.0, 140.0])
    parity = np.array([100.4, 99.8, 100.1, 100.0, 99.7, 130.0])
    expected, _, meta = summarize(parity, strikes, 0.06, 2, 1.0)
    pairs = pl.DataFrame({"datetime": [1] * 6, "strike_price": strikes, "parity_forward": parity})
    forwards, near = estimate_forwards(pairs, near_atm_width=0.06, min_pairs=2, max_residual=1.0)
    assert np.isclose(forwards["forward"][0], expected)
    assert forwards["forward_pair_count"][0] == meta["pair_count"]
    assert forwards["forward_quality"][0] == meta["quality"]
    assert 140.0 not in near["strike_price"].to_list()


def test_forward_quality_flags_insufficient_pairs() -> None:
    import polars as pl

    from niftygex.pricing.forward import estimate_forwards

    pairs = pl.DataFrame({"datetime": [1], "strike_price": [100.0], "parity_forward": [100.2]})
    forwards, _ = estimate_forwards(pairs, near_atm_width=0.03, min_pairs=2, max_residual=25.0)
    assert forwards["forward_quality"][0] == "insufficient_pairs"


def test_iv_status_codes() -> None:
    from niftygex.pricing.implied_vol import invert_with_status

    forward = np.full(5, 100.0)
    strike = np.array([100.0, 90.0, 100.0, 100.0, 100.0])
    t = np.array([30 / 365, 30 / 365, 0.0, 30 / 365, 30 / 365])
    good = float(price(100.0, 100.0, 30 / 365, 0.2, 0.0, True))
    observed = np.array([good, 5.0, good, 0.0, 150.0])
    iv, status = invert_with_status(observed, forward, strike, t, 0.0, np.ones(5, dtype=bool))
    assert status.tolist() == ["ok", "below_intrinsic", "nonpositive_time", "nonpositive_price", "above_upper_bound"]
    assert np.isclose(iv[0], 0.2, atol=1e-5) and np.isnan(iv[1:]).all()


def test_otm_leg_iv_is_assigned_to_both_legs() -> None:
    import polars as pl

    from niftygex.pricing.implied_vol import assign_otm_iv

    legs = pl.DataFrame({
        "datetime": [1, 1, 1, 1, 1],
        "strike_price": [95.0, 95.0, 105.0, 105.0, 110.0],
        "right": ["Call", "Put", "Call", "Put", "Put"],
        "forward": [100.0] * 5,
        "volume": [5.0, 7.0, 9.0, 3.0, 4.0],
        "iv_own": [float("nan"), 0.25, 0.22, 0.30, float("nan")],
        "iv_own_status": ["below_intrinsic", "ok", "ok", "ok", "below_intrinsic"],
    })
    result = assign_otm_iv(legs).sort(["strike_price", "right"])
    # K < F: put is OTM; K >= F: call is OTM. The ITM leg keeps its own solve in iv_own.
    assert result["iv"].to_list()[:4] == [0.25, 0.25, 0.22, 0.22]
    assert result["iv_source_right"].to_list()[:4] == ["Put", "Put", "Call", "Call"]
    assert result["iv_source_volume"].to_list()[:4] == [7.0, 7.0, 9.0, 9.0]
    assert result["iv_status"].to_list()[:4] == ["ok", "ok", "ok", "ok"]
    assert result["iv_own_status"].to_list()[0] == "below_intrinsic"
    # A strike whose OTM leg (the 110 call) has no row on the bar gets no IV.
    assert result["iv_status"].to_list()[4] == "otm_leg_missing" and result["iv"].to_list()[4] is None


def test_strike_at_forward_uses_call() -> None:
    import polars as pl

    from niftygex.pricing.implied_vol import assign_otm_iv

    legs = pl.DataFrame({
        "datetime": [1, 1], "strike_price": [100.0, 100.0], "right": ["Call", "Put"], "forward": [100.0, 100.0],
        "volume": [1.0, 1.0], "iv_own": [0.2, 0.21], "iv_own_status": ["ok", "ok"],
    })
    assert assign_otm_iv(legs)["iv"].to_list() == [0.2, 0.2]
