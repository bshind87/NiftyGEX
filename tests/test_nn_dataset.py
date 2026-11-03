from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest

from niftygex_eval.nn.dataset import Scaler, load_token_tensors, signed_log, state_matrix

IST = ZoneInfo("Asia/Kolkata")
P1, P2 = datetime(2024, 3, 4, 15, 15, tzinfo=IST), datetime(2024, 3, 5, 15, 15, tzinfo=IST)


def write_tokens(root, rows: list[tuple]) -> None:
    frame = pl.DataFrame(rows, schema={"prediction_at": pl.Datetime("us", "Asia/Kolkata"), "lag": pl.Int32,
                                       "token_index": pl.Int32, "iv": pl.Float64, "volume_today": pl.Float64}, orient="row")
    frame.write_parquet(root / "year=2024.parquet")


def test_tokens_are_placed_by_day_snapshot_and_slot(tmp_path) -> None:
    write_tokens(tmp_path, [(P1, 0, 0, 0.10, 100.0), (P1, 0, 1, 0.12, None), (P1, 2, 0, 0.20, 5.0), (P1, 5, 0, 0.9, 1.0),
                            (P2, 1, 0, 0.30, 0.0)])
    data = load_token_tensors(tmp_path, pl.Series([P2, P1, P1 + timedelta(days=9)]), window=3, features=["iv", "volume_today"],
                              signed_log_features=["volume_today"])
    assert data.tokens.shape == (3, 3, 2, 2)
    # Day order follows the table: row 1 is P1. Lag 0 sits at index -1; lag 5 is outside the 3-snapshot window.
    assert data.mask[1].tolist() == [[True, False], [False, False], [True, True]]
    assert data.tokens[1, -1, 0, 0] == pytest.approx(0.10) and data.tokens[1, 0, 0, 0] == pytest.approx(0.20)
    assert data.tokens[1, -1, 0, 1] == pytest.approx(np.log1p(100.0)) and np.isnan(data.tokens[1, -1, 1, 1])
    assert data.mask[0].tolist() == [[False, False], [True, False], [False, False]]  # P2's lag-1 token
    assert not data.mask[2].any()  # a day without tokens stays fully masked


def test_signed_log_keeps_sign() -> None:
    assert signed_log(np.array([-9.0, 0.0, 9.0])) == pytest.approx([-np.log(10.0), 0.0, np.log(10.0)])


def test_state_logs_positive_levels_only() -> None:
    table = pl.DataFrame({"atm_iv": [0.2, None, -1.0], "skew": [0.1, 0.2, 0.3]})
    state = state_matrix(table, ["atm_iv", "skew"], ["atm_iv"])
    assert state[0, 0] == pytest.approx(np.log(0.2)) and np.isnan(state[1, 0]) and np.isnan(state[2, 0])
    assert state[:, 1] == pytest.approx([0.1, 0.2, 0.3])


def test_scaler_uses_training_days_and_zeroes_padding_and_missing() -> None:
    tokens = np.array([[[[1.0], [3.0], [99.0]]], [[[5.0], [np.nan], [7.0]]]], dtype=np.float32)  # [D=2, L=1, N=3, F=1]
    mask = np.array([[[True, True, False]], [[True, True, True]]])
    state = np.array([[1.0], [3.0]], dtype=np.float32)
    scaler = Scaler.fit(tokens[:1], mask[:1], state[:1])          # day 0 is the training day
    assert scaler.token_mean == pytest.approx([2.0]) and scaler.token_std == pytest.approx([1.0])
    scaled, scaled_state = scaler.transform(tokens, mask, state)
    assert scaled[0, 0, :, 0] == pytest.approx([-1.0, 1.0, 0.0])  # the padded 99 becomes 0
    assert scaled[1, 0, :, 0] == pytest.approx([3.0, 0.0, 5.0])   # a missing value becomes 0 (the training mean)
    # One training day: zero spread falls back to a scale of 1, so day 1 sits 2.0 above the training mean.
    assert scaled_state[:, 0] == pytest.approx([0.0, 2.0])
