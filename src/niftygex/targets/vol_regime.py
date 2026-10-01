"""T2 volatility-regime targets: next-day realized volatility vs its trailing level, as causal terciles."""
from __future__ import annotations

from bisect import insort
from datetime import date

import numpy as np

from niftygex.features.realized_vol import daily_realized_vol, trailing_mean  # noqa: F401  (re-exported)

LABELS = ("low", "mid", "high")


def causal_terciles(days: list[date], target_days: list[date | None], values: list[float | None], min_history: int) -> tuple[list, list, list]:
    """Tercile label per row using only values whose target day is strictly before that row's day.

    Rows are snapshot days in order; ``target_days`` gives when each value is known. Returns
    (labels, lower cutoffs, upper cutoffs); labels are None until ``min_history`` values are known.
    """
    order = sorted(range(len(days)), key=lambda i: (target_days[i] or date.max))
    history: list[float] = []
    pointer = 0
    labels, lows, highs = [], [], []
    for i, day in enumerate(days):
        while pointer < len(order) and target_days[order[pointer]] is not None and target_days[order[pointer]] < day:
            value = values[order[pointer]]
            if value is not None and np.isfinite(value):
                insort(history, value)
            pointer += 1
        if len(history) < min_history:
            labels.append(None)
            lows.append(None)
            highs.append(None)
            continue
        low, high = np.quantile(history, [1 / 3, 2 / 3])
        lows.append(float(low))
        highs.append(float(high))
        value = values[i]
        if value is None or not np.isfinite(value):
            labels.append(None)
        else:
            labels.append(LABELS[0] if value <= low else LABELS[1] if value <= high else LABELS[2])
    return labels, lows, highs
