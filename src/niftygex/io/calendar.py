"""Trading calendar built from observed option/futures dates and configured holidays."""
from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable
from datetime import date, timedelta

import polars as pl


def build_trading_calendar(
    observed_dates: Iterable[date],
    holidays: Iterable[date],
    extend_weekdays: int,
) -> pl.DataFrame:
    """Return one row per trading date, with the weekday-rule extension after the last observed date.

    Observed dates are the union of option and futures trading dates. Configured holidays are removed
    from the observed set and skipped by the extension.
    """
    holiday_set = set(holidays)
    observed = sorted(set(observed_dates) - holiday_set)
    if not observed:
        raise ValueError("Cannot build a trading calendar without observed dates")
    extension: list[date] = []
    current = observed[-1]
    while len(extension) < extend_weekdays:
        current += timedelta(days=1)
        if current.weekday() < 5 and current not in holiday_set:
            extension.append(current)
    dates = observed + extension
    return pl.DataFrame(
        {
            "trading_date": dates,
            "observed": [True] * len(observed) + [False] * len(extension),
            "next_trading_date": dates[1:] + [None],
        },
        schema={"trading_date": pl.Date, "observed": pl.Boolean, "next_trading_date": pl.Date},
    )


def next_trading_dates(observation_dates: Iterable[date], trading_dates: list[date]) -> dict[date, date]:
    """Map each observation date to the first trading date strictly after it.

    Observation dates need not be trading dates themselves. Raises when no later trading date exists,
    so a missing availability date can never silently fall back to the observation date.
    """
    ordered = sorted(trading_dates)
    mapping: dict[date, date] = {}
    for value in set(observation_dates):
        index = bisect_right(ordered, value)
        if index >= len(ordered):
            raise ValueError(f"No trading date after {value}; extend the calendar")
        mapping[value] = ordered[index]
    return mapping
