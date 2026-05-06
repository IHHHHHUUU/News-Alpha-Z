"""Trading calendar helpers for daily stock panels."""

from __future__ import annotations

from datetime import date

import pandas as pd


def calendar_from_prices(prices: pd.DataFrame, date_col: str = "date") -> pd.DatetimeIndex:
    """Infer a sorted trading calendar from observed price rows."""

    if date_col not in prices.columns:
        raise KeyError(f"Missing date column: {date_col}")
    return pd.DatetimeIndex(pd.to_datetime(prices[date_col]).dt.normalize().drop_duplicates().sort_values())


def business_day_calendar(
    start: str | date,
    end: str | date,
    holidays: list[str | date] | None = None,
) -> pd.DatetimeIndex:
    """Create a simple business-day calendar with optional holiday exclusions."""

    days = pd.bdate_range(start=start, end=end)
    if holidays:
        holiday_idx = pd.DatetimeIndex(pd.to_datetime(holidays).normalize())
        days = days.difference(holiday_idx)
    return days


def save_calendar(calendar: pd.DatetimeIndex, path: str) -> None:
    pd.DataFrame({"date": calendar.date}).to_parquet(path, index=False)
