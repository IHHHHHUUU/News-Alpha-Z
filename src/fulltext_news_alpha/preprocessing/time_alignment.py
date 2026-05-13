"""No-lookahead news-to-signal-date alignment."""

from __future__ import annotations

from datetime import date, datetime, time
from typing import Any, cast
from zoneinfo import ZoneInfo

import pandas as pd


NY_TZ = ZoneInfo("America/New_York")


def _to_timestamp(value: Any) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.Timestamp(value))


def _timestamp_at_midnight(ts: pd.Timestamp) -> bool:
    return bool(ts.hour == 0 and ts.minute == 0 and ts.second == 0 and ts.microsecond == 0 and ts.nanosecond == 0)


def imprecise_publish_date(publish_time: object) -> pd.Timestamp | None:
    """Return the source calendar date when the payload lacks reliable intraday time."""

    if isinstance(publish_time, date) and not isinstance(publish_time, datetime):
        return _to_timestamp(publish_time)

    ts = _to_timestamp(publish_time)
    if _timestamp_at_midnight(ts):
        return _to_timestamp(ts.date())
    return None


def normalize_trading_days(trading_days: list[date] | pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return sorted midnight timestamps for valid trading days."""

    normalized = sorted({_to_timestamp(day).date() for day in list(trading_days)})
    days = pd.DatetimeIndex(normalized)
    if len(days) == 0:
        raise ValueError("trading_days must contain at least one date")
    return days


def to_new_york_timestamp(value: object, assume_tz: ZoneInfo = NY_TZ) -> pd.Timestamp:
    """Parse a timestamp and express it in America/New_York.

    Naive timestamps are treated as New York local time because most vendor news feeds
    report exchange-local timestamps unless an offset is present.
    """

    ts = _to_timestamp(value)
    if ts.tzinfo is None:
        ts = cast(pd.Timestamp, ts.tz_localize(assume_tz))
    return cast(pd.Timestamp, ts.tz_convert(NY_TZ))


def next_trading_day(anchor: date | pd.Timestamp, trading_days: pd.DatetimeIndex) -> pd.Timestamp:
    """Find the first trading day strictly after ``anchor``."""

    days = normalize_trading_days(trading_days)
    anchor_date = _to_timestamp(anchor).date()
    for day_value in days:
        day = _to_timestamp(day_value)
        if day.date() > anchor_date:
            return day
    raise ValueError(f"No trading day available after {anchor_date}")


def current_or_next_trading_day(anchor: date | pd.Timestamp, trading_days: pd.DatetimeIndex) -> pd.Timestamp:
    """Find the first trading day on or after ``anchor``."""

    days = normalize_trading_days(trading_days)
    anchor_date = _to_timestamp(anchor).date()
    for day_value in days:
        day = _to_timestamp(day_value)
        if day.date() >= anchor_date:
            return day
    raise ValueError(f"No trading day available on or after {anchor_date}")


def assign_signal_date(
    publish_time: object,
    trading_days: list[date] | pd.Series | pd.DatetimeIndex,
    market_close: time = time(16, 0),
) -> date:
    """Map a news publish timestamp to the first tradable signal date.

    Rules:
    - date-only and midnight payloads lack reliable intraday precision and map to
      the first trading day strictly after the source calendar date;
    - news before the regular New York close on a trading day can enter that day's signal;
    - after-close news moves to the next trading day;
    - weekend/holiday news moves to the next trading day.
    """

    days = normalize_trading_days(trading_days)
    imprecise_date = imprecise_publish_date(publish_time)
    if imprecise_date is not None:
        return next_trading_day(imprecise_date, days).date()

    ts = to_new_york_timestamp(publish_time)
    publish_date = _to_timestamp(ts.date())
    if _timestamp_at_midnight(ts):
        return next_trading_day(publish_date, days).date()

    if publish_date in days and ts.time() < market_close:
        return publish_date.date()
    if publish_date in days:
        return next_trading_day(publish_date, days).date()
    return current_or_next_trading_day(publish_date, days).date()


def add_signal_date(
    news: pd.DataFrame,
    trading_days: list[date] | pd.Series | pd.DatetimeIndex,
    publish_col: str = "publish_time",
    output_col: str = "date",
) -> pd.DataFrame:
    """Add a no-lookahead signal date column to a news DataFrame."""

    out = news.copy()
    out[output_col] = [assign_signal_date(ts, trading_days) for ts in out[publish_col]]
    return out
