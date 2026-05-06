"""No-lookahead news-to-signal-date alignment."""

from __future__ import annotations

from datetime import date, time
from zoneinfo import ZoneInfo

import pandas as pd


NY_TZ = ZoneInfo("America/New_York")


def normalize_trading_days(trading_days: list[date] | pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return sorted midnight timestamps for valid trading days."""

    days = pd.to_datetime(list(trading_days)).normalize().drop_duplicates().sort_values()
    if len(days) == 0:
        raise ValueError("trading_days must contain at least one date")
    return pd.DatetimeIndex(days)


def to_new_york_timestamp(value: object, assume_tz: ZoneInfo = NY_TZ) -> pd.Timestamp:
    """Parse a timestamp and express it in America/New_York.

    Naive timestamps are treated as New York local time because most vendor news feeds
    report exchange-local timestamps unless an offset is present.
    """

    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(assume_tz)
    return ts.tz_convert(NY_TZ)


def next_trading_day(anchor: date | pd.Timestamp, trading_days: pd.DatetimeIndex) -> pd.Timestamp:
    """Find the first trading day strictly after ``anchor``."""

    days = normalize_trading_days(trading_days)
    anchor_ts = pd.Timestamp(anchor).normalize()
    idx = days.searchsorted(anchor_ts, side="right")
    if idx >= len(days):
        raise ValueError(f"No trading day available after {anchor_ts.date()}")
    return days[idx]


def current_or_next_trading_day(anchor: date | pd.Timestamp, trading_days: pd.DatetimeIndex) -> pd.Timestamp:
    """Find the first trading day on or after ``anchor``."""

    days = normalize_trading_days(trading_days)
    anchor_ts = pd.Timestamp(anchor).normalize()
    idx = days.searchsorted(anchor_ts, side="left")
    if idx >= len(days):
        raise ValueError(f"No trading day available on or after {anchor_ts.date()}")
    return days[idx]


def assign_signal_date(
    publish_time: object,
    trading_days: list[date] | pd.Series | pd.DatetimeIndex,
    market_close: time = time(16, 0),
) -> date:
    """Map a news publish timestamp to the first tradable signal date.

    Rules:
    - news before the regular New York close on a trading day can enter that day's signal;
    - after-close news moves to the next trading day;
    - weekend/holiday news moves to the next trading day.
    """

    days = normalize_trading_days(trading_days)
    ts = to_new_york_timestamp(publish_time)
    publish_date = pd.Timestamp(ts.date())

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
