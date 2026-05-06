from __future__ import annotations

import pandas as pd

from fulltext_news_alpha.preprocessing.time_alignment import assign_signal_date


def test_pre_close_news_maps_to_same_trading_day() -> None:
    calendar = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    assert assign_signal_date("2024-01-02 15:59:00", calendar).isoformat() == "2024-01-02"


def test_after_close_news_maps_to_next_trading_day() -> None:
    calendar = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    assert assign_signal_date("2024-01-02 16:01:00", calendar).isoformat() == "2024-01-03"


def test_weekend_news_maps_to_next_trading_day() -> None:
    calendar = pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09"])
    assert assign_signal_date("2024-01-06 12:00:00", calendar).isoformat() == "2024-01-08"


def test_timezone_aware_timestamp_uses_new_york_close() -> None:
    calendar = pd.to_datetime(["2024-01-02", "2024-01-03"])
    assert assign_signal_date("2024-01-02T20:30:00Z", calendar).isoformat() == "2024-01-02"
    assert assign_signal_date("2024-01-02T21:30:00Z", calendar).isoformat() == "2024-01-03"
