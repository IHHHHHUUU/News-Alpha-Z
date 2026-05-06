from __future__ import annotations

import numpy as np
import pandas as pd

from fulltext_news_alpha.features.price_volume_factors import build_price_volume_factors
from fulltext_news_alpha.features.return_labels import add_forward_return_labels
from fulltext_news_alpha.preprocessing.time_alignment import assign_signal_date


def test_after_close_news_cannot_enter_same_day_signal() -> None:
    calendar = pd.to_datetime(["2024-01-02", "2024-01-03"])
    assert assign_signal_date("2024-01-02 16:00:01", calendar).isoformat() == "2024-01-03"


def test_price_factor_uses_prior_close_only_for_momentum() -> None:
    dates = pd.bdate_range("2024-01-01", periods=70)
    close = np.arange(100, 170, dtype=float)
    prices = pd.DataFrame(
        {
            "date": dates,
            "ticker": "AAPL",
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.arange(1_000, 1_070),
        }
    )
    factors = build_price_volume_factors(prices, standardize=False)
    check_idx = 10
    date = dates[check_idx].date()
    value = factors.loc[factors["date"] == date, "momentum_5d"].iloc[0]
    expected = close[check_idx - 1] / close[check_idx - 6] - 1.0
    leaked = close[check_idx] / close[check_idx - 5] - 1.0
    assert np.isclose(value, expected)
    assert not np.isclose(value, leaked)


def test_labels_are_forward_returns_and_not_feature_columns() -> None:
    dates = pd.bdate_range("2024-01-01", periods=30)
    prices = pd.DataFrame(
        {
            "date": dates,
            "ticker": "AAPL",
            "open": np.arange(30, dtype=float) + 99,
            "high": np.arange(30, dtype=float) + 101,
            "low": np.arange(30, dtype=float) + 98,
            "close": np.arange(30, dtype=float) + 100,
            "volume": 1_000,
        }
    )
    labels = add_forward_return_labels(prices, horizons=(5,))
    first = labels.loc[0, "future_5d_return"]
    assert np.isclose(first, prices.loc[5, "close"] / prices.loc[0, "close"] - 1.0)
    factors = build_price_volume_factors(prices, standardize=False)
    assert not any(col.startswith("future_") for col in factors.columns)
