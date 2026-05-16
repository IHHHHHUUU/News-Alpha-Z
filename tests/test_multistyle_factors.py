from __future__ import annotations

import numpy as np
import pandas as pd

from fulltext_news_alpha.features.multistyle_factors import (
    RAW_FACTOR_COLUMNS,
    build_multistyle_factors,
)
from fulltext_news_alpha.training.factor_only_baseline import infer_factor_cols


def _prices(tickers: list[str] | None = None, periods: int = 300) -> pd.DataFrame:
    tickers = tickers or ["AAA", "BBB", "CCC", "DDD"]
    dates = pd.bdate_range("2020-01-01", periods=periods)
    rows = []
    for ticker_idx, ticker in enumerate(tickers):
        base = 20.0 + ticker_idx * 5
        for i, date in enumerate(dates):
            close = base + i * (0.1 + ticker_idx * 0.01)
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": close - 0.1,
                    "high": close + 0.2,
                    "low": close - 0.2,
                    "close": close,
                    "volume": 1_000_000 + ticker_idx * 100_000 + i * 100,
                }
            )
    return pd.DataFrame(rows)


def _fundamentals(tickers: list[str] | None = None) -> pd.DataFrame:
    tickers = tickers or ["AAA", "BBB", "CCC", "DDD"]
    rows = []
    for ticker_idx, ticker in enumerate(tickers):
        rows.append(
            {
                "ticker": ticker,
                "filed_date": "2020-03-15",
                "assets": 1_000 + ticker_idx * 100,
                "book_equity": 500 + ticker_idx * 50,
                "revenue_ttm": 900 + ticker_idx * 40,
                "gross_profit_ttm": 450 + ticker_idx * 20,
                "operating_income_ttm": 120 + ticker_idx * 10,
                "net_income_ttm": 80 + ticker_idx * 8,
                "eps_ttm": 2.0 + ticker_idx * 0.1,
                "cfo_ttm": 100 + ticker_idx * 7,
                "capex_ttm": 20 + ticker_idx,
                "depreciation_ttm": 10 + ticker_idx,
                "interest_expense_ttm": 5 + ticker_idx,
                "short_term_debt": 30 + ticker_idx,
                "long_term_debt": 100 + ticker_idx * 4,
                "cash_and_equivalents": 50 + ticker_idx * 2,
                "dividends_ttm": 8 + ticker_idx,
                "buybacks_ttm": 12 + ticker_idx,
                "shares_outstanding": 10_000_000 + ticker_idx * 100_000,
            }
        )
    return pd.DataFrame(rows)


def _analyst(tickers: list[str] | None = None, periods: int = 120) -> pd.DataFrame:
    tickers = tickers or ["AAA", "BBB", "CCC", "DDD"]
    dates = pd.bdate_range("2020-01-01", periods=periods)
    rows = []
    for ticker_idx, ticker in enumerate(tickers):
        for i, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "fy1_eps_estimate": 2.0 + ticker_idx * 0.1 + i * 0.001,
                    "fy2_eps_estimate": 2.5 + ticker_idx * 0.1 + i * 0.001,
                    "target_price": 30.0 + ticker_idx + i * 0.01,
                    "recommendation_score": 3.0 + ticker_idx * 0.01 + i * 0.001,
                }
            )
    return pd.DataFrame(rows)


def test_price_factors_do_not_use_same_day_ohlcv() -> None:
    prices = _prices(["AAA", "BBB", "CCC"], periods=80)
    target_date = pd.bdate_range("2020-01-01", periods=80)[45].date()
    baseline = build_multistyle_factors(prices, standardize=False)

    shocked = prices.copy()
    mask = (pd.to_datetime(shocked["date"]).dt.date == target_date) & (shocked["ticker"] == "AAA")
    shocked.loc[mask, ["close", "high", "low", "volume"]] = [9999.0, 10000.0, 9998.0, 999_999_999]
    changed = build_multistyle_factors(shocked, standardize=False)

    price_cols = [
        "reversal_1m",
        "momentum_3m",
        "reversal_5d",
        "max_daily_return_1m",
        "volatility_1m",
        "dollar_volume_1m",
        "volume_surge",
    ]
    left = baseline.loc[(baseline["date"] == target_date) & (baseline["ticker"] == "AAA"), price_cols]
    right = changed.loc[(changed["date"] == target_date) & (changed["ticker"] == "AAA"), price_cols]
    assert np.allclose(left.to_numpy(dtype=float), right.to_numpy(dtype=float), equal_nan=True)

    next_date = pd.bdate_range("2020-01-01", periods=80)[46].date()
    assert not np.isclose(
        baseline.loc[(baseline["date"] == next_date) & (baseline["ticker"] == "AAA"), "volume_surge"].iloc[0],
        changed.loc[(changed["date"] == next_date) & (changed["ticker"] == "AAA"), "volume_surge"].iloc[0],
    )


def test_fundamental_factors_are_filing_date_point_in_time() -> None:
    prices = _prices(["AAA", "BBB"], periods=70)
    factors = build_multistyle_factors(prices, fundamentals=_fundamentals(["AAA", "BBB"]), standardize=False)

    before = factors.loc[(factors["ticker"] == "AAA") & (factors["date"] == pd.Timestamp("2020-03-13").date())]
    after = factors.loc[(factors["ticker"] == "AAA") & (factors["date"] == pd.Timestamp("2020-03-16").date())]
    assert before["book_to_price"].isna().all()
    assert after["book_to_price"].notna().all()


def test_multistyle_outputs_50_raw_and_zscore_factors_with_optional_data() -> None:
    factors = build_multistyle_factors(
        _prices(periods=300),
        fundamentals=_fundamentals(),
        analyst=_analyst(periods=300),
        industry_map=pd.DataFrame(
            {"ticker": ["AAA", "BBB", "CCC", "DDD"], "industry": ["Tech", "Tech", "Health", "Health"]}
        ),
    )

    raw_cols = [col for col in RAW_FACTOR_COLUMNS if col in factors.columns]
    zscore_cols = [f"{col}_zscore" for col in RAW_FACTOR_COLUMNS if f"{col}_zscore" in factors.columns]
    assert len(raw_cols) == 50
    assert len(zscore_cols) == 50
    assert factors.attrs["metadata"]["analyst_unavailable"] is False
    assert factors.attrs["metadata"]["missing_fundamentals"] is False


def test_missing_analyst_is_recorded_and_revision_factors_unavailable() -> None:
    factors = build_multistyle_factors(_prices(periods=80), fundamentals=None, analyst=None)
    metadata = factors.attrs["metadata"]
    assert metadata["missing_fundamentals"] is True
    assert metadata["analyst_unavailable"] is True
    for col in [
        "fy1_eps_revision_1m",
        "fy2_eps_revision_3m",
        "target_price_revision_1m",
        "recommendation_change",
    ]:
        assert factors[col].isna().all()


def test_factor_only_baseline_excludes_non_factor_columns_from_multistyle_panel() -> None:
    panel = build_multistyle_factors(_prices(periods=80))
    panel["future_20d_market_adjusted_return"] = 0.01
    panel["future_label_zscore"] = 123.0
    panel["mean_emb_0_zscore"] = 456.0
    panel["has_news"] = 1
    panel["news_count"] = 2
    panel["chunk_count"] = 3

    cols = infer_factor_cols(panel)
    assert "future_label_zscore" not in cols
    assert "mean_emb_0_zscore" not in cols
    assert "has_news" not in cols
    assert "news_count" not in cols
    assert "chunk_count" not in cols
