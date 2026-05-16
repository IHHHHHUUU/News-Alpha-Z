from __future__ import annotations

import numpy as np
import pandas as pd

from fulltext_news_alpha.evaluation.ic_metrics import (
    compute_daily_ic_diagnostics,
    compute_ic_by_date,
    summarize_ic,
)
from fulltext_news_alpha.evaluation.plots import compute_factor_coverage
from fulltext_news_alpha.evaluation.portfolio_backtest import filter_rebalance_dates, long_short_returns
from fulltext_news_alpha.factors.factor_standardization import standardize_by_date


def _factor_frame() -> pd.DataFrame:
    rows = []
    for day in ["2024-01-02", "2024-01-03"]:
        for i in range(10):
            rows.append(
                {
                    "date": day,
                    "ticker": f"T{i}",
                    "FullTextNewsAlpha_zscore": float(i),
                    "future_20d_market_adjusted_return": float(i) / 100,
                }
            )
    return pd.DataFrame(rows)


def test_standardize_by_date_outputs_zero_mean_unit_scale() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02"] * 5,
            "ticker": list("ABCDE"),
            "raw": [1.0, 2.0, 3.0, 4.0, 100.0],
        }
    )
    out = standardize_by_date(frame, "raw", output_col="raw_z")
    assert np.isclose(out["raw_z"].mean(), 0.0)
    assert np.isclose(out["raw_z"].std(ddof=0), 1.0)


def test_ic_rankic_and_summary_are_computed() -> None:
    ic = compute_ic_by_date(_factor_frame())
    summary = summarize_ic(ic)
    assert ic["RankIC"].min() > 0.99
    assert summary["RankIC"] > 0.99
    assert summary["RankIC_count"] == 2
    assert summary["RankIC_nan_count"] == 0
    assert summary["RankIC_zero_count"] == 0


def test_factor_coverage_counts_valid_factor_and_return_rows() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02"] * 10 + ["2024-01-03"] * 5,
            "ticker": [f"T{i}" for i in range(10)] + [f"U{i}" for i in range(5)],
            "FullTextNewsAlpha_zscore": [1.0] * 6 + [np.nan] * 4 + [1.0] * 5,
            "future_20d_market_adjusted_return": [1.0] * 8 + [np.nan] * 2 + [1.0] * 5,
        }
    )
    coverage = compute_factor_coverage(frame)
    by_date = dict(zip(coverage["date"], coverage["coverage"], strict=True))
    assert np.isclose(by_date["2024-01-02"], 0.6)
    assert np.isclose(by_date["2024-01-03"], 1.0)


def test_daily_rankic_diagnostics_capture_degenerate_dates() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02"] * 5 + ["2024-01-03"] * 5 + ["2024-01-04"] * 5,
            "ticker": [f"A{i}" for i in range(5)] + [f"B{i}" for i in range(5)] + [f"C{i}" for i in range(5)],
            "FullTextNewsAlpha_zscore": [0, 1, 2, 3, 4] + [1, 1, 1, 1, 1] + [0, 1, 2, 3, 4],
            "future_20d_market_adjusted_return": [0, 1, 2, 3, 4] + [0, 1, 2, 3, 4] + [0, 1, np.nan, np.nan, np.nan],
        }
    )
    diagnostics = compute_daily_ic_diagnostics(frame)
    by_date = {str(row["date"]): row for row in diagnostics.to_dict(orient="records")}
    assert np.isclose(by_date["2024-01-02"]["RankIC"], 1.0)
    assert np.isnan(by_date["2024-01-03"]["RankIC"])
    assert by_date["2024-01-03"]["factor_nunique"] == 1
    assert by_date["2024-01-04"]["valid_count"] == 2
    assert by_date["2024-01-04"]["return_nunique"] == 2


def test_long_short_returns_are_positive_for_monotonic_factor() -> None:
    ls = long_short_returns(_factor_frame())
    assert (ls["long_short_return"] > 0).all()
    assert "cumulative_return" in ls.columns


def test_rebalance_every_filters_to_non_overlapping_dates() -> None:
    frame = pd.concat(
        [
            _factor_frame().assign(date=(pd.Timestamp("2024-01-02") + pd.offsets.BDay(i)).date())
            for i in range(40)
        ],
        ignore_index=True,
    )
    filtered = filter_rebalance_dates(frame, rebalance_every=20)
    assert list(pd.to_datetime(filtered["date"]).drop_duplicates()) == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-30"),
    ]

    ic = compute_ic_by_date(frame, rebalance_every=20)
    ls = long_short_returns(frame, rebalance_every=20)
    assert len(ic) == 2
    assert len(ls) == 2
