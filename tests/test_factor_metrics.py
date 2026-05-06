from __future__ import annotations

import numpy as np
import pandas as pd

from fulltext_news_alpha.evaluation.ic_metrics import compute_ic_by_date, summarize_ic
from fulltext_news_alpha.evaluation.portfolio_backtest import long_short_returns
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


def test_long_short_returns_are_positive_for_monotonic_factor() -> None:
    ls = long_short_returns(_factor_frame())
    assert (ls["long_short_return"] > 0).all()
    assert "cumulative_return" in ls.columns
