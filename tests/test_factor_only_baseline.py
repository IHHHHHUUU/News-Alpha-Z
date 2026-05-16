from __future__ import annotations

import numpy as np
import pandas as pd

from fulltext_news_alpha.evaluation.ic_metrics import compute_ic_by_date, summarize_ic
from fulltext_news_alpha.training.factor_only_baseline import (
    RidgeConfig,
    infer_factor_cols,
    train_factor_only_baseline,
)


def _toy_panel() -> pd.DataFrame:
    dates = [
        "2018-01-02",
        "2019-01-02",
        "2020-01-02",
        "2021-01-04",
        "2022-01-03",
        "2023-01-03",
    ]
    rows = []
    for date in dates:
        for rank, ticker in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE"], start=1):
            signal = float(rank)
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "good_zscore": signal,
                    "weak_zscore": float(6 - rank) * 0.1,
                    "future_5d_return_zscore": signal * 100.0,
                    "mean_emb_0_zscore": signal * -100.0,
                    "has_news": 1,
                    "news_count": 3,
                    "chunk_count": 7,
                    "future_20d_market_adjusted_return": signal / 100.0,
                }
            )
    return pd.DataFrame(rows)


def test_infer_factor_cols_excludes_labels_embeddings_and_news_controls() -> None:
    cols = infer_factor_cols(_toy_panel())
    assert cols == ["good_zscore", "weak_zscore"]
    assert not any(col.startswith("future_") for col in cols)
    assert not any(col.startswith("mean_emb_") for col in cols)
    assert "has_news" not in cols
    assert "news_count" not in cols
    assert "chunk_count" not in cols


def test_ridge_baseline_outputs_non_empty_split_predictions(tmp_path) -> None:
    result = train_factor_only_baseline(
        _toy_panel(),
        output_dir=tmp_path,
        config=RidgeConfig(alpha=0.1),
    )

    predictions = result["predictions"]
    assert set(predictions) == {"train", "valid", "test"}
    for split_name in ["train", "valid", "test"]:
        pred = predictions[split_name]
        assert not pred.empty
        assert {"date", "ticker", "factor_only_pred", "future_20d_market_adjusted_return"} <= set(
            pred.columns
        )
        assert (tmp_path / f"{split_name}.parquet").exists()

    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "single_factor_ic_test.csv").exists()
    assert (tmp_path / "single_factor_rankic_top30_test.csv").exists()


def test_ridge_baseline_rankic_is_positive_for_monotonic_factor() -> None:
    result = train_factor_only_baseline(
        _toy_panel(),
        config=RidgeConfig(alpha=0.1),
    )
    test_pred = result["predictions"]["test"]
    ic = compute_ic_by_date(
        test_pred,
        factor_col="factor_only_pred",
        return_col="future_20d_market_adjusted_return",
    )
    summary = summarize_ic(ic)
    assert np.isfinite(summary["RankIC"])
    assert summary["RankIC"] > 0

    rankic_top30 = result["single_factor_rankic_top30_test"]
    assert rankic_top30.iloc[0]["factor"] == "good_zscore"
    assert rankic_top30.iloc[0]["mean"] > 0
