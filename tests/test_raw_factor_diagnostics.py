from __future__ import annotations

import numpy as np
import pandas as pd

from fulltext_news_alpha.diagnostics.raw_factor_diagnostics import (
    compute_all_daily_ic,
    compute_correlations,
    compute_decile_returns,
    compute_stability_flags,
    recommend_keep_drop,
    summarize_by_factor_split,
)


def _toy_frame() -> pd.DataFrame:
    rows = []
    for split_idx, date_value in enumerate(pd.bdate_range("2020-01-01", periods=3)):
        for i in range(20):
            rows.append(
                {
                    "date": date_value.date(),
                    "ticker": f"T{i:02d}",
                    "good_factor": float(i),
                    "bad_factor": float(-i),
                    "future_20d_market_adjusted_return": float(i + split_idx) / 100.0,
                }
            )
    return pd.DataFrame(rows)


def test_daily_ic_rankic_and_positive_ratio_are_positive_for_monotonic_factor() -> None:
    frame = _toy_frame()
    daily = compute_all_daily_ic(
        {"test": frame},
        ["good_factor"],
        "future_20d_market_adjusted_return",
    )
    summary = summarize_by_factor_split(daily, pd.DataFrame())
    row = summary.iloc[0]
    assert row["IC"] > 0.99
    assert row["RankIC"] > 0.99
    assert row["positive_IC_ratio"] == 1.0
    assert row["positive_RankIC_ratio"] == 1.0


def test_decile_returns_top_decile_exceeds_bottom_decile_for_monotonic_factor() -> None:
    deciles, metrics = compute_decile_returns(
        _toy_frame(),
        "good_factor",
        "future_20d_market_adjusted_return",
        split="test",
        n_deciles=10,
    )
    assert not deciles.empty
    assert metrics["top_decile_return"] > metrics["bottom_decile_return"]
    assert metrics["top_bottom_return"] > 0
    assert metrics["decile_monotonic_score"] > 0.99


def test_correlation_matrix_identifies_perfect_negative_pair() -> None:
    matrices, pairs = compute_correlations(
        {"test": _toy_frame()},
        ["good_factor", "bad_factor"],
        corr_threshold=0.8,
    )
    pearson = matrices[("test", "pearson")]
    assert np.isclose(pearson.loc["good_factor", "bad_factor"], -1.0)
    assert not pairs.empty
    pair = pairs.iloc[0]
    assert pair["factor_a"] == "good_factor"
    assert pair["factor_b"] == "bad_factor"
    assert np.isclose(pair["abs_corr"], 1.0)


def test_stability_flags_detect_train_positive_test_negative() -> None:
    summary = pd.DataFrame(
        {
            "factor": ["flip_factor", "flip_factor", "flip_factor"],
            "split": ["train", "valid", "test"],
            "RankIC": [0.03, 0.02, -0.01],
        }
    )
    flags = compute_stability_flags(summary, ["flip_factor"])
    row = flags.iloc[0]
    assert row["unstable_flag"]
    assert not row["direction_consistent"]
    assert not row["weak_flag"]


def test_keep_drop_heuristic_keeps_one_of_opposite_correlated_pair() -> None:
    factor_cols = ["good_factor", "bad_factor"]
    summary = pd.DataFrame(
        {
            "factor": [
                "good_factor",
                "good_factor",
                "good_factor",
                "bad_factor",
                "bad_factor",
                "bad_factor",
            ],
            "split": ["train", "valid", "test", "train", "valid", "test"],
            "RankIC": [0.02, 0.02, 0.03, -0.02, -0.02, -0.01],
            "RankIC_std": [0.1, 0.1, 0.1, 0.1, 0.1, 0.2],
            "positive_RankIC_ratio": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        }
    )
    flags = compute_stability_flags(summary, factor_cols)
    pairs = pd.DataFrame(
        {
            "split": ["test"],
            "factor_a": ["good_factor"],
            "factor_b": ["bad_factor"],
            "corr": [-1.0],
            "abs_corr": [1.0],
            "method": ["spearman"],
        }
    )
    rec = recommend_keep_drop(factor_cols, summary, flags, pairs)
    assert set(rec["keep_or_drop"]) == {"keep", "drop"}
    kept = rec.loc[rec["keep_or_drop"] == "keep", "factor"].tolist()
    assert kept == ["good_factor"]
