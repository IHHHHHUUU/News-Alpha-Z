from __future__ import annotations

import numpy as np
import pandas as pd

from fulltext_news_alpha.diagnostics.six_factor_alpha_baseline import (
    ALPHA_COLS,
    apply_ridge_score,
    compute_equal_weight_score,
    compute_ic_weights,
    construct_alpha_panel,
    fit_ridge,
)
from fulltext_news_alpha.diagnostics.raw_factor_diagnostics import compute_correlations


def _base_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for date_idx, date_value in enumerate(pd.bdate_range("2020-01-01", periods=4)):
        for i in range(20):
            signal = float(i)
            rows.append(
                {
                    "date": date_value.date(),
                    "ticker": f"T{i:02d}",
                    "RSI_zscore": -signal,
                    "beta_60d_zscore": signal,
                    "max_drawdown_60d_zscore": -signal,
                    "momentum_20d_zscore": -signal,
                    "reversal_5d_zscore": signal,
                    "volatility_20d_zscore": signal,
                    "future_20d_market_adjusted_return": signal + date_idx,
                }
            )
    return rows


def _panel() -> pd.DataFrame:
    return pd.DataFrame(_base_rows())


def test_direction_conversion_and_no_restandardization() -> None:
    panel = _panel()
    alpha = construct_alpha_panel(panel)
    assert np.allclose(alpha["RSI_alpha"], -panel["RSI_zscore"])
    assert np.allclose(alpha["momentum_20d_alpha"], -panel["momentum_20d_zscore"])
    assert np.allclose(alpha["reversal_5d_alpha"], panel["reversal_5d_zscore"])
    assert np.allclose(alpha["beta_60d_alpha"], panel["beta_60d_zscore"])


def test_equal_weight_score_is_mean_of_available_alpha_values() -> None:
    alpha = construct_alpha_panel(_panel())
    score = compute_equal_weight_score(alpha)
    expected = alpha[ALPHA_COLS].mean(axis=1)
    assert np.allclose(score, expected)


def test_ic_weights_use_train_rankic_and_zero_nonpositive_factor() -> None:
    panel = _panel()
    # Make one alpha direction fail on train while other factors remain positive.
    panel["volatility_20d_zscore"] = -panel["volatility_20d_zscore"]
    alpha = construct_alpha_panel(panel)
    weights, warnings = compute_ic_weights(alpha)
    vol = weights[weights["factor"] == "volatility_20d_alpha"].iloc[0]
    beta = weights[weights["factor"] == "beta_60d_alpha"].iloc[0]
    assert vol["train_RankIC"] <= 0
    assert vol["normalized_weight"] == 0.0
    assert not bool(vol["used"])
    assert beta["normalized_weight"] > 0
    assert any("volatility_20d_alpha" in warning for warning in warnings)


def test_ridge_uses_label_zscore_and_outputs_prediction() -> None:
    alpha = construct_alpha_panel(_panel())
    coef, intercept = fit_ridge(alpha, ridge_alpha=1.0)
    pred = apply_ridge_score(alpha, coef, intercept)
    assert coef.shape == (6,)
    assert len(pred) == len(alpha)
    assert pred.notna().all()
    assert abs(float(alpha["label_zscore"].mean())) < 1e-12


def test_correlation_detection_identifies_high_corr_pair() -> None:
    alpha = construct_alpha_panel(_panel())
    matrices, pairs = compute_correlations(
        {"train": alpha},
        ["beta_60d_alpha", "reversal_5d_alpha"],
        corr_threshold=0.8,
    )
    assert np.isclose(matrices[("train", "pearson")].loc["beta_60d_alpha", "reversal_5d_alpha"], 1.0)
    assert not pairs.empty
    assert pairs.iloc[0]["abs_corr"] > 0.8
