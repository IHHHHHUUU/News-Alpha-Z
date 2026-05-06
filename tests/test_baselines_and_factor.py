from __future__ import annotations

import numpy as np
import pandas as pd

from fulltext_news_alpha.factors.build_news_factor import build_full_text_news_alpha
from fulltext_news_alpha.models.news_representations import mean_chunk_pooling
from fulltext_news_alpha.training.baselines import BASELINES
from fulltext_news_alpha.training.train_gate_decoupled import make_gate_targets, mix_predictions


def test_all_required_baselines_are_registered() -> None:
    assert set(BASELINES) == {"B0", "B1", "B2", "B3", "B4", "B5"}


def test_mean_chunk_pooling_groups_by_stock_day() -> None:
    chunks = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02", "2024-01-02"],
            "ticker": ["AAPL", "AAPL", "MSFT"],
            "emb_0": [1.0, 3.0, 10.0],
            "emb_1": [2.0, 4.0, 20.0],
        }
    )
    pooled = mean_chunk_pooling(chunks)
    aapl = pooled[pooled["ticker"] == "AAPL"].iloc[0]
    assert aapl["news_repr_0"] == 2.0
    assert aapl["news_repr_1"] == 3.0


def test_decoupled_gate_targets_favor_lower_error_branch() -> None:
    targets = make_gate_targets(
        factor_only_pred=np.array([0.0, 1.0]),
        fusion_pred=np.array([0.9, 0.0]),
        target=np.array([1.0, 1.0]),
        temperature=0.1,
    )
    assert targets[0] > 0.5
    assert targets[1] < 0.5


def test_full_text_news_alpha_formula_and_standardization() -> None:
    predictions = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02", "2024-01-02"],
            "ticker": ["A", "B", "C"],
            "factor_only_pred": [0.0, 0.0, 0.0],
            "fusion_pred": [1.0, -1.0, 0.0],
            "gate_news_prob": [0.5, 0.5, 0.0],
        }
    )
    predictions["mixed_pred"] = mix_predictions(
        predictions["factor_only_pred"], predictions["fusion_pred"], predictions["gate_news_prob"]
    )
    factor = build_full_text_news_alpha(predictions)
    raw = dict(zip(factor["ticker"], factor["FullTextNewsAlpha_raw"], strict=True))
    assert raw["A"] == 0.5
    assert raw["B"] == -0.5
    assert raw["C"] == 0.0
    assert np.isclose(factor["FullTextNewsAlpha_zscore"].mean(), 0.0)
