from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch optional dependency not installed",
)


def _toy_panel(days: int = 8, names_per_day: int = 20) -> pd.DataFrame:
    rows = []
    for day_idx, date_value in enumerate(pd.bdate_range("2020-01-01", periods=days)):
        for i in range(names_per_day):
            signal = float(i)
            label = signal / 10.0 + day_idx * 0.01
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
                    "future_20d_market_adjusted_return": label,
                }
            )
    return pd.DataFrame(rows)


def test_daily_rank_and_zscore_labels_are_cross_sectional() -> None:
    from fulltext_news_alpha.training.factor_mlp_rank_loss import (
        daily_rank_label,
        daily_zscore_label,
    )

    frame = _toy_panel(days=2, names_per_day=5)
    ranks = daily_rank_label(frame, "future_20d_market_adjusted_return")
    zscores = daily_zscore_label(frame, "future_20d_market_adjusted_return")
    first_day = frame["date"] == frame["date"].iloc[0]
    assert np.allclose(ranks[first_day].to_numpy(), [0.2, 0.4, 0.6, 0.8, 1.0])
    assert np.isclose(zscores[first_day].mean(), 0.0)
    assert np.isclose(zscores[first_day].std(ddof=0), 1.0)


def test_alpha_direction_constructed_without_restandardization() -> None:
    from fulltext_news_alpha.training.factor_mlp_rank_loss import prepare_rank_loss_panel

    panel = _toy_panel(days=1, names_per_day=5)
    prepared, factors = prepare_rank_loss_panel(panel, "future_20d_market_adjusted_return")
    assert "RSI_alpha" in factors
    assert np.allclose(prepared["RSI_alpha"], -panel["RSI_zscore"])
    assert np.allclose(prepared["momentum_20d_alpha"], -panel["momentum_20d_zscore"])
    assert np.allclose(prepared["reversal_5d_alpha"], panel["reversal_5d_zscore"])


def test_ic_loss_prefers_perfect_positive_prediction() -> None:
    import torch

    from fulltext_news_alpha.training.factor_mlp_rank_loss import daily_ic_loss

    y_rank = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4])
    date_id = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    good = y_rank.clone()
    bad = -y_rank
    assert daily_ic_loss(good, y_rank, date_id) < daily_ic_loss(bad, y_rank, date_id)


def test_constant_prediction_triggers_variance_penalty() -> None:
    import torch

    from fulltext_news_alpha.training.factor_mlp_rank_loss import variance_penalty

    pred = torch.ones(8)
    date_id = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    penalty = variance_penalty(pred, date_id, min_pred_std=0.05)
    assert float(penalty) > 0


def test_date_aware_batch_keeps_whole_dates_together() -> None:
    from fulltext_news_alpha.training.factor_mlp_rank_loss import (
        DateAwareBatcher,
        prepare_rank_loss_panel,
    )

    prepared, factors = prepare_rank_loss_panel(_toy_panel(days=4, names_per_day=5), "future_20d_market_adjusted_return")
    batcher = DateAwareBatcher(prepared, factors, batch_dates=2, shuffle=False)
    batch = next(iter(batcher))
    assert set(batch["date_id"].tolist()) == {0, 1}
    assert len(batch["date_id"]) == 10


def test_training_uses_valid_rankic_and_predictions_not_constant(tmp_path: Path) -> None:
    from fulltext_news_alpha.training.factor_mlp_rank_loss import (
        RankLossConfig,
        train_factor_mlp_rank_loss,
    )
    from fulltext_news_alpha.training.torch_utils import SplitConfig

    panel = _toy_panel(days=12, names_per_day=20)
    panel_path = tmp_path / "panel.parquet"
    panel.to_parquet(panel_path, index=False)
    split = SplitConfig(
        train_start="2020-01-01",
        train_end="2020-01-08",
        valid_start="2020-01-09",
        valid_end="2020-01-13",
        test_start="2020-01-14",
        test_end="2020-01-20",
    )
    result = train_factor_mlp_rank_loss(
        panel_path=panel_path,
        output_dir=tmp_path / "rank_loss",
        label_col="future_20d_market_adjusted_return",
        split=split,
        config=RankLossConfig(
            hidden_dim=8,
            dropout=0.0,
            batch_dates=2,
            epochs=5,
            patience=2,
            lr=1e-3,
            seed=3,
            device="cpu",
        ),
    )
    history = pd.read_csv(tmp_path / "rank_loss" / "history.csv")
    preds = pd.read_parquet(tmp_path / "rank_loss" / "predictions_test.parquet")
    assert np.isclose(result["best_valid_RankIC"], history["valid_RankIC"].max())
    assert preds["pred"].std(ddof=0) > 0
    assert result["metrics"]["test"]["daily_pred_std_mean"] > 0
