from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch optional dependency not installed",
)


def _panel() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=12)
    rows = []
    for idx, date_value in enumerate(dates):
        rows.append(
            {
                "date": date_value.date(),
                "ticker": "AAPL",
                "momentum_5d_zscore": float(idx) / 10.0,
                "volatility_20d_zscore": float(idx % 3) / 10.0,
                "future_20d_market_adjusted_return": float(idx + 1) / 100.0,
                "news_count": 1,
                "chunk_count": 2,
                "has_news": 1,
                "mean_emb_0": float(idx),
                "mean_emb_1": float(idx + 1),
                "mean_emb_2": float(idx + 2),
            }
        )
    return pd.DataFrame(rows)


def _manifest(root: Path, panel: pd.DataFrame) -> Path:
    root.mkdir(parents=True)
    records = []
    for _, row in panel.iterrows():
        for chunk_idx in range(2):
            chunk_id = f"{row['ticker']}_{row['date']}_{chunk_idx}"
            path = root / f"{chunk_id}.npy"
            np.save(
                path,
                np.asarray(
                    [
                        float(row["mean_emb_0"]) + chunk_idx,
                        float(row["mean_emb_1"]) + chunk_idx,
                        float(row["mean_emb_2"]) + chunk_idx,
                    ],
                    dtype=np.float32,
                ),
            )
            records.append(
                {
                    "chunk_id": chunk_id,
                    "ticker": row["ticker"],
                    "date": str(row["date"]),
                    "embedding_path": str(path),
                    "embedding_dim": 3,
                    "chunk_index": chunk_idx,
                }
            )
    manifest = root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return manifest


def _split():
    from fulltext_news_alpha.training.torch_utils import SplitConfig

    return SplitConfig(
        train_start="2020-01-01",
        train_end="2020-01-08",
        valid_start="2020-01-09",
        valid_end="2020-01-13",
        test_start="2020-01-14",
        test_end="2020-01-16",
    )


def _config():
    from fulltext_news_alpha.training.torch_utils import TrainConfig

    return TrainConfig(
        batch_size=2,
        max_epochs=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        early_stopping_patience=1,
        device="cpu",
        seed=7,
    )


def test_tcn_backbone_shapes_and_dilations() -> None:
    import torch

    from fulltext_news_alpha.models.tcn_backbone import TCNBackbone

    model = TCNBackbone(input_dim=3, hidden_dim=5, kernel_size=3, dilations=(1, 2, 4, 8))
    encoded, state = model(torch.randn(2, 30, 3))
    assert encoded.shape == (2, 30, 5)
    assert state.shape == (2, 5)
    assert model.dilations == (1, 2, 4, 8)


def test_sequence_datasets_build_b2_and_b3_windows(tmp_path) -> None:
    from fulltext_news_alpha.training.sequence_data import (
        B2SequenceStockDayDataset,
        B3SequenceStockDayDataset,
        load_chunk_embedding_index,
    )

    panel = _panel()
    factor_cols = ["momentum_5d_zscore", "volatility_20d_zscore"]
    emb_cols = ["mean_emb_0", "mean_emb_1", "mean_emb_2"]
    b2 = B2SequenceStockDayDataset(
        sample_frame=panel,
        history_frame=panel,
        factor_cols=factor_cols,
        embedding_cols=emb_cols,
        label_col="future_20d_market_adjusted_return",
        lookback_window=3,
    )
    sample = b2[0]
    assert sample["factor_seq"].shape == (3, 2)
    assert sample["news_seq"].shape == (3, 3)

    chunk_index = load_chunk_embedding_index(_manifest(tmp_path / "chunks", panel))
    b3 = B3SequenceStockDayDataset(
        sample_frame=panel,
        history_frame=panel,
        factor_cols=factor_cols,
        chunk_index=chunk_index,
        label_col="future_20d_market_adjusted_return",
        lookback_window=3,
        max_chunks_per_stock_day=2,
    )
    sample_b3 = b3[0]
    assert sample_b3["chunk_seq"].shape == (3, 2, 3)
    assert sample_b3["chunk_mask_seq"].shape == (3, 2)


def test_temporal_training_smoke_for_four_combinations(tmp_path) -> None:
    from fulltext_news_alpha.training.temporal_training import train_temporal_b4, train_temporal_b5

    panel = _panel()
    panel_path = tmp_path / "panel.parquet"
    panel.to_parquet(panel_path, index=False)
    manifest = _manifest(tmp_path / "manifest_chunks", panel)

    common = {
        "panel_path": panel_path,
        "split": _split(),
        "config": _config(),
        "label_col": "future_20d_market_adjusted_return",
        "news_dim": 4,
        "hidden_dim": 8,
        "bottleneck_hidden_dim": 16,
        "dropout": 0.0,
        "lookback_window": 3,
        "kernel_size": 2,
        "dilations": (1, 2),
    }
    b2_b4 = train_temporal_b4(news_pooling="b2", output_dir=tmp_path / "b2_b4", **common)
    b2_b5 = train_temporal_b5(news_pooling="b2", output_dir=tmp_path / "b2_b5", **common)
    b3_common = {
        **common,
        "chunk_manifest": manifest,
        "max_chunks_per_stock_day": 2,
    }
    b3_b4 = train_temporal_b4(news_pooling="b3", output_dir=tmp_path / "b3_b4", **b3_common)
    b3_b5 = train_temporal_b5(news_pooling="b3", output_dir=tmp_path / "b3_b5", **b3_common)

    assert Path(b2_b4["outputs"]["train"]).exists()
    assert Path(b2_b5["outputs"]["final_train"]).exists()
    assert Path(b3_b4["outputs"]["train"]).exists()
    assert Path(b3_b5["outputs"]["final_train"]).exists()
    assert b3_b4["attention_outputs"]["train"]
    assert b3_b5["attention_outputs"]["train"]
