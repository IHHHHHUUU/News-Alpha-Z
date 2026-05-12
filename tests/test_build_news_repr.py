from __future__ import annotations

import json

import numpy as np
import pandas as pd

from fulltext_news_alpha.features.build_news_repr import build_news_representations


def test_build_news_representations_outputs_repr_and_panel(tmp_path) -> None:
    embeddings_dir = tmp_path / "embeddings" / "2024"
    embeddings_dir.mkdir(parents=True)

    records = []
    for chunk_id, ticker, values in [
        ("c1", "AAPL", [1.0, 2.0, 3.0]),
        ("c2", "AAPL", [3.0, 4.0, 5.0]),
        ("c3", "MSFT", [10.0, 20.0, 30.0]),
    ]:
        path = embeddings_dir / f"{chunk_id}.npy"
        np.save(path, np.asarray(values, dtype=np.float32))
        records.append(
            {
                "chunk_id": chunk_id,
                "embedding_path": str(path),
                "ticker": ticker,
                "date": "2024-01-02",
                "model_name": "test-model",
                "embedding_dim": 3,
            }
        )

    manifest_path = embeddings_dir / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    panel = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02", "2024-01-02"],
            "ticker": ["AAPL", "MSFT", "GOOG"],
            "news_count": [1, 1, 0],
            "chunk_count": [2, 1, 0],
            "feature": [0.1, 0.2, 0.3],
        }
    )
    panel_path = tmp_path / "panel.parquet"
    panel.to_parquet(panel_path, index=False)

    repr_output = tmp_path / "news_repr.parquet"
    panel_output = tmp_path / "panel_with_repr.parquet"
    meta_output = tmp_path / "projection_meta.json"
    metadata = build_news_representations(
        embeddings_root=tmp_path / "embeddings",
        panel_path=panel_path,
        repr_output=repr_output,
        panel_output=panel_output,
        projection_meta_output=meta_output,
        output_dim=2,
        seed=42,
        project_root=tmp_path,
    )

    repr_frame = pd.read_parquet(repr_output)
    assert list(repr_frame.columns) == [
        "date",
        "ticker",
        "news_repr_0",
        "news_repr_1",
        "news_count",
        "chunk_count",
        "embedding_dim",
        "repr_method",
    ]
    assert len(repr_frame) == 2
    assert set(repr_frame["ticker"]) == {"AAPL", "MSFT"}
    assert repr_frame["embedding_dim"].eq(2).all()

    panel_with_repr = pd.read_parquet(panel_output)
    goog = panel_with_repr[panel_with_repr["ticker"] == "GOOG"].iloc[0]
    assert goog["news_repr_0"] == 0.0
    assert goog["news_repr_1"] == 0.0
    assert metadata["qa"]["manifest_rows"] == 3
    assert metadata["qa"]["unique_chunk_ids"] == 3
    assert metadata["qa"]["stock_days_with_embeddings"] == 2
    assert metadata["qa"]["panel_rows_with_nonzero_news_representation"] == 2
