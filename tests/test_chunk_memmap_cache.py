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


def _write_synthetic_manifest(root: Path) -> pd.DataFrame:
    dates = pd.bdate_range("2021-01-04", periods=4)
    ticker_ids = {"AAA": 1, "BBB": 2}
    records: list[dict[str, object]] = []
    panel_rows: list[dict[str, object]] = []
    for ticker, ticker_id in ticker_ids.items():
        for day_idx, date_value in enumerate(dates, start=1):
            panel_rows.append(
                {
                    "ticker": ticker,
                    "date": date_value.date(),
                    "factor_a": float(ticker_id),
                    "factor_b": float(day_idx),
                    "future_20d_market_adjusted_return": float(ticker_id * 10 + day_idx),
                    "has_news": 1,
                }
            )
            for chunk_idx in range(4):
                chunk_dir = root / "2021" / ticker
                chunk_dir.mkdir(parents=True, exist_ok=True)
                embedding = np.asarray(
                    [float(ticker_id), float(day_idx), float(chunk_idx), 1.0],
                    dtype=np.float32,
                )
                embedding_path = chunk_dir / f"{ticker}_{date_value.date()}_{chunk_idx}.npy"
                np.save(embedding_path, embedding)
                records.append(
                    {
                        "chunk_id": f"{ticker}_{date_value.date()}_{chunk_idx}",
                        "ticker": ticker.lower() if ticker == "BBB" else ticker,
                        "date": str(date_value.date()),
                        "publish_time": f"{date_value.date()}T09:{chunk_idx:02d}:00",
                        "source_news_id": f"news-{ticker}-{day_idx}",
                        "chunk_index": chunk_idx,
                        "embedding_path": str(embedding_path),
                    }
                )
    manifest = root / "2021" / "manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    # Reverse the records to verify both paths impose the same deterministic sort.
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in reversed(records)),
        encoding="utf-8",
    )
    return pd.DataFrame(panel_rows)


def _expand_cache_sample(cache_index, sample: dict[str, object], max_chunks: int) -> tuple[np.ndarray, np.ndarray]:
    stockday_ids = sample["stockday_id_seq"].numpy()
    chunk_seq = np.zeros((len(stockday_ids), max_chunks, cache_index.embedding_dim), dtype=np.float32)
    chunk_mask = np.zeros((len(stockday_ids), max_chunks), dtype=bool)
    matrix = np.memmap(
        cache_index.memmap_path,
        dtype=np.float32,
        mode="r",
        shape=(cache_index.chunk_count, cache_index.embedding_dim),
    )
    for day_idx, stockday_id in enumerate(stockday_ids):
        if stockday_id <= 0:
            continue
        offset, length = cache_index.stockday_offsets[int(stockday_id) - 1]
        chunk_seq[day_idx, :length] = np.asarray(matrix[offset : offset + length], dtype=np.float32)
        chunk_mask[day_idx, :length] = True
    return chunk_seq, chunk_mask


def test_chunk_memmap_cache_matches_direct_manifest_path(tmp_path: Path) -> None:
    from fulltext_news_alpha.training.build_chunk_memmap_cache import build_chunk_memmap_cache
    from fulltext_news_alpha.training.sequence_data import (
        B3SequenceStockDayDataset,
        load_chunk_embedding_index,
    )

    embeddings_root = tmp_path / "finbert_by_year"
    panel = _write_synthetic_manifest(embeddings_root)
    cache_dir = tmp_path / "finbert_stockday_cache"
    max_chunks = 3
    metadata = build_chunk_memmap_cache(
        embeddings_root=embeddings_root,
        output_dir=cache_dir,
        project_root=tmp_path,
        max_chunks_per_stock_day=max_chunks,
    )

    direct_index = load_chunk_embedding_index(
        embeddings_root,
        project_root=tmp_path,
        max_chunks_per_stock_day=max_chunks,
    )
    cache_index = load_chunk_embedding_index(
        cache_dir,
        project_root=tmp_path,
        max_chunks_per_stock_day=max_chunks,
    )
    assert direct_index.embedding_dim == cache_index.embedding_dim == 4
    assert metadata["chunk_count"] == 2 * 4 * max_chunks
    assert cache_index.chunk_count == metadata["chunk_count"]
    assert set(direct_index.paths_by_key) == set(cache_index.offsets_by_key or {})

    ticker_to_id = {"AAA": 1, "BBB": 2}
    dataset_direct = B3SequenceStockDayDataset(
        sample_frame=panel,
        history_frame=panel,
        factor_cols=["factor_a", "factor_b"],
        chunk_index=direct_index,
        label_col="future_20d_market_adjusted_return",
        lookback_window=3,
        max_chunks_per_stock_day=max_chunks,
        ticker_to_id=ticker_to_id,
    )
    dataset_cache = B3SequenceStockDayDataset(
        sample_frame=panel,
        history_frame=panel,
        factor_cols=["factor_a", "factor_b"],
        chunk_index=cache_index,
        label_col="future_20d_market_adjusted_return",
        lookback_window=3,
        max_chunks_per_stock_day=max_chunks,
        ticker_to_id=ticker_to_id,
    )
    assert len(dataset_direct) == len(dataset_cache)
    assert dataset_direct.keys.equals(dataset_cache.keys)

    for idx in range(len(dataset_direct)):
        direct_sample = dataset_direct[idx]
        cache_sample = dataset_cache[idx]
        cache_chunk_seq, cache_chunk_mask = _expand_cache_sample(cache_index, cache_sample, max_chunks)
        np.testing.assert_allclose(direct_sample["chunk_seq"].numpy(), cache_chunk_seq, atol=1e-6)
        np.testing.assert_array_equal(direct_sample["chunk_mask_seq"].numpy(), cache_chunk_mask)
        np.testing.assert_allclose(direct_sample["factor_seq"].numpy(), cache_sample["factor_seq"].numpy(), atol=1e-6)
        np.testing.assert_array_equal(direct_sample["ticker_id_seq"].numpy(), cache_sample["ticker_id_seq"].numpy())
        np.testing.assert_allclose(direct_sample["label"].numpy(), cache_sample["label"].numpy(), atol=1e-6)
        np.testing.assert_allclose(direct_sample["has_news"].numpy(), cache_sample["has_news"].numpy(), atol=1e-6)

