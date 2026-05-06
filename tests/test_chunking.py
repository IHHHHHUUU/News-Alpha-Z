from __future__ import annotations

from datetime import datetime

import pandas as pd

from fulltext_news_alpha.preprocessing.chunk_news import (
    build_chunks_for_records,
    chunk_text_by_tokens,
    stable_chunk_id,
)
from fulltext_news_alpha.schemas import RawNewsRecord


def test_chunk_text_never_exceeds_max_tokens() -> None:
    text = " ".join(f"token{i}" for i in range(600))
    chunks = chunk_text_by_tokens(text, max_tokens=256)
    assert len(chunks) == 3
    assert all(len(chunk.split()) <= 256 for chunk in chunks)


def test_chunk_ids_are_stable() -> None:
    first = stable_chunk_id("n1", "AAPL", "2024-01-02", 0)
    second = stable_chunk_id("n1", "AAPL", "2024-01-02", 0)
    assert first == second


def test_stock_day_chunk_cap_is_enforced_without_concatenation() -> None:
    calendar = pd.to_datetime(["2024-01-02", "2024-01-03"])
    records = [
        RawNewsRecord(
            news_id=f"n{i}",
            ticker="AAPL",
            publish_time=datetime(2024, 1, 2, 10, i % 60),
            title=f"title {i}",
            text=" ".join(f"word{j}" for j in range(20)),
        )
        for i in range(10)
    ]
    chunks = build_chunks_for_records(records, calendar, max_tokens=8, max_chunks_per_stock_day=5)
    assert len(chunks) == 5
    assert chunks["chunk_text"].map(lambda value: len(value.split())).max() <= 8
    assert chunks["source_news_id"].nunique() > 1
