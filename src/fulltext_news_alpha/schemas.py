"""Typed records used across the research pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RawNewsRecord:
    news_id: str
    ticker: str
    publish_time: datetime
    title: str
    text: str
    source: str | None = None
    company_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    ticker: str
    date: date
    publish_time: datetime
    source_news_id: str
    chunk_text: str
    chunk_index: int
    token_count: int
    title: str | None = None
    embedding_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StockDayPanel:
    ticker: str
    date: date
    news_count: int
    chunk_count: int
    features: dict[str, float] = field(default_factory=dict)
    labels: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        row = {"ticker": self.ticker, "date": self.date, "news_count": self.news_count, "chunk_count": self.chunk_count}
        row.update(self.features)
        row.update(self.labels)
        return row


@dataclass(frozen=True)
class EmbeddingManifest:
    embedding_id: str
    chunk_id: str
    ticker: str
    date: date
    model_name: str
    embedding_dim: int
    embedding_path: Path

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["embedding_path"] = str(self.embedding_path)
        return payload


@dataclass(frozen=True)
class BranchPrediction:
    ticker: str
    date: date
    factor_only_pred: float
    fusion_pred: float | None = None
    target: float | None = None


@dataclass(frozen=True)
class GatePrediction:
    ticker: str
    date: date
    gate_news_prob: float
    mixed_pred: float


@dataclass(frozen=True)
class NewsFactorRecord:
    date: date
    ticker: str
    FullTextNewsAlpha_raw: float
    FullTextNewsAlpha_zscore: float
    factor_only_pred: float
    fusion_pred: float
    mixed_pred: float
    gate_news_prob: float
    news_count: int
    chunk_count: int
    attention_entropy: float | None = None
