"""News representation baselines for B1-B3 comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd


def embedding_columns(frame: pd.DataFrame, prefix: str = "emb_") -> list[str]:
    cols = [col for col in frame.columns if col.startswith(prefix)]
    if not cols:
        raise KeyError(f"No embedding columns found with prefix {prefix!r}")
    return cols


def mean_chunk_pooling(
    chunks: pd.DataFrame,
    embedding_cols: list[str] | None = None,
    prefix: str = "emb_",
    output_prefix: str = "news_repr_",
) -> pd.DataFrame:
    """B2 baseline: average all chunk embeddings within a stock-day."""

    cols = embedding_cols or embedding_columns(chunks, prefix=prefix)
    pooled = chunks.groupby(["date", "ticker"], as_index=False)[cols].mean()
    rename = {col: f"{output_prefix}{i}" for i, col in enumerate(cols)}
    return pooled.rename(columns=rename)


def headline_embedding_representation(
    news: pd.DataFrame,
    embedding_cols: list[str] | None = None,
    prefix: str = "title_emb_",
    output_prefix: str = "news_repr_",
) -> pd.DataFrame:
    """B1 baseline: use title/headline embeddings averaged at stock-day level."""

    cols = embedding_cols or embedding_columns(news, prefix=prefix)
    pooled = news.groupby(["date", "ticker"], as_index=False)[cols].mean()
    rename = {col: f"{output_prefix}{i}" for i, col in enumerate(cols)}
    return pooled.rename(columns=rename)


def attention_entropy_from_weights(weights: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute attention entropy for saved B3 attention weights."""

    clipped = np.clip(np.asarray(weights, dtype=float), 1e-8, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=axis)
