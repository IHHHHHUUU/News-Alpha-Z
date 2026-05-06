"""Attention top-chunk extraction for interpretability case studies."""

from __future__ import annotations

import pandas as pd


def top_attention_chunks(
    chunks: pd.DataFrame,
    attention_weights: pd.DataFrame,
    top_k: int = 5,
) -> pd.DataFrame:
    """Join chunk text with saved attention weights and return top chunks."""

    required = {"date", "ticker", "chunk_id", "attention_weight"}
    missing = required - set(attention_weights.columns)
    if missing:
        raise KeyError(f"Missing attention columns: {sorted(missing)}")
    merged = chunks.merge(attention_weights, on=["date", "ticker", "chunk_id"], how="inner")
    return (
        merged.sort_values(["date", "ticker", "attention_weight"], ascending=[True, True, False])
        .groupby(["date", "ticker"], group_keys=False)
        .head(top_k)
        .reset_index(drop=True)
    )
