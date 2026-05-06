"""Join news chunks, price features, and labels into stock-day rows."""

from __future__ import annotations

import pandas as pd


def summarize_chunks(chunks: pd.DataFrame) -> pd.DataFrame:
    """Summarize chunk/news coverage at stock-day level."""

    if chunks.empty:
        return pd.DataFrame(columns=["ticker", "date", "news_count", "chunk_count"])
    out = chunks.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    return (
        out.groupby(["ticker", "date"], as_index=False)
        .agg(news_count=("source_news_id", "nunique"), chunk_count=("chunk_id", "nunique"))
        .sort_values(["ticker", "date"])
    )


def build_stock_day_panel(
    factors: pd.DataFrame,
    labels: pd.DataFrame,
    chunks: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a stock-day panel without leaking future label data into features."""

    panel = factors.copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.date
    label_cols = [col for col in labels.columns if col.startswith("future_")]
    labels_view = labels[["ticker", "date"] + label_cols].copy()
    labels_view["date"] = pd.to_datetime(labels_view["date"]).dt.date
    panel = panel.merge(labels_view, on=["ticker", "date"], how="left", validate="one_to_one")

    if chunks is not None:
        coverage = summarize_chunks(chunks)
        panel = panel.merge(coverage, on=["ticker", "date"], how="left")
    else:
        panel["news_count"] = 0
        panel["chunk_count"] = 0
    panel["news_count"] = panel["news_count"].fillna(0).astype(int)
    panel["chunk_count"] = panel["chunk_count"].fillna(0).astype(int)
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)
