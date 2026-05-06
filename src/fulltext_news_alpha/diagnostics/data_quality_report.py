"""Basic panel data quality summaries."""

from __future__ import annotations

import pandas as pd


def panel_quality_summary(panel: pd.DataFrame) -> dict[str, float]:
    return {
        "rows": float(len(panel)),
        "tickers": float(panel["ticker"].nunique()) if "ticker" in panel else 0.0,
        "dates": float(panel["date"].nunique()) if "date" in panel else 0.0,
        "mean_news_count": float(panel.get("news_count", pd.Series(dtype=float)).mean()),
        "mean_chunk_count": float(panel.get("chunk_count", pd.Series(dtype=float)).mean()),
    }
