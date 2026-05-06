"""Load and normalize raw financial news tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fulltext_news_alpha.preprocessing.clean_news import clean_news_text
from fulltext_news_alpha.preprocessing.ticker_alias import normalize_ticker


DEFAULT_COLUMN_MAP = {
    "Date": "publish_time",
    "datetime": "publish_time",
    "published_at": "publish_time",
    "Article_title": "title",
    "Title": "title",
    "headline": "title",
    "Article": "text",
    "content": "text",
    "body": "text",
    "Stock_symbol": "ticker",
    "symbol": "ticker",
    "Url": "news_id",
    "url": "news_id",
}


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def normalize_news_columns(
    frame: pd.DataFrame,
    column_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Normalize likely FNSPID/news columns into the project schema."""

    mapping = dict(DEFAULT_COLUMN_MAP)
    if column_map:
        mapping.update(column_map)
    out = frame.rename(columns={k: v for k, v in mapping.items() if k in frame.columns}).copy()
    required = {"ticker", "publish_time", "title", "text"}
    missing = required - set(out.columns)
    if missing:
        raise KeyError(f"Missing normalized news columns: {sorted(missing)}")
    if "news_id" not in out.columns:
        out["news_id"] = [f"news_{i}" for i in range(len(out))]
    if "source" not in out.columns:
        out["source"] = None
    out["ticker"] = out["ticker"].map(normalize_ticker)
    out["publish_time"] = pd.to_datetime(out["publish_time"])
    out["title"] = out["title"].fillna("").astype(str)
    out["text"] = out["text"].map(clean_news_text)
    return out[["news_id", "ticker", "publish_time", "title", "text", "source"]]


def load_raw_news(path: str | Path, column_map: dict[str, str] | None = None) -> pd.DataFrame:
    return normalize_news_columns(read_table(path), column_map=column_map)
