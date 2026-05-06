"""Full-text news chunking for offline encoder batches."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from fulltext_news_alpha.preprocessing.clean_news import clean_news_text
from fulltext_news_alpha.preprocessing.time_alignment import add_signal_date
from fulltext_news_alpha.schemas import ChunkRecord, RawNewsRecord


def whitespace_tokenize(text: str) -> list[str]:
    """A dependency-light tokenizer used for deterministic chunk planning and tests."""

    return text.split()


def stable_chunk_id(news_id: str, ticker: str, signal_date: object, chunk_index: int) -> str:
    raw = f"{news_id}|{ticker}|{pd.Timestamp(signal_date).date()}|{chunk_index}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def chunk_text_by_tokens(
    text: str,
    max_tokens: int = 256,
    overlap_tokens: int = 0,
) -> list[str]:
    """Split text into independent encoder-sized token chunks."""

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be in [0, max_tokens)")

    tokens = whitespace_tokenize(clean_news_text(text))
    if not tokens:
        return []
    chunks: list[str] = []
    step = max_tokens - overlap_tokens
    for start in range(0, len(tokens), step):
        chunk_tokens = tokens[start : start + max_tokens]
        if chunk_tokens:
            chunks.append(" ".join(chunk_tokens))
        if start + max_tokens >= len(tokens):
            break
    return chunks


def build_chunks_for_records(
    records: list[RawNewsRecord],
    trading_days: list[object] | pd.DatetimeIndex,
    max_tokens: int = 256,
    max_chunks_per_stock_day: int = 64,
    overlap_tokens: int = 0,
) -> pd.DataFrame:
    """Create a capped stock-day chunk table from normalized raw news records."""

    rows: list[dict[str, object]] = []
    for record in records:
        signal_date = add_signal_date(
            pd.DataFrame({"publish_time": [record.publish_time]}), trading_days
        ).loc[0, "date"]
        chunks = chunk_text_by_tokens(record.text, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        for idx, chunk in enumerate(chunks):
            chunk_record = ChunkRecord(
                chunk_id=stable_chunk_id(record.news_id, record.ticker, signal_date, idx),
                ticker=record.ticker,
                date=signal_date,
                publish_time=record.publish_time,
                source_news_id=record.news_id,
                chunk_text=chunk,
                chunk_index=idx,
                token_count=len(whitespace_tokenize(chunk)),
                title=record.title,
            )
            rows.append(asdict(chunk_record))

    if not rows:
        return pd.DataFrame(columns=list(ChunkRecord.__dataclass_fields__))
    chunks = pd.DataFrame(rows).sort_values(
        ["ticker", "date", "publish_time", "source_news_id", "chunk_index"]
    )
    return chunks.groupby(["ticker", "date"], group_keys=False).head(max_chunks_per_stock_day).reset_index(
        drop=True
    )


def chunk_dataframe(
    news: pd.DataFrame,
    trading_days: list[object] | pd.DatetimeIndex,
    max_tokens: int = 256,
    max_chunks_per_stock_day: int = 64,
    overlap_tokens: int = 0,
) -> pd.DataFrame:
    """Chunk a DataFrame with normalized news columns."""

    required = {"news_id", "ticker", "publish_time", "title", "text"}
    missing = required - set(news.columns)
    if missing:
        raise KeyError(f"Missing required news columns: {sorted(missing)}")

    records = [
        RawNewsRecord(
            news_id=str(row.news_id),
            ticker=str(row.ticker),
            publish_time=pd.Timestamp(row.publish_time).to_pydatetime(),
            title="" if pd.isna(row.title) else str(row.title),
            text="" if pd.isna(row.text) else str(row.text),
            source=None if "source" not in news.columns or pd.isna(row.get("source")) else str(row.get("source")),
        )
        for row in news.itertuples(index=False)
    ]
    return build_chunks_for_records(
        records,
        trading_days=trading_days,
        max_tokens=max_tokens,
        max_chunks_per_stock_day=max_chunks_per_stock_day,
        overlap_tokens=overlap_tokens,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk normalized news into FinBERT-sized records.")
    parser.add_argument("--news", required=True, help="Input news parquet/csv with normalized columns.")
    parser.add_argument("--calendar", required=True, help="CSV/parquet with a date column.")
    parser.add_argument("--output", required=True, help="Output parquet path.")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-chunks-per-stock-day", type=int, default=64)
    args = parser.parse_args()

    news_path = Path(args.news)
    news = pd.read_parquet(news_path) if news_path.suffix == ".parquet" else pd.read_csv(news_path)
    cal_path = Path(args.calendar)
    calendar = pd.read_parquet(cal_path) if cal_path.suffix == ".parquet" else pd.read_csv(cal_path)
    chunks = chunk_dataframe(
        news,
        trading_days=pd.to_datetime(calendar["date"]),
        max_tokens=args.max_tokens,
        max_chunks_per_stock_day=args.max_chunks_per_stock_day,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    chunks.to_parquet(output, index=False)


if __name__ == "__main__":
    main()
