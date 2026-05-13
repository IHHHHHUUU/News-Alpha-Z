"""Build the B2 mean-pooling training panel.

The panel joins price-volume factors, future-return labels, news coverage
(``news_count``/``chunk_count``), and the raw 768-d stock-day mean embedding
into one parquet keyed by ``(date, ticker)``.

Stock-days without news are kept (factor-only training still needs them) and
have ``has_news = 0`` together with zero-filled ``mean_emb_*`` columns.

The output schema is:

    date, ticker,
    *factor_cols,
    *label_cols,
    news_count, chunk_count, has_news, embedded_chunk_count,
    mean_emb_0, ..., mean_emb_{D-1}
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_PANEL_PATH = Path("data/processed/panel_with_news_coverage_clean.parquet")
DEFAULT_MEAN_768_PATH = Path("data/features/news_repr_finbert_mean_768.parquet")
DEFAULT_OUTPUT_PATH = Path("data/processed/panel_train_b2_768.parquet")
DEFAULT_META_PATH = Path("data/processed/panel_train_b2_768_meta.json")


def _ensure_columns(frame: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise KeyError(f"{name} is missing required columns: {missing}")


def _normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    return out


def build_training_panel_b2(
    panel_path: str | Path = DEFAULT_PANEL_PATH,
    mean_768_path: str | Path = DEFAULT_MEAN_768_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    meta_path: str | Path = DEFAULT_META_PATH,
) -> dict[str, Any]:
    """Join factors/labels/coverage with the 768-d stock-day mean embedding."""

    panel = pd.read_parquet(panel_path)
    mean_repr = pd.read_parquet(mean_768_path)

    _ensure_columns(panel, ["date", "ticker", "news_count", "chunk_count"], "panel")
    _ensure_columns(mean_repr, ["date", "ticker", "has_news", "embedded_chunk_count"], "mean_repr")

    emb_cols = [col for col in mean_repr.columns if col.startswith("mean_emb_")]
    if not emb_cols:
        raise KeyError("mean_repr parquet has no mean_emb_* columns")

    panel = _normalize_keys(panel)
    mean_repr = _normalize_keys(mean_repr)

    panel = panel.drop_duplicates(["date", "ticker"])
    mean_repr = mean_repr.drop_duplicates(["date", "ticker"])

    join = panel.merge(
        mean_repr[["date", "ticker", "has_news", "embedded_chunk_count", *emb_cols]],
        on=["date", "ticker"],
        how="left",
        validate="one_to_one",
    )

    join["has_news"] = join["has_news"].fillna(0).astype(int)
    join["embedded_chunk_count"] = join["embedded_chunk_count"].fillna(0).astype(int)
    join[emb_cols] = join[emb_cols].fillna(0.0).astype(np.float32)

    label_cols = [col for col in join.columns if col.startswith("future_")]
    factor_cols = [
        col
        for col in join.columns
        if col.endswith("_zscore") and not col.startswith("future_")
    ]

    output_cols = (
        ["date", "ticker"]
        + factor_cols
        + label_cols
        + ["news_count", "chunk_count", "has_news", "embedded_chunk_count"]
        + emb_cols
    )
    missing = [col for col in output_cols if col not in join.columns]
    if missing:
        raise KeyError(f"Joined panel is missing columns: {missing}")
    out = join[output_cols].sort_values(["date", "ticker"]).reset_index(drop=True)

    output_path = Path(output_path)
    meta_path = Path(meta_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    qa = {
        "panel_rows": int(len(out)),
        "unique_tickers": int(out["ticker"].nunique()),
        "date_min": str(out["date"].min()),
        "date_max": str(out["date"].max()),
        "rows_with_news": int((out["has_news"] == 1).sum()),
        "rows_without_news": int((out["has_news"] == 0).sum()),
        "factor_columns": factor_cols,
        "label_columns": label_cols,
        "embedding_dim": len(emb_cols),
        "label_non_null_counts": {
            col: int(out[col].notna().sum()) for col in label_cols
        },
    }
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "panel_path": str(panel_path),
            "mean_768_path": str(mean_768_path),
        },
        "outputs": {"training_panel": str(output_path)},
        "qa": qa,
    }
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the B2 mean-pooling training panel by joining factors/labels/news.",
    )
    parser.add_argument("--panel", default=str(DEFAULT_PANEL_PATH))
    parser.add_argument("--mean-768", default=str(DEFAULT_MEAN_768_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--meta-output", default=str(DEFAULT_META_PATH))
    args = parser.parse_args()

    metadata = build_training_panel_b2(
        panel_path=args.panel,
        mean_768_path=args.mean_768,
        output_path=args.output,
        meta_path=args.meta_output,
    )
    print(json.dumps(metadata["qa"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
