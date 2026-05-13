"""Build raw 768-d stock-day mean-pooled news embeddings.

This is the input to the trainable ``NewsBottleneck`` MLP. No dimensionality
reduction is applied here. Output schema:

    date, ticker, has_news, embedded_chunk_count,
    mean_emb_0, mean_emb_1, ..., mean_emb_{D-1}

Stock-days without any news are *not* emitted by this builder. The downstream
training-panel builder is responsible for joining onto every stock-day and
filling zeros + ``has_news = 0`` where no news exists.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fulltext_news_alpha.features.build_news_repr import (
    build_stock_day_mean_embeddings,
    load_manifests,
    prepare_manifest,
)


DEFAULT_EMBEDDINGS_ROOT = Path("data/embeddings/finbert_by_year")
DEFAULT_REPR_OUTPUT = Path("data/features/news_repr_finbert_mean_768.parquet")
DEFAULT_META_OUTPUT = Path("data/features/news_repr_finbert_mean_768_meta.json")
REPR_METHOD = "mean_pooling_768"


def _columns_for_dim(output_dim: int) -> list[str]:
    return [f"mean_emb_{idx}" for idx in range(output_dim)]


def build_mean_768_repr(
    embeddings_root: str | Path = DEFAULT_EMBEDDINGS_ROOT,
    repr_output: str | Path = DEFAULT_REPR_OUTPUT,
    meta_output: str | Path = DEFAULT_META_OUTPUT,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    """Produce the raw 768-d stock-day mean embedding parquet and metadata."""

    manifest = load_manifests(embeddings_root)
    valid_manifest, qa = prepare_manifest(manifest, project_root=project_root)
    pooled, mean_matrix, pool_qa = build_stock_day_mean_embeddings(valid_manifest)
    qa.update(pool_qa)

    if mean_matrix.ndim != 2:
        raise ValueError("Mean embedding matrix must be two-dimensional.")
    output_dim = int(mean_matrix.shape[1])
    columns = _columns_for_dim(output_dim)

    embedding_frame = pd.DataFrame(mean_matrix.astype(np.float32), columns=pd.Index(columns))
    pooled = pooled.reset_index(drop=True)
    out = pd.concat([pooled, embedding_frame], axis=1)
    out = out.rename(columns={"_embedded_chunk_count": "embedded_chunk_count"})
    out["has_news"] = 1
    out["embedded_chunk_count"] = out["embedded_chunk_count"].astype(int)
    out["ticker"] = out["ticker"].astype(str)
    out["date"] = pd.to_datetime(out["date"]).dt.date
    ordered_cols = ["date", "ticker", "has_news", "embedded_chunk_count", *columns]
    out = out.loc[:, ordered_cols]
    out = out.sort_values(by=["date", "ticker"]).reset_index(drop=True)

    repr_output = Path(repr_output)
    meta_output = Path(meta_output)
    repr_output.parent.mkdir(parents=True, exist_ok=True)
    meta_output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(repr_output, index=False)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repr_method": REPR_METHOD,
        "inputs": {"embeddings_root": str(embeddings_root)},
        "outputs": {"news_repr": str(repr_output)},
        "output_dim": output_dim,
        "qa": qa,
    }
    meta_output.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build raw 768-d stock-day mean-pooled news embeddings (no projection).",
    )
    parser.add_argument("--embeddings-root", default=str(DEFAULT_EMBEDDINGS_ROOT))
    parser.add_argument("--news-repr-output", default=str(DEFAULT_REPR_OUTPUT))
    parser.add_argument("--meta-output", default=str(DEFAULT_META_OUTPUT))
    args = parser.parse_args()

    metadata = build_mean_768_repr(
        embeddings_root=args.embeddings_root,
        repr_output=args.news_repr_output,
        meta_output=args.meta_output,
    )
    print(json.dumps(metadata["qa"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
