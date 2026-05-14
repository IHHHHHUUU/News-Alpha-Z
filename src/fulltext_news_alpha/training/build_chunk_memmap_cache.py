"""Build a B3-friendly chunk embedding memmap cache."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fulltext_news_alpha.training.sequence_data import _read_manifest_jsonl


def _resolve_path(raw_path: str, manifest_dir: str | Path, project_root: str | Path) -> Path | None:
    path = Path(raw_path)
    root = Path(project_root)
    manifest_dir = Path(manifest_dir)
    candidates = (
        (path,)
        if path.is_absolute()
        else (
            root / path,
            manifest_dir / path,
            manifest_dir / path.name,
            path,
        )
    )
    return next((candidate for candidate in candidates if candidate.exists()), None)


def build_chunk_memmap_cache(
    embeddings_root: str | Path,
    output_dir: str | Path,
    project_root: str | Path = ".",
    max_chunks_per_stock_day: int = 64,
) -> dict[str, Any]:
    """Pack per-chunk .npy embeddings into one memmap plus a stock-day index."""

    embeddings_root = Path(embeddings_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_paths = sorted(embeddings_root.rglob("manifest.jsonl"))
    if not manifest_paths:
        raise FileNotFoundError(f"No manifest.jsonl files found under {embeddings_root}")

    manifest = pd.concat([_read_manifest_jsonl(path) for path in manifest_paths], ignore_index=True)
    required = {"chunk_id", "ticker", "date", "embedding_path"}
    missing = required - set(manifest.columns)
    if missing:
        raise KeyError(f"Missing manifest columns: {sorted(missing)}")
    manifest = manifest.drop_duplicates("chunk_id", keep="first").copy()
    manifest["ticker"] = manifest["ticker"].astype(str).str.upper().str.strip()
    manifest["date"] = pd.to_datetime(manifest["date"]).dt.date
    sort_cols = [col for col in ("ticker", "date", "publish_time", "source_news_id", "chunk_index") if col in manifest]
    if sort_cols:
        manifest = manifest.sort_values(sort_cols)

    rows: list[dict[str, Any]] = []
    embedding_dim: int | None = None
    for row in manifest.to_dict(orient="records"):
        resolved = _resolve_path(str(row["embedding_path"]), row.get("_manifest_dir", "."), project_root)
        if resolved is None:
            continue
        if embedding_dim is None:
            embedding_dim = int(np.asarray(np.load(resolved)).reshape(-1).shape[0])
        rows.append(
            {
                "ticker": str(row["ticker"]),
                "date": row["date"],
                "path": resolved,
            }
        )
    if embedding_dim is None:
        raise ValueError("No loadable chunk embedding files found.")

    selected_rows: list[dict[str, Any]] = []
    stockday_rows: list[dict[str, Any]] = []
    offset = 0
    for (ticker, date_value), group in pd.DataFrame(rows).groupby(["ticker", "date"], sort=True):
        group = group.head(int(max_chunks_per_stock_day))
        length = int(len(group))
        if length == 0:
            continue
        stockday_rows.append(
            {
                "ticker": ticker,
                "date": date_value,
                "offset": offset,
                "length": length,
                "embedding_dim": embedding_dim,
            }
        )
        for path in group["path"]:
            selected_rows.append({"path": Path(path)})
        offset += length

    chunk_count = int(len(selected_rows))
    matrix_path = output_dir / "chunk_embeddings.float32.memmap"
    matrix = np.memmap(matrix_path, dtype=np.float32, mode="w+", shape=(chunk_count, embedding_dim))
    for idx, row in enumerate(selected_rows):
        matrix[idx] = np.asarray(np.load(row["path"]), dtype=np.float32).reshape(-1)
        if (idx + 1) % 50_000 == 0:
            matrix.flush()
    matrix.flush()

    index = pd.DataFrame(stockday_rows)
    index["chunk_count"] = chunk_count
    index.to_parquet(output_dir / "stockday_index.parquet", index=False)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"embeddings_root": str(embeddings_root)},
        "outputs": {
            "matrix": str(matrix_path),
            "index": str(output_dir / "stockday_index.parquet"),
        },
        "manifest_file_count": len(manifest_paths),
        "manifest_rows": int(len(manifest)),
        "stock_days": int(len(index)),
        "chunk_count": chunk_count,
        "embedding_dim": embedding_dim,
        "max_chunks_per_stock_day": int(max_chunks_per_stock_day),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a memmap cache for B3 chunk embeddings.")
    parser.add_argument("--embeddings-root", default="data/embeddings/finbert_by_year")
    parser.add_argument("--output-dir", default="data/embeddings/finbert_stockday_cache")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--max-chunks-per-stock-day", type=int, default=64)
    args = parser.parse_args()
    metadata = build_chunk_memmap_cache(
        embeddings_root=args.embeddings_root,
        output_dir=args.output_dir,
        project_root=args.project_root,
        max_chunks_per_stock_day=args.max_chunks_per_stock_day,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
