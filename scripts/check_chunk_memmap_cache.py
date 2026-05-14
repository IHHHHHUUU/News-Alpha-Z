#!/usr/bin/env python
"""Sample QA for B3 direct chunk manifests versus the stock-day memmap cache."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fulltext_news_alpha.training.sequence_data import load_chunk_embedding_index


def _direct_chunks(paths: tuple[Path, ...], embedding_dim: int) -> np.ndarray:
    if not paths:
        return np.empty((0, embedding_dim), dtype=np.float32)
    return np.stack(
        [np.asarray(np.load(path), dtype=np.float32).reshape(-1) for path in paths],
        axis=0,
    )


def _cache_chunks(matrix: np.memmap, offset_length: tuple[int, int]) -> np.ndarray:
    offset, length = offset_length
    return np.asarray(matrix[offset : offset + length], dtype=np.float32)


def check_chunk_memmap_cache(
    embeddings_root: str | Path,
    cache_dir: str | Path,
    project_root: str | Path,
    max_chunks_per_stock_day: int,
    sample_size: int,
    seed: int,
) -> dict[str, Any]:
    direct_index = load_chunk_embedding_index(
        embeddings_root,
        project_root=project_root,
        max_chunks_per_stock_day=max_chunks_per_stock_day,
    )
    cache_index = load_chunk_embedding_index(
        cache_dir,
        project_root=project_root,
        max_chunks_per_stock_day=max_chunks_per_stock_day,
    )
    direct_keys = set(direct_index.paths_by_key)
    cache_keys = set(cache_index.offsets_by_key or {})
    shared_keys = sorted(direct_keys & cache_keys)
    rng = random.Random(seed)
    sampled_keys = (
        rng.sample(shared_keys, min(int(sample_size), len(shared_keys)))
        if sample_size > 0
        else shared_keys
    )
    matrix = np.memmap(
        cache_index.memmap_path,
        dtype=np.float32,
        mode="r",
        shape=(cache_index.chunk_count, cache_index.embedding_dim),
    )

    mismatched_keys: list[dict[str, Any]] = []
    max_abs_diff = 0.0
    for key in sampled_keys:
        direct = _direct_chunks(direct_index.paths_by_key[key], direct_index.embedding_dim)
        cache = _cache_chunks(matrix, cache_index.offsets_by_key[key])  # type: ignore[index]
        mismatch: dict[str, Any] | None = None
        if direct.shape != cache.shape:
            mismatch = {
                "ticker": key[0],
                "date": str(key[1]),
                "reason": "shape",
                "direct_shape": list(direct.shape),
                "cache_shape": list(cache.shape),
            }
        elif direct.shape[1:] != cache.shape[1:]:
            mismatch = {
                "ticker": key[0],
                "date": str(key[1]),
                "reason": "embedding_dim",
                "direct_shape": list(direct.shape),
                "cache_shape": list(cache.shape),
            }
        else:
            if direct.size:
                diff = np.abs(direct - cache)
                max_abs_diff = max(max_abs_diff, float(diff.max()))
            if not np.allclose(direct, cache, atol=1e-6, rtol=1e-6):
                mismatch = {
                    "ticker": key[0],
                    "date": str(key[1]),
                    "reason": "values",
                    "max_abs_diff": float(np.abs(direct - cache).max()) if direct.size else 0.0,
                }
        if mismatch is not None:
            mismatched_keys.append(mismatch)

    return {
        "checked_stock_days": len(sampled_keys),
        "mismatched_keys": mismatched_keys,
        "mismatch_count": len(mismatched_keys),
        "max_abs_diff": max_abs_diff,
        "missing_in_cache": len(direct_keys - cache_keys),
        "missing_in_direct": len(cache_keys - direct_keys),
        "direct_stock_days": len(direct_keys),
        "cache_stock_days": len(cache_keys),
        "direct_embedding_dim": direct_index.embedding_dim,
        "cache_embedding_dim": cache_index.embedding_dim,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate B3 chunk memmap cache alignment against source manifests.")
    parser.add_argument("--embeddings-root", default="data/embeddings/finbert_by_year")
    parser.add_argument("--cache-dir", default="data/embeddings/finbert_stockday_cache")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--max-chunks-per-stock-day", type=int, default=64)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = check_chunk_memmap_cache(
        embeddings_root=args.embeddings_root,
        cache_dir=args.cache_dir,
        project_root=args.project_root,
        max_chunks_per_stock_day=args.max_chunks_per_stock_day,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    failed = (
        summary["mismatch_count"] > 0
        or summary["missing_in_cache"] > 0
        or summary["missing_in_direct"] > 0
        or summary["direct_embedding_dim"] != summary["cache_embedding_dim"]
    )
    if failed:
        raise SystemExit(1)
    print("PASS chunk memmap cache matches direct manifests")


if __name__ == "__main__":
    main()

