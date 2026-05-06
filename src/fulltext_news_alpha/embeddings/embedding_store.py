"""Numpy-backed embedding storage with a JSONL manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


class EmbeddingStore:
    """Persist chunk embeddings to disk one array at a time."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.jsonl"

    def embedding_path(self, embedding_id: str) -> Path:
        safe_id = str(embedding_id).replace("/", "_")
        return self.root / f"{safe_id}.npy"

    def save(self, embedding_id: str, embedding: np.ndarray, metadata: dict[str, object]) -> Path:
        path = self.embedding_path(embedding_id)
        np.save(path, np.asarray(embedding, dtype=np.float32))
        record = dict(metadata)
        record.update(
            {
                "embedding_id": embedding_id,
                "embedding_path": str(path),
                "embedding_dim": int(np.asarray(embedding).shape[-1]),
            }
        )
        with self.manifest_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, default=str) + "\n")
        return path

    def load(self, embedding_id: str) -> np.ndarray:
        return np.load(self.embedding_path(embedding_id))

    def read_manifest(self) -> pd.DataFrame:
        if not self.manifest_path.exists():
            return pd.DataFrame()
        records = [json.loads(line) for line in self.manifest_path.read_text(encoding="utf-8").splitlines() if line]
        return pd.DataFrame(records)

    def existing_ids(self) -> set[str]:
        manifest = self.read_manifest()
        if manifest.empty or "embedding_id" not in manifest.columns:
            return set()
        return set(manifest["embedding_id"].astype(str))


def load_embedding_matrix(paths: Iterable[str | Path]) -> np.ndarray:
    """Load a list of per-chunk `.npy` embeddings into a matrix."""

    arrays = [np.load(Path(path)) for path in paths]
    if not arrays:
        return np.empty((0, 0), dtype=np.float32)
    return np.vstack([np.asarray(arr, dtype=np.float32).reshape(1, -1) for arr in arrays])
