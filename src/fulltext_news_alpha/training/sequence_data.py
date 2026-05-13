"""Sequence datasets for temporal B2/B3 news-alpha training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fulltext_news_alpha.training.torch_utils import torch, to_float_tensor


StockDayKey = tuple[str, object]


@dataclass(frozen=True)
class ChunkEmbeddingIndex:
    """Chunk embedding paths keyed by ``(ticker, date)``."""

    paths_by_key: dict[StockDayKey, tuple[Path, ...]]
    embedding_dim: int


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def _read_manifest_jsonl(path: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            record["_manifest_dir"] = str(path.parent)
            record["_manifest_line"] = line_number
            records.append(record)
    return pd.DataFrame(records)


def load_chunk_embedding_index(
    manifest_or_root: str | Path,
    project_root: str | Path = ".",
    max_chunks_per_stock_day: int = 64,
) -> ChunkEmbeddingIndex:
    """Load a manifest path or manifest root into a chunk embedding index."""

    source = Path(manifest_or_root)
    paths = sorted(source.rglob("manifest.jsonl")) if source.is_dir() else [source]
    if not paths:
        raise FileNotFoundError(f"No manifest.jsonl files found under {source}")
    manifest = pd.concat([_read_manifest_jsonl(path) for path in paths], ignore_index=True)
    required = {"chunk_id", "ticker", "date", "embedding_path"}
    missing = required - set(manifest.columns)
    if missing:
        raise KeyError(f"Missing manifest columns: {sorted(missing)}")

    root = Path(project_root)
    rows: list[tuple[StockDayKey, Path]] = []
    embedding_dim: int | None = None
    manifest = manifest.drop_duplicates("chunk_id", keep="first").copy()
    manifest["ticker"] = manifest["ticker"].astype(str).str.upper().str.strip()
    manifest["date"] = pd.to_datetime(manifest["date"]).dt.date
    sort_cols = [col for col in ("ticker", "date", "publish_time", "source_news_id", "chunk_index") if col in manifest]
    if sort_cols:
        manifest = manifest.sort_values(sort_cols)

    for row in manifest.to_dict(orient="records"):
        raw_path = Path(str(row["embedding_path"]))
        candidates = (
            (raw_path,)
            if raw_path.is_absolute()
            else (
                root / raw_path,
                Path(str(row.get("_manifest_dir", "."))) / raw_path,
                Path(str(row.get("_manifest_dir", "."))) / raw_path.name,
                raw_path,
            )
        )
        resolved = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
        if not resolved.exists():
            continue
        if embedding_dim is None:
            embedding_dim = int(np.asarray(np.load(resolved)).reshape(-1).shape[0])
        key = (str(row["ticker"]), row["date"])
        rows.append((key, resolved))

    if embedding_dim is None:
        raise ValueError("No loadable chunk embedding files found in manifest.")

    paths_by_key: dict[StockDayKey, list[Path]] = {}
    for key, path in rows:
        paths_by_key.setdefault(key, [])
        if len(paths_by_key[key]) < int(max_chunks_per_stock_day):
            paths_by_key[key].append(path)
    return ChunkEmbeddingIndex(
        paths_by_key={key: tuple(paths) for key, paths in paths_by_key.items()},
        embedding_dim=embedding_dim,
    )


class B2SequenceStockDayDataset(torch.utils.data.Dataset):
    """Build 30-trading-day B2 windows from a stock-day panel."""

    def __init__(
        self,
        sample_frame: pd.DataFrame,
        history_frame: pd.DataFrame,
        factor_cols: list[str],
        embedding_cols: list[str],
        label_col: str,
        lookback_window: int = 30,
        has_news_col: str = "has_news",
        drop_missing_label: bool = True,
    ) -> None:
        if lookback_window <= 0:
            raise ValueError("lookback_window must be positive")
        if drop_missing_label:
            sample_frame = sample_frame.dropna(subset=[label_col]).reset_index(drop=True)
        self.factor_cols = list(factor_cols)
        self.embedding_cols = list(embedding_cols)
        self.label_col = str(label_col)
        self.has_news_col = str(has_news_col)
        self.lookback_window = int(lookback_window)

        history = _normalize_frame(history_frame)
        samples = _normalize_frame(sample_frame)
        self._history_by_ticker = {
            ticker: group.reset_index(drop=True)
            for ticker, group in history.groupby("ticker", sort=False)
        }
        self._targets: list[tuple[str, int]] = []
        for row in samples[["ticker", "date"]].to_dict(orient="records"):
            ticker = str(row["ticker"])
            group = self._history_by_ticker.get(ticker)
            if group is None:
                continue
            matches = np.flatnonzero(group["date"].to_numpy() == row["date"])
            if len(matches) == 0:
                continue
            pos = int(matches[-1])
            if pos + 1 < self.lookback_window:
                continue
            self._targets.append((ticker, pos))

        self.keys = pd.DataFrame(
            {
                "date": [self._history_by_ticker[ticker].loc[pos, "date"] for ticker, pos in self._targets],
                "ticker": [ticker for ticker, _ in self._targets],
            }
        )
        self.labels = to_float_tensor(
            np.asarray(
                [self._history_by_ticker[ticker].loc[pos, self.label_col] for ticker, pos in self._targets],
                dtype=np.float32,
            )
        )
        self.has_news = to_float_tensor(
            np.asarray(
                [self._history_by_ticker[ticker].loc[pos, self.has_news_col] for ticker, pos in self._targets],
                dtype=np.float32,
            )
        )

    def __len__(self) -> int:
        return len(self._targets)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ticker, pos = self._targets[idx]
        group = self._history_by_ticker[ticker]
        window = group.iloc[pos + 1 - self.lookback_window : pos + 1]
        return {
            "factor_seq": to_float_tensor(window[self.factor_cols].fillna(0.0).to_numpy()),
            "news_seq": to_float_tensor(window[self.embedding_cols].fillna(0.0).to_numpy()),
            "sequence_mask": torch.ones(self.lookback_window, dtype=torch.bool),
            "label": self.labels[idx],
            "has_news": self.has_news[idx],
        }


class B3SequenceStockDayDataset(B2SequenceStockDayDataset):
    """Build 30-day B3 windows and load frozen chunk embeddings on demand."""

    def __init__(
        self,
        sample_frame: pd.DataFrame,
        history_frame: pd.DataFrame,
        factor_cols: list[str],
        chunk_index: ChunkEmbeddingIndex,
        label_col: str,
        lookback_window: int = 30,
        max_chunks_per_stock_day: int = 64,
        has_news_col: str = "has_news",
        drop_missing_label: bool = True,
    ) -> None:
        self.chunk_index = chunk_index
        self.max_chunks_per_stock_day = int(max_chunks_per_stock_day)
        super().__init__(
            sample_frame=sample_frame,
            history_frame=history_frame,
            factor_cols=factor_cols,
            embedding_cols=[],
            label_col=label_col,
            lookback_window=lookback_window,
            has_news_col=has_news_col,
            drop_missing_label=drop_missing_label,
        )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ticker, pos = self._targets[idx]
        group = self._history_by_ticker[ticker]
        window = group.iloc[pos + 1 - self.lookback_window : pos + 1]
        chunk_seq = np.zeros(
            (self.lookback_window, self.max_chunks_per_stock_day, self.chunk_index.embedding_dim),
            dtype=np.float32,
        )
        chunk_mask = np.zeros((self.lookback_window, self.max_chunks_per_stock_day), dtype=bool)
        for day_idx, date_value in enumerate(window["date"].tolist()):
            paths = self.chunk_index.paths_by_key.get((ticker, date_value), ())
            for chunk_idx, path in enumerate(paths[: self.max_chunks_per_stock_day]):
                chunk_seq[day_idx, chunk_idx] = np.asarray(np.load(path), dtype=np.float32).reshape(-1)
                chunk_mask[day_idx, chunk_idx] = True
        return {
            "factor_seq": to_float_tensor(window[self.factor_cols].fillna(0.0).to_numpy()),
            "chunk_seq": torch.from_numpy(chunk_seq),
            "chunk_mask_seq": torch.from_numpy(chunk_mask),
            "sequence_mask": torch.ones(self.lookback_window, dtype=torch.bool),
            "label": self.labels[idx],
            "has_news": self.has_news[idx],
        }
