"""Offline frozen FinBERT/DeBERTa chunk encoder.

The encoder is loaded lazily and defaults to ``local_files_only=True`` so this
entrypoint never downloads models unless a caller explicitly changes that flag.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fulltext_news_alpha.embeddings.embedding_store import EmbeddingStore


DEFAULT_ENCODER = "ProsusAI/finbert"


class FrozenTransformerEncoder:
    """Mean-pool the final hidden state from a frozen Hugging Face encoder."""

    def __init__(
        self,
        model_name: str = DEFAULT_ENCODER,
        max_length: int = 256,
        device: str | None = None,
        local_files_only: bool = True,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "FrozenTransformerEncoder requires optional dependencies: torch and transformers."
            ) from exc

        self.torch = torch
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        self.model.to(self.device)
        self.model.eval()
        self.model_name = model_name

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.no_grad():
            output = self.model(**encoded)
            hidden = output.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            pooled = summed / counts
        return pooled.detach().cpu().numpy().astype(np.float32)


def encode_chunks_dataframe(
    chunks: pd.DataFrame,
    store: EmbeddingStore,
    encoder: FrozenTransformerEncoder,
    batch_size: int = 32,
    text_col: str = "chunk_text",
    id_col: str = "chunk_id",
) -> pd.DataFrame:
    """Encode chunks, skipping IDs already present in the manifest."""

    required = {text_col, id_col, "ticker", "date"}
    missing = required - set(chunks.columns)
    if missing:
        raise KeyError(f"Missing chunk columns: {sorted(missing)}")

    existing = store.existing_ids()
    records: list[dict[str, object]] = []
    pending = chunks[~chunks[id_col].astype(str).isin(existing)].copy()
    for start in range(0, len(pending), batch_size):
        batch = pending.iloc[start : start + batch_size]
        embeddings = encoder.encode_batch(batch[text_col].fillna("").astype(str).tolist())
        for (_, row), embedding in zip(batch.iterrows(), embeddings, strict=True):
            embedding_id = str(row[id_col])
            path = store.save(
                embedding_id,
                embedding,
                {
                    "chunk_id": row[id_col],
                    "ticker": row["ticker"],
                    "date": row["date"],
                    "model_name": encoder.model_name,
                },
            )
            record = row.to_dict()
            record["embedding_path"] = str(path)
            records.append(record)
    if records:
        return pd.DataFrame(records)
    return chunks.iloc[0:0].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline encode news chunks with a frozen transformer.")
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--output-dir", default="data/embeddings/chunks")
    parser.add_argument("--model-name", default=DEFAULT_ENCODER)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    chunks_path = Path(args.chunks)
    chunks = pd.read_parquet(chunks_path) if chunks_path.suffix == ".parquet" else pd.read_csv(chunks_path)
    encoder = FrozenTransformerEncoder(
        model_name=args.model_name,
        max_length=args.max_length,
        local_files_only=not args.allow_download,
    )
    encoded = encode_chunks_dataframe(chunks, EmbeddingStore(args.output_dir), encoder, batch_size=args.batch_size)
    print(f"Encoded new chunks: {len(encoded)}")


if __name__ == "__main__":
    main()
