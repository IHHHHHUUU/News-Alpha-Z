"""TCN-based sequence models for B2/B3 x B4/B5 experiments."""

from __future__ import annotations

from typing import Any

import numpy as np

from fulltext_news_alpha.models._torch import require_torch
from fulltext_news_alpha.models.chunk_attention_pooler import ChunkAttentionPooler
from fulltext_news_alpha.models.factor_branch import FactorBranch
from fulltext_news_alpha.models.fusion_branch import FusionBranch
from fulltext_news_alpha.models.mixture_gate import MixtureGate
from fulltext_news_alpha.models.news_bottleneck import NewsBottleneck
from fulltext_news_alpha.models.tcn_backbone import TCNBackbone

torch, nn = require_torch()


class FactorSequenceEncoder(nn.Module):
    """Encode 30 trading days of price-volume factors."""

    def __init__(
        self,
        factor_dim: int,
        hidden_dim: int = 128,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.tcn = TCNBackbone(
            input_dim=factor_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
        )

    def forward(self, factor_seq, sequence_mask=None):
        _, factor_state = self.tcn(factor_seq, sequence_mask=sequence_mask)
        return factor_state


class NewsSequenceEncoder(nn.Module):
    """Pool daily news, apply a bottleneck per day, then encode with a TCN."""

    def __init__(
        self,
        news_pooling: str,
        embedding_dim: int,
        factor_dim: int,
        daily_news_dim: int = 64,
        hidden_dim: int = 128,
        bottleneck_hidden_dim: int = 256,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.1,
        chunk_index: Any | None = None,
        max_chunks_per_stock_day: int = 64,
    ) -> None:
        super().__init__()
        if news_pooling not in {"b2", "b3"}:
            raise ValueError("news_pooling must be 'b2' or 'b3'")
        self.news_pooling = news_pooling
        self.embedding_dim = int(embedding_dim)
        self.daily_news_dim = int(daily_news_dim)
        self.max_chunks_per_stock_day = int(max_chunks_per_stock_day)
        self.attention_pooler = (
            ChunkAttentionPooler(
                chunk_dim=self.embedding_dim,
                stock_dim=int(factor_dim),
                hidden_dim=hidden_dim,
            )
            if news_pooling == "b3"
            else None
        )
        self.bottleneck = NewsBottleneck(
            input_dim=self.embedding_dim,
            output_dim=self.daily_news_dim,
            hidden_dim=bottleneck_hidden_dim,
            dropout=dropout,
        )
        self.tcn = TCNBackbone(
            input_dim=self.daily_news_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
        )
        self._register_stockday_cache(chunk_index)

    def _register_stockday_cache(self, chunk_index: Any | None) -> None:
        if self.news_pooling != "b3" or chunk_index is None:
            return
        memmap_path = getattr(chunk_index, "memmap_path", None)
        stockday_offsets = getattr(chunk_index, "stockday_offsets", ())
        chunk_count = int(getattr(chunk_index, "chunk_count", 0))
        if memmap_path is None or not stockday_offsets or chunk_count <= 0:
            return

        matrix = np.memmap(
            memmap_path,
            dtype=np.float32,
            mode="r",
            shape=(chunk_count, self.embedding_dim),
        )
        cache = torch.zeros(
            (len(stockday_offsets) + 1, self.max_chunks_per_stock_day, self.embedding_dim),
            dtype=torch.float32,
        )
        mask = torch.zeros((len(stockday_offsets) + 1, self.max_chunks_per_stock_day), dtype=torch.bool)
        for stockday_id, (offset, length) in enumerate(stockday_offsets, start=1):
            if length <= 0:
                continue
            values = torch.from_numpy(np.asarray(matrix[offset : offset + length], dtype=np.float32).copy())
            cache[stockday_id, :length] = values
            mask[stockday_id, :length] = True
        self.register_buffer("stockday_chunk_cache", cache, persistent=False)
        self.register_buffer("stockday_chunk_mask_cache", mask, persistent=False)

    def _daily_pooled_news(self, batch: dict[str, Any]) -> dict[str, Any]:
        factor_seq = batch["factor_seq"]
        if self.news_pooling == "b2":
            return {"pooled_news_seq": batch["news_seq"]}

        if "stockday_id_seq" in batch:
            if not hasattr(self, "stockday_chunk_cache") or not hasattr(self, "stockday_chunk_mask_cache"):
                raise RuntimeError("stockday_id_seq was provided, but no stock-day chunk cache is loaded")
            stockday_ids = batch["stockday_id_seq"].long()
            chunk_seq = self.stockday_chunk_cache[stockday_ids]
            if chunk_seq.dtype != factor_seq.dtype:
                chunk_seq = chunk_seq.to(factor_seq.dtype)
            chunk_mask_seq = self.stockday_chunk_mask_cache[stockday_ids]
        else:
            chunk_seq = batch["chunk_seq"]
            chunk_mask_seq = batch["chunk_mask_seq"]
        batch_size, seq_len, max_chunks, chunk_dim = chunk_seq.shape
        flat_chunks = chunk_seq.reshape(batch_size * seq_len, max_chunks, chunk_dim)
        flat_mask = chunk_mask_seq.reshape(batch_size * seq_len, max_chunks)
        flat_stock = factor_seq.reshape(batch_size * seq_len, factor_seq.shape[-1])
        if self.attention_pooler is None:
            raise RuntimeError("attention_pooler is required for B3 news pooling")
        pooled, weights, entropy = self.attention_pooler(flat_chunks, flat_stock, mask=flat_mask)
        return {
            "pooled_news_seq": pooled.reshape(batch_size, seq_len, chunk_dim),
            "attention_weights_seq": weights.reshape(batch_size, seq_len, max_chunks),
            "attention_entropy_seq": entropy.reshape(batch_size, seq_len),
        }

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        pooled = self._daily_pooled_news(batch)
        news_seq = pooled["pooled_news_seq"]
        batch_size, seq_len, embedding_dim = news_seq.shape
        daily_repr = self.bottleneck(news_seq.reshape(batch_size * seq_len, embedding_dim))
        daily_repr = daily_repr.reshape(batch_size, seq_len, self.daily_news_dim)
        _, news_state = self.tcn(daily_repr, sequence_mask=batch.get("sequence_mask"))
        outputs = {"full_text_news_repr": news_state}
        if "attention_entropy_seq" in pooled:
            outputs["attention_entropy"] = pooled["attention_entropy_seq"][:, -1]
            outputs["target_attention_weights"] = pooled["attention_weights_seq"][:, -1, :]
        return outputs


class TemporalMixtureModel(nn.Module):
    """B4 conventional mixture model with factor/news TCN streams."""

    def __init__(
        self,
        news_pooling: str,
        factor_dim: int,
        embedding_dim: int,
        daily_news_dim: int = 64,
        hidden_dim: int = 128,
        bottleneck_hidden_dim: int = 256,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.1,
        chunk_index: Any | None = None,
        max_chunks_per_stock_day: int = 64,
    ) -> None:
        super().__init__()
        self.factor_encoder = FactorSequenceEncoder(
            factor_dim=factor_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
        )
        self.news_encoder = NewsSequenceEncoder(
            news_pooling=news_pooling,
            embedding_dim=embedding_dim,
            factor_dim=factor_dim,
            daily_news_dim=daily_news_dim,
            hidden_dim=hidden_dim,
            bottleneck_hidden_dim=bottleneck_hidden_dim,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
            chunk_index=chunk_index,
            max_chunks_per_stock_day=max_chunks_per_stock_day,
        )
        self.factor_branch = FactorBranch(factor_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.fusion_branch = FusionBranch(
            factor_dim=hidden_dim,
            news_dim=hidden_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.gate = MixtureGate(
            factor_dim=hidden_dim,
            news_dim=hidden_dim,
            hidden_dim=max(1, hidden_dim // 2),
            dropout=dropout,
        )

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        factor_state = self.factor_encoder(batch["factor_seq"], sequence_mask=batch.get("sequence_mask"))
        news_outputs = self.news_encoder(batch)
        news_state = news_outputs["full_text_news_repr"]
        factor_only_pred = self.factor_branch(factor_state)
        fusion_pred_raw = self.fusion_branch(factor_state, news_state)
        gate_prob_raw = self.gate(factor_state, news_state)
        mask = batch["has_news"].clamp(0.0, 1.0).to(factor_state.dtype)
        fusion_pred = mask * fusion_pred_raw + (1.0 - mask) * factor_only_pred
        gate_news_prob = mask * gate_prob_raw
        mixed_pred = (1.0 - gate_news_prob) * factor_only_pred + gate_news_prob * fusion_pred
        outputs = {
            "factor_only_pred": factor_only_pred,
            "fusion_pred": fusion_pred,
            "gate_news_prob": gate_news_prob,
            "mixed_pred": mixed_pred,
            "full_text_news_repr": news_state,
            "factor_state": factor_state,
        }
        outputs.update({key: value for key, value in news_outputs.items() if key not in outputs})
        return outputs


class TemporalFactorModel(nn.Module):
    """B5 stage-1 factor-only model."""

    def __init__(self, factor_dim: int, hidden_dim: int = 128, kernel_size: int = 3, dilations=(1, 2, 4, 8), dropout: float = 0.1) -> None:
        super().__init__()
        self.factor_encoder = FactorSequenceEncoder(
            factor_dim=factor_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            dilations=tuple(int(value) for value in dilations),
            dropout=dropout,
        )
        self.factor_branch = FactorBranch(factor_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        factor_state = self.factor_encoder(batch["factor_seq"], sequence_mask=batch.get("sequence_mask"))
        return {"factor_only_pred": self.factor_branch(factor_state), "factor_state": factor_state}


class TemporalFusionModel(nn.Module):
    """B5 stage-2 factor-news fusion model."""

    def __init__(
        self,
        news_pooling: str,
        factor_dim: int,
        embedding_dim: int,
        daily_news_dim: int = 64,
        hidden_dim: int = 128,
        bottleneck_hidden_dim: int = 256,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.1,
        chunk_index: Any | None = None,
        max_chunks_per_stock_day: int = 64,
    ) -> None:
        super().__init__()
        self.factor_encoder = FactorSequenceEncoder(
            factor_dim=factor_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
        )
        self.news_encoder = NewsSequenceEncoder(
            news_pooling=news_pooling,
            embedding_dim=embedding_dim,
            factor_dim=factor_dim,
            daily_news_dim=daily_news_dim,
            hidden_dim=hidden_dim,
            bottleneck_hidden_dim=bottleneck_hidden_dim,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
            chunk_index=chunk_index,
            max_chunks_per_stock_day=max_chunks_per_stock_day,
        )
        self.fusion_branch = FusionBranch(
            factor_dim=hidden_dim,
            news_dim=hidden_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        factor_state = self.factor_encoder(batch["factor_seq"], sequence_mask=batch.get("sequence_mask"))
        news_outputs = self.news_encoder(batch)
        news_state = news_outputs["full_text_news_repr"]
        outputs = {
            "fusion_pred": self.fusion_branch(factor_state, news_state),
            "full_text_news_repr": news_state,
            "factor_state": factor_state,
        }
        outputs.update({key: value for key, value in news_outputs.items() if key not in outputs})
        return outputs
