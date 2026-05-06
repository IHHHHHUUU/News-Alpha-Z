"""Combined full-text news alpha model."""

from __future__ import annotations

from fulltext_news_alpha.models._torch import require_torch
from fulltext_news_alpha.models.chunk_attention_pooler import ChunkAttentionPooler
from fulltext_news_alpha.models.factor_branch import FactorBranch
from fulltext_news_alpha.models.fusion_branch import FusionBranch
from fulltext_news_alpha.models.mixture_gate import MixtureGate
from fulltext_news_alpha.models.news_bottleneck import NewsBottleneck

torch, nn = require_torch()


class FullTextNewsAlphaModel(nn.Module):
    """Stock-aware attention + RAM-style factor/fusion/gate model."""

    def __init__(
        self,
        factor_dim: int,
        chunk_dim: int,
        stock_dim: int,
        news_dim: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.pooler = ChunkAttentionPooler(chunk_dim=chunk_dim, stock_dim=stock_dim, hidden_dim=hidden_dim)
        self.news_bottleneck = NewsBottleneck(input_dim=chunk_dim, output_dim=news_dim)
        self.factor_branch = FactorBranch(factor_dim=factor_dim, hidden_dim=hidden_dim)
        self.fusion_branch = FusionBranch(factor_dim=factor_dim, news_dim=news_dim, hidden_dim=hidden_dim)
        self.gate = MixtureGate(factor_dim=factor_dim, news_dim=news_dim, hidden_dim=hidden_dim // 2)

    def forward(self, factors, chunk_embeddings, stock_vector, chunk_mask=None):
        pooled_news, attention_weights, attention_entropy = self.pooler(
            chunk_embeddings, stock_vector, mask=chunk_mask
        )
        full_text_news_repr = self.news_bottleneck(pooled_news)
        factor_only_pred = self.factor_branch(factors)
        fusion_pred = self.fusion_branch(factors, full_text_news_repr)
        gate_news_prob = self.gate(factors, full_text_news_repr)
        mixed_pred = (1.0 - gate_news_prob) * factor_only_pred + gate_news_prob * fusion_pred
        return {
            "factor_only_pred": factor_only_pred,
            "fusion_pred": fusion_pred,
            "gate_news_prob": gate_news_prob,
            "mixed_pred": mixed_pred,
            "FullTextNewsAlpha": gate_news_prob * (fusion_pred - factor_only_pred),
            "full_text_news_repr": full_text_news_repr,
            "attention_weights": attention_weights,
            "attention_entropy": attention_entropy,
        }
