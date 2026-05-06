"""Stock-aware chunk attention pooling inspired by news-fusion literature."""

from __future__ import annotations

from fulltext_news_alpha.models._torch import require_torch

torch, nn = require_torch()


class ChunkAttentionPooler(nn.Module):
    """Pool stock-day chunk embeddings using a stock/company query vector."""

    def __init__(self, chunk_dim: int, stock_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.chunk_proj = nn.Linear(chunk_dim, hidden_dim)
        self.stock_proj = nn.Linear(stock_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, chunk_embeddings, stock_vector, mask=None):
        """Return pooled vector, attention weights, and attention entropy."""

        if chunk_embeddings.ndim != 3:
            raise ValueError("chunk_embeddings must have shape [batch, chunks, dim]")
        query = self.stock_proj(stock_vector).unsqueeze(1)
        keys = self.chunk_proj(chunk_embeddings)
        scores = self.score(torch.tanh(keys + query)).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        if mask is not None:
            weights = weights * mask.float()
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        pooled = torch.sum(chunk_embeddings * weights.unsqueeze(-1), dim=1)
        entropy = -(weights.clamp(min=1e-8) * weights.clamp(min=1e-8).log()).sum(dim=-1)
        return pooled, weights, entropy
