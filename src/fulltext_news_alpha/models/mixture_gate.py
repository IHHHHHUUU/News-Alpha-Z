"""Decoupled mixture gate that estimates news-branch reliability."""

from __future__ import annotations

from fulltext_news_alpha.models._torch import require_torch

torch, nn = require_torch()


class MixtureGate(nn.Module):
    def __init__(self, factor_dim: int, news_dim: int = 64, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(factor_dim + news_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, factors, full_text_news_repr):
        logits = self.net(torch.cat([factors, full_text_news_repr], dim=-1)).squeeze(-1)
        return torch.sigmoid(logits)
