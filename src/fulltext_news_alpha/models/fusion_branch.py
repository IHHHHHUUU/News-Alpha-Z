"""RAM-style factor-news fusion prediction branch."""

from __future__ import annotations

from fulltext_news_alpha.models._torch import require_torch

torch, nn = require_torch()


class FusionBranch(nn.Module):
    def __init__(self, factor_dim: int, news_dim: int = 64, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(factor_dim + news_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, factors, full_text_news_repr):
        return self.net(torch.cat([factors, full_text_news_repr], dim=-1)).squeeze(-1)
