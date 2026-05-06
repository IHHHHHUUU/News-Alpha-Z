"""RAM-style factor-only prediction branch."""

from __future__ import annotations

from fulltext_news_alpha.models._torch import require_torch

_, nn = require_torch()


class FactorBranch(nn.Module):
    def __init__(self, factor_dim: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(factor_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, factors):
        return self.net(factors).squeeze(-1)
