"""64-dimensional news representation bottleneck."""

from __future__ import annotations

from fulltext_news_alpha.models._torch import require_torch

_, nn = require_torch()


class NewsBottleneck(nn.Module):
    """Map pooled full-text news vectors to `full_text_news_repr`."""

    def __init__(self, input_dim: int, output_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
        )

    def forward(self, pooled_news):
        return self.net(pooled_news)
