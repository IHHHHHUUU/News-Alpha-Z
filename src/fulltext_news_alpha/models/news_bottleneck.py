"""Two-layer bottleneck MLP mapping pooled news vectors to ``full_text_news_repr``.

The bottleneck is the only learnable dimensionality reducer in this project. It
sits between mean-pooled or attention-pooled chunk embeddings and the downstream
factor-news fusion branch, so its parameters are trained jointly with the fusion
head.

Architecture (matches the project specification):

    LayerNorm(input_dim)
    Linear(input_dim -> hidden_dim)
    GELU
    Dropout(p)
    Linear(hidden_dim -> output_dim)
    LayerNorm(output_dim)
"""

from __future__ import annotations

from fulltext_news_alpha.models._torch import require_torch

_, nn = require_torch()


class NewsBottleneck(nn.Module):
    """Map pooled full-text news vectors to ``full_text_news_repr``."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.dropout_p = float(dropout)
        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
        )

    def forward(self, pooled_news):
        return self.net(pooled_news)
