"""Model modules for factor/news fusion."""

from __future__ import annotations

try:
    from fulltext_news_alpha.models.full_model import FullTextNewsAlphaModel
except RuntimeError:
    FullTextNewsAlphaModel = None  # type: ignore[assignment]

__all__ = ["FullTextNewsAlphaModel"]
