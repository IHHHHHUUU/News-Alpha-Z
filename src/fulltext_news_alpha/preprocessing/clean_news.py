"""Utilities for cleaning financial news text before chunking."""

from __future__ import annotations

import html
import re


_WHITESPACE_RE = re.compile(r"\s+")
_BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*Reuters\s*[-:]?\s*", re.IGNORECASE),
    re.compile(r"\(Reuters\)\s*[-:]?\s*", re.IGNORECASE),
]


def clean_news_text(text: str | None) -> str:
    """Normalize one full-text news body while preserving semantic content."""

    if text is None:
        return ""
    cleaned = html.unescape(str(text))
    cleaned = cleaned.replace("\u00a0", " ")
    for pattern in _BOILERPLATE_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned)
    return cleaned.strip()
