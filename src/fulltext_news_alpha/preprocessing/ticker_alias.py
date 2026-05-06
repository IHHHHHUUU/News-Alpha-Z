"""Ticker and company-name alias helpers."""

from __future__ import annotations

import re


def normalize_ticker(ticker: str) -> str:
    """Normalize ticker symbols while preserving class-share dots as hyphens."""

    return re.sub(r"\s+", "", str(ticker).upper()).replace(".", "-")


def build_alias_map(rows: list[tuple[str, str]]) -> dict[str, str]:
    """Build a lowercase company/ticker alias map to canonical tickers."""

    aliases: dict[str, str] = {}
    for ticker, company in rows:
        canonical = normalize_ticker(ticker)
        aliases[canonical.lower()] = canonical
        if company:
            aliases[str(company).strip().lower()] = canonical
    return aliases
