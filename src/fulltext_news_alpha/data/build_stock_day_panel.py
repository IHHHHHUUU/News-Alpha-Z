"""CLI for assembling a stock-day research panel."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fulltext_news_alpha.data.align_news_price import build_stock_day_panel
from fulltext_news_alpha.features.price_volume_factors import build_price_volume_factors
from fulltext_news_alpha.features.return_labels import add_forward_return_labels


def _read(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stock-day panel from prices and chunk coverage.")
    parser.add_argument("--prices", required=True)
    parser.add_argument("--chunks")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    prices = _read(args.prices)
    labels = add_forward_return_labels(prices)
    factors = build_price_volume_factors(prices)
    chunks = _read(args.chunks) if args.chunks else None
    panel = build_stock_day_panel(factors, labels, chunks)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(args.output, index=False)


if __name__ == "__main__":
    main()
