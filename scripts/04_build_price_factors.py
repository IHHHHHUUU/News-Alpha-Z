from pathlib import Path

import pandas as pd

from fulltext_news_alpha.features.price_volume_factors import build_price_volume_factors


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build price-volume factors.")
    parser.add_argument("--prices", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prices = pd.read_parquet(args.prices)
    factors = build_price_volume_factors(prices)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    factors.to_parquet(args.output, index=False)


if __name__ == "__main__":
    main()
