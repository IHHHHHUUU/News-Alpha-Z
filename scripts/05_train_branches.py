"""Train Stage 1 branch baselines from prepared feature tables."""

from pathlib import Path

import pandas as pd

from fulltext_news_alpha.training.train_factor_branch import train_factor_branch
from fulltext_news_alpha.training.train_fusion_branch import train_fusion_branch


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train factor-only and factor-news branches.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--predict", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label-col", default="future_20d_market_adjusted_return")
    parser.add_argument("--factor-cols", nargs="+", required=True)
    parser.add_argument("--news-repr-cols", nargs="+", required=True)
    args = parser.parse_args()

    train = pd.read_parquet(args.train)
    predict = pd.read_parquet(args.predict)
    factor = train_factor_branch(train, predict, args.factor_cols, args.label_col)
    fusion = train_fusion_branch(train, predict, args.factor_cols, args.news_repr_cols, args.label_col)
    out = factor.merge(fusion, on=["date", "ticker"], how="inner")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)


if __name__ == "__main__":
    main()
