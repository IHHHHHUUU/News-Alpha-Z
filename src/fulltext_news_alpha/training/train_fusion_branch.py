"""Stage 1 factor-news fusion branch training."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fulltext_news_alpha.training.train_factor_branch import fit_ridge_regression, predict_ridge


def train_fusion_branch(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    factor_cols: list[str],
    news_repr_cols: list[str],
    label_col: str,
    l2: float = 1e-3,
) -> pd.DataFrame:
    cols = factor_cols + news_repr_cols
    weights = fit_ridge_regression(train[cols].fillna(0.0).to_numpy(), train[label_col].fillna(0.0).to_numpy(), l2=l2)
    out = predict[["date", "ticker"]].copy()
    out["fusion_pred"] = predict_ridge(predict[cols].fillna(0.0).to_numpy(), weights)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train deterministic factor-news ridge branch.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--predict", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label-col", default="future_20d_market_adjusted_return")
    parser.add_argument("--factor-cols", nargs="+", required=True)
    parser.add_argument("--news-repr-cols", nargs="+", required=True)
    args = parser.parse_args()
    train = pd.read_parquet(args.train)
    predict = pd.read_parquet(args.predict)
    preds = train_fusion_branch(train, predict, args.factor_cols, args.news_repr_cols, args.label_col)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(args.output, index=False)


if __name__ == "__main__":
    main()
