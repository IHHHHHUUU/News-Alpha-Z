"""Stage 1 factor-only branch training.

The production neural branch lives under ``models/``. This module also provides
a small deterministic ridge baseline for smoke tests and interface validation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def fit_ridge_regression(x: np.ndarray, y: np.ndarray, l2: float = 1e-3) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(design.shape[1]) * l2
    penalty[0, 0] = 0.0
    return np.linalg.pinv(design.T @ design + penalty) @ design.T @ y


def predict_ridge(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), np.asarray(x, dtype=float)])
    return design @ weights


def train_factor_branch(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    factor_cols: list[str],
    label_col: str,
    l2: float = 1e-3,
) -> pd.DataFrame:
    weights = fit_ridge_regression(train[factor_cols].fillna(0.0).to_numpy(), train[label_col].fillna(0.0).to_numpy(), l2=l2)
    out = predict[["date", "ticker"]].copy()
    out["factor_only_pred"] = predict_ridge(predict[factor_cols].fillna(0.0).to_numpy(), weights)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train deterministic factor-only ridge branch.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--predict", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label-col", default="future_20d_market_adjusted_return")
    parser.add_argument("--factor-cols", nargs="+", required=True)
    args = parser.parse_args()
    train = pd.read_parquet(args.train)
    predict = pd.read_parquet(args.predict)
    preds = train_factor_branch(train, predict, args.factor_cols, args.label_col)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(args.output, index=False)


if __name__ == "__main__":
    main()
