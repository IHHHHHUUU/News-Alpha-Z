"""Stage 2 decoupled gate target construction and deterministic gate fitting."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fulltext_news_alpha.training.train_factor_branch import fit_ridge_regression, predict_ridge


def make_gate_targets(
    factor_only_pred: np.ndarray | pd.Series,
    fusion_pred: np.ndarray | pd.Series,
    target: np.ndarray | pd.Series,
    temperature: float = 1.0,
) -> np.ndarray:
    """Compute soft targets favoring the lower-error branch.

    Output is the target probability for trusting the news fusion branch.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    factor_error = np.square(np.asarray(factor_only_pred, dtype=float) - np.asarray(target, dtype=float))
    fusion_error = np.square(np.asarray(fusion_pred, dtype=float) - np.asarray(target, dtype=float))
    logits = np.column_stack([-factor_error / temperature, -fusion_error / temperature])
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs[:, 1]


def mix_predictions(
    factor_only_pred: np.ndarray | pd.Series,
    fusion_pred: np.ndarray | pd.Series,
    gate_news_prob: np.ndarray | pd.Series,
) -> np.ndarray:
    factor_arr = np.asarray(factor_only_pred, dtype=float)
    fusion_arr = np.asarray(fusion_pred, dtype=float)
    gate_arr = np.asarray(gate_news_prob, dtype=float)
    return (1.0 - gate_arr) * factor_arr + gate_arr * fusion_arr


def train_decoupled_gate(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    gate_feature_cols: list[str],
    temperature: float = 1.0,
    l2: float = 1e-3,
    label_col: str = "target",
) -> pd.DataFrame:
    train_targets = make_gate_targets(
        train["factor_only_pred"], train["fusion_pred"], train[label_col], temperature=temperature
    )
    weights = fit_ridge_regression(train[gate_feature_cols].fillna(0.0).to_numpy(), train_targets, l2=l2)
    logits = predict_ridge(predict[gate_feature_cols].fillna(0.0).to_numpy(), weights)
    gate = 1.0 / (1.0 + np.exp(-logits))
    out = predict[["date", "ticker", "factor_only_pred", "fusion_pred"]].copy()
    out["gate_news_prob"] = gate
    out["mixed_pred"] = mix_predictions(out["factor_only_pred"], out["fusion_pred"], out["gate_news_prob"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train decoupled mixture gate.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--predict", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gate-feature-cols", nargs="+", required=True)
    parser.add_argument("--label-col", default="future_20d_market_adjusted_return")
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()
    train = pd.read_parquet(args.train).rename(columns={args.label_col: "target"})
    predict = pd.read_parquet(args.predict)
    preds = train_decoupled_gate(train, predict, args.gate_feature_cols, temperature=args.temperature)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(args.output, index=False)


if __name__ == "__main__":
    main()
