"""Construct the single FullTextNewsAlpha factor."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fulltext_news_alpha.factors.factor_standardization import standardize_by_date


def build_full_text_news_alpha(
    predictions: pd.DataFrame,
    coverage: pd.DataFrame | None = None,
    attention: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build raw and standardized FullTextNewsAlpha from branch disagreement."""

    required = {"date", "ticker", "factor_only_pred", "fusion_pred", "mixed_pred", "gate_news_prob"}
    missing = required - set(predictions.columns)
    if missing:
        raise KeyError(f"Missing prediction columns: {sorted(missing)}")
    out = predictions.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out["FullTextNewsAlpha_raw"] = out["gate_news_prob"] * (out["fusion_pred"] - out["factor_only_pred"])

    if coverage is not None:
        cov = coverage.copy()
        cov["date"] = pd.to_datetime(cov["date"]).dt.date
        out = out.merge(cov[["date", "ticker", "news_count", "chunk_count"]], on=["date", "ticker"], how="left")
    else:
        out["news_count"] = 0
        out["chunk_count"] = 0

    if attention is not None and "attention_entropy" in attention.columns:
        attn = attention[["date", "ticker", "attention_entropy"]].copy()
        attn["date"] = pd.to_datetime(attn["date"]).dt.date
        out = out.merge(attn, on=["date", "ticker"], how="left")
    else:
        out["attention_entropy"] = pd.NA

    out = standardize_by_date(
        out,
        "FullTextNewsAlpha_raw",
        output_col="FullTextNewsAlpha_zscore",
    )
    columns = [
        "date",
        "ticker",
        "FullTextNewsAlpha_raw",
        "FullTextNewsAlpha_zscore",
        "factor_only_pred",
        "fusion_pred",
        "mixed_pred",
        "gate_news_prob",
        "news_count",
        "chunk_count",
        "attention_entropy",
    ]
    return out[columns].sort_values(["date", "ticker"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Construct FullTextNewsAlpha factor table.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    preds = pd.read_parquet(args.predictions)
    factor = build_full_text_news_alpha(preds)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    factor.to_parquet(args.output, index=False)


if __name__ == "__main__":
    main()
