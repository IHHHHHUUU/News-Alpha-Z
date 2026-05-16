"""Construct the single FullTextNewsAlpha factor."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import pandas as pd

from fulltext_news_alpha.factors.factor_standardization import standardize_by_date


def _normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.loc[:, "date"] = pd.Series(pd.to_datetime(out["date"])).dt.date
    out.loc[:, "ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    return out


def build_full_text_news_alpha(
    predictions: pd.DataFrame,
    coverage: pd.DataFrame | None = None,
    attention: pd.DataFrame | None = None,
    label_frame: pd.DataFrame | None = None,
    label_col: str | None = None,
) -> pd.DataFrame:
    """Build raw and standardized FullTextNewsAlpha from branch disagreement."""

    required = {"date", "ticker", "factor_only_pred", "fusion_pred", "mixed_pred", "gate_news_prob"}
    missing = required - set(predictions.columns)
    if missing:
        raise KeyError(f"Missing prediction columns: {sorted(missing)}")
    out = _normalize_keys(predictions)
    out["FullTextNewsAlpha_raw"] = out["gate_news_prob"] * (out["fusion_pred"] - out["factor_only_pred"])

    if coverage is not None:
        cov = _normalize_keys(coverage)
        out = out.merge(cov[["date", "ticker", "news_count", "chunk_count"]], on=["date", "ticker"], how="left")
    else:
        out["news_count"] = 0
        out["chunk_count"] = 0

    if "attention_entropy" not in out.columns and attention is not None and "attention_entropy" in attention.columns:
        attn = _normalize_keys(cast(pd.DataFrame, attention[["date", "ticker", "attention_entropy"]]))
        out = out.merge(attn, on=["date", "ticker"], how="left")
    if "attention_entropy" not in out.columns:
        out["attention_entropy"] = pd.NA

    if label_frame is not None and label_col is not None:
        if label_col not in label_frame.columns:
            raise KeyError(f"Panel is missing requested label column: {label_col}")
        labels = _normalize_keys(cast(pd.DataFrame, label_frame[["date", "ticker", label_col]]))
        labels = labels.drop_duplicates(subset=["date", "ticker"])
        out = out.merge(labels, on=["date", "ticker"], how="left", validate="one_to_one")
        label_coverage = float(out[label_col].notna().mean()) if len(out) else 0.0
        if label_coverage < 0.95:
            print(
                f"warning: label coverage for {label_col} after merge is {label_coverage:.2%}",
                flush=True,
            )

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
    if label_col is not None and label_col in out.columns:
        columns.append(label_col)
    return out[columns].sort_values(by=["date", "ticker"]).reset_index(drop=True)  # type: ignore[call-overload]


def main() -> None:
    parser = argparse.ArgumentParser(description="Construct FullTextNewsAlpha factor table.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--panel", default=None, help="Optional panel parquet to merge labels/coverage.")
    parser.add_argument("--label-col", default="future_20d_market_adjusted_return")
    args = parser.parse_args()
    preds = pd.read_parquet(args.predictions)
    panel = pd.read_parquet(args.panel) if args.panel else None
    factor = build_full_text_news_alpha(
        preds,
        coverage=panel,
        label_frame=panel,
        label_col=args.label_col if panel is not None else None,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    factor.to_parquet(args.output, index=False)


if __name__ == "__main__":
    main()
