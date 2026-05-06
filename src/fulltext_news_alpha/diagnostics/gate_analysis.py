"""Gate behavior diagnostics."""

from __future__ import annotations

import pandas as pd


def summarize_gate_by_decile(frame: pd.DataFrame, factor_col: str = "FullTextNewsAlpha_zscore") -> pd.DataFrame:
    out = frame.dropna(subset=[factor_col, "gate_news_prob"]).copy()
    out["factor_decile"] = out.groupby("date")[factor_col].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop") + 1
    )
    return out.groupby("factor_decile", as_index=False)["gate_news_prob"].mean()
