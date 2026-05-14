"""Factor decay diagnostics."""

from __future__ import annotations

import pandas as pd

from fulltext_news_alpha.evaluation.ic_metrics import compute_ic_by_date


def factor_decay_diagnostics(
    frame: pd.DataFrame,
    factor_col: str = "FullTextNewsAlpha_zscore",
    horizon_return_cols: tuple[str, ...] = (
        "future_5d_market_adjusted_return",
        "future_20d_market_adjusted_return",
    ),
    rebalance_every: int | None = None,
) -> pd.DataFrame:
    """Compute RankIC by available label horizon."""

    rows: list[pd.DataFrame] = []
    for return_col in horizon_return_cols:
        if return_col not in frame.columns:
            continue
        ic = compute_ic_by_date(
            frame,
            factor_col=factor_col,
            return_col=return_col,
            rebalance_every=rebalance_every,
        )
        ic["return_col"] = return_col
        rows.append(ic)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
