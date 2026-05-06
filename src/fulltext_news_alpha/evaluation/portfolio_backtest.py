"""Simple cross-sectional decile and long-short factor backtests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def assign_deciles(group: pd.DataFrame, factor_col: str, n_deciles: int = 10) -> pd.Series:
    valid = group[factor_col].rank(method="first")
    if valid.notna().sum() < n_deciles:
        return pd.Series(pd.NA, index=group.index, dtype="Int64")
    return pd.qcut(valid, q=n_deciles, labels=False, duplicates="drop") + 1


def decile_returns(
    frame: pd.DataFrame,
    factor_col: str = "FullTextNewsAlpha_zscore",
    return_col: str = "future_20d_market_adjusted_return",
    n_deciles: int = 10,
) -> pd.DataFrame:
    out = frame.copy()
    out["decile"] = pd.NA
    for _, group in out.groupby("date"):
        out.loc[group.index, "decile"] = assign_deciles(group, factor_col, n_deciles=n_deciles)
    return (
        out.dropna(subset=["decile", return_col])
        .groupby(["date", "decile"], as_index=False)[return_col]
        .mean()
        .rename(columns={return_col: "return"})
    )


def long_short_returns(
    frame: pd.DataFrame,
    factor_col: str = "FullTextNewsAlpha_zscore",
    return_col: str = "future_20d_market_adjusted_return",
    n_deciles: int = 10,
) -> pd.DataFrame:
    deciles = decile_returns(frame, factor_col=factor_col, return_col=return_col, n_deciles=n_deciles)
    if deciles.empty:
        return pd.DataFrame(columns=["date", "long_return", "short_return", "long_short_return", "cumulative_return"])
    wide = deciles.pivot(index="date", columns="decile", values="return")
    out = pd.DataFrame(index=wide.index)
    out["long_return"] = wide.get(n_deciles)
    out["short_return"] = wide.get(1)
    out["long_short_return"] = out["long_return"] - out["short_return"]
    out["cumulative_return"] = (1.0 + out["long_short_return"].fillna(0.0)).cumprod() - 1.0
    return out.reset_index()


def performance_summary(returns: pd.Series, periods_per_year: int = 252) -> dict[str, float]:
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {"annualized_return": np.nan, "Sharpe": np.nan, "max_drawdown": np.nan}
    cumulative = (1.0 + clean).cumprod()
    ann_return = float(cumulative.iloc[-1] ** (periods_per_year / len(clean)) - 1.0)
    vol = clean.std(ddof=1)
    sharpe = float(clean.mean() / vol * np.sqrt(periods_per_year)) if vol and np.isfinite(vol) else np.nan
    drawdown = cumulative / cumulative.cummax() - 1.0
    return {
        "annualized_return": ann_return,
        "Sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
    }
