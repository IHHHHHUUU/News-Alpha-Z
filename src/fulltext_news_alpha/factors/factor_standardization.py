"""Cross-sectional factor winsorization and z-score utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize_series(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Clip a series to cross-sectional quantiles, preserving missing values."""

    clean = series.dropna()
    if clean.empty:
        return series.copy()
    lo = clean.quantile(lower)
    hi = clean.quantile(upper)
    return series.clip(lower=lo, upper=hi)


def zscore_series(series: pd.Series) -> pd.Series:
    """Z-score a series with stable handling for zero or undefined dispersion."""

    mean = series.mean(skipna=True)
    std = series.std(skipna=True, ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.zeros(len(series), dtype=float), index=series.index)
    return (series - mean) / std


def standardize_by_date(
    frame: pd.DataFrame,
    value_col: str,
    date_col: str = "date",
    output_col: str | None = None,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.DataFrame:
    """Winsorize then z-score a factor within each date cross-section."""

    if value_col not in frame.columns:
        raise KeyError(f"Missing value column: {value_col}")
    out = frame.copy()
    output_col = output_col or f"{value_col}_zscore"

    def _standardize(series: pd.Series) -> pd.Series:
        return zscore_series(winsorize_series(series, lower=lower, upper=upper))

    out[output_col] = out.groupby(date_col)[value_col].transform(_standardize)
    return out
