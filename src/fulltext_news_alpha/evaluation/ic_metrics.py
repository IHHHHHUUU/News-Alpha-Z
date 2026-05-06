"""IC, RankIC, and rolling diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _corr(group: pd.DataFrame, factor_col: str, return_col: str, method: str) -> float:
    valid = group[[factor_col, return_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 2 or valid[factor_col].nunique() < 2 or valid[return_col].nunique() < 2:
        return np.nan
    return float(valid[factor_col].corr(valid[return_col], method=method))


def compute_ic_by_date(
    frame: pd.DataFrame,
    factor_col: str = "FullTextNewsAlpha_zscore",
    return_col: str = "future_20d_market_adjusted_return",
) -> pd.DataFrame:
    """Compute daily Pearson IC and Spearman RankIC."""

    rows: list[dict[str, object]] = []
    for date, group in frame.groupby("date"):
        rows.append(
            {
                "date": date,
                "IC": _corr(group, factor_col, return_col, "pearson"),
                "RankIC": _corr(group, factor_col, return_col, "spearman"),
                "coverage": int(group[[factor_col, return_col]].dropna().shape[0]),
            }
        )
    return pd.DataFrame(rows)


def summarize_ic(ic_frame: pd.DataFrame) -> dict[str, float]:
    """Summarize IC and RankIC with information ratios."""

    summary: dict[str, float] = {}
    for col in ["IC", "RankIC"]:
        series = ic_frame[col].dropna()
        mean = float(series.mean()) if not series.empty else np.nan
        std = float(series.std(ddof=1)) if len(series) > 1 else np.nan
        summary[col] = mean
        summary[f"{col}IR"] = mean / std * np.sqrt(252) if std and np.isfinite(std) else np.nan
    summary["coverage"] = float(ic_frame["coverage"].mean()) if "coverage" in ic_frame else np.nan
    return summary


def rolling_rankic(
    ic_frame: pd.DataFrame,
    windows: tuple[int, ...] = (20, 60, 120),
    rankic_col: str = "RankIC",
) -> pd.DataFrame:
    """Add rolling RankIC averages for common decay windows."""

    out = ic_frame.sort_values("date").copy()
    for window in windows:
        out[f"rolling_{window}d_RankIC"] = out[rankic_col].rolling(window, min_periods=max(2, window // 4)).mean()
    return out
