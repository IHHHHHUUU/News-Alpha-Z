"""IC, RankIC, and rolling diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fulltext_news_alpha.evaluation.portfolio_backtest import filter_rebalance_dates


def _corr(group: pd.DataFrame, factor_col: str, return_col: str, method: str) -> float:
    valid = group[[factor_col, return_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 2 or valid[factor_col].nunique() < 2 or valid[return_col].nunique() < 2:
        return np.nan
    return float(valid[factor_col].corr(valid[return_col], method=method))


def compute_ic_by_date(
    frame: pd.DataFrame,
    factor_col: str = "FullTextNewsAlpha_zscore",
    return_col: str = "future_20d_market_adjusted_return",
    rebalance_every: int | None = None,
) -> pd.DataFrame:
    """Compute daily Pearson IC and Spearman RankIC."""

    frame = filter_rebalance_dates(frame, rebalance_every=rebalance_every)
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


def compute_daily_ic_diagnostics(
    frame: pd.DataFrame,
    factor_col: str = "FullTextNewsAlpha_zscore",
    return_col: str = "future_20d_market_adjusted_return",
    rebalance_every: int | None = None,
) -> pd.DataFrame:
    """Return per-date IC inputs and degeneracy diagnostics."""

    frame = filter_rebalance_dates(frame, rebalance_every=rebalance_every)
    rows: list[dict[str, object]] = []
    for date, group in frame.groupby("date"):
        valid = group[[factor_col, return_col]].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(
            {
                "date": date,
                "row_count": int(len(group)),
                "valid_count": int(len(valid)),
                "factor_nunique": int(valid[factor_col].nunique()) if len(valid) else 0,
                "return_nunique": int(valid[return_col].nunique()) if len(valid) else 0,
                "factor_std": float(valid[factor_col].std(ddof=0)) if len(valid) else np.nan,
                "return_std": float(valid[return_col].std(ddof=0)) if len(valid) else np.nan,
                "IC": _corr(group, factor_col, return_col, "pearson"),
                "RankIC": _corr(group, factor_col, return_col, "spearman"),
            }
        )
    return pd.DataFrame(rows)


def summarize_factor_diagnostics(
    frame: pd.DataFrame,
    daily_diagnostics: pd.DataFrame,
    factor_col: str = "FullTextNewsAlpha_zscore",
    return_col: str = "future_20d_market_adjusted_return",
    rebalance_every: int | None = None,
) -> dict[str, float | int | str]:
    """Summarize factor table health and daily IC diagnostics."""

    filtered = filter_rebalance_dates(frame, rebalance_every=rebalance_every)
    factor_non_null = frame[factor_col].notna() if factor_col in frame else pd.Series([], dtype=bool)
    return_non_null = frame[return_col].notna() if return_col in frame else pd.Series([], dtype=bool)
    return {
        "total_rows": int(len(frame)),
        "date_count": int(pd.to_datetime(frame["date"]).nunique()) if "date" in frame else 0,
        "return_col": str(return_col),
        "factor_col": str(factor_col),
        "factor_non_null_ratio": float(factor_non_null.mean()) if len(frame) else np.nan,
        "return_non_null_ratio": float(return_non_null.mean()) if len(frame) else np.nan,
        "rows_after_rebalance": int(len(filtered)),
        "date_count_after_rebalance": int(pd.to_datetime(filtered["date"]).nunique()) if "date" in filtered else 0,
        "avg_daily_valid_count": float(daily_diagnostics["valid_count"].mean()) if len(daily_diagnostics) else np.nan,
        "min_daily_valid_count": int(daily_diagnostics["valid_count"].min()) if len(daily_diagnostics) else 0,
        "max_daily_valid_count": int(daily_diagnostics["valid_count"].max()) if len(daily_diagnostics) else 0,
        "daily_factor_std_mean": float(daily_diagnostics["factor_std"].mean()) if len(daily_diagnostics) else np.nan,
        "daily_factor_std_min": float(daily_diagnostics["factor_std"].min()) if len(daily_diagnostics) else np.nan,
        "daily_factor_std_zero_count": int((daily_diagnostics["factor_std"] == 0).sum()) if len(daily_diagnostics) else 0,
        "daily_return_std_mean": float(daily_diagnostics["return_std"].mean()) if len(daily_diagnostics) else np.nan,
        "daily_return_std_min": float(daily_diagnostics["return_std"].min()) if len(daily_diagnostics) else np.nan,
        "daily_return_std_zero_count": int((daily_diagnostics["return_std"] == 0).sum()) if len(daily_diagnostics) else 0,
        "rankic_nan_count": int(daily_diagnostics["RankIC"].isna().sum()) if len(daily_diagnostics) else 0,
        "rankic_zero_count": int((daily_diagnostics["RankIC"] == 0).sum()) if len(daily_diagnostics) else 0,
        "rankic_non_null_count": int(daily_diagnostics["RankIC"].notna().sum()) if len(daily_diagnostics) else 0,
    }


def summarize_ic(ic_frame: pd.DataFrame) -> dict[str, float]:
    """Summarize IC and RankIC with information ratios."""

    summary: dict[str, float] = {}
    for col in ["IC", "RankIC"]:
        series = ic_frame[col].dropna()
        mean = float(series.mean()) if not series.empty else np.nan
        std = float(series.std(ddof=1)) if len(series) > 1 else np.nan
        summary[col] = mean
        summary[f"{col}IR"] = mean / std * np.sqrt(252) if std and np.isfinite(std) else np.nan
        summary[f"{col}_count"] = float(series.shape[0])
        summary[f"{col}_nan_count"] = float(ic_frame[col].isna().sum())
    summary["RankIC_zero_count"] = float((ic_frame["RankIC"] == 0).sum()) if "RankIC" in ic_frame else np.nan
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
