"""Single-factor diagnostics for raw price-volume z-score factors."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RAW_PRICE_VOLUME_FACTORS = [
    "momentum_5d_zscore",
    "momentum_20d_zscore",
    "momentum_60d_zscore",
    "reversal_5d_zscore",
    "volatility_20d_zscore",
    "volume_zscore_20d_zscore",
    "beta_60d_zscore",
    "max_drawdown_60d_zscore",
    "RSI_zscore",
    "overnight_gap_zscore",
]


@dataclass(frozen=True)
class RawFactorSplitConfig:
    """Date-based split used by raw factor diagnostics."""

    train_start: str = "2018-01-01"
    train_end: str = "2020-12-31"
    valid_start: str = "2021-01-01"
    valid_end: str = "2021-12-31"
    test_start: str = "2022-01-01"
    test_end: str = "2023-12-31"


def _safe_corr(valid: pd.DataFrame, factor_col: str, label_col: str, method: str) -> float:
    if len(valid) < 10:
        return np.nan
    if valid[factor_col].nunique() < 2 or valid[label_col].nunique() < 2:
        return np.nan
    return float(valid[factor_col].corr(valid[label_col], method=method))


def _ir(mean: float, std: float) -> float:
    return float(mean / std * np.sqrt(252)) if std and np.isfinite(std) else np.nan


def _spearman_by_values(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> float:
    valid = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 2 or valid["x"].nunique() < 2 or valid["y"].nunique() < 2:
        return np.nan
    return float(valid["x"].corr(valid["y"], method="spearman"))


def normalize_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize key columns for diagnostics."""

    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


def split_panel(frame: pd.DataFrame, split: RawFactorSplitConfig) -> dict[str, pd.DataFrame]:
    """Slice train/valid/test frames by date."""

    dates = pd.to_datetime(frame["date"])

    def _mask(start: str, end: str) -> pd.Series:
        return (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))

    return {
        "train": frame.loc[_mask(split.train_start, split.train_end)].reset_index(drop=True),
        "valid": frame.loc[_mask(split.valid_start, split.valid_end)].reset_index(drop=True),
        "test": frame.loc[_mask(split.test_start, split.test_end)].reset_index(drop=True),
    }


def compute_daily_factor_ic(
    frame: pd.DataFrame,
    factor_col: str,
    label_col: str,
    split: str,
    min_coverage: int = 10,
) -> pd.DataFrame:
    """Compute per-date Pearson IC and Spearman RankIC for one factor."""

    rows: list[dict[str, Any]] = []
    for date_value, group in frame.groupby("date", sort=True):
        valid = group[[factor_col, label_col]].replace([np.inf, -np.inf], np.nan).dropna()
        coverage = int(len(valid))
        if coverage < min_coverage:
            ic = np.nan
            rankic = np.nan
        else:
            ic = _safe_corr(valid, factor_col, label_col, "pearson")
            rankic = _safe_corr(valid, factor_col, label_col, "spearman")
        rows.append(
            {
                "date": date_value,
                "split": split,
                "factor": factor_col,
                "IC": ic,
                "RankIC": rankic,
                "coverage": coverage,
                "total_rows": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def compute_all_daily_ic(
    splits: dict[str, pd.DataFrame],
    factor_cols: list[str],
    label_col: str,
    min_coverage: int = 10,
) -> pd.DataFrame:
    """Compute daily IC/RankIC for all factors and splits."""

    frames = [
        compute_daily_factor_ic(frame, factor, label_col, split, min_coverage=min_coverage)
        for split, frame in splits.items()
        for factor in factor_cols
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def compute_rolling_rankic(
    daily_ic: pd.DataFrame,
    rolling_window: int = 60,
    min_periods: int = 15,
) -> pd.DataFrame:
    """Add rolling RankIC per factor/split."""

    out = daily_ic.sort_values(["factor", "split", "date"]).copy()
    out["rolling_60d_RankIC"] = (
        out.groupby(["factor", "split"], group_keys=False)["RankIC"]
        .rolling(int(rolling_window), min_periods=int(min_periods))
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    return out[["date", "split", "factor", "RankIC", "rolling_60d_RankIC", "coverage"]]


def compute_decile_returns(
    frame: pd.DataFrame,
    factor_col: str,
    label_col: str,
    split: str,
    n_deciles: int = 10,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compute per-date decile returns and aggregate monotonic diagnostics."""

    daily_rows: list[dict[str, Any]] = []
    top_bottom: list[float] = []
    for date_value, group in frame.groupby("date", sort=True):
        valid = group[[factor_col, label_col]].replace([np.inf, -np.inf], np.nan).dropna().copy()
        if len(valid) < n_deciles or valid[factor_col].nunique() < n_deciles:
            continue
        ranked = valid[factor_col].rank(method="first")
        valid["decile"] = pd.qcut(ranked, q=n_deciles, labels=False, duplicates="drop") + 1
        decile_mean = valid.groupby("decile", observed=True)[label_col].mean()
        if 1 in decile_mean.index and n_deciles in decile_mean.index:
            top_bottom.append(float(decile_mean.loc[n_deciles] - decile_mean.loc[1]))
        for decile, value in decile_mean.items():
            daily_rows.append(
                {
                    "date": date_value,
                    "split": split,
                    "factor": factor_col,
                    "decile": int(decile),
                    "mean_return": float(value),
                    "count": int((valid["decile"] == decile).sum()),
                }
            )

    daily = pd.DataFrame(daily_rows)
    if daily.empty:
        return daily, {
            "decile_monotonic_score": np.nan,
            "top_bottom_return": np.nan,
            "top_bottom_tstat": np.nan,
            "top_decile_return": np.nan,
            "bottom_decile_return": np.nan,
        }

    aggregate = (
        daily.groupby(["split", "factor", "decile"], as_index=False)
        .agg(mean_return=("mean_return", "mean"), days=("mean_return", "count"), avg_count=("count", "mean"))
        .sort_values(["split", "factor", "decile"])
    )
    deciles = aggregate["decile"].to_numpy(dtype=float)
    returns = aggregate["mean_return"].to_numpy(dtype=float)
    monotonic = _spearman_by_values(deciles, returns)
    top_value = aggregate.loc[aggregate["decile"] == n_deciles, "mean_return"]
    bottom_value = aggregate.loc[aggregate["decile"] == 1, "mean_return"]
    top_return = float(top_value.iloc[0]) if not top_value.empty else np.nan
    bottom_return = float(bottom_value.iloc[0]) if not bottom_value.empty else np.nan
    top_bottom_arr = np.asarray(top_bottom, dtype=float)
    top_bottom_mean = float(np.nanmean(top_bottom_arr)) if top_bottom_arr.size else np.nan
    top_bottom_std = float(np.nanstd(top_bottom_arr, ddof=1)) if top_bottom_arr.size > 1 else np.nan
    top_bottom_tstat = (
        float(top_bottom_mean / top_bottom_std * np.sqrt(top_bottom_arr.size))
        if top_bottom_std and np.isfinite(top_bottom_std)
        else np.nan
    )
    metrics = {
        "decile_monotonic_score": monotonic,
        "top_bottom_return": top_bottom_mean,
        "top_bottom_tstat": top_bottom_tstat,
        "top_decile_return": top_return,
        "bottom_decile_return": bottom_return,
    }
    return aggregate, metrics


def compute_all_decile_returns(
    splits: dict[str, pd.DataFrame],
    factor_cols: list[str],
    label_col: str,
    n_deciles: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute decile returns and factor/split decile summary metrics."""

    decile_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    for split, frame in splits.items():
        for factor in factor_cols:
            deciles, metrics = compute_decile_returns(
                frame, factor, label_col, split, n_deciles=n_deciles
            )
            if not deciles.empty:
                decile_frames.append(deciles)
            metric_rows.append({"split": split, "factor": factor, **metrics})
    decile_returns = pd.concat(decile_frames, ignore_index=True) if decile_frames else pd.DataFrame()
    return decile_returns, pd.DataFrame(metric_rows)


def summarize_by_factor_split(
    daily_ic: pd.DataFrame,
    decile_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Build factor/split IC, coverage, and decile summary."""

    rows: list[dict[str, Any]] = []
    for (factor, split), group in daily_ic.groupby(["factor", "split"], sort=True):
        ic = group["IC"].dropna()
        rankic = group["RankIC"].dropna()
        ic_mean = float(ic.mean()) if not ic.empty else np.nan
        rankic_mean = float(rankic.mean()) if not rankic.empty else np.nan
        ic_std = float(ic.std(ddof=1)) if len(ic) > 1 else np.nan
        rankic_std = float(rankic.std(ddof=1)) if len(rankic) > 1 else np.nan
        total_rows = float(group["total_rows"].sum())
        valid_rows = float(group["coverage"].sum())
        row = {
            "factor": factor,
            "split": split,
            "IC": ic_mean,
            "ICIR": _ir(ic_mean, ic_std),
            "RankIC": rankic_mean,
            "RankICIR": _ir(rankic_mean, rankic_std),
            "IC_std": ic_std,
            "RankIC_std": rankic_std,
            "positive_IC_ratio": float((ic > 0).mean()) if not ic.empty else np.nan,
            "positive_RankIC_ratio": float((rankic > 0).mean()) if not rankic.empty else np.nan,
            "days": int(len(group)),
            "avg_coverage": float(group["coverage"].mean()),
            "min_coverage": int(group["coverage"].min()),
            "max_coverage": int(group["coverage"].max()),
            "coverage_ratio": valid_rows / total_rows if total_rows else np.nan,
        }
        rows.append(row)
    summary = pd.DataFrame(rows)
    if not decile_metrics.empty:
        summary = summary.merge(decile_metrics, on=["factor", "split"], how="left")
    return summary.sort_values(["split", "factor"]).reset_index(drop=True)


def compute_average_factor_corr(
    frame: pd.DataFrame,
    factor_cols: list[str],
    method: str,
    min_coverage: int = 10,
) -> pd.DataFrame:
    """Average daily cross-sectional factor correlation matrices."""

    matrices: list[pd.DataFrame] = []
    for _, group in frame.groupby("date", sort=True):
        valid = group[factor_cols].replace([np.inf, -np.inf], np.nan)
        if int(valid.dropna(how="all").shape[0]) < min_coverage:
            continue
        matrices.append(valid.corr(method=method, min_periods=min_coverage))
    if not matrices:
        return pd.DataFrame(np.nan, index=factor_cols, columns=factor_cols)
    stacked = np.stack([matrix.reindex(index=factor_cols, columns=factor_cols).to_numpy() for matrix in matrices])
    avg = np.nanmean(stacked, axis=0)
    return pd.DataFrame(avg, index=factor_cols, columns=factor_cols)


def compute_correlations(
    splits: dict[str, pd.DataFrame],
    factor_cols: list[str],
    corr_threshold: float = 0.8,
) -> tuple[dict[tuple[str, str], pd.DataFrame], pd.DataFrame]:
    """Compute average Pearson/Spearman matrices and high-correlation pairs."""

    matrices: dict[tuple[str, str], pd.DataFrame] = {}
    pair_rows: list[dict[str, Any]] = []
    for split, frame in splits.items():
        for method in ("pearson", "spearman"):
            matrix = compute_average_factor_corr(frame, factor_cols, method=method)
            matrices[(split, method)] = matrix
            for i, factor_a in enumerate(factor_cols):
                for factor_b in factor_cols[i + 1 :]:
                    corr = matrix.loc[factor_a, factor_b]
                    if pd.notna(corr) and abs(float(corr)) >= corr_threshold:
                        pair_rows.append(
                            {
                                "split": split,
                                "factor_a": factor_a,
                                "factor_b": factor_b,
                                "corr": float(corr),
                                "abs_corr": abs(float(corr)),
                                "method": method,
                            }
                        )
    pairs = pd.DataFrame(pair_rows)
    if not pairs.empty:
        pairs = pairs.sort_values(["abs_corr", "split", "method"], ascending=[False, True, True])
    return matrices, pairs


def compute_stability_flags(summary: pd.DataFrame, factor_cols: list[str]) -> pd.DataFrame:
    """Compare train/valid/test RankIC signs and weak-factor flags."""

    rows: list[dict[str, Any]] = []
    pivot = summary.pivot(index="factor", columns="split", values="RankIC")
    for factor in factor_cols:
        train = float(pivot.loc[factor, "train"]) if factor in pivot.index and "train" in pivot else np.nan
        valid = float(pivot.loc[factor, "valid"]) if factor in pivot.index and "valid" in pivot else np.nan
        test = float(pivot.loc[factor, "test"]) if factor in pivot.index and "test" in pivot else np.nan
        values = np.asarray([train, valid, test], dtype=float)
        signs = np.sign(values[np.isfinite(values) & (values != 0)])
        direction_consistent = bool(len(signs) == 3 and (np.all(signs > 0) or np.all(signs < 0)))
        unstable_flag = bool(
            (np.isfinite(train) and np.isfinite(test) and train * test < 0)
            or (np.isfinite(valid) and np.isfinite(test) and valid * test < 0)
        )
        weak_flag = bool(not np.isfinite(test) or abs(test) < 0.005)
        median_rankic = float(np.nanmedian(values)) if np.isfinite(values).any() else np.nan
        rows.append(
            {
                "factor": factor,
                "train_RankIC": train,
                "valid_RankIC": valid,
                "test_RankIC": test,
                "direction_consistent": direction_consistent,
                "unstable_flag": unstable_flag,
                "weak_flag": weak_flag,
                "selected_direction": int(np.sign(median_rankic)) if np.isfinite(median_rankic) else 0,
            }
        )
    return pd.DataFrame(rows)


def _stability_score(row: pd.Series, summary: pd.DataFrame) -> float:
    test_rows = summary[(summary["factor"] == row["factor"]) & (summary["split"] == "test")]
    if test_rows.empty:
        return -np.inf
    test = test_rows.iloc[0]
    rankic_std = float(test["RankIC_std"]) if pd.notna(test["RankIC_std"]) else np.inf
    score = 0.0
    score += abs(float(row["test_RankIC"])) * 100.0 if pd.notna(row["test_RankIC"]) else 0.0
    score += float(test["positive_RankIC_ratio"]) if pd.notna(test["positive_RankIC_ratio"]) else 0.0
    score += 1.0 if bool(row["direction_consistent"]) else 0.0
    score += 1.0 / (1.0 + rankic_std) if np.isfinite(rankic_std) else 0.0
    return float(score)


def recommend_keep_drop(
    factor_cols: list[str],
    summary: pd.DataFrame,
    stability_flags: pd.DataFrame,
    high_corr_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Recommend one factor to keep within each high-correlation component."""

    factors = set(factor_cols)
    edges: list[tuple[str, str]] = []
    if not high_corr_pairs.empty:
        edges = list(zip(high_corr_pairs["factor_a"], high_corr_pairs["factor_b"], strict=True))

    parent = {factor: factor for factor in factor_cols}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for factor_a, factor_b in edges:
        if factor_a in factors and factor_b in factors:
            union(factor_a, factor_b)

    flags = stability_flags.copy()
    flags["stability_score"] = [float(_stability_score(row, summary)) for _, row in flags.iterrows()]
    components: dict[str, list[str]] = {}
    for factor in factor_cols:
        components.setdefault(find(factor), []).append(factor)

    rows: list[dict[str, Any]] = []
    for members in components.values():
        ranked = flags[flags["factor"].isin(members)].sort_values(
            ["stability_score", "test_RankIC"], ascending=[False, False]
        )
        keep = str(ranked.iloc[0]["factor"])
        for factor in members:
            flag = flags[flags["factor"] == factor].iloc[0]
            correlated = [member for member in members if member != factor]
            if factor == keep:
                reason = (
                    "best stability score among correlated factors"
                    if correlated
                    else "no abs(corr) >= threshold pair"
                )
                keep_or_drop = "keep"
            else:
                reason = f"drop in favor of {keep}"
                keep_or_drop = "drop"
            rows.append(
                {
                    "factor": factor,
                    "keep_or_drop": keep_or_drop,
                    "reason": reason,
                    "correlated_with": ";".join(correlated),
                    "test_RankIC": float(flag["test_RankIC"]),
                    "direction_consistent": bool(flag["direction_consistent"]),
                    "stability_score": float(flag["stability_score"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["keep_or_drop", "factor"], ascending=[False, True])


def run_raw_factor_diagnostics(
    panel: pd.DataFrame,
    output_dir: str | Path,
    factor_cols: list[str] | None = None,
    label_col: str = "future_20d_market_adjusted_return",
    split: RawFactorSplitConfig = RawFactorSplitConfig(),
    n_deciles: int = 10,
    rolling_window: int = 60,
    min_periods: int = 15,
    corr_threshold: float = 0.8,
) -> dict[str, Any]:
    """Run all raw factor diagnostics and write CSV/JSON outputs."""

    factors = list(factor_cols or RAW_PRICE_VOLUME_FACTORS)
    missing = [col for col in ["date", "ticker", label_col, *factors] if col not in panel.columns]
    if missing:
        raise KeyError(f"Panel missing required columns: {missing}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = normalize_panel(panel)
    splits = split_panel(frame, split)

    daily_ic = compute_all_daily_ic(splits, factors, label_col)
    rolling = compute_rolling_rankic(daily_ic, rolling_window=rolling_window, min_periods=min_periods)
    decile_returns, decile_metrics = compute_all_decile_returns(
        splits, factors, label_col, n_deciles=n_deciles
    )
    summary = summarize_by_factor_split(daily_ic, decile_metrics)
    corr_matrices, high_pairs = compute_correlations(
        splits, factors, corr_threshold=corr_threshold
    )
    stability = compute_stability_flags(summary, factors)
    recommendations = recommend_keep_drop(factors, summary, stability, high_pairs)

    summary.to_csv(output / "summary_by_factor_split.csv", index=False)
    daily_ic.to_csv(output / "raw_factor_daily_ic.csv", index=False)
    rolling.to_csv(output / "raw_factor_rolling60_rankic.csv", index=False)
    decile_returns.to_csv(output / "raw_factor_decile_returns.csv", index=False)
    high_pairs.to_csv(output / "highly_correlated_pairs.csv", index=False)
    stability.to_csv(output / "factor_stability_flags.csv", index=False)
    recommendations.to_csv(output / "recommended_factor_keep_drop.csv", index=False)
    for (split_name, method), matrix in corr_matrices.items():
        matrix.to_csv(output / f"factor_corr_{split_name}_{method}.csv")

    diagnostics = {
        "panel_rows": int(len(frame)),
        "factor_cols": factors,
        "label_col": label_col,
        "split": asdict(split),
        "n_deciles": int(n_deciles),
        "rolling_window": int(rolling_window),
        "min_periods": int(min_periods),
        "corr_threshold": float(corr_threshold),
        "split_rows": {name: int(len(split_frame)) for name, split_frame in splits.items()},
        "unstable_factors": stability.loc[stability["unstable_flag"], "factor"].tolist(),
        "weak_test_factors": stability.loc[stability["weak_flag"], "factor"].tolist(),
        "highly_correlated_pair_count": int(len(high_pairs)),
        "recommended_keep": recommendations.loc[
            recommendations["keep_or_drop"] == "keep", "factor"
        ].tolist(),
        "recommended_drop": recommendations.loc[
            recommendations["keep_or_drop"] == "drop", "factor"
        ].tolist(),
    }
    (output / "diagnostics_summary.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose raw price-volume single factors.")
    parser.add_argument("--panel", default="data/processed/panel_train_b2_768.parquet")
    parser.add_argument("--output-dir", default="data/reports/raw_factor_diagnostics")
    parser.add_argument("--label-col", default="future_20d_market_adjusted_return")
    parser.add_argument("--factor-cols", nargs="+", default=None)
    parser.add_argument("--n-deciles", type=int, default=10)
    parser.add_argument("--rolling-window", type=int, default=60)
    parser.add_argument("--min-periods", type=int, default=15)
    parser.add_argument("--corr-threshold", type=float, default=0.8)
    args = parser.parse_args()

    panel = pd.read_parquet(args.panel)
    diagnostics = run_raw_factor_diagnostics(
        panel=panel,
        output_dir=args.output_dir,
        factor_cols=args.factor_cols,
        label_col=args.label_col,
        n_deciles=args.n_deciles,
        rolling_window=args.rolling_window,
        min_periods=args.min_periods,
        corr_threshold=args.corr_threshold,
    )
    print(json.dumps(diagnostics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
