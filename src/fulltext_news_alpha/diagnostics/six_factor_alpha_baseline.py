"""Six-factor alpha direction diagnostics and factor-only baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fulltext_news_alpha.diagnostics.raw_factor_diagnostics import (
    RawFactorSplitConfig,
    compute_all_daily_ic,
    compute_correlations,
    normalize_panel,
    split_panel,
    summarize_by_factor_split,
)
from fulltext_news_alpha.evaluation.portfolio_backtest import filter_rebalance_dates


ALPHA_FACTOR_MAP = {
    "RSI_alpha": ("RSI_zscore", -1.0),
    "beta_60d_alpha": ("beta_60d_zscore", 1.0),
    "max_drawdown_60d_alpha": ("max_drawdown_60d_zscore", -1.0),
    "momentum_20d_alpha": ("momentum_20d_zscore", -1.0),
    "reversal_5d_alpha": ("reversal_5d_zscore", 1.0),
    "volatility_20d_alpha": ("volatility_20d_zscore", 1.0),
}
ALPHA_COLS = list(ALPHA_FACTOR_MAP)


def _ir(mean: float, std: float) -> float:
    return float(mean / std * np.sqrt(252)) if std and np.isfinite(std) else np.nan


def _safe_corr(valid: pd.DataFrame, score_col: str, label_col: str, method: str) -> float:
    if len(valid) < 10:
        return np.nan
    if valid[score_col].nunique() < 2 or valid[label_col].nunique() < 2:
        return np.nan
    return float(valid[score_col].corr(valid[label_col], method=method))


def _daily_zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if not std or not np.isfinite(std):
        return pd.Series(np.nan, index=series.index, dtype=float)
    return (series - series.mean()) / std


def construct_alpha_panel(
    panel: pd.DataFrame,
    label_col: str = "future_20d_market_adjusted_return",
) -> pd.DataFrame:
    """Create directional alpha columns without re-standardizing factor values."""

    missing = ["date", "ticker", label_col]
    missing += [source for source, _ in ALPHA_FACTOR_MAP.values()]
    missing_cols = [col for col in missing if col not in panel.columns]
    if missing_cols:
        raise KeyError(f"Panel missing required columns: {missing_cols}")

    out = normalize_panel(panel)
    for alpha_col, (source_col, direction) in ALPHA_FACTOR_MAP.items():
        out[alpha_col] = out[source_col].astype(float) * float(direction)
    out["label"] = out[label_col].astype(float)
    out["label_zscore"] = out.groupby("date", group_keys=False)["label"].transform(_daily_zscore)
    return out


def summarize_alpha_single_factors(
    splits: dict[str, pd.DataFrame],
    alpha_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute single-alpha IC/RankIC and direction flags."""

    factors = list(alpha_cols or ALPHA_COLS)
    daily_ic = compute_all_daily_ic(splits, factors, "label")
    summary = summarize_by_factor_split(daily_ic, pd.DataFrame())
    pivot = summary.pivot(index="factor", columns="split", values="RankIC")
    flag_rows: list[dict[str, Any]] = []
    for factor in factors:
        train = pivot.loc[factor, "train"] if factor in pivot.index and "train" in pivot else np.nan
        valid = pivot.loc[factor, "valid"] if factor in pivot.index and "valid" in pivot else np.nan
        test = pivot.loc[factor, "test"] if factor in pivot.index and "test" in pivot else np.nan
        values = np.asarray([train, valid, test], dtype=float)
        signs = np.sign(values[np.isfinite(values) & (values != 0)])
        unstable = bool(not (len(signs) == 3 and (np.all(signs > 0) or np.all(signs < 0))))
        flag_rows.append(
            {
                "factor": factor,
                "direction_failed_test": bool(np.isfinite(test) and test < 0),
                "unstable_direction": unstable,
            }
        )
    return summary.merge(pd.DataFrame(flag_rows), on="factor", how="left")


def compute_equal_weight_score(frame: pd.DataFrame, alpha_cols: list[str] | None = None) -> pd.Series:
    """Average available alpha columns row-wise."""

    factors = list(alpha_cols or ALPHA_COLS)
    return frame[factors].mean(axis=1, skipna=True)


def compute_ic_weights(
    train_frame: pd.DataFrame,
    alpha_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Use train RankIC as non-negative normalized IC weights."""

    factors = list(alpha_cols or ALPHA_COLS)
    daily_ic = compute_all_daily_ic({"train": train_frame}, factors, "label")
    summary = summarize_by_factor_split(daily_ic, pd.DataFrame())
    rankic_by_factor = dict(zip(summary["factor"], summary["RankIC"], strict=False))
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    raw_weights: list[float] = []
    for factor in factors:
        train_rankic = float(rankic_by_factor.get(factor, np.nan))
        raw_weight = train_rankic if np.isfinite(train_rankic) and train_rankic > 0 else 0.0
        if raw_weight == 0.0:
            warnings.append(f"{factor} train RankIC <= 0; IC weight set to 0")
        raw_weights.append(raw_weight)
        rows.append(
            {
                "factor": factor,
                "train_RankIC": train_rankic,
                "raw_weight": raw_weight,
            }
        )
    weight_sum = float(np.sum(raw_weights))
    if weight_sum <= 0:
        warnings.append("all train RankIC <= 0; fallback to equal weight")
        normalized = [1.0 / len(factors)] * len(factors)
        used = [True] * len(factors)
    else:
        normalized = [weight / weight_sum for weight in raw_weights]
        used = [weight > 0 for weight in raw_weights]
    weights = pd.DataFrame(rows)
    weights["normalized_weight"] = normalized
    weights["used"] = used
    return weights, warnings


def apply_ic_weight_score(
    frame: pd.DataFrame,
    weights: pd.DataFrame,
    alpha_cols: list[str] | None = None,
) -> pd.Series:
    """Apply normalized IC weights to alpha columns."""

    factors = list(alpha_cols or ALPHA_COLS)
    weight_map = dict(zip(weights["factor"], weights["normalized_weight"], strict=False))
    values = frame[factors].fillna(0.0).to_numpy(dtype=float)
    weight_vec = np.asarray([weight_map.get(factor, 0.0) for factor in factors], dtype=float)
    return pd.Series(values @ weight_vec, index=frame.index)


def fit_ridge(
    train_frame: pd.DataFrame,
    alpha_cols: list[str] | None = None,
    ridge_alpha: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Fit Ridge closed-form model on train label_zscore."""

    factors = list(alpha_cols or ALPHA_COLS)
    train = train_frame.dropna(subset=["label_zscore"]).copy()
    if train.empty:
        raise ValueError("No train rows with non-null label_zscore")
    x = train[factors].fillna(0.0).to_numpy(dtype=float)
    y = train["label_zscore"].to_numpy(dtype=float)
    x_mean = x.mean(axis=0)
    y_mean = float(y.mean())
    x_centered = x - x_mean
    y_centered = y - y_mean
    penalty = float(ridge_alpha) * np.eye(len(factors))
    coef = np.linalg.solve(x_centered.T @ x_centered + penalty, x_centered.T @ y_centered)
    intercept = y_mean - float(x_mean @ coef)
    return coef.astype(float), intercept


def apply_ridge_score(
    frame: pd.DataFrame,
    coefficients: np.ndarray,
    intercept: float,
    alpha_cols: list[str] | None = None,
) -> pd.Series:
    factors = list(alpha_cols or ALPHA_COLS)
    values = frame[factors].fillna(0.0).to_numpy(dtype=float)
    return pd.Series(values @ coefficients + intercept, index=frame.index)


def compute_daily_score_ic(
    frame: pd.DataFrame,
    score_col: str,
    label_col: str = "label",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date_value, group in frame.groupby("date", sort=True):
        valid = group[[score_col, label_col]].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(
            {
                "date": date_value,
                "IC": _safe_corr(valid, score_col, label_col, "pearson"),
                "RankIC": _safe_corr(valid, score_col, label_col, "spearman"),
                "coverage": int(len(valid)),
            }
        )
    return pd.DataFrame(rows)


def summarize_score_ic(ic: pd.DataFrame) -> dict[str, float]:
    result: dict[str, float] = {}
    for col in ("IC", "RankIC"):
        series = ic[col].dropna()
        mean = float(series.mean()) if not series.empty else np.nan
        std = float(series.std(ddof=1)) if len(series) > 1 else np.nan
        result[col] = mean
        result[f"{col}IR"] = _ir(mean, std)
    result["coverage"] = float(ic["coverage"].mean()) if "coverage" in ic else np.nan
    return result


def evaluate_long_short(
    frame: pd.DataFrame,
    score_col: str,
    rebalance_every: int = 20,
    n_deciles: int = 10,
) -> dict[str, float]:
    """Evaluate non-overlapping long-short decile spread."""

    filtered = filter_rebalance_dates(frame, rebalance_every=rebalance_every)
    top_bottom: list[float] = []
    top_returns: list[float] = []
    bottom_returns: list[float] = []
    for _, group in filtered.groupby("date", sort=True):
        valid = group[[score_col, "label"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
        if len(valid) < n_deciles or valid[score_col].nunique() < n_deciles:
            continue
        ranked = valid[score_col].rank(method="first")
        valid["decile"] = pd.qcut(ranked, q=n_deciles, labels=False, duplicates="drop") + 1
        decile_mean = valid.groupby("decile", observed=True)["label"].mean()
        if 1 not in decile_mean.index or n_deciles not in decile_mean.index:
            continue
        bottom = float(decile_mean.loc[1])
        top = float(decile_mean.loc[n_deciles])
        top_returns.append(top)
        bottom_returns.append(bottom)
        top_bottom.append(top - bottom)
    spread = np.asarray(top_bottom, dtype=float)
    spread_mean = float(np.nanmean(spread)) if spread.size else np.nan
    spread_std = float(np.nanstd(spread, ddof=1)) if spread.size > 1 else np.nan
    tstat = (
        float(spread_mean / spread_std * np.sqrt(spread.size))
        if spread_std and np.isfinite(spread_std)
        else np.nan
    )
    return {
        "long_short_return": spread_mean,
        "long_short_tstat": tstat,
        "top_decile_return": float(np.nanmean(top_returns)) if top_returns else np.nan,
        "bottom_decile_return": float(np.nanmean(bottom_returns)) if bottom_returns else np.nan,
        "top_bottom_return": spread_mean,
    }


def evaluate_score_by_split(
    splits: dict[str, pd.DataFrame],
    score_col: str,
    rebalance_every: int = 20,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, frame in splits.items():
        ic = summarize_score_ic(compute_daily_score_ic(frame, score_col))
        ls = evaluate_long_short(frame, score_col, rebalance_every=rebalance_every)
        rows.append({"split": split, **ic, **ls})
    return pd.DataFrame(rows)


def build_baseline_comparison(
    evaluations: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name, eval_frame in evaluations.items():
        by_split = eval_frame.set_index("split")
        test_rankic = float(by_split.loc["test", "RankIC"])
        signs = np.sign(
            [
                by_split.loc["train", "RankIC"],
                by_split.loc["valid", "RankIC"],
                by_split.loc["test", "RankIC"],
            ]
        )
        direction_stable = bool(np.all(signs > 0) or np.all(signs < 0))
        row = {
            "model": model_name,
            "train_IC": float(by_split.loc["train", "IC"]),
            "train_RankIC": float(by_split.loc["train", "RankIC"]),
            "valid_IC": float(by_split.loc["valid", "IC"]),
            "valid_RankIC": float(by_split.loc["valid", "RankIC"]),
            "test_IC": float(by_split.loc["test", "IC"]),
            "test_RankIC": test_rankic,
            "test_ICIR": float(by_split.loc["test", "ICIR"]),
            "test_RankICIR": float(by_split.loc["test", "RankICIR"]),
            "test_long_short_return_mean": float(by_split.loc["test", "long_short_return"]),
            "test_long_short_tstat": float(by_split.loc["test", "long_short_tstat"]),
            "test_top_bottom_return": float(by_split.loc["test", "top_bottom_return"]),
            "conclusion": "positive" if test_rankic > 0 and direction_stable else "fail",
        }
        if test_rankic > 0 and not direction_stable:
            row["conclusion"] = "unstable"
        rows.append(row)
    comparison = pd.DataFrame(rows)
    if not comparison.empty:
        best_idx = comparison["test_RankIC"].idxmax()
        if comparison.loc[best_idx, "test_RankIC"] > 0:
            comparison.loc[best_idx, "conclusion"] = "best"
    return comparison


def run_six_factor_alpha_baseline(
    panel: pd.DataFrame,
    output_dir: str | Path,
    label_col: str = "future_20d_market_adjusted_return",
    ridge_alpha: float = 1.0,
    rebalance_every: int = 20,
    corr_threshold: float = 0.8,
    split: RawFactorSplitConfig = RawFactorSplitConfig(),
) -> dict[str, Any]:
    """Run six-alpha diagnostics and factor-only baselines."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    alpha_panel = construct_alpha_panel(panel, label_col=label_col)
    alpha_panel.to_parquet(output / "alpha_panel.parquet", index=False)
    splits = split_panel(alpha_panel, split)

    single_summary = summarize_alpha_single_factors(splits)
    single_summary.to_csv(output / "alpha_single_factor_summary.csv", index=False)

    corr_matrices, high_pairs = compute_correlations(splits, ALPHA_COLS, corr_threshold=corr_threshold)
    for (split_name, method), matrix in corr_matrices.items():
        matrix.to_csv(output / f"alpha_corr_{split_name}_{method}.csv")
    high_pairs = high_pairs.rename(columns={"factor": "unused"})
    high_pairs.to_csv(output / "alpha_high_corr_pairs.csv", index=False)

    weights, warnings = compute_ic_weights(splits["train"])
    weights.to_csv(output / "ic_weights.csv", index=False)
    coef, intercept = fit_ridge(splits["train"], ridge_alpha=ridge_alpha)
    ridge_coef = pd.DataFrame({"factor": ALPHA_COLS, "coefficient": coef})
    ridge_coef.to_csv(output / "ridge_coefficients.csv", index=False)

    scored_splits: dict[str, pd.DataFrame] = {}
    for split_name, frame in splits.items():
        scored = frame.copy()
        scored["score_equal"] = compute_equal_weight_score(scored)
        scored["score_ic_weight"] = apply_ic_weight_score(scored, weights)
        scored["score_ridge"] = apply_ridge_score(scored, coef, intercept)
        scored_splits[split_name] = scored
        output_cols = [
            "date",
            "ticker",
            "label",
            "label_zscore",
            *ALPHA_COLS,
            "score_equal",
            "score_ic_weight",
            "score_ridge",
        ]
        scored[output_cols].to_parquet(output / f"predictions_{split_name}.parquet", index=False)

    evaluations = {
        "Equal Weight": evaluate_score_by_split(scored_splits, "score_equal", rebalance_every),
        "IC Weight": evaluate_score_by_split(scored_splits, "score_ic_weight", rebalance_every),
        "Ridge": evaluate_score_by_split(scored_splits, "score_ridge", rebalance_every),
    }
    comparison = build_baseline_comparison(evaluations)
    comparison.to_csv(output / "baseline_comparison.csv", index=False)

    diagnostics = {
        "panel_rows": int(len(alpha_panel)),
        "label_col": label_col,
        "alpha_cols": ALPHA_COLS,
        "ridge_alpha": float(ridge_alpha),
        "ridge_intercept": float(intercept),
        "rebalance_every": int(rebalance_every),
        "corr_threshold": float(corr_threshold),
        "split_rows": {name: int(len(frame)) for name, frame in splits.items()},
        "high_corr_pair_count": int(len(high_pairs)),
        "ic_weight_warnings": warnings,
        "best_model": (
            str(comparison.loc[comparison["test_RankIC"].idxmax(), "model"])
            if not comparison.empty
            else None
        ),
    }
    (output / "diagnostics_summary.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run six-factor alpha baselines.")
    parser.add_argument("--panel", default="data/processed/panel_train_b2_768.parquet")
    parser.add_argument("--output-dir", default="data/reports/six_factor_alpha_baseline")
    parser.add_argument("--label-col", default="future_20d_market_adjusted_return")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--rebalance-every", type=int, default=20)
    parser.add_argument("--corr-threshold", type=float, default=0.8)
    args = parser.parse_args()

    panel = pd.read_parquet(args.panel)
    diagnostics = run_six_factor_alpha_baseline(
        panel=panel,
        output_dir=args.output_dir,
        label_col=args.label_col,
        ridge_alpha=args.ridge_alpha,
        rebalance_every=args.rebalance_every,
        corr_threshold=args.corr_threshold,
    )
    print(json.dumps(diagnostics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
