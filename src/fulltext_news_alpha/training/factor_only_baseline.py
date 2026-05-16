"""Tabular factor-only baselines for B2/B3+B5 diagnostics.

This module intentionally avoids news features. It is meant to answer whether
the lagged price-volume ``*_zscore`` factors carry signal before comparing them
with the TCN news/fusion branches.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fulltext_news_alpha.evaluation.ic_metrics import compute_ic_by_date, summarize_ic

try:
    from fulltext_news_alpha.training.torch_utils import SplitConfig, split_by_date
except RuntimeError:  # pragma: no cover - only used when optional torch is absent.

    @dataclass(frozen=True)
    class SplitConfig:  # type: ignore[no-redef]
        """Date-based train / validation / test split."""

        train_start: str = "2018-01-01"
        train_end: str = "2020-12-31"
        valid_start: str = "2021-01-01"
        valid_end: str = "2021-12-31"
        test_start: str = "2022-01-01"
        test_end: str = "2023-12-31"

    def split_by_date(frame: pd.DataFrame, split: SplitConfig) -> dict[str, pd.DataFrame]:  # type: ignore[no-redef]
        dates = pd.to_datetime(frame["date"])

        def _mask(start: str, end: str) -> pd.Series:
            return (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))

        return {
            "train": frame.loc[_mask(split.train_start, split.train_end)].reset_index(drop=True),
            "valid": frame.loc[_mask(split.valid_start, split.valid_end)].reset_index(drop=True),
            "test": frame.loc[_mask(split.test_start, split.test_end)].reset_index(drop=True),
        }


DEFAULT_PANEL = "data/processed/panel_train_b2_768.parquet"
DEFAULT_LABEL_COL = "future_20d_market_adjusted_return"
DEFAULT_OUTPUT_DIR = "data/predictions/factor_only_baseline"
FORBIDDEN_EXACT_FEATURES = {"has_news", "news_count", "chunk_count"}
FORBIDDEN_PREFIXES = ("future_", "mean_emb_")


@dataclass(frozen=True)
class RidgeConfig:
    """Configuration for the closed-form Ridge baseline."""

    alpha: float = 1.0
    feature_mode: str = "current"
    lookback_window: int = 30


@dataclass
class RidgeBaseline:
    """Small dependency-free Ridge regressor with train-set imputation/scaling."""

    alpha: float = 1.0
    feature_means_: np.ndarray | None = None
    feature_scales_: np.ndarray | None = None
    coef_: np.ndarray | None = None
    intercept_: float = 0.0

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> "RidgeBaseline":
        valid = labels.replace([np.inf, -np.inf], np.nan).notna()
        if not bool(valid.any()):
            raise ValueError("Ridge baseline needs at least one non-null training label.")

        x = features.loc[valid].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64)
        y = labels.loc[valid].to_numpy(dtype=np.float64)
        means = features.loc[valid].replace([np.inf, -np.inf], np.nan).mean(skipna=True).to_numpy(
            dtype=np.float64
        )
        means = np.where(np.isfinite(means), means, 0.0)
        x = np.where(np.isnan(x), means, x)
        scales = np.nanstd(x, axis=0)
        scales = np.where(np.isfinite(scales) & (scales > 0), scales, 1.0)
        x_scaled = (x - means) / scales

        y_mean = float(np.mean(y))
        centered_y = y - y_mean
        penalty = float(self.alpha) * np.eye(x_scaled.shape[1], dtype=np.float64)
        xtx = x_scaled.T @ x_scaled
        xty = x_scaled.T @ centered_y
        try:
            coef = np.linalg.solve(xtx + penalty, xty)
        except np.linalg.LinAlgError:
            coef = np.linalg.pinv(xtx + penalty) @ xty

        self.feature_means_ = means
        self.feature_scales_ = scales
        self.coef_ = coef
        self.intercept_ = y_mean
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.feature_means_ is None or self.feature_scales_ is None or self.coef_ is None:
            raise RuntimeError("RidgeBaseline must be fit before predict.")
        x = features.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64)
        x = np.where(np.isnan(x), self.feature_means_, x)
        x_scaled = (x - self.feature_means_) / self.feature_scales_
        return (x_scaled @ self.coef_ + self.intercept_).astype(float)

    def metadata(self) -> dict[str, Any]:
        coef = self.coef_ if self.coef_ is not None else np.array([], dtype=float)
        return {
            "model": "ridge",
            "alpha": float(self.alpha),
            "intercept": float(self.intercept_),
            "coef_l2_norm": float(np.linalg.norm(coef)) if coef.size else 0.0,
        }


def normalize_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize key columns without changing feature values."""

    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def infer_factor_cols(frame: pd.DataFrame) -> list[str]:
    """Select safe tabular ``*_zscore`` factors and exclude labels/news features."""

    cols = []
    for col in frame.columns:
        if not col.endswith("_zscore"):
            continue
        if col in FORBIDDEN_EXACT_FEATURES:
            continue
        if any(col.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
            continue
        cols.append(col)
    if not cols:
        raise KeyError("No eligible '*_zscore' factor columns found.")
    return sorted(cols)


def make_feature_frame(
    frame: pd.DataFrame,
    factor_cols: list[str],
    feature_mode: str = "current",
    lookback_window: int = 30,
) -> pd.DataFrame:
    """Return current-day features or flattened per-ticker lagged factor features."""

    if feature_mode == "current":
        return frame[factor_cols].copy()
    if feature_mode != "sequence":
        raise ValueError("feature_mode must be 'current' or 'sequence'.")
    if lookback_window <= 0:
        raise ValueError("lookback_window must be positive.")

    ordered = frame.sort_values(["ticker", "date"]).copy()
    lagged: dict[str, pd.Series] = {}
    grouped = ordered.groupby("ticker", sort=False)
    for lag in range(lookback_window):
        shifted = grouped[factor_cols].shift(lag)
        for col in factor_cols:
            lagged[f"{col}_lag{lag}"] = shifted[col]
    features = pd.DataFrame(lagged, index=ordered.index)
    return features.loc[frame.index].copy()


def _ic_summary_for_predictions(
    predictions: pd.DataFrame,
    label_col: str,
    rebalance_every: int | None = None,
) -> dict[str, float]:
    ic = compute_ic_by_date(
        predictions,
        factor_col="factor_only_pred",
        return_col=label_col,
        rebalance_every=rebalance_every,
    )
    summary = summarize_ic(ic)
    return {
        "IC": float(summary.get("IC", np.nan)),
        "ICIR": float(summary.get("ICIR", np.nan)),
        "RankIC": float(summary.get("RankIC", np.nan)),
        "RankICIR": float(summary.get("RankICIR", np.nan)),
        "coverage": float(summary.get("coverage", np.nan)),
        "days": float(summary.get("IC_count", np.nan)),
    }


def _metric_row(ic_frame: pd.DataFrame, factor: str, metric: str) -> dict[str, Any]:
    series = ic_frame[metric].dropna()
    mean = float(series.mean()) if not series.empty else np.nan
    std = float(series.std(ddof=1)) if len(series) > 1 else np.nan
    ir = mean / std * np.sqrt(252) if std and np.isfinite(std) else np.nan
    return {
        "factor": factor,
        "metric": metric,
        "mean": float(mean),
        "ir": float(ir),
        "std": float(std),
        "positive_ratio": float((series > 0).mean()) if not series.empty else np.nan,
        "days": int(series.shape[0]),
        "avg_coverage": float(ic_frame["coverage"].mean()) if "coverage" in ic_frame else np.nan,
    }


def single_factor_diagnostics(
    frame: pd.DataFrame,
    factor_cols: list[str],
    label_col: str,
    rebalance_every: int | None = None,
) -> pd.DataFrame:
    """Compute daily IC/RankIC summaries for each raw factor column."""

    rows: list[dict[str, Any]] = []
    for factor_col in factor_cols:
        ic = compute_ic_by_date(
            frame,
            factor_col=factor_col,
            return_col=label_col,
            rebalance_every=rebalance_every,
        )
        rows.append(_metric_row(ic, factor_col, "IC"))
        rows.append(_metric_row(ic, factor_col, "RankIC"))
    return pd.DataFrame(rows)


def train_factor_only_baseline(
    panel: pd.DataFrame,
    output_dir: str | Path | None = None,
    label_col: str = DEFAULT_LABEL_COL,
    factor_cols: list[str] | None = None,
    split: SplitConfig | None = None,
    config: RidgeConfig | None = None,
    rebalance_every: int | None = None,
) -> dict[str, Any]:
    """Fit Ridge on factor columns, optionally save predictions and diagnostics."""

    split = split or SplitConfig()
    config = config or RidgeConfig()
    frame = normalize_panel(panel)
    frame["_row_id"] = np.arange(len(frame))
    if label_col not in frame.columns:
        raise KeyError(f"Panel is missing label column: {label_col}")
    selected_factor_cols = list(factor_cols) if factor_cols is not None else infer_factor_cols(frame)
    features = make_feature_frame(
        frame,
        selected_factor_cols,
        feature_mode=config.feature_mode,
        lookback_window=config.lookback_window,
    )
    feature_cols = list(features.columns)
    splits = split_by_date(frame, split)
    feature_splits = {
        name: features.iloc[part["_row_id"].to_numpy()].reset_index(drop=True)
        for name, part in splits.items()
    }

    model = RidgeBaseline(alpha=config.alpha).fit(feature_splits["train"], splits["train"][label_col])
    predictions: dict[str, pd.DataFrame] = {}
    split_metrics: dict[str, dict[str, float]] = {}
    for name, part in splits.items():
        pred = part[["date", "ticker", label_col]].copy()
        if len(part):
            pred["factor_only_pred"] = model.predict(feature_splits[name])
        else:
            pred["factor_only_pred"] = pd.Series(dtype=float)
        pred = pred[["date", "ticker", "factor_only_pred", label_col]]
        predictions[name] = pred.reset_index(drop=True)
        split_metrics[name] = _ic_summary_for_predictions(pred, label_col, rebalance_every)

    single_factor_ic_test = single_factor_diagnostics(
        splits["test"],
        selected_factor_cols,
        label_col=label_col,
        rebalance_every=rebalance_every,
    )
    rankic_top30 = (
        single_factor_ic_test.loc[single_factor_ic_test["metric"] == "RankIC"]
        .sort_values("mean", ascending=False)
        .head(30)
        .reset_index(drop=True)
    )

    summary: dict[str, Any] = {
        "model": model.metadata(),
        "label_col": label_col,
        "feature_mode": config.feature_mode,
        "lookback_window": int(config.lookback_window),
        "factor_cols": selected_factor_cols,
        "feature_cols": feature_cols,
        "split": asdict(split),
        "rebalance_every": rebalance_every,
        "split_rows": {name: int(len(part)) for name, part in splits.items()},
        "split_metrics": split_metrics,
        "single_factor_rankic_top30_test": rankic_top30.to_dict(orient="records"),
    }

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, pred in predictions.items():
            pred.to_parquet(out_dir / f"{name}.parquet", index=False)
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        single_factor_ic_test.to_csv(out_dir / "single_factor_ic_test.csv", index=False)
        rankic_top30.to_csv(out_dir / "single_factor_rankic_top30_test.csv", index=False)

    return {
        "summary": summary,
        "predictions": predictions,
        "single_factor_ic_test": single_factor_ic_test,
        "single_factor_rankic_top30_test": rankic_top30,
    }


def _parse_rebalance_every(value: str | None) -> int | None:
    if value is None or value.lower() in {"none", "null", ""}:
        return None
    parsed = int(value)
    return None if parsed <= 1 else parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tabular factor-only Ridge baseline.")
    parser.add_argument("--panel", default=DEFAULT_PANEL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label-col", default=DEFAULT_LABEL_COL)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--feature-mode", choices=["current", "sequence"], default="current")
    parser.add_argument("--lookback-window", type=int, default=30)
    parser.add_argument("--rebalance-every", default=None)
    args = parser.parse_args()

    panel = pd.read_parquet(args.panel)
    result = train_factor_only_baseline(
        panel,
        output_dir=args.output_dir,
        label_col=args.label_col,
        config=RidgeConfig(
            alpha=args.alpha,
            feature_mode=args.feature_mode,
            lookback_window=args.lookback_window,
        ),
        rebalance_every=_parse_rebalance_every(args.rebalance_every),
    )
    metrics = result["summary"]["split_metrics"]
    print(json.dumps(metrics, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
