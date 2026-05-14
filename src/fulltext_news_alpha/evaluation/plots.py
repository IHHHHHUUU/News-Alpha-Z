"""Report plot generation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from fulltext_news_alpha.evaluation.ic_metrics import compute_ic_by_date, rolling_rankic
from fulltext_news_alpha.evaluation.portfolio_backtest import decile_returns, long_short_returns


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_cumulative_long_short(ls_returns: pd.DataFrame, output: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(pd.to_datetime(ls_returns["date"]), ls_returns["cumulative_return"], label="Long-short")
    ax.set_title("Cumulative Long-Short Return")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return")
    ax.grid(True, alpha=0.3)
    _save(fig, Path(output))


def plot_rolling_rankic(ic_frame: pd.DataFrame, output: str | Path, window: int = 60) -> None:
    rolled = rolling_rankic(ic_frame, windows=(window,))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(pd.to_datetime(rolled["date"]), rolled[f"rolling_{window}d_RankIC"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Rolling {window}D RankIC")
    ax.set_xlabel("Date")
    ax.set_ylabel("RankIC")
    ax.grid(True, alpha=0.3)
    _save(fig, Path(output))


def plot_decile_bar(deciles: pd.DataFrame, output: str | Path) -> None:
    avg = deciles.groupby("decile")["return"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    avg.plot(kind="bar", ax=ax)
    ax.set_title("Average Decile Returns")
    ax.set_xlabel("Decile")
    ax.set_ylabel("Mean return")
    _save(fig, Path(output))


def plot_series_by_date(frame: pd.DataFrame, value_col: str, output: str | Path, title: str) -> None:
    series = frame.groupby("date")[value_col].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(pd.to_datetime(series.index), series.values)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    _save(fig, Path(output))


def plot_attention_entropy(frame: pd.DataFrame, output: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    frame["attention_entropy"].dropna().hist(ax=ax, bins=30)
    ax.set_title("Attention Entropy Distribution")
    ax.set_xlabel("Entropy")
    _save(fig, Path(output))


def generate_standard_plots(
    factor_frame: pd.DataFrame,
    output_dir: str | Path,
    return_col: str = "future_20d_market_adjusted_return",
    rebalance_every: int | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    ic = compute_ic_by_date(factor_frame, return_col=return_col, rebalance_every=rebalance_every)
    deciles = decile_returns(factor_frame, return_col=return_col, rebalance_every=rebalance_every)
    ls = long_short_returns(factor_frame, return_col=return_col, rebalance_every=rebalance_every)
    outputs = {
        "cumulative_long_short": output_dir / "cumulative_long_short.png",
        "rolling_60d_rankic": output_dir / "rolling_60d_rankic.png",
        "decile_returns": output_dir / "decile_returns.png",
        "coverage": output_dir / "coverage.png",
        "average_gate_news_prob": output_dir / "average_gate_news_prob.png",
    }
    plot_cumulative_long_short(ls, outputs["cumulative_long_short"])
    plot_rolling_rankic(ic, outputs["rolling_60d_rankic"])
    plot_decile_bar(deciles, outputs["decile_returns"])
    plot_series_by_date(factor_frame.assign(coverage=1), "coverage", outputs["coverage"], "Factor Coverage")
    plot_series_by_date(factor_frame, "gate_news_prob", outputs["average_gate_news_prob"], "Average Gate News Probability")
    if "attention_entropy" in factor_frame:
        outputs["attention_entropy"] = output_dir / "attention_entropy.png"
        plot_attention_entropy(factor_frame, outputs["attention_entropy"])
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate standard factor evaluation plots.")
    parser.add_argument("--factor-table", required=True)
    parser.add_argument("--output-dir", default="data/reports/plots")
    parser.add_argument("--return-col", default="future_20d_market_adjusted_return")
    parser.add_argument("--rebalance-every", type=int, default=20)
    args = parser.parse_args()
    frame = pd.read_parquet(args.factor_table)
    outputs = generate_standard_plots(
        frame,
        args.output_dir,
        return_col=args.return_col,
        rebalance_every=args.rebalance_every,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
