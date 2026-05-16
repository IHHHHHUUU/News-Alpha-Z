"""Evaluate a FullTextNewsAlpha factor table."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from fulltext_news_alpha.evaluation.ic_metrics import (
    compute_daily_ic_diagnostics,
    compute_ic_by_date,
    summarize_factor_diagnostics,
    summarize_ic,
)
from fulltext_news_alpha.evaluation.plots import generate_standard_plots
from fulltext_news_alpha.evaluation.portfolio_backtest import long_short_returns, performance_summary
from fulltext_news_alpha.factors.factor_standardization import standardize_by_date


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate FullTextNewsAlpha.")
    parser.add_argument("--factor-table", required=True)
    parser.add_argument("--output-dir", default="data/reports/factor_eval")
    parser.add_argument("--factor-col", default="FullTextNewsAlpha_zscore")
    parser.add_argument(
        "--raw-factor-col",
        default=None,
        help="Optional raw score column to z-score by date before evaluation.",
    )
    parser.add_argument(
        "--zscore-output-col",
        default=None,
        help="Output column for --raw-factor-col daily z-score. Defaults to --factor-col.",
    )
    parser.add_argument("--return-col", default="future_20d_market_adjusted_return")
    parser.add_argument(
        "--rebalance-every",
        type=int,
        default=20,
        help="Evaluate every Nth date; use 20 for non-overlapping 20D forward returns.",
    )
    args = parser.parse_args()
    frame = pd.read_parquet(args.factor_table)
    factor_col = args.factor_col
    if args.raw_factor_col:
        output_col = args.zscore_output_col or args.factor_col
        frame = standardize_by_date(frame, args.raw_factor_col, output_col=output_col)
        factor_col = output_col
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ic = compute_ic_by_date(
        frame,
        factor_col=factor_col,
        return_col=args.return_col,
        rebalance_every=args.rebalance_every,
    )
    daily_diagnostics = compute_daily_ic_diagnostics(
        frame,
        factor_col=factor_col,
        return_col=args.return_col,
        rebalance_every=args.rebalance_every,
    )
    ls = long_short_returns(
        frame,
        factor_col=factor_col,
        return_col=args.return_col,
        rebalance_every=args.rebalance_every,
    )
    summary = summarize_ic(ic)
    periods_per_year = 252 / max(1, int(args.rebalance_every))
    summary.update(performance_summary(pd.Series(ls["long_short_return"]), periods_per_year=periods_per_year))
    summary["rebalance_every"] = float(args.rebalance_every)
    summary["periods_per_year"] = float(periods_per_year)
    diagnostics = summarize_factor_diagnostics(
        frame,
        daily_diagnostics,
        factor_col=factor_col,
        return_col=args.return_col,
        rebalance_every=args.rebalance_every,
    )
    pd.Series(summary).to_frame("value").to_csv(output_dir / "summary.csv")
    ic.to_csv(output_dir / "ic_by_date.csv", index=False)
    daily_diagnostics.to_csv(output_dir / "daily_diagnostics.csv", index=False)
    (output_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    ls.to_csv(output_dir / "long_short_returns.csv", index=False)
    generate_standard_plots(
        frame,
        output_dir / "plots",
        factor_col=factor_col,
        return_col=args.return_col,
        rebalance_every=args.rebalance_every,
    )
    if args.raw_factor_col:
        frame.to_parquet(output_dir / "evaluated_factor_table.parquet", index=False)


if __name__ == "__main__":
    main()
