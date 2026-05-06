"""Evaluate a FullTextNewsAlpha factor table."""

from pathlib import Path

import pandas as pd

from fulltext_news_alpha.evaluation.ic_metrics import compute_ic_by_date, summarize_ic
from fulltext_news_alpha.evaluation.plots import generate_standard_plots
from fulltext_news_alpha.evaluation.portfolio_backtest import long_short_returns, performance_summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate FullTextNewsAlpha.")
    parser.add_argument("--factor-table", required=True)
    parser.add_argument("--output-dir", default="data/reports/factor_eval")
    parser.add_argument("--return-col", default="future_20d_market_adjusted_return")
    args = parser.parse_args()
    frame = pd.read_parquet(args.factor_table)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ic = compute_ic_by_date(frame, return_col=args.return_col)
    ls = long_short_returns(frame, return_col=args.return_col)
    summary = summarize_ic(ic)
    summary.update(performance_summary(ls["long_short_return"]))
    pd.Series(summary).to_frame("value").to_csv(output_dir / "summary.csv")
    ic.to_csv(output_dir / "ic_by_date.csv", index=False)
    ls.to_csv(output_dir / "long_short_returns.csv", index=False)
    generate_standard_plots(frame, output_dir / "plots", return_col=args.return_col)


if __name__ == "__main__":
    main()
