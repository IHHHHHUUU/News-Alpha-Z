"""Forward-return labels with market adjustment."""

from __future__ import annotations

import pandas as pd


def add_forward_return_labels(
    prices: pd.DataFrame,
    horizons: tuple[int, ...] = (5, 20),
    ticker_col: str = "ticker",
    date_col: str = "date",
    close_col: str = "close",
    market_ticker: str | None = None,
) -> pd.DataFrame:
    """Add future return and market-adjusted labels for each horizon."""

    out = prices.copy()
    out[date_col] = pd.to_datetime(out[date_col]).dt.date
    out = out.sort_values([ticker_col, date_col])
    grouped = out.groupby(ticker_col, group_keys=False)

    for horizon in horizons:
        future_close = grouped[close_col].shift(-horizon)
        out[f"future_{horizon}d_return"] = future_close / out[close_col] - 1.0

    if market_ticker and market_ticker in set(out[ticker_col]):
        market = out[out[ticker_col] == market_ticker][[date_col] + [f"future_{h}d_return" for h in horizons]]
        market = market.rename(columns={f"future_{h}d_return": f"market_future_{h}d_return" for h in horizons})
        out = out.merge(market, on=date_col, how="left")
        for horizon in horizons:
            out[f"future_{horizon}d_market_adjusted_return"] = (
                out[f"future_{horizon}d_return"] - out[f"market_future_{horizon}d_return"]
            )
    else:
        for horizon in horizons:
            daily_market = out.groupby(date_col)[f"future_{horizon}d_return"].transform("mean")
            out[f"future_{horizon}d_market_adjusted_return"] = (
                out[f"future_{horizon}d_return"] - daily_market
            )
    return out
