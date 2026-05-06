"""No-lookahead price-volume factor engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fulltext_news_alpha.factors.factor_standardization import standardize_by_date


PRICE_FACTOR_COLUMNS = [
    "momentum_5d",
    "momentum_20d",
    "momentum_60d",
    "reversal_5d",
    "volatility_20d",
    "volume_zscore_20d",
    "beta_60d",
    "max_drawdown_60d",
    "RSI",
    "overnight_gap",
]


def _rsi_from_returns(returns: pd.Series, window: int = 14) -> pd.Series:
    gains = returns.clip(lower=0).rolling(window).mean()
    losses = (-returns.clip(upper=0)).rolling(window).mean()
    rs = gains / losses.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _rolling_max_drawdown(close: pd.Series, window: int = 60) -> pd.Series:
    def _mdd(values: np.ndarray) -> float:
        running_max = np.maximum.accumulate(values)
        drawdowns = values / running_max - 1.0
        return float(np.nanmin(drawdowns))

    return close.rolling(window).apply(_mdd, raw=True)


def build_price_volume_factors(
    prices: pd.DataFrame,
    market_returns: pd.DataFrame | None = None,
    standardize: bool = True,
) -> pd.DataFrame:
    """Create basic daily factors using only data available before each signal date."""

    required = {"date", "ticker", "open", "high", "low", "close", "volume"}
    missing = required - set(prices.columns)
    if missing:
        raise KeyError(f"Missing price columns: {sorted(missing)}")

    out = prices.copy().sort_values(["ticker", "date"])
    out["date"] = pd.to_datetime(out["date"]).dt.date
    grouped = out.groupby("ticker", group_keys=False)
    out["ret_1d"] = grouped["close"].pct_change()
    out["prev_close"] = grouped["close"].shift(1)
    out["prev_open"] = grouped["open"].shift(1)

    out["momentum_5d"] = grouped["close"].shift(1) / grouped["close"].shift(6) - 1.0
    out["momentum_20d"] = grouped["close"].shift(1) / grouped["close"].shift(21) - 1.0
    out["momentum_60d"] = grouped["close"].shift(1) / grouped["close"].shift(61) - 1.0
    out["reversal_5d"] = -out["momentum_5d"]
    out["volatility_20d"] = grouped["ret_1d"].transform(lambda s: s.shift(1).rolling(20).std())
    volume_mean = grouped["volume"].transform(lambda s: s.shift(1).rolling(20).mean())
    volume_std = grouped["volume"].transform(lambda s: s.shift(1).rolling(20).std())
    prev_volume = grouped["volume"].shift(1)
    out["volume_zscore_20d"] = (prev_volume - volume_mean) / volume_std
    out["max_drawdown_60d"] = grouped["close"].transform(
        lambda s: _rolling_max_drawdown(s.shift(1), window=60)
    )
    out["RSI"] = grouped["ret_1d"].transform(lambda s: _rsi_from_returns(s.shift(1)))
    out["overnight_gap"] = out["open"] / out["prev_close"] - 1.0

    if market_returns is None:
        market = out.groupby("date", as_index=False)["ret_1d"].mean().rename(columns={"ret_1d": "market_ret_1d"})
    else:
        market = market_returns.rename(columns={"return": "market_ret_1d"}).copy()
        market["date"] = pd.to_datetime(market["date"]).dt.date
    out = out.merge(market[["date", "market_ret_1d"]], on="date", how="left")

    beta = pd.Series(index=out.index, dtype=float)
    for _, index in out.groupby("ticker").groups.items():
        group = out.loc[index]
        market_shifted = group["market_ret_1d"].shift(1)
        cov = group["ret_1d"].shift(1).rolling(60).cov(market_shifted)
        var = market_shifted.rolling(60).var()
        beta.loc[index] = cov / var
    out["beta_60d"] = beta

    factor_frame = out[["date", "ticker"] + PRICE_FACTOR_COLUMNS].copy()
    if standardize:
        for col in PRICE_FACTOR_COLUMNS:
            factor_frame = standardize_by_date(factor_frame, col, output_col=f"{col}_zscore")
    return factor_frame
