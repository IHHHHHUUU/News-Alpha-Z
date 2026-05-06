"""Optional yfinance price interface."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def fetch_yfinance_prices(
    tickers: list[str],
    start: str,
    end: str,
    output: str | Path | None = None,
    execute: bool = False,
) -> pd.DataFrame:
    """Fetch prices only when explicitly enabled.

    The default mode returns an empty schema so scripts can validate wiring without
    network access or accidental data downloads.
    """

    columns = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    if not execute:
        return pd.DataFrame(columns=columns)

    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install the optional `prices` extra to use yfinance.") from exc

    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        if data.empty:
            continue
        frame = data.reset_index().rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
        frame["ticker"] = ticker
        frames.append(frame[columns])
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(output, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Optional yfinance price fetcher.")
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    prices = fetch_yfinance_prices(args.tickers, args.start, args.end, args.output, execute=args.execute)
    print(f"Rows: {len(prices)}")


if __name__ == "__main__":
    main()
