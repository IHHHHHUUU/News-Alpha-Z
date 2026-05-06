"""Portfolio turnover diagnostics."""

from __future__ import annotations

import pandas as pd


def top_bottom_members(
    frame: pd.DataFrame,
    factor_col: str = "FullTextNewsAlpha_zscore",
    quantile: float = 0.1,
) -> pd.DataFrame:
    """Return top and bottom portfolio membership by date."""

    rows: list[dict[str, object]] = []
    for date, group in frame.dropna(subset=[factor_col]).groupby("date"):
        n = max(1, int(len(group) * quantile))
        sorted_group = group.sort_values(factor_col)
        for side, part in [("short", sorted_group.head(n)), ("long", sorted_group.tail(n))]:
            for ticker in part["ticker"]:
                rows.append({"date": date, "ticker": ticker, "side": side})
    return pd.DataFrame(rows)


def portfolio_turnover(members: pd.DataFrame) -> pd.DataFrame:
    """Compute one-way turnover of long and short membership sets."""

    if members.empty:
        return pd.DataFrame(columns=["date", "side", "turnover"])
    rows: list[dict[str, object]] = []
    for side, side_frame in members.groupby("side"):
        previous: set[str] | None = None
        for date, group in side_frame.sort_values("date").groupby("date"):
            current = set(group["ticker"].astype(str))
            turnover = 0.0 if previous is None else 1.0 - len(current & previous) / max(len(current), 1)
            rows.append({"date": date, "side": side, "turnover": turnover})
            previous = current
    return pd.DataFrame(rows)
