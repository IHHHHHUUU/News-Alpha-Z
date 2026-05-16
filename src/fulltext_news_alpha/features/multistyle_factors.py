"""RAM-style multi-style factor library with point-in-time safeguards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FACTOR_GROUPS: dict[str, list[str]] = {
    "valuation": [
        "book_to_price",
        "earnings_yield",
        "sales_to_price",
        "cash_flow_yield",
        "free_cash_flow_yield",
        "ebitda_ev",
        "dividend_yield",
        "shareholder_yield",
    ],
    "momentum_reversal": [
        "reversal_1m",
        "momentum_3m",
        "momentum_6m",
        "momentum_12m_skip_1m",
        "industry_relative_momentum",
        "momentum_acceleration",
        "high_52w_proximity",
        "reversal_5d",
        "max_daily_return_1m",
    ],
    "quality_profitability": [
        "roe",
        "roa",
        "roic",
        "gross_margin",
        "operating_margin",
        "net_margin",
        "asset_turnover",
        "accruals",
        "cfo_assets",
    ],
    "growth_investment": [
        "revenue_growth_yoy",
        "eps_growth_yoy",
        "ebitda_growth_yoy",
        "gross_profit_growth_yoy",
        "cfo_growth_yoy",
        "asset_growth_yoy",
        "capex_assets",
    ],
    "risk_volatility": [
        "market_beta_1y",
        "volatility_1m",
        "volatility_3m",
        "idiosyncratic_volatility",
        "downside_volatility",
        "max_drawdown_1y",
        "leverage",
        "interest_coverage",
    ],
    "size_liquidity_volume": [
        "log_market_cap",
        "turnover_1m",
        "dollar_volume_1m",
        "amihud_illiquidity",
        "volume_surge",
    ],
    "revision": [
        "fy1_eps_revision_1m",
        "fy2_eps_revision_3m",
        "target_price_revision_1m",
        "recommendation_change",
    ],
}

RAW_FACTOR_COLUMNS = [factor for factors in FACTOR_GROUPS.values() for factor in factors]

FUNDAMENTAL_ALIASES: dict[str, tuple[str, ...]] = {
    "assets": ("assets", "Assets"),
    "assets_current": ("assets_current", "AssetsCurrent"),
    "liabilities": ("liabilities", "Liabilities"),
    "liabilities_current": ("liabilities_current", "LiabilitiesCurrent"),
    "book_equity": (
        "book_equity",
        "stockholders_equity",
        "StockholdersEquity",
        "CommonStocksIncludingAdditionalPaidInCapital",
    ),
    "revenue_ttm": (
        "revenue_ttm",
        "revenues_ttm",
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
    ),
    "gross_profit_ttm": ("gross_profit_ttm", "GrossProfit"),
    "operating_income_ttm": ("operating_income_ttm", "OperatingIncomeLoss"),
    "net_income_ttm": ("net_income_ttm", "NetIncomeLoss"),
    "eps_ttm": ("eps_ttm", "EarningsPerShareDiluted", "EarningsPerShareBasic"),
    "cfo_ttm": ("cfo_ttm", "NetCashProvidedByUsedInOperatingActivities"),
    "capex_ttm": ("capex_ttm", "PaymentsToAcquirePropertyPlantAndEquipment"),
    "depreciation_ttm": ("depreciation_ttm", "DepreciationDepletionAndAmortization"),
    "interest_expense_ttm": (
        "interest_expense_ttm",
        "InterestExpenseNonOperating",
        "InterestExpense",
    ),
    "short_term_debt": ("short_term_debt", "LongTermDebtCurrent", "DebtCurrent"),
    "long_term_debt": ("long_term_debt", "LongTermDebtNoncurrent"),
    "cash_and_equivalents": (
        "cash_and_equivalents",
        "CashAndCashEquivalentsAtCarryingValue",
    ),
    "dividends_ttm": ("dividends_ttm", "PaymentsOfDividends"),
    "buybacks_ttm": ("buybacks_ttm", "PaymentsForRepurchaseOfCommonStock"),
    "shares_outstanding": (
        "shares_outstanding",
        "CommonStocksOutstanding",
        "EntityCommonStockSharesOutstanding",
        "WeightedAverageNumberOfDilutedSharesOutstanding",
    ),
}

ANALYST_COLUMNS = [
    "fy1_eps_estimate",
    "fy2_eps_estimate",
    "target_price",
    "recommendation_score",
]


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def _normalise_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    return out


def _rolling_max_drawdown(close: pd.Series, window: int) -> pd.Series:
    def _mdd(values: np.ndarray) -> float:
        values = values[np.isfinite(values)]
        if values.size == 0:
            return np.nan
        running_max = np.maximum.accumulate(values)
        return float(np.nanmin(values / running_max - 1.0))

    return close.rolling(window, min_periods=max(5, window // 4)).apply(_mdd, raw=True)


def _standardize_factor(frame: pd.DataFrame, raw_col: str, output_col: str) -> pd.Series:
    def _one_date(series: pd.Series) -> pd.Series:
        clean = series.replace([np.inf, -np.inf], np.nan)
        if clean.notna().sum() == 0:
            return pd.Series(np.nan, index=series.index, dtype=float)
        lower = clean.quantile(0.01)
        upper = clean.quantile(0.99)
        clipped = clean.clip(lower=lower, upper=upper)
        std = clipped.std(skipna=True, ddof=0)
        if not np.isfinite(std) or std == 0:
            return pd.Series(0.0, index=series.index, dtype=float).where(clean.notna(), np.nan)
        return (clipped - clipped.mean(skipna=True)) / std

    return frame.groupby("date", group_keys=False)[raw_col].transform(_one_date).rename(output_col)


def _coverage(frame: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    return {col: float(frame[col].notna().mean()) if col in frame else 0.0 for col in cols}


def _coverage_by_group(frame: pd.DataFrame) -> dict[str, float]:
    grouped = {}
    for group, cols in FACTOR_GROUPS.items():
        values = [frame[col].notna().mean() for col in cols if col in frame]
        grouped[group] = float(np.mean(values)) if values else 0.0
    return grouped


def _pick_column(frame: pd.DataFrame, aliases: tuple[str, ...]) -> pd.Series:
    for name in aliases:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _align_pit_table(
    stock_days: pd.DataFrame,
    facts: pd.DataFrame,
    lag_days: int = 1,
) -> pd.DataFrame:
    """Asof-align sparse PIT rows by ticker using available_date <= signal date."""

    if facts is None or facts.empty:
        return pd.DataFrame(index=stock_days.index)
    facts = facts.copy()
    facts["ticker"] = facts["ticker"].astype(str).str.upper().str.strip()
    date_source = None
    for candidate in ["available_date", "filed_date", "source_filed_date", "latest_filing_date", "date"]:
        if candidate in facts.columns:
            date_source = candidate
            break
    if date_source is None:
        raise KeyError("PIT table needs one of available_date/filed_date/source_filed_date/date.")

    facts["available_date"] = pd.to_datetime(facts[date_source])
    if date_source != "available_date":
        facts["available_date"] = facts["available_date"] + pd.Timedelta(days=lag_days)
    if "source_filed_date" not in facts.columns:
        facts["source_filed_date"] = pd.to_datetime(facts[date_source]).dt.date
    if "latest_filing_date" not in facts.columns:
        facts["latest_filing_date"] = pd.to_datetime(facts[date_source]).dt.date

    left = stock_days[["date", "ticker"]].copy()
    left["_row_id"] = np.arange(len(left))
    left["date"] = pd.to_datetime(left["date"])
    aligned_parts = []
    for ticker, left_group in left.groupby("ticker", sort=False):
        right_group = facts.loc[facts["ticker"] == ticker].sort_values("available_date")
        if right_group.empty:
            empty = left_group[["_row_id"]].copy()
            aligned_parts.append(empty)
            continue
        merged = pd.merge_asof(
            left_group.sort_values("date"),
            right_group.drop(columns=["date"], errors="ignore").sort_values("available_date"),
            left_on="date",
            right_on="available_date",
            by="ticker",
            direction="backward",
        )
        aligned_parts.append(merged)
    if not aligned_parts:
        return pd.DataFrame(index=stock_days.index)
    aligned = pd.concat(aligned_parts, ignore_index=True).sort_values("_row_id")
    aligned.index = stock_days.index
    return aligned.drop(columns=["_row_id", "date", "ticker"], errors="ignore")


def _prepare_price_frame(prices: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "ticker", "close", "volume"}
    missing = required - set(prices.columns)
    if missing:
        raise KeyError(f"Missing price columns: {sorted(missing)}")
    out = _normalise_keys(prices).sort_values(["ticker", "date"]).reset_index(drop=True)
    grouped = out.groupby("ticker", group_keys=False)
    out["close_lag1"] = grouped["close"].shift(1)
    out["volume_lag1"] = grouped["volume"].shift(1)
    out["ret_1d"] = grouped["close"].pct_change()
    out["ret_lag1"] = grouped["ret_1d"].shift(1)

    out["reversal_1m"] = -(grouped["close"].shift(1) / grouped["close"].shift(22) - 1.0)
    out["momentum_3m"] = grouped["close"].shift(1) / grouped["close"].shift(64) - 1.0
    out["momentum_6m"] = grouped["close"].shift(1) / grouped["close"].shift(127) - 1.0
    out["momentum_12m_skip_1m"] = grouped["close"].shift(21) / grouped["close"].shift(253) - 1.0
    out["momentum_acceleration"] = out["momentum_3m"] - out["momentum_6m"]
    high_source = "high" if "high" in out.columns else "close"
    high_lagged = grouped[high_source].shift(1)
    out["high_52w_proximity"] = out["close_lag1"] / high_lagged.groupby(out["ticker"]).transform(
        lambda s: s.rolling(252, min_periods=63).max()
    )
    out["reversal_5d"] = -(grouped["close"].shift(1) / grouped["close"].shift(6) - 1.0)
    out["max_daily_return_1m"] = grouped["ret_lag1"].transform(
        lambda s: s.rolling(21, min_periods=5).max()
    )

    market = out.groupby("date", as_index=False)["ret_1d"].mean().rename(
        columns={"ret_1d": "market_ret_1d"}
    )
    out = out.merge(market, on="date", how="left")
    grouped = out.groupby("ticker", group_keys=False)
    out["market_ret_lag1"] = grouped["market_ret_1d"].shift(1)
    beta = pd.Series(index=out.index, dtype=float)
    idio_vol = pd.Series(index=out.index, dtype=float)
    for _, index in out.groupby("ticker").groups.items():
        group = out.loc[index]
        beta.loc[index] = group["ret_lag1"].rolling(252, min_periods=63).cov(
            group["market_ret_lag1"]
        ) / group["market_ret_lag1"].rolling(252, min_periods=63).var()
        idio_vol.loc[index] = _idiosyncratic_volatility(group)
    out["market_beta_1y"] = beta
    out["volatility_1m"] = grouped["ret_lag1"].transform(lambda s: s.rolling(21, min_periods=5).std())
    out["volatility_3m"] = grouped["ret_lag1"].transform(lambda s: s.rolling(63, min_periods=21).std())
    out["idiosyncratic_volatility"] = idio_vol
    out["downside_volatility"] = grouped["ret_lag1"].transform(
        lambda s: s.where(s < 0).rolling(63, min_periods=10).std()
    )
    out["max_drawdown_1y"] = grouped["close_lag1"].transform(
        lambda s: _rolling_max_drawdown(s, window=252)
    )
    out["dollar_volume_1m"] = (out["close_lag1"] * out["volume_lag1"]).groupby(out["ticker"]).transform(
        lambda s: s.rolling(21, min_periods=5).mean()
    )
    out["amihud_illiquidity"] = (
        out["ret_lag1"].abs() / (out["close_lag1"] * out["volume_lag1"]).replace(0, np.nan)
    ).groupby(out["ticker"]).transform(lambda s: s.rolling(21, min_periods=5).mean())
    out["volume_surge"] = out["volume_lag1"] / grouped["volume"].shift(1).groupby(out["ticker"]).transform(
        lambda s: s.rolling(21, min_periods=5).mean()
    )
    if "market_cap" in out.columns:
        out["market_cap_price_source"] = grouped["market_cap"].shift(1)
    else:
        out["market_cap_price_source"] = np.nan
    return out


def _idiosyncratic_volatility(group: pd.DataFrame) -> pd.Series:
    cov = group["ret_lag1"].rolling(252, min_periods=63).cov(group["market_ret_lag1"])
    var = group["market_ret_lag1"].rolling(252, min_periods=63).var()
    beta = cov / var
    alpha = group["ret_lag1"].rolling(252, min_periods=63).mean() - beta * group[
        "market_ret_lag1"
    ].rolling(252, min_periods=63).mean()
    residual = group["ret_lag1"] - alpha - beta * group["market_ret_lag1"]
    return residual.rolling(252, min_periods=63).std()


def _add_industry_relative_momentum(out: pd.DataFrame, industry_map: pd.DataFrame | None) -> pd.Series:
    if industry_map is None or industry_map.empty:
        return pd.Series(np.nan, index=out.index, dtype=float)
    mapping = industry_map.copy()
    mapping["ticker"] = mapping["ticker"].astype(str).str.upper().str.strip()
    industry_col = "industry"
    if industry_col not in mapping.columns:
        non_key = [col for col in mapping.columns if col != "ticker"]
        if not non_key:
            return pd.Series(np.nan, index=out.index, dtype=float)
        industry_col = non_key[0]
    tmp = out[["date", "ticker", "momentum_6m"]].merge(
        mapping[["ticker", industry_col]],
        on="ticker",
        how="left",
    )
    industry_mean = tmp.groupby(["date", industry_col])["momentum_6m"].transform("mean")
    return tmp["momentum_6m"] - industry_mean


def _add_fundamental_factors(
    out: pd.DataFrame,
    fundamentals: pd.DataFrame | None,
    lag_days: int = 1,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata: dict[str, Any] = {"missing_fundamentals": fundamentals is None or fundamentals.empty}
    if fundamentals is None or fundamentals.empty:
        for col in [
            "source_filed_date",
            "latest_filing_date",
            "shares_outstanding",
            "market_cap",
            "enterprise_value",
        ]:
            out[col] = np.nan
        return out, metadata

    aligned = _align_pit_table(out[["date", "ticker"]], fundamentals, lag_days=lag_days)
    out = pd.concat([out, aligned.add_prefix("fund_")], axis=1)
    for canonical, aliases in FUNDAMENTAL_ALIASES.items():
        prefixed_aliases = tuple(f"fund_{alias}" for alias in aliases)
        out[canonical] = _pick_column(out, prefixed_aliases)
    for date_col in ["source_filed_date", "latest_filing_date"]:
        prefixed = f"fund_{date_col}"
        out[date_col] = out[prefixed] if prefixed in out.columns else pd.NaT

    out["total_debt"] = out["short_term_debt"].fillna(0) + out["long_term_debt"].fillna(0)
    out["ebitda_ttm"] = out["operating_income_ttm"] + out["depreciation_ttm"].fillna(0)
    out["nopat_ttm"] = out["operating_income_ttm"] * 0.79
    out["invested_capital"] = out["book_equity"] + out["total_debt"] - out["cash_and_equivalents"].fillna(0)
    out["market_cap"] = out["market_cap_price_source"]
    missing_market_cap = out["market_cap"].isna()
    out.loc[missing_market_cap, "market_cap"] = (
        out.loc[missing_market_cap, "shares_outstanding"]
        * out.loc[missing_market_cap, "close_lag1"]
    )
    out["enterprise_value"] = out["market_cap"] + out["total_debt"] - out["cash_and_equivalents"]

    out["book_to_price"] = _safe_divide(out["book_equity"], out["market_cap"])
    out["earnings_yield"] = _safe_divide(out["net_income_ttm"], out["market_cap"])
    out["sales_to_price"] = _safe_divide(out["revenue_ttm"], out["market_cap"])
    out["cash_flow_yield"] = _safe_divide(out["cfo_ttm"], out["market_cap"])
    out["free_cash_flow_yield"] = _safe_divide(out["cfo_ttm"] - out["capex_ttm"], out["market_cap"])
    out["ebitda_ev"] = _safe_divide(out["ebitda_ttm"], out["enterprise_value"])
    out["dividend_yield"] = _safe_divide(out["dividends_ttm"], out["market_cap"])
    out["shareholder_yield"] = _safe_divide(
        out["dividends_ttm"].fillna(0) + out["buybacks_ttm"].fillna(0),
        out["market_cap"],
    )
    out["roe"] = _safe_divide(out["net_income_ttm"], out["book_equity"])
    out["roa"] = _safe_divide(out["net_income_ttm"], out["assets"])
    out["roic"] = _safe_divide(out["nopat_ttm"], out["invested_capital"])
    out["gross_margin"] = _safe_divide(out["gross_profit_ttm"], out["revenue_ttm"])
    out["operating_margin"] = _safe_divide(out["operating_income_ttm"], out["revenue_ttm"])
    out["net_margin"] = _safe_divide(out["net_income_ttm"], out["revenue_ttm"])
    out["asset_turnover"] = _safe_divide(out["revenue_ttm"], out["assets"])
    out["accruals"] = _safe_divide(out["net_income_ttm"] - out["cfo_ttm"], out["assets"])
    out["cfo_assets"] = _safe_divide(out["cfo_ttm"], out["assets"])
    out["asset_growth_yoy"] = out.groupby("ticker")["assets"].pct_change(252, fill_method=None)
    for base, output in [
        ("revenue_ttm", "revenue_growth_yoy"),
        ("eps_ttm", "eps_growth_yoy"),
        ("ebitda_ttm", "ebitda_growth_yoy"),
        ("gross_profit_ttm", "gross_profit_growth_yoy"),
        ("cfo_ttm", "cfo_growth_yoy"),
    ]:
        out[output] = out.groupby("ticker")[base].pct_change(252, fill_method=None)
    out["capex_assets"] = _safe_divide(out["capex_ttm"], out["assets"])
    out["leverage"] = _safe_divide(out["total_debt"], out["assets"])
    out["interest_coverage"] = _safe_divide(out["operating_income_ttm"], out["interest_expense_ttm"])
    out["log_market_cap"] = np.log(out["market_cap"].where(out["market_cap"] > 0))
    out["turnover_1m"] = _safe_divide(
        out.groupby("ticker")["volume_lag1"].transform(lambda s: s.rolling(21, min_periods=5).mean()),
        out["shares_outstanding"],
    )
    metadata["fundamental_lag_days"] = int(lag_days)
    return out, metadata


def _add_analyst_factors(
    out: pd.DataFrame,
    analyst: pd.DataFrame | None,
    lag_days: int = 1,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata: dict[str, Any] = {"analyst_unavailable": analyst is None or analyst.empty}
    if analyst is None or analyst.empty:
        return out, metadata
    source = analyst.copy()
    if "as_of_date" in source.columns and "date" not in source.columns:
        source = source.rename(columns={"as_of_date": "date"})
    missing = {"date", "ticker"} - set(source.columns)
    if missing:
        raise KeyError(f"Analyst PIT file missing columns: {sorted(missing)}")
    aligned = _align_pit_table(out[["date", "ticker"]], source, lag_days=lag_days)
    out = pd.concat([out, aligned.add_prefix("analyst_")], axis=1)
    for col in ANALYST_COLUMNS:
        out[col] = pd.to_numeric(out.get(f"analyst_{col}", np.nan), errors="coerce")
    grouped = out.groupby("ticker", group_keys=False)
    out["fy1_eps_revision_1m"] = grouped["fy1_eps_estimate"].pct_change(21, fill_method=None)
    out["fy2_eps_revision_3m"] = grouped["fy2_eps_estimate"].pct_change(63, fill_method=None)
    out["target_price_revision_1m"] = grouped["target_price"].pct_change(21, fill_method=None)
    out["recommendation_change"] = out["recommendation_score"] - grouped["recommendation_score"].shift(21)
    metadata["analyst_lag_days"] = int(lag_days)
    return out, metadata


def build_multistyle_factors(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame | None = None,
    analyst: pd.DataFrame | None = None,
    industry_map: pd.DataFrame | None = None,
    standardize: bool = True,
) -> pd.DataFrame:
    """Build 50 RAM-style factor columns using PIT-aligned data where needed."""

    out = _prepare_price_frame(prices)
    out["industry_relative_momentum"] = _add_industry_relative_momentum(out, industry_map)
    out, fundamental_meta = _add_fundamental_factors(out, fundamentals)
    out, analyst_meta = _add_analyst_factors(out, analyst)

    for col in RAW_FACTOR_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    if standardize:
        zscores: dict[str, pd.Series] = {}
        for col in RAW_FACTOR_COLUMNS:
            zscores[f"{col}_zscore"] = _standardize_factor(out, col, f"{col}_zscore")
        out = pd.concat([out, pd.DataFrame(zscores, index=out.index)], axis=1)

    keep_cols = ["date", "ticker", "source_filed_date", "latest_filing_date"]
    zscore_cols = [f"{col}_zscore" for col in RAW_FACTOR_COLUMNS] if standardize else []
    output = out[keep_cols + RAW_FACTOR_COLUMNS + zscore_cols].copy()
    metadata = {
        **fundamental_meta,
        **analyst_meta,
        "raw_factor_count": len(RAW_FACTOR_COLUMNS),
        "zscore_factor_count": len(zscore_cols),
        "raw_factor_coverage": _coverage(output, RAW_FACTOR_COLUMNS),
        "coverage_by_group": _coverage_by_group(output),
        "unavailable_factors": [col for col in RAW_FACTOR_COLUMNS if output[col].notna().sum() == 0],
    }
    output.attrs["metadata"] = metadata
    return output


def read_table(path: str | Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def write_metadata(path: str | Path, factors: pd.DataFrame) -> None:
    metadata = factors.attrs.get("metadata", {})
    Path(path).write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str), encoding="utf-8")
