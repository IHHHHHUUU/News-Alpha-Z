"""Utilities for SEC companyfacts point-in-time fundamental inputs.

The downloader is intentionally small and requires an explicit user agent. The
factor builder consumes the normalized output and applies the signal-date lag.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


def load_company_tickers(path: str | Path | None = None) -> pd.DataFrame:
    """Load SEC ticker/CIK mapping from a local file or the official endpoint."""

    if path is not None and Path(path).exists():
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    else:
        request = Request(SEC_COMPANY_TICKERS_URL, headers={"User-Agent": "News-Alpha-Z research"})
        with urlopen(request, timeout=30) as response:  # nosec B310 - official read-only SEC API.
            payload = json.loads(response.read().decode("utf-8"))
    frame = pd.DataFrame(payload.values() if isinstance(payload, dict) else payload)
    frame = frame.rename(columns={"ticker": "ticker", "cik_str": "cik"})
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    frame["cik"] = frame["cik"].astype(int)
    return frame[["ticker", "cik", "title"] if "title" in frame.columns else ["ticker", "cik"]]


def fetch_companyfacts(cik: int, user_agent: str, sleep_seconds: float = 0.1) -> dict:
    """Fetch one SEC companyfacts JSON document."""

    if not user_agent:
        raise ValueError("SEC requests require a descriptive user_agent.")
    request = Request(
        SEC_COMPANYFACTS_URL.format(cik=int(cik)),
        headers={"User-Agent": user_agent},
    )
    with urlopen(request, timeout=60) as response:  # nosec B310 - official read-only SEC API.
        payload = json.loads(response.read().decode("utf-8"))
    if sleep_seconds > 0:
        time.sleep(float(sleep_seconds))
    return payload


def companyfacts_to_long_frame(payload: dict, ticker: str) -> pd.DataFrame:
    """Convert SEC companyfacts JSON to a long point-in-time fact table."""

    rows = []
    facts = payload.get("facts", {}).get("us-gaap", {})
    for concept, concept_payload in facts.items():
        units = concept_payload.get("units", {})
        for unit, entries in units.items():
            for entry in entries:
                filed = entry.get("filed")
                value = entry.get("val")
                if filed is None or value is None:
                    continue
                rows.append(
                    {
                        "ticker": ticker.upper().strip(),
                        "concept": concept,
                        "unit": unit,
                        "value": value,
                        "fy": entry.get("fy"),
                        "fp": entry.get("fp"),
                        "form": entry.get("form"),
                        "filed_date": filed,
                        "end_date": entry.get("end"),
                        "frame": entry.get("frame"),
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["filed_date"] = pd.to_datetime(frame["filed_date"]).dt.date
    frame["end_date"] = pd.to_datetime(frame["end_date"], errors="coerce").dt.date
    return frame.sort_values(["ticker", "filed_date", "concept"]).reset_index(drop=True)
