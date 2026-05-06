"""SEC EDGAR full-text hook reserved for a later version."""

from __future__ import annotations

import pandas as pd


def load_edgar_8k_stub(*_: object, **__: object) -> pd.DataFrame:
    """Return the expected schema without downloading filings."""

    return pd.DataFrame(
        columns=["news_id", "ticker", "publish_time", "title", "text", "source", "company_name"]
    )
