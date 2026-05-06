"""Optional cross-sectional neutralization utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def neutralize_by_group(
    frame: pd.DataFrame,
    value_col: str,
    group_col: str,
    date_col: str = "date",
    output_col: str | None = None,
) -> pd.DataFrame:
    """Remove daily group means from a factor."""

    out = frame.copy()
    output_col = output_col or f"{value_col}_neutralized"
    group_mean = out.groupby([date_col, group_col])[value_col].transform("mean")
    out[output_col] = out[value_col] - group_mean
    out[output_col] = out[output_col].replace([np.inf, -np.inf], np.nan)
    return out
