"""Factor table persistence helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_factor_table(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)


def load_factor_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
