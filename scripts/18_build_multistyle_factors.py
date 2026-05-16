"""Build RAM-style multistyle factors and optionally a training panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fulltext_news_alpha.features.multistyle_factors import (
    RAW_FACTOR_COLUMNS,
    build_multistyle_factors,
    read_table,
)


def _read_required(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _normalise_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    return out


def _join_without_duplicate_columns(
    base: pd.DataFrame,
    extra: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    keep = ["date", "ticker"] + [col for col in columns if col in extra.columns and col not in base.columns]
    if len(keep) <= 2:
        return base
    return base.merge(extra[keep], on=["date", "ticker"], how="left")


def _build_panel(
    factors: pd.DataFrame,
    panel: pd.DataFrame | None,
    mean_768: pd.DataFrame | None,
) -> pd.DataFrame:
    factor_cols = [f"{col}_zscore" for col in RAW_FACTOR_COLUMNS if f"{col}_zscore" in factors.columns]
    metadata_cols = [col for col in ["source_filed_date", "latest_filing_date"] if col in factors.columns]
    factor_join = factors[["date", "ticker"] + metadata_cols + factor_cols]
    if panel is None:
        base = factor_join.copy()
    else:
        base = _normalise_keys(panel)
        old_factor_cols = [col for col in factor_cols + metadata_cols if col in base.columns]
        base = base.drop(columns=old_factor_cols)
        base = base.merge(factor_join, on=["date", "ticker"], how="left")
    if mean_768 is not None:
        mean = _normalise_keys(mean_768)
        mean_cols = [
            col
            for col in mean.columns
            if col.startswith("mean_emb_") or col in {"has_news", "embedded_chunk_count"}
        ]
        base = _join_without_duplicate_columns(base, mean, mean_cols)
    return base.sort_values(["date", "ticker"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RAM-style multistyle factor library.")
    parser.add_argument("--prices", required=True)
    parser.add_argument("--panel", default=None)
    parser.add_argument("--fundamentals", default="data/features/fundamental_factors_pit.parquet")
    parser.add_argument("--analyst", default=None)
    parser.add_argument("--industry-map", default=None)
    parser.add_argument("--output", default="data/features/multistyle_factors.parquet")
    parser.add_argument("--panel-output", default="data/processed/panel_train_multistyle_768.parquet")
    parser.add_argument("--mean-768", default="data/features/news_repr_finbert_mean_768.parquet")
    args = parser.parse_args()

    prices = _read_required(args.prices)
    fundamentals = read_table(args.fundamentals)
    analyst = read_table(args.analyst)
    industry_map = read_table(args.industry_map)
    factors = build_multistyle_factors(
        prices,
        fundamentals=fundamentals,
        analyst=analyst,
        industry_map=industry_map,
        standardize=True,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    factors.to_parquet(output, index=False)
    metadata = factors.attrs.get("metadata", {})
    metadata_path = output.with_suffix(".meta.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    if args.panel_output:
        panel = read_table(args.panel)
        mean_768 = read_table(args.mean_768)
        panel_out = _build_panel(factors, panel, mean_768)
        panel_output = Path(args.panel_output)
        panel_output.parent.mkdir(parents=True, exist_ok=True)
        panel_out.to_parquet(panel_output, index=False)
        panel_meta = {
            "source_factor_file": str(output),
            "panel_rows": int(len(panel_out)),
            "zscore_factor_count": int(len([c for c in panel_out.columns if c.endswith("_zscore")])),
            "multistyle_metadata": metadata,
        }
        panel_output.with_suffix(".meta.json").write_text(
            json.dumps(panel_meta, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )

    print(json.dumps(metadata, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
