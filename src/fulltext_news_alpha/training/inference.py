"""Final prediction assembly."""

from __future__ import annotations

import pandas as pd

from fulltext_news_alpha.training.train_gate_decoupled import mix_predictions


def assemble_predictions(
    factor_preds: pd.DataFrame,
    fusion_preds: pd.DataFrame,
    gate_preds: pd.DataFrame,
) -> pd.DataFrame:
    """Merge branch and gate outputs into final model predictions."""

    out = factor_preds.merge(fusion_preds, on=["date", "ticker"], how="inner")
    out = out.merge(gate_preds[["date", "ticker", "gate_news_prob"]], on=["date", "ticker"], how="inner")
    out["mixed_pred"] = mix_predictions(out["factor_only_pred"], out["fusion_pred"], out["gate_news_prob"])
    return out
