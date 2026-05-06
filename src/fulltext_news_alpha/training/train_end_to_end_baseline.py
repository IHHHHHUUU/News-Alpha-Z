"""B4 conventional mixture baseline utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fulltext_news_alpha.training.train_gate_decoupled import mix_predictions


def conventional_mixture_predictions(frame: pd.DataFrame, default_gate: float = 0.5) -> pd.DataFrame:
    """Blend two branch predictions with a fixed gate for an interface baseline."""

    out = frame[["date", "ticker", "factor_only_pred", "fusion_pred"]].copy()
    out["gate_news_prob"] = np.clip(default_gate, 0.0, 1.0)
    out["mixed_pred"] = mix_predictions(out["factor_only_pred"], out["fusion_pred"], out["gate_news_prob"])
    return out
