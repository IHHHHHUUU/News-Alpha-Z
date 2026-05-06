from __future__ import annotations

import importlib.util

import pytest


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch optional dependency not installed")
def test_full_model_forward_outputs_required_fields() -> None:
    import torch

    from fulltext_news_alpha.models.full_model import FullTextNewsAlphaModel

    model = FullTextNewsAlphaModel(factor_dim=4, chunk_dim=8, stock_dim=3)
    output = model(
        factors=torch.randn(2, 4),
        chunk_embeddings=torch.randn(2, 5, 8),
        stock_vector=torch.randn(2, 3),
        chunk_mask=torch.ones(2, 5, dtype=torch.bool),
    )
    required = {
        "factor_only_pred",
        "fusion_pred",
        "gate_news_prob",
        "mixed_pred",
        "FullTextNewsAlpha",
        "full_text_news_repr",
        "attention_weights",
        "attention_entropy",
    }
    assert required <= set(output)
    assert output["full_text_news_repr"].shape == (2, 64)
    assert output["attention_weights"].shape == (2, 5)
