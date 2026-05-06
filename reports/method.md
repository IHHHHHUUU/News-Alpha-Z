# Method Notes

## Research Question

Can full-text chunk-attentive news representations add incremental predictive information over lagged price-volume
factors for stock-day cross-sectional return prediction?

## Representation

The text pipeline uses full article bodies rather than titles alone. Articles are cleaned, split into independent
max-256-token chunks, encoded offline by a frozen financial transformer, and stored on disk. Training code consumes only
the saved embeddings.

## Stock-Aware Attention

For each stock-day, chunk embeddings are pooled with stock/company context. The attention module is designed to reduce
weights on repeated background paragraphs, unrelated company mentions, and generic market commentary. It saves attention
weights and entropy for diagnostics.

## RAM-Style Decoupled Mixture

The factor-only branch and factor-news fusion branch are trained independently. The decoupled gate is trained after branch
prediction, using soft targets derived from relative branch errors:

```text
target_news_prob = softmax([-factor_error / T, -fusion_error / T])[fusion]
```

The final prediction is:

```text
mixed_pred = (1 - gate_news_prob) * factor_only_pred + gate_news_prob * fusion_pred
```

The factor is:

```text
FullTextNewsAlpha_raw = gate_news_prob * (fusion_pred - factor_only_pred)
```

## No-Lookahead Controls

- Pre-close news can enter same-day signals.
- After-close, weekend, and holiday news moves to the next trading day.
- Price-volume factors use lagged close/return/volume data only.
- Future returns are kept as labels and never emitted by factor feature builders.
