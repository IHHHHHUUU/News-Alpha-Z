"""B3 + B5: trainable stock-aware attention pooling + decoupled TCN mixture."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from fulltext_news_alpha.training.temporal_training import (
    add_temporal_args,
    config_from_args,
    print_training_summary,
    train_temporal_b5,
    wandb_from_args,
)
from fulltext_news_alpha.training.torch_utils import SplitConfig, TrainConfig, WandbConfig


def train_b3_b5(
    panel_path: str | Path,
    output_dir: str | Path,
    split: SplitConfig,
    config: TrainConfig,
    chunk_manifest: str | Path,
    label_col: str = "future_20d_market_adjusted_return",
    news_dim: int = 64,
    hidden_dim: int = 128,
    bottleneck_hidden_dim: int = 256,
    dropout: float = 0.1,
    gate_temperature: float = 0.5,
    lookback_window: int = 30,
    kernel_size: int = 3,
    dilations: tuple[int, ...] = (1, 2, 4, 8),
    project_root: str | Path = ".",
    max_chunks_per_stock_day: int = 64,
    wandb_config: WandbConfig | None = None,
) -> dict[str, Any]:
    """Run the B3+B5 temporal pipeline."""

    return train_temporal_b5(
        news_pooling="b3",
        panel_path=panel_path,
        output_dir=output_dir,
        split=split,
        config=config,
        label_col=label_col,
        news_dim=news_dim,
        hidden_dim=hidden_dim,
        bottleneck_hidden_dim=bottleneck_hidden_dim,
        dropout=dropout,
        gate_temperature=gate_temperature,
        lookback_window=lookback_window,
        kernel_size=kernel_size,
        dilations=dilations,
        chunk_manifest=chunk_manifest,
        project_root=project_root,
        max_chunks_per_stock_day=max_chunks_per_stock_day,
        wandb_config=wandb_config,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the B3+B5 TCN baseline.")
    add_temporal_args(parser, news_pooling="b3")
    parser.set_defaults(output_dir="data/predictions/b3_b5")
    parser.add_argument("--gate-temperature", type=float, default=0.5)
    args = parser.parse_args()
    metadata = train_b3_b5(
        panel_path=args.panel,
        output_dir=args.output_dir,
        split=SplitConfig(),
        config=config_from_args(args),
        chunk_manifest=args.chunk_manifest,
        label_col=args.label_col,
        news_dim=args.news_dim,
        hidden_dim=args.hidden_dim,
        bottleneck_hidden_dim=args.bottleneck_hidden_dim,
        dropout=args.dropout,
        gate_temperature=args.gate_temperature,
        lookback_window=args.lookback_window,
        kernel_size=args.kernel_size,
        dilations=tuple(args.dilations),
        project_root=args.project_root,
        max_chunks_per_stock_day=args.max_chunks_per_stock_day,
        wandb_config=wandb_from_args(args),
    )
    print_training_summary(metadata)


if __name__ == "__main__":
    main()
