"""B2 + B4: mean chunk pooling + conventional end-to-end mixture training.

The full model is trained jointly with a single MSE loss on ``mixed_pred``:

    full_text_news_repr = NewsBottleneck(mean_emb)
    factor_only_pred    = FactorBranch(factors)
    fusion_pred_raw     = FusionBranch(factors, full_text_news_repr)
    fusion_pred         = where(has_news, fusion_pred_raw, factor_only_pred)
    gate_news_prob      = MixtureGate(factors, full_text_news_repr) * has_news
    mixed_pred          = (1 - gate_news_prob) * factor_only_pred
                         + gate_news_prob * fusion_pred
    loss                = MSE(mixed_pred, future_return)

Stock-days without news effectively fall back to ``factor_only_pred`` because
both ``fusion_pred`` and ``gate_news_prob`` are masked out for them.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised only without torch installed.
    raise RuntimeError(
        "fulltext_news_alpha.training.train_b2_b4 requires the optional torch dependency."
    ) from exc

from fulltext_news_alpha.models.factor_branch import FactorBranch
from fulltext_news_alpha.models.fusion_branch import FusionBranch
from fulltext_news_alpha.models.mixture_gate import MixtureGate
from fulltext_news_alpha.models.news_bottleneck import NewsBottleneck
from fulltext_news_alpha.training.temporal_training import train_temporal_b4
from fulltext_news_alpha.training.torch_utils import (
    SplitConfig,
    StockDayDataset,
    TrainConfig,
    WandbConfig,
    build_dataloader,
    collect_predictions,
    dump_json,
    emb_column_names,
    finish_wandb_run,
    infer_factor_columns,
    init_wandb_run,
    load_training_panel,
    make_wandb_callback,
    resolve_device,
    save_checkpoint,
    set_global_seed,
    split_by_date,
    train_loop,
)


class B2B4Model(nn.Module):
    """B2 + B4 combined model: bottleneck + factor / fusion / gate branches."""

    def __init__(
        self,
        factor_dim: int,
        embedding_dim: int = 768,
        news_dim: int = 64,
        hidden_dim: int = 128,
        bottleneck_hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.bottleneck = NewsBottleneck(
            input_dim=embedding_dim,
            output_dim=news_dim,
            hidden_dim=bottleneck_hidden_dim,
            dropout=dropout,
        )
        self.factor_branch = FactorBranch(
            factor_dim=factor_dim, hidden_dim=hidden_dim, dropout=dropout,
        )
        self.fusion_branch = FusionBranch(
            factor_dim=factor_dim,
            news_dim=news_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.gate = MixtureGate(
            factor_dim=factor_dim, news_dim=news_dim, hidden_dim=hidden_dim // 2, dropout=dropout,
        )

    def forward(
        self,
        factors: torch.Tensor,
        news_emb: torch.Tensor,
        has_news: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        full_text_news_repr = self.bottleneck(news_emb)
        factor_only_pred = self.factor_branch(factors)
        fusion_pred_raw = self.fusion_branch(factors, full_text_news_repr)
        gate_prob_raw = self.gate(factors, full_text_news_repr)
        mask = has_news.clamp(0.0, 1.0).to(factors.dtype)
        fusion_pred = mask * fusion_pred_raw + (1.0 - mask) * factor_only_pred
        gate_news_prob = mask * gate_prob_raw
        mixed_pred = (1.0 - gate_news_prob) * factor_only_pred + gate_news_prob * fusion_pred
        return {
            "factor_only_pred": factor_only_pred,
            "fusion_pred": fusion_pred,
            "gate_news_prob": gate_news_prob,
            "mixed_pred": mixed_pred,
            "full_text_news_repr": full_text_news_repr,
        }


def _mse_loss(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    preds = model(batch["factors"], batch["news_emb"], batch["has_news"])
    target = batch["label"]
    return torch.mean((preds["mixed_pred"] - target) ** 2)


def _forward_only(model: nn.Module, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return model(batch["factors"], batch["news_emb"], batch["has_news"])


def _predictions_to_frame(
    keys: pd.DataFrame,
    outputs: dict[str, np.ndarray],
    columns: tuple[str, ...] = (
        "factor_only_pred",
        "fusion_pred",
        "gate_news_prob",
        "mixed_pred",
    ),
) -> pd.DataFrame:
    out = keys.copy().reset_index(drop=True)
    for column in columns:
        if column not in outputs:
            raise KeyError(f"Missing prediction column: {column}")
        out[column] = outputs[column].astype(np.float32)
    return out


def train_b2_b4(
    panel_path: str | Path,
    output_dir: str | Path,
    split: SplitConfig,
    config: TrainConfig,
    label_col: str = "future_20d_market_adjusted_return",
    news_dim: int = 64,
    hidden_dim: int = 128,
    bottleneck_hidden_dim: int = 256,
    dropout: float = 0.1,
    lookback_window: int = 30,
    kernel_size: int = 3,
    dilations: tuple[int, ...] = (1, 2, 4, 8),
    wandb_config: WandbConfig | None = None,
) -> dict[str, Any]:
    """Run the full B2+B4 training pipeline and save predictions / checkpoint."""

    return train_temporal_b4(
        news_pooling="b2",
        panel_path=panel_path,
        output_dir=output_dir,
        split=split,
        config=config,
        label_col=label_col,
        news_dim=news_dim,
        hidden_dim=hidden_dim,
        bottleneck_hidden_dim=bottleneck_hidden_dim,
        dropout=dropout,
        lookback_window=lookback_window,
        kernel_size=kernel_size,
        dilations=dilations,
        wandb_config=wandb_config,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(config.seed)

    panel = load_training_panel(panel_path)
    if label_col not in panel.columns:
        raise KeyError(f"label_col '{label_col}' not in panel columns: {list(panel.columns)[:8]} ...")
    factor_cols = infer_factor_columns(panel)
    embedding_cols = emb_column_names(panel, prefix="mean_emb_")

    splits = split_by_date(panel, split)
    datasets = {
        name: StockDayDataset(
            frame,
            factor_cols=factor_cols,
            embedding_cols=embedding_cols,
            label_col=label_col,
            has_news_col="has_news",
            drop_missing_label=(name != "test"),
        )
        for name, frame in splits.items()
    }
    loaders = {
        name: build_dataloader(
            dataset,
            batch_size=config.batch_size,
            shuffle=(name == "train"),
            num_workers=config.num_workers,
            drop_last=False,
        )
        for name, dataset in datasets.items()
    }

    model = B2B4Model(
        factor_dim=len(factor_cols),
        embedding_dim=len(embedding_cols),
        news_dim=news_dim,
        hidden_dim=hidden_dim,
        bottleneck_hidden_dim=bottleneck_hidden_dim,
        dropout=dropout,
    )

    run = init_wandb_run(
        wandb_config or WandbConfig(),
        {
            "model": "B2+B4 conventional mixture",
            "label_col": label_col,
            "factor_dim": len(factor_cols),
            "embedding_dim": len(embedding_cols),
            "news_dim": news_dim,
            "hidden_dim": hidden_dim,
            "bottleneck_hidden_dim": bottleneck_hidden_dim,
            "dropout": dropout,
            "split": asdict(split),
            "train_config": asdict(config),
            "split_sizes": {name: int(len(frame)) for name, frame in splits.items()},
            "dataset_sizes": {name: int(len(dataset)) for name, dataset in datasets.items()},
        },
    )
    try:
        history = train_loop(
            model=model,
            train_loader=loaders["train"],
            valid_loader=loaders["valid"],
            compute_loss=_mse_loss,
            config=config,
            progress_callback=make_wandb_callback(run, "b2_b4"),
        )
        if run is not None:
            run.summary.update(
                {
                    "best_valid_loss": history.get("best_valid_loss"),
                    "best_epoch": history.get("best_epoch"),
                }
            )
    finally:
        finish_wandb_run(run)

    device = resolve_device(config.device)
    model.to(device)
    predictions: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "valid", "test"):
        outputs = collect_predictions(
            model=model,
            loader=loaders[split_name],
            forward_fn=_forward_only,
            device=device,
            extra_outputs=("factor_only_pred", "fusion_pred", "gate_news_prob", "mixed_pred"),
        )
        keys = datasets[split_name].keys
        frame = _predictions_to_frame(keys, outputs)
        predictions[split_name] = frame
        frame.to_parquet(output_dir / f"{split_name}.parquet", index=False)

    save_checkpoint(
        model,
        output_dir / "b2_b4_checkpoint.pt",
        metadata={
            "factor_cols": factor_cols,
            "embedding_cols": embedding_cols,
            "label_col": label_col,
        },
    )

    metadata = {
        "model": "B2+B4 conventional mixture",
        "label_col": label_col,
        "factor_cols": factor_cols,
        "embedding_dim": len(embedding_cols),
        "news_dim": news_dim,
        "hidden_dim": hidden_dim,
        "bottleneck_hidden_dim": bottleneck_hidden_dim,
        "dropout": dropout,
        "split": asdict(split),
        "train_config": asdict(config),
        "split_sizes": {name: int(len(frame)) for name, frame in splits.items()},
        "dataset_sizes": {name: int(len(dataset)) for name, dataset in datasets.items()},
        "history": history,
        "outputs": {
            split_name: str(output_dir / f"{split_name}.parquet")
            for split_name in ("train", "valid", "test")
        },
        "checkpoint": str(output_dir / "b2_b4_checkpoint.pt"),
    }
    dump_json(output_dir / "metadata.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the B2+B4 baseline.")
    parser.add_argument("--panel", default="data/processed/panel_train_b2_768.parquet")
    parser.add_argument("--output-dir", default="data/predictions/b2_b4")
    parser.add_argument("--label-col", default="future_20d_market_adjusted_return")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--bottleneck-hidden-dim", type=int, default=256)
    parser.add_argument("--news-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lookback-window", type=int, default=30)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dilations", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default="news-alpha-z")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-mode", default=None, help="wandb mode, e.g. online/offline/disabled.")
    args = parser.parse_args()

    split = SplitConfig()
    config = TrainConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.patience,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
    )
    metadata = train_b2_b4(
        panel_path=args.panel,
        output_dir=args.output_dir,
        split=split,
        config=config,
        label_col=args.label_col,
        news_dim=args.news_dim,
        hidden_dim=args.hidden_dim,
        bottleneck_hidden_dim=args.bottleneck_hidden_dim,
        dropout=args.dropout,
        lookback_window=args.lookback_window,
        kernel_size=args.kernel_size,
        dilations=tuple(args.dilations),
        wandb_config=WandbConfig(
            enabled=args.wandb,
            project=args.wandb_project,
            entity=args.wandb_entity,
            run_name=args.wandb_run_name,
            group=args.wandb_group,
            mode=args.wandb_mode,
        ),
    )
    summary = {
        "best_valid_loss": metadata["history"].get("best_valid_loss"),
        "best_epoch": metadata["history"].get("best_epoch"),
        "split_sizes": metadata["split_sizes"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
