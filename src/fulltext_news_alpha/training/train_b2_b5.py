"""B2 + B5: mean chunk pooling with three-stage decoupled training.

Stage 1: train ``FactorBranch`` (factors -> future_return) with MSE.
Stage 2: train ``NewsBottleneck + FusionBranch`` (factors + news_emb -> future_return)
         with MSE on loaders containing only ``has_news == 1`` rows with labels.
Stage 3: freeze Stage 1 and Stage 2. Build gate targets from per-row branch
         errors and train ``MixtureGate`` with a binary cross-entropy loss
         against the soft target ``softmax([-err_factor/tau, -err_fusion/tau])``.

Final predictions are written per split:

    factor_only_pred, fusion_pred, gate_news_prob, mixed_pred
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
        "fulltext_news_alpha.training.train_b2_b5 requires the optional torch dependency."
    ) from exc

from fulltext_news_alpha.models.factor_branch import FactorBranch
from fulltext_news_alpha.models.fusion_branch import FusionBranch
from fulltext_news_alpha.models.mixture_gate import MixtureGate
from fulltext_news_alpha.models.news_bottleneck import NewsBottleneck
from fulltext_news_alpha.training.torch_utils import (
    SplitConfig,
    StockDayDataset,
    TrainConfig,
    build_dataloader,
    collect_predictions,
    dump_json,
    emb_column_names,
    infer_factor_columns,
    load_training_panel,
    resolve_device,
    save_checkpoint,
    set_global_seed,
    split_by_date,
    train_loop,
)


class FusionWithBottleneck(nn.Module):
    """Stage-2 module: bottleneck + fusion branch fitted together."""

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
        self.fusion = FusionBranch(
            factor_dim=factor_dim,
            news_dim=news_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        factors: torch.Tensor,
        news_emb: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        repr_64 = self.bottleneck(news_emb)
        fusion_pred = self.fusion(factors, repr_64)
        return {"fusion_pred": fusion_pred, "full_text_news_repr": repr_64}


def _factor_loss(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    pred = model(batch["factors"])
    return torch.mean((pred - batch["label"]) ** 2)


def _fusion_loss(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    outputs = model(batch["factors"], batch["news_emb"])
    return torch.mean((outputs["fusion_pred"] - batch["label"]) ** 2)


def _news_labeled_frame(frame: pd.DataFrame, label_col: str) -> pd.DataFrame:
    return frame.loc[frame["has_news"] == 1].dropna(subset=[label_col]).reset_index(drop=True)


def _gate_targets_from_errors(
    factor_pred: np.ndarray, fusion_pred: np.ndarray, label: np.ndarray, temperature: float,
) -> np.ndarray:
    err_factor = (factor_pred - label) ** 2
    err_fusion = (fusion_pred - label) ** 2
    logits = np.stack([-err_factor / temperature, -err_fusion / temperature], axis=1)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    return probs[:, 1].astype(np.float32)


def _gate_target_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "p05": float("nan"),
            "p25": float("nan"),
            "p50": float("nan"),
            "p75": float("nan"),
            "p95": float("nan"),
        }
    percentiles = np.quantile(arr, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p05": float(percentiles[0]),
        "p25": float(percentiles[1]),
        "p50": float(percentiles[2]),
        "p75": float(percentiles[3]),
        "p95": float(percentiles[4]),
    }


class GateInputDataset(torch.utils.data.Dataset):
    """Dataset for the gate training stage.

    Each row is ``(factors, full_text_news_repr, has_news, gate_target)``.
    Rows where ``has_news == 0`` are filtered out by the caller, so the gate
    never sees zero-news inputs during training.
    """

    def __init__(
        self,
        factors: np.ndarray,
        full_text_news_repr: np.ndarray,
        gate_target: np.ndarray,
    ) -> None:
        self.factors = torch.from_numpy(np.asarray(factors, dtype=np.float32))
        self.repr = torch.from_numpy(np.asarray(full_text_news_repr, dtype=np.float32))
        self.target = torch.from_numpy(np.asarray(gate_target, dtype=np.float32))

    def __len__(self) -> int:
        return int(self.factors.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "factors": self.factors[idx],
            "full_text_news_repr": self.repr[idx],
            "gate_target": self.target[idx],
        }


def _gate_loss(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    prob = model(batch["factors"], batch["full_text_news_repr"])
    target = batch["gate_target"].clamp(1e-6, 1 - 1e-6)
    return nn.functional.binary_cross_entropy(prob.clamp(1e-6, 1 - 1e-6), target)


def _predict_factor_only(
    model: nn.Module, datasets: dict[str, StockDayDataset], device: torch.device,
) -> dict[str, np.ndarray]:
    outputs: dict[str, np.ndarray] = {}
    model.eval()
    with torch.no_grad():
        for name, dataset in datasets.items():
            preds = []
            for start in range(0, len(dataset), 8192):
                end = min(start + 8192, len(dataset))
                factors = dataset.factors[start:end].to(device)
                preds.append(model(factors).detach().cpu().numpy())
            outputs[name] = np.concatenate(preds, axis=0) if preds else np.empty((0,), dtype=np.float32)
    return outputs


def _predict_fusion(
    model: nn.Module, datasets: dict[str, StockDayDataset], device: torch.device,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    model.eval()
    with torch.no_grad():
        for name, dataset in datasets.items():
            fusion_chunks: list[np.ndarray] = []
            repr_chunks: list[np.ndarray] = []
            for start in range(0, len(dataset), 8192):
                end = min(start + 8192, len(dataset))
                factors = dataset.factors[start:end].to(device)
                emb = dataset.news_emb[start:end].to(device)
                preds = model(factors, emb)
                fusion_chunks.append(preds["fusion_pred"].detach().cpu().numpy())
                repr_chunks.append(preds["full_text_news_repr"].detach().cpu().numpy())
            if fusion_chunks:
                outputs[name] = (
                    np.concatenate(fusion_chunks, axis=0),
                    np.concatenate(repr_chunks, axis=0),
                )
            else:
                outputs[name] = (np.empty((0,), dtype=np.float32), np.empty((0, 0), dtype=np.float32))
    return outputs


def train_b2_b5(
    panel_path: str | Path,
    output_dir: str | Path,
    split: SplitConfig,
    config: TrainConfig,
    label_col: str = "future_20d_market_adjusted_return",
    news_dim: int = 64,
    hidden_dim: int = 128,
    bottleneck_hidden_dim: int = 256,
    dropout: float = 0.1,
    gate_temperature: float = 0.5,
) -> dict[str, Any]:
    """Run the three-stage B2+B5 pipeline and persist all artifacts."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(config.seed)

    panel = load_training_panel(panel_path)
    if label_col not in panel.columns:
        raise KeyError(f"label_col '{label_col}' not in panel columns")
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
    fusion_frames = {
        "train": _news_labeled_frame(splits["train"], label_col),
        "valid": _news_labeled_frame(splits["valid"], label_col),
    }
    fusion_datasets = {
        name: StockDayDataset(
            frame,
            factor_cols=factor_cols,
            embedding_cols=embedding_cols,
            label_col=label_col,
            has_news_col="has_news",
            drop_missing_label=False,
        )
        for name, frame in fusion_frames.items()
    }
    fusion_loaders = {
        name: build_dataloader(
            dataset,
            batch_size=config.batch_size,
            shuffle=(name == "train"),
            num_workers=config.num_workers,
            drop_last=False,
        )
        for name, dataset in fusion_datasets.items()
    }
    device = resolve_device(config.device)

    # Stage 1 - factor branch.
    factor_model = FactorBranch(factor_dim=len(factor_cols), hidden_dim=hidden_dim, dropout=dropout)

    def _factor_forward(model: nn.Module, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"factor_only_pred": model(batch["factors"])}

    stage1_history = train_loop(
        model=factor_model,
        train_loader=loaders["train"],
        valid_loader=loaders["valid"],
        compute_loss=_factor_loss,
        config=config,
    )
    factor_model.to(device)
    factor_preds = _predict_factor_only(factor_model, datasets, device)

    # Stage 2 - bottleneck + fusion branch on news-bearing stock-days.
    fusion_model = FusionWithBottleneck(
        factor_dim=len(factor_cols),
        embedding_dim=len(embedding_cols),
        news_dim=news_dim,
        hidden_dim=hidden_dim,
        bottleneck_hidden_dim=bottleneck_hidden_dim,
        dropout=dropout,
    )
    stage2_history = train_loop(
        model=fusion_model,
        train_loader=fusion_loaders["train"],
        valid_loader=fusion_loaders["valid"],
        compute_loss=_fusion_loss,
        config=config,
    )
    fusion_model.to(device)
    fusion_outputs = _predict_fusion(fusion_model, datasets, device)

    # Stage 3 - decoupled gate.
    train_factor_pred = factor_preds["train"]
    train_fusion_pred, train_repr = fusion_outputs["train"]
    train_has_news = datasets["train"].has_news.numpy()
    train_labels = datasets["train"].labels.numpy()

    train_gate_mask = train_has_news > 0.5
    gate_target_train = _gate_targets_from_errors(
        factor_pred=train_factor_pred[train_gate_mask],
        fusion_pred=train_fusion_pred[train_gate_mask],
        label=train_labels[train_gate_mask],
        temperature=gate_temperature,
    )
    gate_train_dataset = GateInputDataset(
        factors=datasets["train"].factors.numpy()[train_gate_mask],
        full_text_news_repr=train_repr[train_gate_mask],
        gate_target=gate_target_train,
    )

    valid_factor_pred = factor_preds["valid"]
    valid_fusion_pred, valid_repr = fusion_outputs["valid"]
    valid_has_news = datasets["valid"].has_news.numpy()
    valid_labels = datasets["valid"].labels.numpy()
    valid_gate_mask = valid_has_news > 0.5
    gate_target_valid = _gate_targets_from_errors(
        factor_pred=valid_factor_pred[valid_gate_mask],
        fusion_pred=valid_fusion_pred[valid_gate_mask],
        label=valid_labels[valid_gate_mask],
        temperature=gate_temperature,
    )
    gate_target_distribution = {
        "train": _gate_target_stats(gate_target_train),
        "valid": _gate_target_stats(gate_target_valid),
    }
    gate_valid_dataset = GateInputDataset(
        factors=datasets["valid"].factors.numpy()[valid_gate_mask],
        full_text_news_repr=valid_repr[valid_gate_mask],
        gate_target=gate_target_valid,
    )

    gate_train_loader = build_dataloader(
        gate_train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    gate_valid_loader = build_dataloader(
        gate_valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    gate_model = MixtureGate(
        factor_dim=len(factor_cols),
        news_dim=news_dim,
        hidden_dim=hidden_dim // 2,
        dropout=dropout,
    )
    stage3_history = train_loop(
        model=gate_model,
        train_loader=gate_train_loader,
        valid_loader=gate_valid_loader,
        compute_loss=_gate_loss,
        config=config,
    )

    # Build final predictions for train / valid / test.
    gate_model.to(device)
    for split_name in ("train", "valid", "test"):
        factor_pred = factor_preds[split_name]
        fusion_pred, repr_matrix = fusion_outputs[split_name]
        has_news_vec = datasets[split_name].has_news.numpy()
        if len(repr_matrix) == 0 or repr_matrix.ndim != 2:
            gate_prob = np.zeros_like(factor_pred, dtype=np.float32)
        else:
            with torch.no_grad():
                factors_tensor = datasets[split_name].factors.to(device)
                repr_tensor = torch.from_numpy(np.asarray(repr_matrix, dtype=np.float32)).to(device)
                gate_prob = gate_model(factors_tensor, repr_tensor).detach().cpu().numpy()
        gate_prob = gate_prob.astype(np.float32) * has_news_vec.astype(np.float32)
        fusion_pred_full = np.where(
            has_news_vec > 0.5,
            fusion_pred,
            factor_pred,
        ).astype(np.float32)
        mixed_pred = ((1.0 - gate_prob) * factor_pred + gate_prob * fusion_pred_full).astype(np.float32)
        frame = datasets[split_name].keys.copy().reset_index(drop=True)
        frame["factor_only_pred"] = factor_pred.astype(np.float32)
        frame["fusion_pred"] = fusion_pred_full
        frame["gate_news_prob"] = gate_prob
        frame["mixed_pred"] = mixed_pred
        frame.to_parquet(output_dir / f"final_{split_name}.parquet", index=False)

    # Persist stage-specific predictions for traceability.
    for split_name in ("train", "valid", "test"):
        stage1_frame = datasets[split_name].keys.copy().reset_index(drop=True)
        stage1_frame["factor_only_pred"] = factor_preds[split_name].astype(np.float32)
        stage1_frame.to_parquet(output_dir / f"stage1_factor_{split_name}.parquet", index=False)

        fusion_pred, repr_matrix = fusion_outputs[split_name]
        stage2_frame = datasets[split_name].keys.copy().reset_index(drop=True)
        stage2_frame["fusion_pred_raw"] = fusion_pred.astype(np.float32)
        stage2_frame.to_parquet(output_dir / f"stage2_fusion_{split_name}.parquet", index=False)

    save_checkpoint(
        factor_model,
        output_dir / "b2_b5_stage1_factor.pt",
        metadata={"factor_cols": factor_cols, "label_col": label_col},
    )
    save_checkpoint(
        fusion_model,
        output_dir / "b2_b5_stage2_fusion.pt",
        metadata={
            "factor_cols": factor_cols,
            "embedding_dim": len(embedding_cols),
            "news_dim": news_dim,
            "label_col": label_col,
            "fusion_train_rows": len(fusion_datasets["train"]),
            "fusion_valid_rows": len(fusion_datasets["valid"]),
        },
    )
    save_checkpoint(
        gate_model,
        output_dir / "b2_b5_stage3_gate.pt",
        metadata={
            "factor_cols": factor_cols,
            "news_dim": news_dim,
            "gate_temperature": gate_temperature,
            "gate_target_distribution": gate_target_distribution,
        },
    )

    metadata = {
        "model": "B2+B5 decoupled mixture",
        "label_col": label_col,
        "factor_cols": factor_cols,
        "embedding_dim": len(embedding_cols),
        "news_dim": news_dim,
        "hidden_dim": hidden_dim,
        "bottleneck_hidden_dim": bottleneck_hidden_dim,
        "dropout": dropout,
        "gate_temperature": gate_temperature,
        "gate_target_distribution": gate_target_distribution,
        "split": asdict(split),
        "train_config": asdict(config),
        "split_sizes": {name: int(len(frame)) for name, frame in splits.items()},
        "dataset_sizes": {name: int(len(dataset)) for name, dataset in datasets.items()},
        "stage2_fusion_dataset_sizes": {
            name: int(len(dataset)) for name, dataset in fusion_datasets.items()
        },
        "histories": {
            "stage1": stage1_history,
            "stage2": stage2_history,
            "stage3": stage3_history,
        },
        "outputs": {
            f"final_{split_name}": str(output_dir / f"final_{split_name}.parquet")
            for split_name in ("train", "valid", "test")
        },
        "checkpoints": {
            "stage1": str(output_dir / "b2_b5_stage1_factor.pt"),
            "stage2": str(output_dir / "b2_b5_stage2_fusion.pt"),
            "stage3": str(output_dir / "b2_b5_stage3_gate.pt"),
        },
    }
    dump_json(output_dir / "metadata.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the B2+B5 decoupled mixture baseline.")
    parser.add_argument("--panel", default="data/processed/panel_train_b2_768.parquet")
    parser.add_argument("--output-dir", default="data/predictions/b2_b5")
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
    parser.add_argument("--gate-temperature", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
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
    metadata = train_b2_b5(
        panel_path=args.panel,
        output_dir=args.output_dir,
        split=split,
        config=config,
        label_col=args.label_col,
        news_dim=args.news_dim,
        hidden_dim=args.hidden_dim,
        bottleneck_hidden_dim=args.bottleneck_hidden_dim,
        dropout=args.dropout,
        gate_temperature=args.gate_temperature,
    )
    summary = {
        "stage1_best_valid_loss": metadata["histories"]["stage1"].get("best_valid_loss"),
        "stage2_best_valid_loss": metadata["histories"]["stage2"].get("best_valid_loss"),
        "stage3_best_valid_loss": metadata["histories"]["stage3"].get("best_valid_loss"),
        "split_sizes": metadata["split_sizes"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
