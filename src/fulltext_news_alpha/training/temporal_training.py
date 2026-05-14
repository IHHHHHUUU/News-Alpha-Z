"""Shared TCN training pipelines for B2/B3 x B4/B5 combinations."""

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
        "fulltext_news_alpha.training.temporal_training requires the optional torch dependency."
    ) from exc

from fulltext_news_alpha.models.mixture_gate import MixtureGate
from fulltext_news_alpha.models.temporal_mixture import (
    TemporalFactorModel,
    TemporalFusionModel,
    TemporalMixtureModel,
)
from fulltext_news_alpha.training.sequence_data import (
    B2SequenceStockDayDataset,
    B3SequenceStockDayDataset,
    ChunkEmbeddingIndex,
    load_chunk_embedding_index,
)
from fulltext_news_alpha.training.torch_utils import (
    SplitConfig,
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


DEFAULT_DILATIONS = (1, 2, 4, 8)


def _model_name(news_pooling: str, training_method: str) -> str:
    return f"{news_pooling.upper()}+{training_method.upper()} TCN"


def _mse_mixed_loss(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    preds = model(batch)
    return torch.mean((preds["mixed_pred"] - batch["label"]) ** 2)


def _factor_loss(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    preds = model(batch)
    return torch.mean((preds["factor_only_pred"] - batch["label"]) ** 2)


def _fusion_loss(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    preds = model(batch)
    return torch.mean((preds["fusion_pred"] - batch["label"]) ** 2)


def _forward_only(model: nn.Module, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return model(batch)


def _news_labeled_frame(frame: pd.DataFrame, label_col: str) -> pd.DataFrame:
    return frame.loc[frame["has_news"] == 1].dropna(subset=[label_col]).reset_index(drop=True)


def _gate_targets_from_errors(
    factor_pred: np.ndarray,
    fusion_pred: np.ndarray,
    label: np.ndarray,
    temperature: float,
) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("gate temperature must be positive")
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


def _b5_data_diagnostics(
    splits: dict[str, pd.DataFrame],
    datasets: dict[str, B2SequenceStockDayDataset | B3SequenceStockDayDataset],
    label_col: str,
    lookback_window: int,
    fusion_frames: dict[str, pd.DataFrame] | None = None,
    fusion_datasets: dict[str, B2SequenceStockDayDataset | B3SequenceStockDayDataset] | None = None,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "lookback_window": int(lookback_window),
        "split_sizes": {name: int(len(frame)) for name, frame in splits.items()},
        "dataset_sizes": {name: int(len(dataset)) for name, dataset in datasets.items()},
        "has_news_rows": {
            name: int((frame["has_news"].fillna(0).astype(float) > 0.5).sum())
            for name, frame in splits.items()
        },
        "label_available_rows": {
            name: int(frame[label_col].notna().sum())
            for name, frame in splits.items()
        },
    }
    if fusion_frames is not None:
        diagnostics["stage2_fusion_frame_sizes"] = {
            name: int(len(frame)) for name, frame in fusion_frames.items()
        }
    if fusion_datasets is not None:
        diagnostics["stage2_fusion_dataset_sizes"] = {
            name: int(len(dataset)) for name, dataset in fusion_datasets.items()
        }
    return diagnostics


def _ensure_non_empty_b5_dataset(
    datasets: dict[str, B2SequenceStockDayDataset | B3SequenceStockDayDataset],
    required_splits: tuple[str, ...],
    diagnostics: dict[str, Any],
    context: str,
) -> None:
    empty = [name for name in required_splits if len(datasets[name]) == 0]
    if empty:
        raise ValueError(f"Empty B5 {context} dataset(s): {empty}; diagnostics={diagnostics}")


def _validate_gate_training_inputs(
    split_name: str,
    factor_pred: np.ndarray,
    fusion_pred: np.ndarray,
    labels: np.ndarray,
    gate_mask: np.ndarray,
    factor_state: np.ndarray,
    news_state: np.ndarray,
    diagnostics: dict[str, Any],
) -> np.ndarray:
    if not bool(np.any(gate_mask)):
        raise ValueError(f"Empty Stage 3 gate {split_name} mask; diagnostics={diagnostics}")
    selected = {
        "factor_pred": np.asarray(factor_pred)[gate_mask],
        "fusion_pred": np.asarray(fusion_pred)[gate_mask],
        "labels": np.asarray(labels)[gate_mask],
        "factor_state": np.asarray(factor_state)[gate_mask],
        "news_state": np.asarray(news_state)[gate_mask],
    }
    non_finite = {
        name: int((~np.isfinite(values)).sum())
        for name, values in selected.items()
    }
    if any(count > 0 for count in non_finite.values()):
        raise ValueError(
            f"Non-finite Stage 3 gate {split_name} inputs: {non_finite}; "
            f"diagnostics={diagnostics}"
        )
    return gate_mask


class TemporalGateInputDataset(torch.utils.data.Dataset):
    """Gate dataset over encoded factor/news states."""

    def __init__(
        self,
        factor_state: np.ndarray,
        full_text_news_repr: np.ndarray,
        gate_target: np.ndarray,
    ) -> None:
        self.factor_state = torch.from_numpy(np.asarray(factor_state, dtype=np.float32))
        self.news_state = torch.from_numpy(np.asarray(full_text_news_repr, dtype=np.float32))
        self.target = torch.from_numpy(np.asarray(gate_target, dtype=np.float32))

    def __len__(self) -> int:
        return int(self.factor_state.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "factor_state": self.factor_state[idx],
            "full_text_news_repr": self.news_state[idx],
            "gate_target": self.target[idx],
        }


def _gate_loss(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    prob = model(batch["factor_state"], batch["full_text_news_repr"])
    target = batch["gate_target"].clamp(1e-6, 1 - 1e-6)
    return nn.functional.binary_cross_entropy(prob.clamp(1e-6, 1 - 1e-6), target)


def _build_dataset(
    news_pooling: str,
    sample_frame: pd.DataFrame,
    history_frame: pd.DataFrame,
    factor_cols: list[str],
    embedding_cols: list[str],
    label_col: str,
    lookback_window: int,
    drop_missing_label: bool,
    chunk_index: ChunkEmbeddingIndex | None,
    max_chunks_per_stock_day: int,
    ticker_to_id: dict[str, int] | None,
) -> B2SequenceStockDayDataset | B3SequenceStockDayDataset:
    if news_pooling == "b2":
        return B2SequenceStockDayDataset(
            sample_frame=sample_frame,
            history_frame=history_frame,
            factor_cols=factor_cols,
            embedding_cols=embedding_cols,
            label_col=label_col,
            lookback_window=lookback_window,
            drop_missing_label=drop_missing_label,
            ticker_to_id=ticker_to_id,
        )
    if chunk_index is None:
        raise ValueError("chunk_index is required for B3 sequence datasets")
    return B3SequenceStockDayDataset(
        sample_frame=sample_frame,
        history_frame=history_frame,
        factor_cols=factor_cols,
        chunk_index=chunk_index,
        label_col=label_col,
        lookback_window=lookback_window,
        max_chunks_per_stock_day=max_chunks_per_stock_day,
        drop_missing_label=drop_missing_label,
        ticker_to_id=ticker_to_id,
    )


def _prepare_data(
    news_pooling: str,
    panel_path: str | Path,
    split: SplitConfig,
    label_col: str,
    lookback_window: int,
    chunk_manifest: str | Path | None,
    project_root: str | Path,
    max_chunks_per_stock_day: int,
) -> tuple[
    pd.DataFrame,
    dict[str, pd.DataFrame],
    list[str],
    list[str],
    int,
    ChunkEmbeddingIndex | None,
    dict[str, int],
]:
    panel = load_training_panel(panel_path)
    if label_col not in panel.columns:
        raise KeyError(f"label_col '{label_col}' not in panel columns")
    tickers = sorted(panel["ticker"].astype(str).str.upper().str.strip().unique())
    ticker_to_id = {ticker: idx for idx, ticker in enumerate(tickers, start=1)}
    factor_cols = infer_factor_columns(panel)
    embedding_cols: list[str] = []
    chunk_index: ChunkEmbeddingIndex | None = None
    if news_pooling == "b2":
        embedding_cols = emb_column_names(panel, prefix="mean_emb_")
        embedding_dim = len(embedding_cols)
    elif news_pooling == "b3":
        if chunk_manifest is None:
            raise ValueError("--chunk-manifest is required for B3 training")
        chunk_index = load_chunk_embedding_index(
            chunk_manifest,
            project_root=project_root,
            max_chunks_per_stock_day=max_chunks_per_stock_day,
        )
        embedding_dim = chunk_index.embedding_dim
    else:
        raise ValueError("news_pooling must be 'b2' or 'b3'")
    splits = split_by_date(panel, split)
    return panel, splits, factor_cols, embedding_cols, embedding_dim, chunk_index, ticker_to_id


def _build_split_datasets(
    news_pooling: str,
    splits: dict[str, pd.DataFrame],
    panel: pd.DataFrame,
    factor_cols: list[str],
    embedding_cols: list[str],
    label_col: str,
    lookback_window: int,
    chunk_index: ChunkEmbeddingIndex | None,
    max_chunks_per_stock_day: int,
    ticker_to_id: dict[str, int] | None,
) -> dict[str, B2SequenceStockDayDataset | B3SequenceStockDayDataset]:
    return {
        name: _build_dataset(
            news_pooling=news_pooling,
            sample_frame=frame,
            history_frame=panel,
            factor_cols=factor_cols,
            embedding_cols=embedding_cols,
            label_col=label_col,
            lookback_window=lookback_window,
            drop_missing_label=(name != "test"),
            chunk_index=chunk_index,
            max_chunks_per_stock_day=max_chunks_per_stock_day,
            ticker_to_id=ticker_to_id,
        )
        for name, frame in splits.items()
    }


def _build_loaders(
    datasets: dict[str, torch.utils.data.Dataset],
    config: TrainConfig,
) -> dict[str, torch.utils.data.DataLoader]:
    return {
        name: build_dataloader(
            dataset,
            batch_size=config.batch_size,
            shuffle=(name == "train"),
            num_workers=config.num_workers,
            drop_last=False,
        )
        for name, dataset in datasets.items()
    }


def _prediction_frame(
    keys: pd.DataFrame,
    outputs: dict[str, np.ndarray],
    include_attention: bool,
) -> pd.DataFrame:
    out = keys.copy().reset_index(drop=True)
    for column in ("factor_only_pred", "fusion_pred", "gate_news_prob", "mixed_pred"):
        out[column] = outputs[column].astype(np.float32)
    if include_attention and "attention_entropy" in outputs:
        out["attention_entropy"] = outputs["attention_entropy"].astype(np.float32)
    return out


def _save_attention_weights(
    output_dir: Path,
    split_name: str,
    keys: pd.DataFrame,
    outputs: dict[str, np.ndarray],
) -> str | None:
    weights = outputs.get("target_attention_weights")
    if weights is None:
        return None
    weight_cols = [f"attention_weight_{idx}" for idx in range(weights.shape[1])]
    frame = pd.concat(
        [
            keys.reset_index(drop=True),
            pd.DataFrame(weights.astype(np.float32), columns=pd.Index(weight_cols)),
        ],
        axis=1,
    )
    path = output_dir / f"attention_weights_{split_name}.parquet"
    frame.to_parquet(path, index=False)
    return str(path)


def train_temporal_b4(
    news_pooling: str,
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
    dilations: tuple[int, ...] = DEFAULT_DILATIONS,
    chunk_manifest: str | Path | None = None,
    project_root: str | Path = ".",
    max_chunks_per_stock_day: int = 64,
    wandb_config: WandbConfig | None = None,
) -> dict[str, Any]:
    """Run B2/B3 + B4 TCN training and save predictions / checkpoint."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(config.seed)
    panel, splits, factor_cols, embedding_cols, embedding_dim, chunk_index, ticker_to_id = _prepare_data(
        news_pooling=news_pooling,
        panel_path=panel_path,
        split=split,
        label_col=label_col,
        lookback_window=lookback_window,
        chunk_manifest=chunk_manifest,
        project_root=project_root,
        max_chunks_per_stock_day=max_chunks_per_stock_day,
    )
    datasets = _build_split_datasets(
        news_pooling=news_pooling,
        splits=splits,
        panel=panel,
        factor_cols=factor_cols,
        embedding_cols=embedding_cols,
        label_col=label_col,
        lookback_window=lookback_window,
        chunk_index=chunk_index,
        max_chunks_per_stock_day=max_chunks_per_stock_day,
        ticker_to_id=ticker_to_id,
    )
    loaders = _build_loaders(datasets, config)
    model = TemporalMixtureModel(
        news_pooling=news_pooling,
        factor_dim=len(factor_cols),
        embedding_dim=embedding_dim,
        daily_news_dim=news_dim,
        hidden_dim=hidden_dim,
        bottleneck_hidden_dim=bottleneck_hidden_dim,
        kernel_size=kernel_size,
        dilations=dilations,
        dropout=dropout,
        chunk_index=chunk_index,
        max_chunks_per_stock_day=max_chunks_per_stock_day,
        num_tickers=len(ticker_to_id),
    )
    run = init_wandb_run(
        wandb_config or WandbConfig(),
        {
            "model": _model_name(news_pooling, "b4"),
            "label_col": label_col,
            "factor_dim": len(factor_cols),
            "embedding_dim": embedding_dim,
            "num_tickers": len(ticker_to_id),
            "news_dim": news_dim,
            "hidden_dim": hidden_dim,
            "lookback_window": lookback_window,
            "kernel_size": kernel_size,
            "dilations": list(dilations),
            "dropout": dropout,
            "split": asdict(split),
            "train_config": asdict(config),
        },
    )
    try:
        history = train_loop(
            model=model,
            train_loader=loaders["train"],
            valid_loader=loaders["valid"],
            compute_loss=_mse_mixed_loss,
            config=config,
            progress_callback=make_wandb_callback(run, f"{news_pooling}_b4"),
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
    include_attention = news_pooling == "b3"
    outputs_by_split: dict[str, str] = {}
    attention_outputs: dict[str, str] = {}
    for split_name in ("train", "valid", "test"):
        outputs = collect_predictions(
            model=model,
            loader=loaders[split_name],
            forward_fn=_forward_only,
            device=device,
            extra_outputs=(
                "factor_only_pred",
                "fusion_pred",
                "gate_news_prob",
                "mixed_pred",
                "attention_entropy",
                "target_attention_weights",
            ),
        )
        frame = _prediction_frame(datasets[split_name].keys, outputs, include_attention)
        path = output_dir / f"{split_name}.parquet"
        frame.to_parquet(path, index=False)
        outputs_by_split[split_name] = str(path)
        weights_path = _save_attention_weights(
            output_dir,
            split_name,
            datasets[split_name].keys,
            outputs,
        )
        if weights_path is not None:
            attention_outputs[split_name] = weights_path

    checkpoint_path = output_dir / f"{news_pooling}_b4_tcn_checkpoint.pt"
    save_checkpoint(
        model,
        checkpoint_path,
        metadata={
            "factor_cols": factor_cols,
            "embedding_cols": embedding_cols,
            "label_col": label_col,
            "news_pooling": news_pooling,
            "num_tickers": len(ticker_to_id),
        },
    )
    metadata = {
        "model": _model_name(news_pooling, "b4"),
        "label_col": label_col,
        "factor_cols": factor_cols,
        "embedding_dim": embedding_dim,
        "num_tickers": len(ticker_to_id),
        "news_dim": news_dim,
        "hidden_dim": hidden_dim,
        "bottleneck_hidden_dim": bottleneck_hidden_dim,
        "dropout": dropout,
        "lookback_window": lookback_window,
        "kernel_size": kernel_size,
        "dilations": list(dilations),
        "split": asdict(split),
        "train_config": asdict(config),
        "split_sizes": {name: int(len(frame)) for name, frame in splits.items()},
        "dataset_sizes": {name: int(len(dataset)) for name, dataset in datasets.items()},
        "history": history,
        "outputs": outputs_by_split,
        "attention_outputs": attention_outputs,
        "checkpoint": str(checkpoint_path),
    }
    dump_json(output_dir / "metadata.json", metadata)
    return metadata


def _predict_factor(
    model: nn.Module,
    datasets: dict[str, B2SequenceStockDayDataset | B3SequenceStockDayDataset],
    loaders: dict[str, torch.utils.data.DataLoader],
    device: torch.device,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, loader in loaders.items():
        pred = collect_predictions(
            model=model,
            loader=loader,
            forward_fn=_forward_only,
            device=device,
            extra_outputs=("factor_only_pred", "factor_state"),
        )
        outputs[name] = (pred["factor_only_pred"], pred["factor_state"])
    return outputs


def _predict_fusion(
    model: nn.Module,
    loaders: dict[str, torch.utils.data.DataLoader],
    device: torch.device,
) -> dict[str, dict[str, np.ndarray]]:
    outputs: dict[str, dict[str, np.ndarray]] = {}
    for name, loader in loaders.items():
        outputs[name] = collect_predictions(
            model=model,
            loader=loader,
            forward_fn=_forward_only,
            device=device,
            extra_outputs=(
                "fusion_pred",
                "full_text_news_repr",
                "factor_state",
                "attention_entropy",
                "target_attention_weights",
            ),
        )
    return outputs


def train_temporal_b5(
    news_pooling: str,
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
    lookback_window: int = 30,
    kernel_size: int = 3,
    dilations: tuple[int, ...] = DEFAULT_DILATIONS,
    chunk_manifest: str | Path | None = None,
    project_root: str | Path = ".",
    max_chunks_per_stock_day: int = 64,
    wandb_config: WandbConfig | None = None,
) -> dict[str, Any]:
    """Run B2/B3 + B5 TCN training and save predictions / checkpoints."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(config.seed)
    panel, splits, factor_cols, embedding_cols, embedding_dim, chunk_index, ticker_to_id = _prepare_data(
        news_pooling=news_pooling,
        panel_path=panel_path,
        split=split,
        label_col=label_col,
        lookback_window=lookback_window,
        chunk_manifest=chunk_manifest,
        project_root=project_root,
        max_chunks_per_stock_day=max_chunks_per_stock_day,
    )
    datasets = _build_split_datasets(
        news_pooling=news_pooling,
        splits=splits,
        panel=panel,
        factor_cols=factor_cols,
        embedding_cols=embedding_cols,
        label_col=label_col,
        lookback_window=lookback_window,
        chunk_index=chunk_index,
        max_chunks_per_stock_day=max_chunks_per_stock_day,
        ticker_to_id=ticker_to_id,
    )
    fusion_frames = {
        "train": _news_labeled_frame(splits["train"], label_col),
        "valid": _news_labeled_frame(splits["valid"], label_col),
    }
    fusion_datasets = {
        name: _build_dataset(
            news_pooling=news_pooling,
            sample_frame=frame,
            history_frame=panel,
            factor_cols=factor_cols,
            embedding_cols=embedding_cols,
            label_col=label_col,
            lookback_window=lookback_window,
            drop_missing_label=False,
            chunk_index=chunk_index,
            max_chunks_per_stock_day=max_chunks_per_stock_day,
            ticker_to_id=ticker_to_id,
        )
        for name, frame in fusion_frames.items()
    }
    diagnostics = _b5_data_diagnostics(
        splits=splits,
        datasets=datasets,
        label_col=label_col,
        lookback_window=lookback_window,
        fusion_frames=fusion_frames,
        fusion_datasets=fusion_datasets,
    )
    _ensure_non_empty_b5_dataset(datasets, ("train", "valid", "test"), diagnostics, "base")
    _ensure_non_empty_b5_dataset(fusion_datasets, ("train", "valid"), diagnostics, "Stage 2 fusion")
    loaders = _build_loaders(datasets, config)
    fusion_loaders = _build_loaders(fusion_datasets, config)
    device = resolve_device(config.device)
    run = init_wandb_run(
        wandb_config or WandbConfig(),
        {
            "model": _model_name(news_pooling, "b5"),
            "label_col": label_col,
            "factor_dim": len(factor_cols),
            "embedding_dim": embedding_dim,
            "num_tickers": len(ticker_to_id),
            "news_dim": news_dim,
            "hidden_dim": hidden_dim,
            "gate_temperature": gate_temperature,
            "lookback_window": lookback_window,
            "kernel_size": kernel_size,
            "dilations": list(dilations),
            "dropout": dropout,
            "split": asdict(split),
            "train_config": asdict(config),
        },
    )

    factor_model = TemporalFactorModel(
        factor_dim=len(factor_cols),
        hidden_dim=hidden_dim,
        kernel_size=kernel_size,
        dilations=dilations,
        dropout=dropout,
    )
    stage1_history = train_loop(
        model=factor_model,
        train_loader=loaders["train"],
        valid_loader=loaders["valid"],
        compute_loss=_factor_loss,
        config=config,
        progress_callback=make_wandb_callback(run, f"{news_pooling}_b5_stage1_factor"),
    )
    factor_model.to(device)
    factor_outputs = _predict_factor(factor_model, datasets, loaders, device)

    fusion_model = TemporalFusionModel(
        news_pooling=news_pooling,
        factor_dim=len(factor_cols),
        embedding_dim=embedding_dim,
        daily_news_dim=news_dim,
        hidden_dim=hidden_dim,
        bottleneck_hidden_dim=bottleneck_hidden_dim,
        kernel_size=kernel_size,
        dilations=dilations,
        dropout=dropout,
        chunk_index=chunk_index,
        max_chunks_per_stock_day=max_chunks_per_stock_day,
        num_tickers=len(ticker_to_id),
    )
    stage2_history = train_loop(
        model=fusion_model,
        train_loader=fusion_loaders["train"],
        valid_loader=fusion_loaders["valid"],
        compute_loss=_fusion_loss,
        config=config,
        progress_callback=make_wandb_callback(run, f"{news_pooling}_b5_stage2_fusion"),
    )
    fusion_model.to(device)
    fusion_outputs = _predict_fusion(fusion_model, loaders, device)

    train_factor_pred, train_factor_state = factor_outputs["train"]
    train_fusion = fusion_outputs["train"]
    train_has_news = datasets["train"].has_news.numpy()
    train_labels = datasets["train"].labels.numpy()
    train_gate_mask = train_has_news > 0.5
    train_gate_mask = _validate_gate_training_inputs(
        split_name="train",
        factor_pred=train_factor_pred,
        fusion_pred=train_fusion["fusion_pred"],
        labels=train_labels,
        gate_mask=train_gate_mask,
        factor_state=train_factor_state,
        news_state=train_fusion["full_text_news_repr"],
        diagnostics=diagnostics,
    )
    gate_target_train = _gate_targets_from_errors(
        factor_pred=train_factor_pred[train_gate_mask],
        fusion_pred=train_fusion["fusion_pred"][train_gate_mask],
        label=train_labels[train_gate_mask],
        temperature=gate_temperature,
    )
    gate_train_dataset = TemporalGateInputDataset(
        factor_state=train_factor_state[train_gate_mask],
        full_text_news_repr=train_fusion["full_text_news_repr"][train_gate_mask],
        gate_target=gate_target_train,
    )

    valid_factor_pred, valid_factor_state = factor_outputs["valid"]
    valid_fusion = fusion_outputs["valid"]
    valid_has_news = datasets["valid"].has_news.numpy()
    valid_labels = datasets["valid"].labels.numpy()
    valid_gate_mask = valid_has_news > 0.5
    valid_gate_mask = _validate_gate_training_inputs(
        split_name="valid",
        factor_pred=valid_factor_pred,
        fusion_pred=valid_fusion["fusion_pred"],
        labels=valid_labels,
        gate_mask=valid_gate_mask,
        factor_state=valid_factor_state,
        news_state=valid_fusion["full_text_news_repr"],
        diagnostics=diagnostics,
    )
    gate_target_valid = _gate_targets_from_errors(
        factor_pred=valid_factor_pred[valid_gate_mask],
        fusion_pred=valid_fusion["fusion_pred"][valid_gate_mask],
        label=valid_labels[valid_gate_mask],
        temperature=gate_temperature,
    )
    gate_target_distribution = {
        "train": _gate_target_stats(gate_target_train),
        "valid": _gate_target_stats(gate_target_valid),
    }
    gate_valid_dataset = TemporalGateInputDataset(
        factor_state=valid_factor_state[valid_gate_mask],
        full_text_news_repr=valid_fusion["full_text_news_repr"][valid_gate_mask],
        gate_target=gate_target_valid,
    )
    gate_model = MixtureGate(
        factor_dim=hidden_dim,
        news_dim=hidden_dim,
        hidden_dim=max(1, hidden_dim // 2),
        dropout=dropout,
    )
    stage3_history = train_loop(
        model=gate_model,
        train_loader=build_dataloader(
            gate_train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        ),
        valid_loader=build_dataloader(
            gate_valid_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        ),
        compute_loss=_gate_loss,
        config=config,
        progress_callback=make_wandb_callback(run, f"{news_pooling}_b5_stage3_gate"),
    )
    gate_model.to(device)

    include_attention = news_pooling == "b3"
    final_outputs: dict[str, str] = {}
    attention_outputs: dict[str, str] = {}
    for split_name in ("train", "valid", "test"):
        factor_pred, factor_state = factor_outputs[split_name]
        fusion = fusion_outputs[split_name]
        has_news_vec = datasets[split_name].has_news.numpy()
        if len(factor_pred) == 0:
            gate_prob = np.empty((0,), dtype=np.float32)
        else:
            with torch.no_grad():
                factors_tensor = torch.from_numpy(np.asarray(factor_state, dtype=np.float32)).to(device)
                news_tensor = torch.from_numpy(
                    np.asarray(fusion["full_text_news_repr"], dtype=np.float32)
                ).to(device)
                gate_prob = gate_model(factors_tensor, news_tensor).detach().cpu().numpy()
        gate_prob = gate_prob.astype(np.float32) * has_news_vec.astype(np.float32)
        fusion_pred_full = np.where(has_news_vec > 0.5, fusion["fusion_pred"], factor_pred)
        mixed_pred = ((1.0 - gate_prob) * factor_pred + gate_prob * fusion_pred_full).astype(np.float32)
        frame = datasets[split_name].keys.copy().reset_index(drop=True)
        frame["factor_only_pred"] = factor_pred.astype(np.float32)
        frame["fusion_pred"] = fusion_pred_full.astype(np.float32)
        frame["gate_news_prob"] = gate_prob
        frame["mixed_pred"] = mixed_pred
        if include_attention and "attention_entropy" in fusion:
            frame["attention_entropy"] = fusion["attention_entropy"].astype(np.float32)
        path = output_dir / f"final_{split_name}.parquet"
        frame.to_parquet(path, index=False)
        final_outputs[f"final_{split_name}"] = str(path)
        weights_path = _save_attention_weights(
            output_dir,
            split_name,
            datasets[split_name].keys,
            fusion,
        )
        if weights_path is not None:
            attention_outputs[split_name] = weights_path

        stage1_frame = datasets[split_name].keys.copy().reset_index(drop=True)
        stage1_frame["factor_only_pred"] = factor_pred.astype(np.float32)
        stage1_frame.to_parquet(output_dir / f"stage1_factor_{split_name}.parquet", index=False)
        stage2_frame = datasets[split_name].keys.copy().reset_index(drop=True)
        stage2_frame["fusion_pred_raw"] = fusion["fusion_pred"].astype(np.float32)
        if include_attention and "attention_entropy" in fusion:
            stage2_frame["attention_entropy"] = fusion["attention_entropy"].astype(np.float32)
        stage2_frame.to_parquet(output_dir / f"stage2_fusion_{split_name}.parquet", index=False)

    checkpoints = {
        "stage1": str(output_dir / f"{news_pooling}_b5_stage1_factor.pt"),
        "stage2": str(output_dir / f"{news_pooling}_b5_stage2_fusion.pt"),
        "stage3": str(output_dir / f"{news_pooling}_b5_stage3_gate.pt"),
    }
    save_checkpoint(factor_model, checkpoints["stage1"], metadata={"factor_cols": factor_cols})
    save_checkpoint(
        fusion_model,
        checkpoints["stage2"],
        metadata={
            "factor_cols": factor_cols,
            "embedding_dim": embedding_dim,
            "news_dim": news_dim,
            "news_pooling": news_pooling,
            "num_tickers": len(ticker_to_id),
        },
    )
    save_checkpoint(
        gate_model,
        checkpoints["stage3"],
        metadata={
            "news_dim": hidden_dim,
            "gate_temperature": gate_temperature,
            "gate_target_distribution": gate_target_distribution,
        },
    )
    if run is not None:
        run.summary.update(
            {
                "stage1_best_valid_loss": stage1_history.get("best_valid_loss"),
                "stage2_best_valid_loss": stage2_history.get("best_valid_loss"),
                "stage3_best_valid_loss": stage3_history.get("best_valid_loss"),
            }
        )
        finish_wandb_run(run)
    metadata = {
        "model": _model_name(news_pooling, "b5"),
        "label_col": label_col,
        "factor_cols": factor_cols,
        "embedding_dim": embedding_dim,
        "num_tickers": len(ticker_to_id),
        "news_dim": news_dim,
        "hidden_dim": hidden_dim,
        "bottleneck_hidden_dim": bottleneck_hidden_dim,
        "dropout": dropout,
        "gate_temperature": gate_temperature,
        "lookback_window": lookback_window,
        "kernel_size": kernel_size,
        "dilations": list(dilations),
        "split": asdict(split),
        "train_config": asdict(config),
        "split_sizes": {name: int(len(frame)) for name, frame in splits.items()},
        "dataset_sizes": {name: int(len(dataset)) for name, dataset in datasets.items()},
        "stage2_fusion_dataset_sizes": {
            name: int(len(dataset)) for name, dataset in fusion_datasets.items()
        },
        "gate_target_distribution": gate_target_distribution,
        "histories": {
            "stage1": stage1_history,
            "stage2": stage2_history,
            "stage3": stage3_history,
        },
        "outputs": final_outputs,
        "attention_outputs": attention_outputs,
        "checkpoints": checkpoints,
    }
    dump_json(output_dir / "metadata.json", metadata)
    return metadata


def add_temporal_args(parser: argparse.ArgumentParser, news_pooling: str) -> None:
    parser.add_argument("--panel", default="data/processed/panel_train_b2_768.parquet")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--label-col", default="future_20d_market_adjusted_return")
    parser.add_argument("--batch-size", type=int, default=16 if news_pooling == "b3" else 4096)
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
    parser.add_argument("--dilations", type=int, nargs="+", default=list(DEFAULT_DILATIONS))
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    if news_pooling == "b3":
        parser.add_argument("--chunk-manifest", required=True)
        parser.add_argument("--project-root", default=".")
        parser.add_argument("--max-chunks-per-stock-day", type=int, default=64)
    else:
        parser.add_argument("--chunk-manifest", default=None)
        parser.add_argument("--project-root", default=".")
        parser.add_argument("--max-chunks-per-stock-day", type=int, default=64)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default="news-alpha-z")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-mode", default=None, help="wandb mode, e.g. online/offline/disabled.")


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.patience,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
    )


def wandb_from_args(args: argparse.Namespace) -> WandbConfig:
    return WandbConfig(
        enabled=args.wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        group=args.wandb_group,
        mode=args.wandb_mode,
    )


def print_training_summary(metadata: dict[str, Any]) -> None:
    if "history" in metadata:
        summary = {
            "best_valid_loss": metadata["history"].get("best_valid_loss"),
            "best_epoch": metadata["history"].get("best_epoch"),
            "dataset_sizes": metadata["dataset_sizes"],
        }
    else:
        summary = {
            "stage1_best_valid_loss": metadata["histories"]["stage1"].get("best_valid_loss"),
            "stage2_best_valid_loss": metadata["histories"]["stage2"].get("best_valid_loss"),
            "stage3_best_valid_loss": metadata["histories"]["stage3"].get("best_valid_loss"),
            "dataset_sizes": metadata["dataset_sizes"],
        }
    print(json.dumps(summary, indent=2))
