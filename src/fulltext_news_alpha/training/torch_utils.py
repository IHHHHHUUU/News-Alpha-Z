"""Shared PyTorch utilities for B2 / B3 baseline training.

The helpers in this module are intentionally lightweight: they handle reading
the training panel parquet, slicing into train/valid/test splits, exposing
tensor-friendly Datasets, and running a generic train loop with early stopping
on a configurable validation loss.

Heavy lifting (model definition, loss design, prediction tables) lives in the
per-baseline training modules.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.nn.utils.clip_grad import clip_grad_norm_
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover - exercised only without torch installed.
    raise RuntimeError(
        "fulltext_news_alpha.training.torch_utils requires the optional torch dependency. "
        'Install it with `pip install -e ".[text]"`.'
    ) from exc


@dataclass(frozen=True)
class SplitConfig:
    """Date-based train / validation / test split."""

    train_start: str = "2018-01-01"
    train_end: str = "2020-12-31"
    valid_start: str = "2021-01-01"
    valid_end: str = "2021-12-31"
    test_start: str = "2022-01-01"
    test_end: str = "2023-12-31"


@dataclass(frozen=True)
class TrainConfig:
    """Generic training-loop hyperparameters."""

    batch_size: int = 4096
    max_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 5
    grad_clip_norm: float | None = 1.0
    num_workers: int = 0
    device: str | None = None
    seed: int = 42


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_device(preferred: str | None = None) -> torch.device:
    """Return the requested torch device, defaulting to CUDA when available."""

    if preferred:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_training_panel(path: str | Path) -> pd.DataFrame:
    """Load the B2 training panel and normalize key columns."""

    frame = pd.read_parquet(path)
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    return frame


def split_by_date(frame: pd.DataFrame, split: SplitConfig) -> dict[str, pd.DataFrame]:
    """Slice the training panel into train/valid/test sub-frames by date."""

    dates = pd.to_datetime(frame["date"])

    def _mask(start: str, end: str) -> pd.Series:
        return (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))

    return {
        "train": frame.loc[_mask(split.train_start, split.train_end)].reset_index(drop=True),
        "valid": frame.loc[_mask(split.valid_start, split.valid_end)].reset_index(drop=True),
        "test": frame.loc[_mask(split.test_start, split.test_end)].reset_index(drop=True),
    }


def emb_column_names(frame: pd.DataFrame, prefix: str = "mean_emb_") -> list[str]:
    cols = [col for col in frame.columns if col.startswith(prefix)]
    if not cols:
        raise KeyError(f"No columns found with prefix '{prefix}'")
    return cols


def to_float_tensor(values: np.ndarray | pd.DataFrame | pd.Series) -> torch.Tensor:
    array = np.asarray(values, dtype=np.float32)
    return torch.from_numpy(array)


class StockDayDataset(Dataset):
    """Stock-day Dataset exposing (factors, news_emb, has_news, label) tensors.

    Train/validation callers should keep ``drop_missing_label=True`` so losses
    never see fake targets. Test/inference callers may keep missing labels; they
    remain NaN rather than being filled with zero.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        factor_cols: list[str],
        embedding_cols: list[str],
        label_col: str,
        has_news_col: str = "has_news",
        drop_missing_label: bool = True,
    ) -> None:
        if drop_missing_label:
            frame = frame.dropna(subset=[label_col]).reset_index(drop=True)

        self.factor_cols = list(factor_cols)
        self.embedding_cols = list(embedding_cols)
        self.label_col = str(label_col)
        self.has_news_col = str(has_news_col)

        self.factors = to_float_tensor(frame[self.factor_cols].fillna(0.0).to_numpy())
        self.news_emb = to_float_tensor(frame[self.embedding_cols].fillna(0.0).to_numpy())
        self.labels = to_float_tensor(frame[self.label_col].to_numpy())
        self.has_news = to_float_tensor(frame[self.has_news_col].fillna(0).to_numpy())
        self.keys = frame[["date", "ticker"]].copy()

    def __len__(self) -> int:
        return int(self.factors.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "factors": self.factors[idx],
            "news_emb": self.news_emb[idx],
            "label": self.labels[idx],
            "has_news": self.has_news[idx],
        }


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        drop_last=bool(drop_last),
        pin_memory=torch.cuda.is_available(),
    )


@dataclass
class EarlyStopper:
    """Track validation loss and signal when to stop training."""

    patience: int = 5
    min_delta: float = 1e-6
    best_loss: float = math.inf
    best_epoch: int = -1
    bad_epochs: int = 0
    best_state: dict[str, torch.Tensor] | None = None

    def step(
        self,
        model: nn.Module,
        epoch: int,
        val_loss: float,
    ) -> tuple[bool, bool]:
        """Return (improved, should_stop) for this epoch."""

        improved = val_loss + self.min_delta < self.best_loss
        if improved:
            self.best_loss = float(val_loss)
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
            self.best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            self.bad_epochs += 1
        should_stop = self.bad_epochs >= self.patience
        return improved, should_stop

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    compute_loss: Callable[[nn.Module, dict[str, torch.Tensor]], torch.Tensor],
    config: TrainConfig,
    log_every: int = 50,
    progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    """Generic train + validation loop with early stopping.

    The caller supplies a ``compute_loss`` callable that takes the model and a
    batch dict and returns a scalar loss tensor. This keeps the loop agnostic
    to whether we are training B2+B4 (mixed pred MSE), B2+B5 stage 1 (factor
    MSE), stage 2 (fusion MSE), or stage 3 (gate BCE).
    """

    device = resolve_device(config.device)
    set_global_seed(config.seed)

    model.to(device)
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    early = EarlyStopper(patience=config.early_stopping_patience)
    history: list[dict[str, Any]] = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for step, batch in enumerate(train_loader, start=1):
            batch_on_device = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = compute_loss(model, batch_on_device)
            loss.backward()
            if config.grad_clip_norm:
                clip_grad_norm_(
                    model.parameters(),
                    max_norm=float(config.grad_clip_norm),
                )
            optimizer.step()
            batch_size = int(next(iter(batch_on_device.values())).shape[0])
            train_loss_sum += float(loss.detach().item()) * batch_size
            train_count += batch_size

        train_loss = train_loss_sum / max(1, train_count)
        val_loss = evaluate_loss(model, valid_loader, compute_loss, device=device)
        improved, should_stop = early.step(model, epoch, val_loss)
        entry = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "valid_loss": float(val_loss),
            "improved": bool(improved),
            "best_valid_loss": float(early.best_loss),
            "best_epoch": int(early.best_epoch),
        }
        history.append(entry)
        if progress_callback is not None:
            progress_callback(entry)
        if should_stop:
            break

    early.restore_best(model)
    return {
        "history": history,
        "best_valid_loss": float(early.best_loss),
        "best_epoch": int(early.best_epoch),
    }


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    compute_loss: Callable[[nn.Module, dict[str, torch.Tensor]], torch.Tensor],
    device: torch.device,
) -> float:
    """Run the loss callable over a DataLoader without gradient updates."""

    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch_on_device = {key: value.to(device) for key, value in batch.items()}
            loss = compute_loss(model, batch_on_device)
            batch_size = int(next(iter(batch_on_device.values())).shape[0])
            total += float(loss.detach().item()) * batch_size
            count += batch_size
    return total / max(1, count)


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    forward_fn: Callable[[nn.Module, dict[str, torch.Tensor]], dict[str, torch.Tensor]],
    device: torch.device,
    extra_outputs: Iterable[str] = (),
) -> dict[str, np.ndarray]:
    """Run the model over a loader and concatenate selected output tensors."""

    model.eval()
    outputs: dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        for batch in loader:
            batch_on_device = {key: value.to(device) for key, value in batch.items()}
            preds = forward_fn(model, batch_on_device)
            for name in extra_outputs:
                if name not in preds:
                    continue
                value = preds[name]
                if value is None:
                    continue
                outputs.setdefault(name, []).append(value.detach().cpu().numpy())
    return {name: np.concatenate(parts, axis=0) for name, parts in outputs.items()}


def save_checkpoint(
    model: nn.Module,
    output_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict()}
    if metadata is not None:
        payload["metadata"] = metadata
    torch.save(payload, output_path)


def dump_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def infer_factor_columns(frame: pd.DataFrame) -> list[str]:
    """Return zscore factor columns excluding label columns."""

    cols = [
        col
        for col in frame.columns
        if col.endswith("_zscore") and not col.startswith("future_")
    ]
    if not cols:
        raise KeyError("No '*_zscore' factor columns found in training panel.")
    return cols
