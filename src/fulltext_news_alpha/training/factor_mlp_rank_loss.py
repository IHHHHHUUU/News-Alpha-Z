"""Factor-only MLP baseline trained with daily RankIC-oriented loss."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised only without torch installed.
    raise RuntimeError(
        "fulltext_news_alpha.training.factor_mlp_rank_loss requires torch."
    ) from exc

from fulltext_news_alpha.diagnostics.six_factor_alpha_baseline import ALPHA_FACTOR_MAP
from fulltext_news_alpha.evaluation.ic_metrics import compute_ic_by_date, summarize_ic
from fulltext_news_alpha.evaluation.portfolio_backtest import (
    long_short_returns,
    performance_summary,
)
from fulltext_news_alpha.training.torch_utils import (
    SplitConfig,
    load_training_panel,
    resolve_device,
    set_global_seed,
    split_by_date,
)


DEFAULT_ALPHA_COLS = list(ALPHA_FACTOR_MAP)


@dataclass(frozen=True)
class RankLossConfig:
    """Training hyperparameters for the factor MLP rank-loss baseline."""

    hidden_dim: int = 64
    dropout: float = 0.1
    batch_dates: int = 16
    epochs: int = 100
    patience: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    ic_weight: float = 0.70
    huber_weight: float = 0.20
    var_weight: float = 0.10
    min_pred_std: float = 0.05
    seed: int = 42
    device: str | None = None
    rebalance_every: int = 20


class FactorMLP(nn.Module):
    """Two-hidden-layer MLP for factor-only prediction."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values).squeeze(-1)


def _daily_zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if not std or not np.isfinite(std):
        return pd.Series(np.nan, index=series.index, dtype=float)
    return (series - series.mean()) / std


def daily_rank_label(frame: pd.DataFrame, label_col: str) -> pd.Series:
    """Return per-date cross-sectional percentile rank labels."""

    return frame.groupby("date", group_keys=False)[label_col].rank(method="average", pct=True)


def daily_zscore_label(frame: pd.DataFrame, label_col: str) -> pd.Series:
    """Return per-date cross-sectional z-scored labels."""

    return frame.groupby("date", group_keys=False)[label_col].transform(_daily_zscore)


def ensure_alpha_factors(frame: pd.DataFrame) -> pd.DataFrame:
    """Construct default six directional alpha factors if they are missing."""

    out = frame.copy()
    for alpha_col, (source_col, direction) in ALPHA_FACTOR_MAP.items():
        if alpha_col not in out.columns:
            if source_col not in out.columns:
                raise KeyError(f"Missing both alpha column '{alpha_col}' and source '{source_col}'")
            out[alpha_col] = out[source_col].astype(float) * float(direction)
    return out


def prepare_rank_loss_panel(
    panel: pd.DataFrame,
    label_col: str,
    factor_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Normalize keys, construct alpha columns, and attach y_rank / y_z."""

    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    if factor_cols is None:
        out = ensure_alpha_factors(out)
        factors = list(DEFAULT_ALPHA_COLS)
    else:
        factors = list(factor_cols)
    missing = [col for col in ["date", "ticker", label_col, *factors] if col not in out.columns]
    if missing:
        raise KeyError(f"Panel missing required columns: {missing}")
    out["label"] = out[label_col].astype(float)
    out["y_rank"] = daily_rank_label(out, label_col)
    out["y_z"] = daily_zscore_label(out, label_col)
    return out.sort_values(["date", "ticker"]).reset_index(drop=True), factors


class DateAwareBatcher:
    """Yield batches made of whole dates so daily IC loss is grouped correctly."""

    def __init__(
        self,
        frame: pd.DataFrame,
        factor_cols: list[str],
        batch_dates: int = 16,
        shuffle: bool = False,
        seed: int = 42,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.factor_cols = list(factor_cols)
        self.batch_dates = int(batch_dates)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.dates = pd.Index(pd.to_datetime(self.frame["date"]).drop_duplicates().sort_values())
        self._indices_by_date = {
            date.date(): np.flatnonzero(pd.to_datetime(self.frame["date"]).dt.date.to_numpy() == date.date())
            for date in self.dates
        }
        self._epoch = 0

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        dates = np.asarray([date.date() for date in self.dates], dtype=object)
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self._epoch)
            rng.shuffle(dates)
        self._epoch += 1
        for start in range(0, len(dates), self.batch_dates):
            batch_dates = dates[start : start + self.batch_dates]
            parts = [self._indices_by_date[date] for date in batch_dates]
            indices = np.concatenate(parts) if parts else np.asarray([], dtype=int)
            batch = self.frame.iloc[indices]
            date_codes = pd.Categorical(batch["date"], categories=batch_dates, ordered=True).codes
            yield {
                "x": torch.from_numpy(batch[self.factor_cols].fillna(0.0).to_numpy(dtype=np.float32)),
                "y_rank": torch.from_numpy(batch["y_rank"].to_numpy(dtype=np.float32)),
                "y_z": torch.from_numpy(batch["y_z"].to_numpy(dtype=np.float32)),
                "date_id": torch.from_numpy(date_codes.astype(np.int64)),
            }

    def __len__(self) -> int:
        return int(np.ceil(len(self.dates) / max(1, self.batch_dates)))


def _finite_mask(*values: torch.Tensor) -> torch.Tensor:
    mask = torch.ones_like(values[0], dtype=torch.bool)
    for value in values:
        mask = mask & torch.isfinite(value)
    return mask


def _corr_torch(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor | None:
    mask = _finite_mask(x, y)
    x = x[mask]
    y = y[mask]
    if x.numel() < 3:
        return None
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    x_std = torch.sqrt(torch.mean(x_centered.square()))
    y_std = torch.sqrt(torch.mean(y_centered.square()))
    if bool((x_std <= eps).detach().cpu()) or bool((y_std <= eps).detach().cpu()):
        return None
    return torch.mean(x_centered * y_centered) / (x_std * y_std + eps)


def _daily_zscore_torch(values: torch.Tensor, date_id: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    out = torch.zeros_like(values)
    for day in torch.unique(date_id):
        mask = date_id == day
        day_values = values[mask]
        std = torch.sqrt(torch.mean((day_values - day_values.mean()).square()))
        if bool((std > eps).detach().cpu()):
            out[mask] = (day_values - day_values.mean()) / (std + eps)
    return out


def daily_ic_loss(pred: torch.Tensor, y_rank: torch.Tensor, date_id: torch.Tensor) -> torch.Tensor:
    """Compute 1 - mean daily Pearson correlation between pred and y_rank."""

    corrs: list[torch.Tensor] = []
    for day in torch.unique(date_id):
        mask = date_id == day
        corr = _corr_torch(pred[mask], y_rank[mask])
        if corr is not None:
            corrs.append(corr)
    if not corrs:
        return pred.sum() * 0.0 + 1.0
    return 1.0 - torch.stack(corrs).mean()


def huber_auxiliary_loss(pred: torch.Tensor, y_z: torch.Tensor, date_id: torch.Tensor) -> torch.Tensor:
    """Huber loss between daily-zscored predictions and daily-zscored labels."""

    pred_z = _daily_zscore_torch(pred, date_id)
    mask = _finite_mask(pred_z, y_z)
    if not bool(mask.any().detach().cpu()):
        return pred.sum() * 0.0
    return nn.functional.huber_loss(pred_z[mask], y_z[mask], delta=1.0)


def variance_penalty(
    pred: torch.Tensor,
    date_id: torch.Tensor,
    min_pred_std: float = 0.05,
) -> torch.Tensor:
    """Penalize dates whose cross-sectional prediction std is too small."""

    penalties: list[torch.Tensor] = []
    for day in torch.unique(date_id):
        day_pred = pred[date_id == day]
        if day_pred.numel() < 2:
            continue
        std = torch.sqrt(torch.mean((day_pred - day_pred.mean()).square()))
        penalties.append(torch.relu(torch.as_tensor(min_pred_std, device=pred.device) - std).square())
    if not penalties:
        return pred.sum() * 0.0
    return torch.stack(penalties).mean()


def combined_rank_loss(
    pred: torch.Tensor,
    y_rank: torch.Tensor,
    y_z: torch.Tensor,
    date_id: torch.Tensor,
    config: RankLossConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    ic = daily_ic_loss(pred, y_rank, date_id)
    huber = huber_auxiliary_loss(pred, y_z, date_id)
    var = variance_penalty(pred, date_id, min_pred_std=config.min_pred_std)
    total = config.ic_weight * ic + config.huber_weight * huber + config.var_weight * var
    return total, {
        "ic_loss": float(ic.detach().cpu()),
        "huber_loss": float(huber.detach().cpu()),
        "variance_penalty": float(var.detach().cpu()),
        "total_loss": float(total.detach().cpu()),
    }


def predict_frame(
    model: nn.Module,
    frame: pd.DataFrame,
    factor_cols: list[str],
    device: torch.device,
    batch_rows: int = 8192,
) -> pd.DataFrame:
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(frame), batch_rows):
            batch = frame.iloc[start : start + batch_rows]
            x = torch.from_numpy(batch[factor_cols].fillna(0.0).to_numpy(dtype=np.float32)).to(device)
            preds.append(model(x).detach().cpu().numpy())
    out = frame[["date", "ticker", "label", "y_rank", "y_z", *factor_cols]].copy()
    out["pred"] = np.concatenate(preds) if preds else np.asarray([], dtype=np.float32)
    return out.rename(columns={"y_rank": "label_rank", "y_z": "label_zscore"})


def evaluate_predictions(
    predictions: pd.DataFrame,
    rebalance_every: int = 20,
) -> dict[str, float]:
    eval_frame = predictions.rename(columns={"label": "future_return"}).copy()
    ic = compute_ic_by_date(eval_frame, factor_col="pred", return_col="future_return")
    summary = summarize_ic(ic)
    rankic = ic["RankIC"].dropna()
    ls = long_short_returns(
        eval_frame,
        factor_col="pred",
        return_col="future_return",
        rebalance_every=rebalance_every,
    )
    periods_per_year = 252 / max(1, int(rebalance_every))
    perf = performance_summary(ls["long_short_return"], periods_per_year=periods_per_year)
    daily_pred_std = predictions.groupby("date")["pred"].std(ddof=0)
    return {
        "IC": float(summary.get("IC", np.nan)),
        "ICIR": float(summary.get("ICIR", np.nan)),
        "RankIC": float(summary.get("RankIC", np.nan)),
        "RankICIR": float(summary.get("RankICIR", np.nan)),
        "coverage": float(summary.get("coverage", np.nan)),
        "days": int(len(ic)),
        "pred_mean": float(predictions["pred"].mean()),
        "pred_std": float(predictions["pred"].std(ddof=0)),
        "daily_pred_std_mean": float(daily_pred_std.mean()),
        "daily_pred_std_min": float(daily_pred_std.min()),
        "positive_rankic_ratio": float((rankic > 0).mean()) if len(rankic) else np.nan,
        "long_short_return_mean": float(ls["long_short_return"].dropna().mean()) if not ls.empty else np.nan,
        "long_short_sharpe": float(perf["Sharpe"]),
        "max_drawdown": float(perf["max_drawdown"]),
        "rebalance_every": int(rebalance_every),
    }


def train_factor_mlp_rank_loss(
    panel_path: str | Path,
    output_dir: str | Path,
    label_col: str = "future_20d_market_adjusted_return",
    factor_cols: list[str] | None = None,
    split: SplitConfig = SplitConfig(),
    config: RankLossConfig = RankLossConfig(),
) -> dict[str, Any]:
    """Train FactorMLP with daily IC loss and valid RankIC early stopping."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    set_global_seed(config.seed)
    device = resolve_device(config.device)

    panel = load_training_panel(panel_path)
    frame, factors = prepare_rank_loss_panel(panel, label_col=label_col, factor_cols=factor_cols)
    splits = split_by_date(frame, split)
    train_frame = splits["train"].dropna(subset=["label", "y_rank", "y_z"]).reset_index(drop=True)
    valid_frame = splits["valid"].dropna(subset=["label", "y_rank", "y_z"]).reset_index(drop=True)
    test_frame = splits["test"].reset_index(drop=True)

    model = FactorMLP(len(factors), hidden_dim=config.hidden_dim, dropout=config.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_rankic = -np.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0
    history_rows: list[dict[str, Any]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loader = DateAwareBatcher(
            train_frame,
            factors,
            batch_dates=config.batch_dates,
            shuffle=True,
            seed=config.seed + epoch,
        )
        loss_sums = {"total_loss": 0.0, "ic_loss": 0.0, "huber_loss": 0.0, "variance_penalty": 0.0}
        steps = 0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch["x"])
            loss, parts = combined_rank_loss(
                pred,
                batch["y_rank"],
                batch["y_z"],
                batch["date_id"],
                config,
            )
            loss.backward()
            optimizer.step()
            for key in loss_sums:
                loss_sums[key] += parts[key]
            steps += 1

        valid_pred = predict_frame(model, valid_frame, factors, device)
        valid_metrics = evaluate_predictions(valid_pred, rebalance_every=config.rebalance_every)
        valid_rankic = float(valid_metrics["RankIC"])
        improved = np.isfinite(valid_rankic) and valid_rankic > best_rankic + 1e-8
        if improved:
            best_rankic = valid_rankic
            best_epoch = epoch
            bad_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            bad_epochs += 1
        row = {
            "epoch": epoch,
            **{key: value / max(1, steps) for key, value in loss_sums.items()},
            "valid_RankIC": valid_rankic,
            "valid_IC": float(valid_metrics["IC"]),
            "valid_daily_pred_std_mean": float(valid_metrics["daily_pred_std_mean"]),
            "best_valid_RankIC": float(best_rankic),
            "best_epoch": int(best_epoch),
            "improved": bool(improved),
        }
        history_rows.append(row)
        print(
            "epoch={epoch} total_loss={total_loss:.6g} valid_RankIC={valid_RankIC:.6g} "
            "best_valid_RankIC={best_valid_RankIC:.6g} best_epoch={best_epoch}".format(**row),
            flush=True,
        )
        if bad_epochs >= config.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    predictions = {
        "train": predict_frame(model, train_frame, factors, device),
        "valid": predict_frame(model, valid_frame, factors, device),
        "test": predict_frame(model, test_frame, factors, device),
    }
    metrics = {
        split_name: evaluate_predictions(pred, rebalance_every=config.rebalance_every)
        for split_name, pred in predictions.items()
    }

    for split_name, pred in predictions.items():
        pred.to_parquet(output / f"predictions_{split_name}.parquet", index=False)
    pd.DataFrame(history_rows).to_csv(output / "history.csv", index=False)
    torch.save({"state_dict": model.state_dict()}, output / "best_model.pt")
    factor_config = {
        "label_col": label_col,
        "factor_cols": factors,
        "used_default_alpha_factors": factor_cols is None,
        "split": asdict(split),
        "config": asdict(config),
        "early_stopping_metric": "valid_RankIC",
        "loss": {
            "ic_weight": config.ic_weight,
            "huber_weight": config.huber_weight,
            "var_weight": config.var_weight,
            "min_pred_std": config.min_pred_std,
        },
    }
    (output / "factor_config.json").write_text(
        json.dumps(factor_config, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    payload = {
        "metrics": metrics,
        "best_epoch": int(best_epoch),
        "best_valid_RankIC": float(best_rankic),
        "history_rows": int(len(history_rows)),
    }
    (output / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train factor-only MLP with daily IC rank loss. Example: "
            "python scripts/22_train_factor_mlp_rank_loss.py "
            "--panel data/processed/panel_train_b2_768.parquet "
            "--output-dir data/predictions/factor_mlp_rank_loss "
            "--label-col future_20d_market_adjusted_return --hidden-dim 64 "
            "--dropout 0.1 --batch-dates 16 --epochs 100 --patience 10"
        )
    )
    parser.add_argument("--panel", default="data/processed/panel_train_b2_768.parquet")
    parser.add_argument("--output-dir", default="data/predictions/factor_mlp_rank_loss")
    parser.add_argument("--label-col", default="future_20d_market_adjusted_return")
    parser.add_argument("--factor-cols", nargs="+", default=None)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-dates", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ic-weight", type=float, default=0.70)
    parser.add_argument("--huber-weight", type=float, default=0.20)
    parser.add_argument("--var-weight", type=float, default=0.10)
    parser.add_argument("--min-pred-std", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebalance-every", type=int, default=20)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    payload = train_factor_mlp_rank_loss(
        panel_path=args.panel,
        output_dir=args.output_dir,
        label_col=args.label_col,
        factor_cols=args.factor_cols,
        config=RankLossConfig(
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            batch_dates=args.batch_dates,
            epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            weight_decay=args.weight_decay,
            ic_weight=args.ic_weight,
            huber_weight=args.huber_weight,
            var_weight=args.var_weight,
            min_pred_std=args.min_pred_std,
            seed=args.seed,
            device=args.device,
            rebalance_every=args.rebalance_every,
        ),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
