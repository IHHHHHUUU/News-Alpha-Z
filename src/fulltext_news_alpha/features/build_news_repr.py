"""Build stock-day news representations from saved chunk embeddings."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_EMBEDDINGS_ROOT = Path("data/embeddings")
DEFAULT_PANEL_PATH = Path("data/processed/panel_with_news_coverage_clean.parquet")
DEFAULT_REPR_OUTPUT = Path("data/features/news_repr_mean64_clean.parquet")
DEFAULT_PANEL_OUTPUT = Path("data/processed/panel_with_news_repr_mean64_clean.parquet")
DEFAULT_META_OUTPUT = Path("data/features/news_repr_projection_meta.json")
REPR_METHOD = "mean_pooling_64"


def find_manifest_paths(embeddings_root: str | Path) -> list[Path]:
    """Find embedding manifests below an embeddings root."""

    root = Path(embeddings_root)
    if not root.exists():
        raise FileNotFoundError(f"Embedding root does not exist: {root}")
    return sorted(root.rglob("manifest.jsonl"))


def read_manifest_jsonl(path: str | Path) -> pd.DataFrame:
    """Read one JSONL manifest and keep the source directory for path resolution."""

    manifest_path = Path(path)
    records: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            record["_manifest_path"] = str(manifest_path)
            record["_manifest_dir"] = str(manifest_path.parent)
            record["_manifest_line"] = line_number
            records.append(record)
    return pd.DataFrame(records)


def load_manifests(embeddings_root: str | Path) -> pd.DataFrame:
    """Load and concatenate every manifest below ``embeddings_root``."""

    paths = find_manifest_paths(embeddings_root)
    if not paths:
        raise FileNotFoundError(f"No manifest.jsonl files found under {embeddings_root}")
    frames = [read_manifest_jsonl(path) for path in paths]
    manifest = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    required = {"chunk_id", "ticker", "date", "embedding_path"}
    missing = required - set(manifest.columns)
    if missing:
        raise KeyError(f"Missing manifest columns: {sorted(missing)}")
    return manifest


def resolve_embedding_path(raw_path: object, manifest_dir: object, project_root: str | Path) -> Path:
    """Resolve absolute, repo-relative, or manifest-relative embedding paths."""

    path = Path(str(raw_path))
    if path.is_absolute():
        return path

    root = Path(project_root)
    candidates = [
        root / path,
        Path(str(manifest_dir)) / path,
        Path(str(manifest_dir)) / path.name,
        path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def prepare_manifest(manifest: pd.DataFrame, project_root: str | Path = ".") -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate manifest rows, de-duplicate chunk IDs, and drop missing files."""

    out = manifest.copy()
    out["chunk_id"] = out["chunk_id"].astype(str)
    out["ticker"] = out["ticker"].astype(str)
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out["_resolved_embedding_path"] = [
        resolve_embedding_path(path, manifest_dir, project_root)
        for path, manifest_dir in zip(out["embedding_path"], out["_manifest_dir"], strict=True)
    ]
    out["_embedding_file_exists"] = [path.exists() for path in out["_resolved_embedding_path"]]

    manifest_rows = int(len(out))
    unique_chunk_ids = int(out["chunk_id"].nunique())
    duplicate_chunk_id_count = int(manifest_rows - unique_chunk_ids)
    missing_embedding_files_count = int((~out["_embedding_file_exists"]).sum())
    embedding_dim_distribution = (
        out["embedding_dim"].value_counts(dropna=False).sort_index().astype(int).to_dict()
        if "embedding_dim" in out.columns
        else {}
    )

    valid = out.drop_duplicates("chunk_id", keep="first")
    valid = valid[valid["_embedding_file_exists"]].reset_index(drop=True)
    if valid.empty:
        raise ValueError("No valid embedding rows remain after duplicate and missing-file checks.")

    qa = {
        "manifest_rows": manifest_rows,
        "unique_chunk_ids": unique_chunk_ids,
        "duplicate_chunk_id_count": duplicate_chunk_id_count,
        "missing_embedding_files_count": missing_embedding_files_count,
        "valid_embedding_rows": int(len(valid)),
        "manifest_file_count": int(out["_manifest_path"].nunique()),
        "embedding_dim_distribution": {str(key): int(value) for key, value in embedding_dim_distribution.items()},
    }
    return valid, qa


def build_stock_day_mean_embeddings(manifest: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Mean-pool chunk embeddings within each ticker-date stock day."""

    sums: dict[tuple[str, object], np.ndarray] = {}
    counts: dict[tuple[str, object], int] = {}
    loaded_dims: list[int] = []

    for path_value, ticker, date_value in zip(
        manifest["_resolved_embedding_path"],
        manifest["ticker"],
        manifest["date"],
        strict=True,
    ):
        path = Path(path_value)
        embedding = np.asarray(np.load(path), dtype=np.float32).reshape(-1)
        if embedding.size == 0:
            raise ValueError(f"Empty embedding array: {path}")
        key = (str(ticker), date_value)
        loaded_dims.append(int(embedding.shape[0]))
        if key not in sums:
            sums[key] = embedding.copy()
            counts[key] = 1
        else:
            if sums[key].shape != embedding.shape:
                raise ValueError(
                    f"Embedding dimension mismatch for {key}: "
                    f"{sums[key].shape[0]} vs {embedding.shape[0]} at {path}"
                )
            sums[key] += embedding
            counts[key] += 1

    keys = sorted(sums, key=lambda item: (pd.Timestamp(item[1]), item[0]))
    mean_matrix = np.vstack([sums[key] / counts[key] for key in keys]).astype(np.float32)
    pooled = pd.DataFrame(
        {
            "ticker": [key[0] for key in keys],
            "date": [key[1] for key in keys],
            "_embedded_chunk_count": [counts[key] for key in keys],
        }
    )
    loaded_dim_distribution = pd.Series(loaded_dims).value_counts().sort_index().astype(int).to_dict()
    qa = {
        "stock_days_with_embeddings": int(len(pooled)),
        "loaded_embedding_dim_distribution": {
            str(key): int(value) for key, value in loaded_dim_distribution.items()
        },
    }
    return pooled, mean_matrix, qa


def project_embeddings(
    matrix: np.ndarray,
    output_dim: int = 64,
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reduce stock-day mean embeddings to a fixed-width representation."""

    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    input_dim = int(matrix.shape[1])
    if output_dim <= 0:
        raise ValueError("output_dim must be positive")

    try:
        from sklearn.decomposition import PCA
    except ImportError:
        rng = np.random.default_rng(seed)
        projection = rng.normal(loc=0.0, scale=1.0 / np.sqrt(input_dim), size=(input_dim, output_dim))
        projected = matrix @ projection.astype(np.float32)
        metadata = {
            "method": "gaussian_random_projection",
            "seed": seed,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "sklearn_available": False,
        }
        return projected.astype(np.float32), metadata

    fit_components = min(output_dim, matrix.shape[0], matrix.shape[1])
    pca = PCA(n_components=fit_components, random_state=seed)
    projected = pca.fit_transform(matrix).astype(np.float32)
    if fit_components < output_dim:
        padded = np.zeros((matrix.shape[0], output_dim), dtype=np.float32)
        padded[:, :fit_components] = projected
        projected = padded

    metadata = {
        "method": "pca",
        "seed": seed,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "fit_components": int(fit_components),
        "sklearn_available": True,
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "explained_variance_ratio": [float(value) for value in pca.explained_variance_ratio_],
    }
    return projected, metadata


def attach_coverage_counts(repr_frame: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Attach panel news_count/chunk_count coverage to representation rows."""

    coverage_cols = ["date", "ticker", "news_count", "chunk_count"]
    missing = set(coverage_cols) - set(panel.columns)
    if missing:
        raise KeyError(f"Missing panel coverage columns: {sorted(missing)}")

    coverage = panel[coverage_cols].copy()
    coverage["date"] = pd.to_datetime(coverage["date"]).dt.date
    coverage["ticker"] = coverage["ticker"].astype(str)
    coverage = coverage.drop_duplicates(["date", "ticker"])

    out = repr_frame.merge(coverage, on=["date", "ticker"], how="left")
    out["news_count"] = out["news_count"].fillna(0).astype(int)
    out["chunk_count"] = out["chunk_count"].fillna(out["_embedded_chunk_count"]).astype(int)
    return out.drop(columns=["_embedded_chunk_count"])


def merge_repr_into_panel(
    panel: pd.DataFrame,
    repr_frame: pd.DataFrame,
    repr_cols: list[str],
    output_dim: int,
) -> tuple[pd.DataFrame, int]:
    """Left join representation columns into the stock-day panel."""

    panel_out = panel.copy()
    panel_out["date"] = pd.to_datetime(panel_out["date"]).dt.date
    panel_out["ticker"] = panel_out["ticker"].astype(str)

    join_cols = ["date", "ticker"] + repr_cols + ["embedding_dim", "repr_method"]
    panel_out = panel_out.merge(repr_frame[join_cols], on=["date", "ticker"], how="left")
    panel_out[repr_cols] = panel_out[repr_cols].fillna(0.0).astype(np.float32)
    panel_out["embedding_dim"] = panel_out["embedding_dim"].fillna(output_dim).astype(int)
    panel_out["repr_method"] = panel_out["repr_method"].fillna(REPR_METHOD)
    nonzero_rows = int(np.any(panel_out[repr_cols].to_numpy(dtype=np.float32) != 0.0, axis=1).sum())
    return panel_out, nonzero_rows


def build_news_representations(
    embeddings_root: str | Path = DEFAULT_EMBEDDINGS_ROOT,
    panel_path: str | Path = DEFAULT_PANEL_PATH,
    repr_output: str | Path = DEFAULT_REPR_OUTPUT,
    panel_output: str | Path = DEFAULT_PANEL_OUTPUT,
    projection_meta_output: str | Path = DEFAULT_META_OUTPUT,
    output_dim: int = 64,
    seed: int = 42,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    """Build mean-pooled 64-d stock-day news representations and panel join."""

    manifest = load_manifests(embeddings_root)
    valid_manifest, qa = prepare_manifest(manifest, project_root=project_root)
    pooled, mean_matrix, pool_qa = build_stock_day_mean_embeddings(valid_manifest)
    qa.update(pool_qa)

    projected, projection_meta = project_embeddings(mean_matrix, output_dim=output_dim, seed=seed)
    repr_cols = [f"news_repr_{idx}" for idx in range(output_dim)]
    repr_values = pd.DataFrame(projected, columns=repr_cols)
    repr_frame = pd.concat([pooled.reset_index(drop=True), repr_values], axis=1)

    panel = pd.read_parquet(panel_path)
    repr_frame = attach_coverage_counts(repr_frame, panel)
    repr_frame["embedding_dim"] = output_dim
    repr_frame["repr_method"] = REPR_METHOD
    repr_frame = repr_frame[
        ["date", "ticker", *repr_cols, "news_count", "chunk_count", "embedding_dim", "repr_method"]
    ].sort_values(["date", "ticker"])

    panel_out, nonzero_panel_rows = merge_repr_into_panel(panel, repr_frame, repr_cols, output_dim)
    qa["panel_rows_with_nonzero_news_representation"] = nonzero_panel_rows

    repr_output = Path(repr_output)
    panel_output = Path(panel_output)
    projection_meta_output = Path(projection_meta_output)
    repr_output.parent.mkdir(parents=True, exist_ok=True)
    panel_output.parent.mkdir(parents=True, exist_ok=True)
    projection_meta_output.parent.mkdir(parents=True, exist_ok=True)
    repr_frame.to_parquet(repr_output, index=False)
    panel_out.to_parquet(panel_output, index=False)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repr_method": REPR_METHOD,
        "inputs": {
            "embeddings_root": str(embeddings_root),
            "panel_path": str(panel_path),
        },
        "outputs": {
            "news_repr": str(repr_output),
            "panel_with_news_repr": str(panel_output),
        },
        "qa": qa,
        "projection": projection_meta,
    }
    projection_meta_output.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stock-day mean-pooled news representations.")
    parser.add_argument("--embeddings-root", default=str(DEFAULT_EMBEDDINGS_ROOT))
    parser.add_argument("--panel", default=str(DEFAULT_PANEL_PATH))
    parser.add_argument("--news-repr-output", default=str(DEFAULT_REPR_OUTPUT))
    parser.add_argument("--panel-output", default=str(DEFAULT_PANEL_OUTPUT))
    parser.add_argument("--projection-meta-output", default=str(DEFAULT_META_OUTPUT))
    parser.add_argument("--output-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metadata = build_news_representations(
        embeddings_root=args.embeddings_root,
        panel_path=args.panel,
        repr_output=args.news_repr_output,
        panel_output=args.panel_output,
        projection_meta_output=args.projection_meta_output,
        output_dim=args.output_dim,
        seed=args.seed,
    )
    print(json.dumps(metadata["qa"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
