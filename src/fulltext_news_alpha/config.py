"""Configuration helpers for the News-Alpha-Z pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a plain dictionary."""

    with Path(path).open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a mapping config at {path}, got {type(loaded)!r}")
    return loaded


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return ``base`` recursively updated by ``override`` without mutating inputs."""

    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged
