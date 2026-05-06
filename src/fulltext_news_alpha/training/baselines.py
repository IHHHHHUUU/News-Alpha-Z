"""Baseline registry for the report comparisons."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineSpec:
    code: str
    name: str
    description: str


BASELINES = {
    "B0": BaselineSpec("B0", "factor-only", "Price-volume factors only."),
    "B1": BaselineSpec("B1", "title/headline news", "Title/headline embeddings replace full-text chunks."),
    "B2": BaselineSpec("B2", "mean chunk pooling", "Full-text chunk embeddings are averaged by stock-day."),
    "B3": BaselineSpec("B3", "stock-aware chunk attention", "Main full-text representation with stock-aware attention."),
    "B4": BaselineSpec("B4", "conventional mixture", "Mixture predictions trained or blended without decoupled gate targets."),
    "B5": BaselineSpec("B5", "decoupled mixture", "Final method with independently trained branches and decoupled gate."),
}


def list_baselines() -> list[BaselineSpec]:
    return list(BASELINES.values())
