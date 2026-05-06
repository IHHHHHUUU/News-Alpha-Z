"""Torch import guard."""

from __future__ import annotations


def require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("This model module requires the optional `torch` dependency.") from exc
    return torch, nn
