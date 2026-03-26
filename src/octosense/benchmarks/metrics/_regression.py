"""Shared regression-style metric helpers for benchmark task families."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


def as_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (int, float, bool)):
        return torch.as_tensor(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        try:
            return torch.as_tensor(value)
        except (TypeError, ValueError):
            return None
    return None


def reshape_target_like(prediction: torch.Tensor, target: Any) -> torch.Tensor | None:
    target_tensor = as_tensor(target)
    if target_tensor is None:
        return None
    return target_tensor.to(dtype=prediction.dtype).reshape_as(prediction)


def mean_absolute_error(prediction: Any, target: Any) -> tuple[float, float]:
    prediction_tensor = as_tensor(prediction)
    if prediction_tensor is None:
        return 0.0, 0.0
    prediction_tensor = prediction_tensor.float()
    target_tensor = reshape_target_like(prediction_tensor, target)
    if target_tensor is None:
        return 0.0, 0.0
    error = (prediction_tensor - target_tensor).abs()
    return float(error.mean().item()), float(error.numel())


def root_mean_square_error(prediction: Any, target: Any) -> float:
    prediction_tensor = as_tensor(prediction)
    if prediction_tensor is None:
        return 0.0
    prediction_tensor = prediction_tensor.float()
    target_tensor = reshape_target_like(prediction_tensor, target)
    if target_tensor is None:
        return 0.0
    squared = torch.square(prediction_tensor - target_tensor)
    return float(torch.sqrt(squared.mean()).item())


__all__ = [
    "as_tensor",
    "mean_absolute_error",
    "reshape_target_like",
    "root_mean_square_error",
]
