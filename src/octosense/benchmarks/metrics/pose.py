"""Pose benchmark metrics."""

from __future__ import annotations

from typing import Any

import torch

from octosense.benchmarks.metrics._regression import (
    as_tensor,
    reshape_target_like,
    root_mean_square_error,
)


def _mean_per_joint_position_error(predictions: Any, targets: Any) -> tuple[float, float]:
    prediction_tensor = as_tensor(predictions)
    if prediction_tensor is None:
        return 0.0, 0.0
    prediction_tensor = prediction_tensor.float()
    target_tensor = reshape_target_like(prediction_tensor, targets)
    if target_tensor is None:
        return 0.0, 0.0
    if prediction_tensor.ndim == 0:
        return 0.0, 0.0
    joint_errors = torch.linalg.vector_norm(prediction_tensor - target_tensor, dim=-1)
    return float(joint_errors.mean().item()), float(joint_errors.numel())


def pose_metrics(predictions: Any, targets: Any) -> dict[str, float]:
    mpjpe, count = _mean_per_joint_position_error(predictions, targets)
    return {
        "mpjpe": mpjpe,
        "rmse": root_mean_square_error(predictions, targets),
        "count": count,
    }


__all__ = ["pose_metrics"]
