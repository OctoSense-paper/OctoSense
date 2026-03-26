"""Localization benchmark metrics."""

from __future__ import annotations

from typing import Any

from octosense.benchmarks.metrics._regression import (
    mean_absolute_error,
    root_mean_square_error,
)


def localization_metrics(predictions: Any, targets: Any) -> dict[str, float]:
    mae, count = mean_absolute_error(predictions, targets)
    return {
        "mae": mae,
        "rmse": root_mean_square_error(predictions, targets),
        "count": count,
    }


__all__ = ["localization_metrics"]
