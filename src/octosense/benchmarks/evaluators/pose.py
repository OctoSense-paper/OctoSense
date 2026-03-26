"""Pose benchmark evaluator."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from octosense.benchmarks.evaluators._shared import (
    build_evaluation_result,
    require_prediction_target,
    resolve_protocol,
)
from octosense.benchmarks.metrics.pose import pose_metrics


def evaluate_pose(outputs: Mapping[str, Any], protocol: Mapping[str, Any] | None = None) -> dict[str, Any]:
    resolved_protocol = resolve_protocol(protocol, default_task_kind="pose")
    predictions, targets = require_prediction_target(
        outputs,
        protocol=resolved_protocol,
        task_label="Pose",
    )
    return build_evaluation_result(
        protocol=resolved_protocol,
        metrics=pose_metrics(predictions, targets),
    )


__all__ = ["evaluate_pose"]
