"""Localization benchmark evaluator."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from octosense.benchmarks.evaluators._shared import (
    build_evaluation_result,
    require_prediction_target,
    resolve_protocol,
)
from octosense.benchmarks.metrics.localization import localization_metrics


def evaluate_localization(outputs: Mapping[str, Any], protocol: Mapping[str, Any] | None = None) -> dict[str, Any]:
    resolved_protocol = resolve_protocol(protocol, default_task_kind="localization")
    predictions, targets = require_prediction_target(
        outputs,
        protocol=resolved_protocol,
        task_label="Localization",
    )
    return build_evaluation_result(
        protocol=resolved_protocol,
        metrics=localization_metrics(predictions, targets),
    )


__all__ = ["evaluate_localization"]
