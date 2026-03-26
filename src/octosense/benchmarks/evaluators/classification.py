"""Classification benchmark evaluator."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from octosense.benchmarks.evaluators._shared import (
    build_evaluation_result,
    require_prediction_target,
    resolve_protocol,
)
from octosense.benchmarks.metrics.classification import classification_metrics


def evaluate_classification(outputs: Mapping[str, Any], protocol: Mapping[str, Any] | None = None) -> dict[str, Any]:
    resolved_protocol = resolve_protocol(protocol, default_task_kind="classification")
    predictions, targets = require_prediction_target(
        outputs,
        protocol=resolved_protocol,
        task_label="Classification",
    )
    return build_evaluation_result(
        protocol=resolved_protocol,
        metrics=classification_metrics(predictions, targets),
    )


__all__ = ["evaluate_classification"]
