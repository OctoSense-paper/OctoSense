"""Respiration benchmark evaluator."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from octosense.benchmarks.evaluators._shared import (
    build_evaluation_result,
    require_prediction_target,
    resolve_protocol,
)
from octosense.benchmarks.metrics.respiration import respiration_metrics


def evaluate_respiration(outputs: Mapping[str, Any], protocol: Mapping[str, Any] | None = None) -> dict[str, Any]:
    resolved_protocol = resolve_protocol(protocol, default_task_kind="respiration")
    predictions, targets = require_prediction_target(
        outputs,
        protocol=resolved_protocol,
        task_label="Respiration",
    )
    return build_evaluation_result(
        protocol=resolved_protocol,
        metrics=respiration_metrics(predictions, targets),
    )


__all__ = ["evaluate_respiration"]
