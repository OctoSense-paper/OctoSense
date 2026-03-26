"""Shared helpers for canonical benchmark evaluators."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from octosense.benchmarks.protocols import (
    canonical_protocol_for_task_kind,
    canonical_task_kinds_for_protocol_id,
)

def resolve_protocol(
    protocol: Mapping[str, Any] | None,
    *,
    default_task_kind: str,
) -> dict[str, Any]:
    requested = dict(protocol or {})
    task_kind = str(requested.get("task_kind") or default_task_kind).strip()
    if not task_kind:
        raise ValueError("benchmark evaluator must resolve a non-empty task_kind")
    canonical_protocol = canonical_protocol_for_task_kind(task_kind)
    resolved = dict(canonical_protocol)
    for key in ("name", "protocol_id", "task_kind", "primary_metric", "metric_keys", "artifact_groups"):
        value = requested.get(key)
        if value is None:
            continue
        expected = canonical_protocol[key]
        if key in {"metric_keys", "artifact_groups"}:
            if list(value) != list(expected):
                raise ValueError(
                    f"benchmark evaluator protocol field {key!r} must match the canonical owner "
                    f"for task kind {task_kind!r}"
                )
            continue
        if str(value).strip() != str(expected):
            raise ValueError(
                f"benchmark evaluator protocol field {key!r} must match the canonical owner "
                f"for task kind {task_kind!r}"
            )
    protocol_id = str(resolved["protocol_id"])
    if task_kind not in canonical_task_kinds_for_protocol_id(protocol_id):
        raise ValueError(
            f"benchmark evaluator protocol id {protocol_id!r} does not own task kind {task_kind!r}"
        )
    for key in ("prediction_key", "target_key"):
        if key not in requested:
            continue
        value = str(requested[key] or "").strip()
        if not value:
            resolved[key] = ""
            continue
        resolved[key] = value
    return resolved


def require_prediction_target(
    outputs: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    task_label: str,
) -> tuple[Any, Any]:
    payload = dict(outputs)
    prediction_key = str(protocol.get("prediction_key", "")).strip()
    target_key = str(protocol.get("target_key", "")).strip()
    if not prediction_key or not target_key:
        raise ValueError(
            f"{task_label} benchmark protocol must declare non-empty prediction_key and target_key"
        )
    prediction = payload.get(prediction_key)
    target = payload.get(target_key)
    if prediction is None or target is None:
        raise ValueError(
            f"{task_label} benchmark outputs must include '{prediction_key}' and '{target_key}'"
        )
    return prediction, target


def build_evaluation_result(
    *,
    protocol: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    resolved_protocol = dict(protocol)
    normalized_metrics = {str(key): float(value) for key, value in metrics.items()}
    primary_metric = str(resolved_protocol.get("primary_metric", ""))
    primary_value = normalized_metrics.get(primary_metric)
    result: dict[str, Any] = {
        "protocol": resolved_protocol,
        "metrics": normalized_metrics,
        "task_kind": resolved_protocol.get("task_kind"),
        "primary_metric": primary_metric,
    }
    if primary_value is not None:
        result["primary_metric_value"] = float(primary_value)
    return result


__all__ = [
    "build_evaluation_result",
    "require_prediction_target",
    "resolve_protocol",
]
