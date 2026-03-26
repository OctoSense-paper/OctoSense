"""Canonical benchmark protocol catalog.

This module is the benchmark-owned source of truth for protocol families and
their task-kind bindings. Other owners may import these helpers, but should not
redefine the canonical protocol table elsewhere.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

_CANONICAL_PROTOCOLS_BY_ID: dict[str, dict[str, Any]] = {
    "classification": {
        "name": "classification",
        "protocol_id": "classification",
        "task_kind": "classification",
        "primary_metric": "accuracy",
        "prediction_key": "predictions",
        "metric_keys": ["accuracy", "macro_f1", "correct", "count"],
        "artifact_groups": ["metrics", "provenance", "environment", "run_manifest", "report"],
    },
    "localization": {
        "name": "localization",
        "protocol_id": "localization",
        "task_kind": "localization",
        "primary_metric": "mae",
        "prediction_key": "predictions",
        "metric_keys": ["mae", "rmse", "count"],
        "artifact_groups": ["metrics", "provenance", "environment", "run_manifest", "report"],
    },
    "pose": {
        "name": "pose",
        "protocol_id": "pose",
        "task_kind": "pose",
        "primary_metric": "mpjpe",
        "prediction_key": "predictions",
        "metric_keys": ["mpjpe", "rmse", "count"],
        "artifact_groups": ["metrics", "provenance", "environment", "run_manifest", "report"],
    },
    "respiration": {
        "name": "respiration",
        "protocol_id": "respiration",
        "task_kind": "respiration",
        "primary_metric": "rmse",
        "prediction_key": "predictions",
        "metric_keys": ["rmse", "mae", "count"],
        "artifact_groups": ["metrics", "provenance", "environment", "run_manifest", "report"],
    },
}

_PROTOCOL_ID_BY_TASK_KIND: dict[str, str] = {
    "classification": "classification",
    "gait": "classification",
    "pose": "pose",
    "localization": "localization",
    "respiration": "respiration",
}

_TARGET_KEY_BY_TASK_KIND: dict[str, str] = {
    "classification": "class_label",
    "gait": "identity_label",
    "pose": "pose_keypoints",
    "localization": "position",
    "respiration": "respiration_signal",
}

_EXECUTION_ADAPTER_BY_TASK_KIND: dict[str, str] = {
    "classification": "scalar_index_supervised",
    "gait": "scalar_index_supervised",
    "pose": "structured_tensor_mapping",
    "localization": "structured_tensor_mapping",
    "respiration": "structured_tensor_mapping",
}

_PRIMARY_METRIC_TO_PROTOCOL_ID: dict[str, str] = {
    str(protocol["primary_metric"]): protocol_id
    for protocol_id, protocol in _CANONICAL_PROTOCOLS_BY_ID.items()
}

_TASK_KINDS_BY_PROTOCOL_ID: dict[str, tuple[str, ...]] = {
    protocol_id: tuple(
        task_kind
        for task_kind, owned_protocol_id in _PROTOCOL_ID_BY_TASK_KIND.items()
        if owned_protocol_id == protocol_id
    )
    for protocol_id in _CANONICAL_PROTOCOLS_BY_ID
}


def _normalized_non_empty(value: str, *, label: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{label} must be a non-empty string")
    return normalized


def canonical_protocol_for_protocol_id(protocol_id: str) -> dict[str, Any]:
    normalized = _normalized_non_empty(protocol_id, label="protocol_id")
    protocol = _CANONICAL_PROTOCOLS_BY_ID.get(normalized)
    if protocol is None:
        supported = ", ".join(sorted(_CANONICAL_PROTOCOLS_BY_ID))
        raise ValueError(
            f"Protocol id {protocol_id!r} does not have a canonical benchmark owner. "
            f"Supported protocol ids: {supported}"
        )
    return dict(protocol)


def canonical_protocol_id_for_task_kind(task_kind: str) -> str:
    normalized = _normalized_non_empty(task_kind, label="task_kind")
    protocol_id = _PROTOCOL_ID_BY_TASK_KIND.get(normalized)
    if protocol_id is None:
        supported = ", ".join(sorted(_PROTOCOL_ID_BY_TASK_KIND))
        raise ValueError(
            f"Task kind {task_kind!r} does not have a canonical benchmark protocol owner. "
            f"Supported task kinds: {supported}"
        )
    return protocol_id


def canonical_protocol_id_for_primary_metric(primary_metric: str) -> str:
    normalized = _normalized_non_empty(primary_metric, label="primary_metric")
    protocol_id = _PRIMARY_METRIC_TO_PROTOCOL_ID.get(normalized)
    if protocol_id is None:
        supported = ", ".join(sorted(_PRIMARY_METRIC_TO_PROTOCOL_ID))
        raise ValueError(
            f"Primary metric {primary_metric!r} does not have a canonical benchmark protocol owner. "
            f"Supported primary metrics: {supported}"
        )
    return protocol_id


def canonical_target_key_for_task_kind(task_kind: str) -> str:
    normalized = _normalized_non_empty(task_kind, label="task_kind")
    target_key = _TARGET_KEY_BY_TASK_KIND.get(normalized)
    if target_key is None:
        supported = ", ".join(sorted(_TARGET_KEY_BY_TASK_KIND))
        raise ValueError(
            f"Task kind {task_kind!r} does not have a canonical benchmark target field. "
            f"Supported task kinds: {supported}"
        )
    return target_key


def canonical_task_kinds_for_protocol_id(protocol_id: str) -> tuple[str, ...]:
    normalized = _normalized_non_empty(protocol_id, label="protocol_id")
    task_kinds = _TASK_KINDS_BY_PROTOCOL_ID.get(normalized)
    if task_kinds is None:
        supported = ", ".join(sorted(_TASK_KINDS_BY_PROTOCOL_ID))
        raise ValueError(
            f"Protocol id {protocol_id!r} does not have canonical task-kind owners. "
            f"Supported protocol ids: {supported}"
        )
    return task_kinds


def canonical_protocol_for_task_kind(task_kind: str) -> dict[str, Any]:
    normalized = _normalized_non_empty(task_kind, label="task_kind")
    protocol = canonical_protocol_for_protocol_id(canonical_protocol_id_for_task_kind(normalized))
    if normalized != str(protocol["task_kind"]):
        protocol["task_kind"] = normalized
    protocol["target_key"] = canonical_target_key_for_task_kind(normalized)
    return dict(protocol)


def canonical_execution_adapter_for_task_kind(task_kind: str) -> str:
    normalized = _normalized_non_empty(task_kind, label="task_kind")
    adapter = _EXECUTION_ADAPTER_BY_TASK_KIND.get(normalized)
    if adapter is None:
        supported = ", ".join(sorted(_EXECUTION_ADAPTER_BY_TASK_KIND))
        raise ValueError(
            f"Task kind {task_kind!r} does not have a canonical execution adapter. "
            f"Supported task kinds: {supported}"
        )
    return adapter


def resolve_runtime_protocol_payload(
    protocol_payload: Mapping[str, Any] | None,
    *,
    task_kind: str,
    primary_metric: str,
    target_fields: Sequence[str],
    execution_adapter: str,
    field_path: str = "spec.protocol",
) -> dict[str, Any]:
    """Canonicalize one execution-time protocol payload against benchmark-owned contracts."""

    canonical_protocol = canonical_protocol_for_task_kind(task_kind)
    resolved = dict(protocol_payload or {})
    expected_fields = (
        "name",
        "protocol_id",
        "task_kind",
        "primary_metric",
        "metric_keys",
        "artifact_groups",
    )
    for key in expected_fields:
        expected = canonical_protocol.get(key)
        actual = resolved.get(key)
        if actual is None:
            resolved[key] = list(expected) if isinstance(expected, list) else expected
            continue
        if key in {"metric_keys", "artifact_groups"}:
            if list(actual) != list(expected):
                raise ValueError(
                    f"{field_path}['{key}'] must match the canonical benchmark owner "
                    f"for task kind {task_kind!r}."
                )
            resolved[key] = list(expected)
            continue
        if str(actual).strip() != str(expected):
            raise ValueError(
                f"{field_path}['{key}'] must match the canonical benchmark owner "
                f"for task kind {task_kind!r}; got {actual!r}, expected {expected!r}."
            )
        resolved[key] = expected

    expected_adapter = canonical_execution_adapter_for_task_kind(task_kind)
    if execution_adapter != expected_adapter:
        raise ValueError(
            "execution adapter must match the canonical benchmark contract for task kind "
            f"{task_kind!r}; got {execution_adapter!r}, expected {expected_adapter!r}."
        )
    adapter = str(resolved.get("execution_adapter", expected_adapter) or "").strip()
    if adapter != expected_adapter:
        raise ValueError(
            f"{field_path}['execution_adapter'] must match the canonical benchmark "
            f"contract for task kind {task_kind!r}; got {adapter!r}, expected "
            f"{expected_adapter!r}."
        )
    resolved["execution_adapter"] = expected_adapter

    # Keep primary_metric aligned with the task-owned output contract.
    normalized_primary_metric = _normalized_non_empty(primary_metric, label="primary_metric")
    if normalized_primary_metric != str(canonical_protocol["primary_metric"]):
        raise ValueError(
            "primary metric must match the canonical benchmark contract for task kind "
            f"{task_kind!r}; got {normalized_primary_metric!r}, expected "
            f"{canonical_protocol['primary_metric']!r}."
        )

    prediction_key = str(
        resolved.get("prediction_key", canonical_protocol.get("prediction_key", "predictions")) or ""
    ).strip()
    if not prediction_key:
        raise ValueError(f"{field_path}['prediction_key'] must be a non-empty string.")
    resolved["prediction_key"] = prediction_key

    normalized_target_fields = [str(field).strip() for field in target_fields if str(field).strip()]
    canonical_target_key = canonical_target_key_for_task_kind(task_kind)
    target_key = str(resolved.get("target_key", canonical_target_key) or "").strip()
    if not target_key:
        raise ValueError(f"{field_path}['target_key'] must resolve to a canonical target field.")
    if normalized_target_fields and target_key not in normalized_target_fields:
        raise ValueError(
            f"{field_path}['target_key'] must match TaskSpec.target_schema.fields. "
            f"Got {target_key!r}; declared fields: {normalized_target_fields!r}."
        )
    resolved["target_key"] = target_key
    return resolved


__all__ = [
    "canonical_execution_adapter_for_task_kind",
    "canonical_protocol_for_protocol_id",
    "canonical_protocol_for_task_kind",
    "canonical_protocol_id_for_primary_metric",
    "canonical_protocol_id_for_task_kind",
    "canonical_target_key_for_task_kind",
    "canonical_task_kinds_for_protocol_id",
    "resolve_runtime_protocol_payload",
]
