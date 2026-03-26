"""Dataset-owned task-binding/materialization helpers."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any


def _binding_string_list(
    payload: Any,
    *,
    field_name: str,
    owner: str,
) -> list[str]:
    if payload is None or payload == "":
        return []
    if not isinstance(payload, list):
        raise TypeError(f"{owner} field '{field_name}' must be a list.")
    return [str(value) for value in payload]


def _binding_mapping(
    payload: Any,
    *,
    field_name: str,
    owner: str,
) -> dict[str, Any]:
    if payload is None or payload == "":
        return {}
    if not isinstance(payload, dict):
        raise TypeError(f"{owner} field '{field_name}' must be a mapping.")
    return {str(key): value for key, value in payload.items()}


def _binding_shape_value(
    payload: Any,
    *,
    field_name: str,
    owner: str,
) -> list[object]:
    if payload is None or payload == "":
        return []
    if not isinstance(payload, list):
        raise TypeError(f"{owner} field '{field_name}' must be a list.")
    return list(payload)


def _default_dataset_slot_shape(*, target_kind: str) -> list[object] | None:
    if str(target_kind).strip() == "categorical_label":
        return []
    return None


def _binding_optional_mapping(
    payload: Any,
    *,
    field_name: str,
    owner: str,
) -> dict[str, Any] | None:
    if payload is None or payload == "":
        return None
    if not isinstance(payload, dict):
        raise TypeError(f"{owner} field '{field_name}' must be a mapping.")
    return copy.deepcopy(payload)


def resolve_dataset_task_binding_payload(
    task_binding: dict[str, Any],
    *,
    owner: str,
) -> dict[str, object]:
    """Compile dataset-local binding payload without duplicating canonical task semantics."""

    from octosense.tasks.definitions import load as load_task_definition
    from octosense.tasks.leakage import resolve_dataset_leakage_keys

    binding_id = str(task_binding.get("binding_id") or "").strip()
    if not binding_id:
        raise ValueError(f"{owner} task_binding payload is missing binding_id.")
    task_id = str(task_binding.get("task_id") or "").strip()
    if not task_id:
        raise ValueError(f"{owner} task binding '{binding_id}' must declare canonical task_id.")

    task_spec = load_task_definition(task_id)
    semantic_schema = task_spec.target_schema

    label_space = task_binding.get("label_space", {})
    if label_space is None or label_space == "":
        label_space = {}
    if not isinstance(label_space, dict):
        raise TypeError(f"{owner} task binding '{binding_id}' field 'label_space' must be a mapping.")

    target_binding = task_binding.get("target_binding", {})
    if target_binding is None or target_binding == "":
        target_binding = {}
    if not isinstance(target_binding, dict):
        raise TypeError(
            f"{owner} task binding '{binding_id}' field 'target_binding' must be a mapping."
        )

    field_mapping = _binding_mapping(
        target_binding.get("fields"),
        field_name="target_binding.fields",
        owner=f"{owner} task binding '{binding_id}'",
    )
    shape_source = _binding_mapping(
        target_binding.get("shape_source"),
        field_name="target_binding.shape_source",
        owner=f"{owner} task binding '{binding_id}'",
    )
    slots: dict[str, object] = {}
    dataset_target_schema: dict[str, object] = {}
    for semantic_field in semantic_schema.fields:
        concrete_field = field_mapping.get(semantic_field)
        if concrete_field in {None, ""}:
            raise ValueError(
                f"{owner} task binding '{binding_id}' must declare target_binding.fields."
                f"{semantic_field} for canonical task_id '{task_id}'."
            )
        concrete_field_name = str(concrete_field)
        slot_payload: dict[str, object] = {"concrete_field": concrete_field_name}
        slot_shape = _default_dataset_slot_shape(target_kind=semantic_schema.target_kind)
        if semantic_field in shape_source:
            raw_shape_source = shape_source[semantic_field]
            if raw_shape_source is not None and raw_shape_source != "":
                if not isinstance(raw_shape_source, dict):
                    raise TypeError(
                        f"{owner} task binding '{binding_id}' field "
                        f"'target_binding.shape_source.{semantic_field}' must be a mapping."
                    )
                resolved_shape_source = copy.deepcopy(raw_shape_source)
                slot_payload["shape_source"] = resolved_shape_source
                if "value" in resolved_shape_source:
                    slot_shape = _binding_shape_value(
                        resolved_shape_source.get("value"),
                        field_name=f"target_binding.shape_source.{semantic_field}.value",
                        owner=f"{owner} task binding '{binding_id}'",
                    )
        if slot_shape is not None:
            slot_payload["shape"] = list(slot_shape)
            dataset_target_schema[concrete_field_name] = list(slot_shape)
        slots[semantic_field] = slot_payload

    concrete_schema: dict[str, object] = {"slots": slots}
    binding_payload: dict[str, object] = {
        "task_binding": binding_id,
        "task_id": task_id,
        "target_schema": concrete_schema,
    }

    vocabulary = label_space.get("vocabulary")
    if vocabulary is not None and vocabulary != "":
        if not isinstance(vocabulary, dict):
            raise TypeError(
                f"{owner} task binding '{binding_id}' field 'label_space.vocabulary' must be a mapping."
            )
        binding_payload["vocabulary"] = copy.deepcopy(vocabulary)

    name_source = _binding_optional_mapping(
        label_space.get("name_source"),
        field_name="label_space.name_source",
        owner=f"{owner} task binding '{binding_id}'",
    )
    if name_source is not None:
        binding_payload["name_source"] = name_source

    labels = _binding_string_list(
        label_space.get("labels"),
        field_name="label_space.labels",
        owner=f"{owner} task binding '{binding_id}'",
    )
    if labels:
        binding_payload["labels"] = labels

    leakage_keys = resolve_dataset_leakage_keys(
        task_id,
        _binding_string_list(
            task_binding.get("leakage_keys"),
            field_name="leakage_keys",
            owner=f"{owner} task binding '{binding_id}'",
        ),
        owner=f"{owner} task binding '{binding_id}'",
    )
    if target_binding.get("source") not in {None, ""}:
        binding_payload["target_source"] = str(target_binding["source"])
    if leakage_keys:
        binding_payload["leakage_keys"] = leakage_keys

    for field_name in ("id_field", "name_field"):
        value = label_space.get(field_name)
        if value not in {None, ""}:
            binding_payload[field_name] = str(value)

    if task_binding.get("label_field") not in {None, ""}:
        binding_payload["label_field"] = str(task_binding["label_field"])

    if dataset_target_schema:
        binding_payload["dataset_target_schema"] = dataset_target_schema
    return binding_payload


def resolve_materialized_task_identity(
    materialization: Mapping[str, Any],
    *,
    owner: str,
) -> tuple[str, dict[str, object]]:
    """Resolve canonical task identity plus dataset-local target layout."""

    task_id = str(materialization.get("task_id", "")).strip()
    if not task_id:
        raise ValueError(f"{owner} is missing canonical task_id.")
    target_schema = materialization.get("target_schema")
    if target_schema is None or target_schema == "":
        return task_id, {}
    if not isinstance(target_schema, Mapping):
        raise ValueError(f"{owner} is missing dataset-local target_schema.")

    return task_id, {str(key): value for key, value in target_schema.items()}


def canonical_task_semantics(task_id: str) -> tuple[str, str]:
    """Resolve canonical task/target semantics from ``octosense.tasks`` only."""

    from octosense.tasks.definitions import load as load_task_definition

    task_spec = load_task_definition(task_id)
    return str(task_spec.kind), str(task_spec.target_schema.target_kind)


def _resolve_target_field_bridge_payload(
    binding_payload: Mapping[str, Any],
    *,
    owner: str,
) -> dict[str, str]:
    target_binding = binding_payload.get("target_binding")
    if not isinstance(target_binding, dict):
        raise TypeError(f"{owner} task binding must define mapping field 'target_binding'.")
    field_mapping = target_binding.get("fields")
    if not isinstance(field_mapping, dict):
        raise TypeError(f"{owner} task binding must define mapping field 'target_binding.fields'.")

    bridge = {
        str(semantic_field): str(concrete_field)
        for semantic_field, concrete_field in field_mapping.items()
        if str(semantic_field).strip() and str(concrete_field).strip()
    }
    if not bridge:
        raise ValueError(f"{owner} task binding must declare at least one target_binding.fields entry.")
    return bridge


def resolve_target_field_bridge_from_task_binding(
    dataset_id: str,
    task_binding: str,
    *,
    owner: str,
) -> dict[str, str]:
    """Resolve canonical->dataset target field mapping from a dataset task binding."""

    from octosense.datasets.catalog import get_dataset_binding_payload

    resolved_dataset_id = str(dataset_id)
    resolved_binding_id = str(task_binding)
    try:
        binding_payload = get_dataset_binding_payload(
            resolved_dataset_id,
            binding_kind="task_binding",
            binding_id=resolved_binding_id,
        )
    except KeyError as exc:
        message = str(exc)
        if message.startswith("'") and message.endswith("'"):
            message = message[1:-1]
        raise ValueError(
            f"Unsupported binding_id {resolved_binding_id!r} while resolving {owner}. {message}"
        ) from exc
    return _resolve_target_field_bridge_payload(binding_payload, owner=owner)


__all__ = [
    "canonical_task_semantics",
    "resolve_target_field_bridge_from_task_binding",
    "resolve_dataset_task_binding_payload",
    "resolve_materialized_task_identity",
]
