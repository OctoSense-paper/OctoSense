"""Builtin owner entry for the HuPR dataset definition."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from ...base import (
    resolve_dataset_root,
)
from ...catalog import get_dataset_binding_payload, list_dataset_binding_ids
from ...core.builder import (
    DatasetBuildArtifact,
    DatasetBuildRequest,
)
from ...core.materialization import (
    artifact_split_payload,
    build_manifest_backed_view,
    merge_materialized_split_views,
)
from ...core.manifest import manifest_from_rows
from ...core.task_binding import (
    canonical_task_semantics,
    resolve_dataset_task_binding_payload,
    resolve_materialized_task_identity,
)
from .ingest import (
    HUPR_BANDWIDTH,
    HUPR_CENTER_FREQ,
    HuPRDataset,
    _hupr_declared_concrete_target_layout,
)
from .manifest import (
    build_hupr_plan,
)

DATASET_ID = "hupr"

if TYPE_CHECKING:
    from octosense.datasets.base import DatasetLoadRequest
    from octosense.datasets.views.dataset_view import DatasetView


def _require_hupr_task_binding_fields(
    task_binding: dict[str, Any],
    *,
    owner: str,
) -> dict[str, str]:
    target_binding = task_binding.get("target_binding")
    if not isinstance(target_binding, dict):
        raise TypeError(f"{owner} must define mapping field 'target_binding'.")
    field_mapping = target_binding.get("fields")
    if not isinstance(field_mapping, dict):
        raise TypeError(f"{owner} must define mapping field 'target_binding.fields'.")
    resolved = {
        str(semantic_field): str(concrete_field)
        for semantic_field, concrete_field in field_mapping.items()
        if str(semantic_field).strip() and str(concrete_field).strip()
    }
    if not resolved:
        raise ValueError(f"{owner} must declare at least one target_binding.fields entry.")
    return resolved


def _resolve_task_binding_contract(binding_id: str | None) -> dict[str, Any]:
    candidate = "" if binding_id in {None, ""} else str(binding_id).strip()
    available = list_dataset_binding_ids(DATASET_ID, binding_kind="task_binding")
    if not candidate:
        supported = ", ".join(available) or "<none>"
        raise ValueError(
            "HuPR requires an explicit task_binding; "
            "implicit default/singleton fallback is not supported. "
            f"Supported bindings: {supported}."
        )
    if candidate not in available:
        supported = ", ".join(available) or "<none>"
        raise ValueError(
            f"HuPR task_binding must be one of: {supported}. Received {candidate!r}."
        )
    return get_dataset_binding_payload(
        DATASET_ID,
        binding_kind="task_binding",
        binding_id=candidate,
    )


def _resolve_split_scheme_id(binding_id: str | None) -> str:
    candidate = "" if binding_id in {None, ""} else str(binding_id).strip()
    available = list_dataset_binding_ids(DATASET_ID, binding_kind="split_scheme")
    if not candidate:
        supported = ", ".join(available) or "<none>"
        raise ValueError(
            "HuPR requires an explicit split_scheme; "
            "implicit default/singleton fallback is not supported. "
            f"Supported bindings: {supported}."
        )
    if candidate not in available:
        supported = ", ".join(available) or "<none>"
        raise ValueError(
            f"HuPR split_scheme must be one of: {supported}. Received {candidate!r}."
        )
    payload = get_dataset_binding_payload(
        DATASET_ID,
        binding_kind="split_scheme",
        binding_id=candidate,
    )
    return str(payload["binding_id"])


def _artifact_materialization(artifact: DatasetBuildArtifact) -> dict[str, Any]:
    payload = getattr(artifact, "materialization", None)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("HuPR artifact materialization payload must be a mapping.")
    return dict(payload)


def _resolve_split_file_indexes(materialization: dict[str, Any]) -> dict[str, list[int]]:
    payload = materialization.get("split_file_indexes")
    if not isinstance(payload, dict):
        raise ValueError("HuPR artifact materialization payload is missing split_file_indexes.")
    resolved: dict[str, list[int]] = {}
    for split_name, raw_indexes in payload.items():
        if not isinstance(raw_indexes, list):
            raise ValueError(
                "HuPR artifact materialization split_file_indexes values must be lists. "
                f"split={split_name!r}"
            )
        resolved[str(split_name)] = [int(index) for index in raw_indexes]
    return resolved


def _resolve_hupr_concrete_target_layout(
    binding_payload: dict[str, object],
    task_binding: dict[str, Any],
) -> dict[str, object]:
    owner = f"HuPR task binding '{task_binding['binding_id']}'"
    dataset_target_schema = binding_payload.get("dataset_target_schema")
    if not isinstance(dataset_target_schema, dict):
        raise ValueError(
            f"{owner} requires shared compiler dataset_target_schema derived from target_binding.shape_source."
        )
    field_mapping = _require_hupr_task_binding_fields(task_binding, owner=owner)

    base_schema = _hupr_declared_concrete_target_layout()
    declared_fields: set[str] = set()
    for semantic_field, concrete_field_value in field_mapping.items():
        concrete_field = str(concrete_field_value or "").strip()
        if not concrete_field:
            raise ValueError(
                f"{owner} is missing concrete field mapping for semantic slot "
                f"{semantic_field!r}."
            )
        if concrete_field not in base_schema:
            declared = ", ".join(sorted(base_schema))
            raise ValueError(
                f"{owner} declares concrete field {concrete_field!r} missing from HuPR "
                f"dataset-local target layout schema. Declared: {declared}"
            )
        declared_fields.add(concrete_field)
    missing_shapes = sorted(field for field in declared_fields if field not in dataset_target_schema)
    if missing_shapes:
        joined = ", ".join(missing_shapes)
        raise ValueError(
            f"{owner} requires explicit target_binding.shape_source value payloads for: {joined}"
        )
    unexpected_shapes = sorted(
        field for field in dataset_target_schema if field not in declared_fields
    )
    if unexpected_shapes:
        joined = ", ".join(unexpected_shapes)
        raise ValueError(
            f"{owner} shared compiler emitted undeclared dataset_target_schema fields: {joined}"
        )
    return {field: list(dataset_target_schema[field]) for field in sorted(declared_fields)}


def _resolve_hupr_target_field_bridge(
    task_binding: dict[str, Any],
) -> dict[str, str]:
    owner = f"HuPR task binding '{task_binding['binding_id']}'"
    return _require_hupr_task_binding_fields(task_binding, owner=owner)


def _resolve_hupr_artifact_target_field_bridge(
    materialization: dict[str, Any],
    *,
    owner: str,
) -> dict[str, str]:
    payload = materialization.get("target_field_bridge")
    if not isinstance(payload, dict):
        raise ValueError(f"{owner} is missing target_field_bridge.")
    resolved: dict[str, str] = {}
    for canonical_field, concrete_field in payload.items():
        canonical_name = str(canonical_field or "").strip()
        concrete_name = str(concrete_field or "").strip()
        if not canonical_name or not concrete_name:
            raise ValueError(
                f"{owner} target_field_bridge entries must map non-empty canonical and "
                "concrete field names."
            )
        resolved[canonical_name] = concrete_name
    if not resolved:
        raise ValueError(f"{owner} target_field_bridge must not be empty.")
    return resolved


def _validate_hupr_target_field_bridge(
    target_field_bridge: dict[str, str],
    *,
    task_id: str,
    concrete_target_schema: dict[str, object],
    owner: str,
) -> dict[str, str]:
    from octosense.tasks.definitions import load as load_task_definition

    task_spec = load_task_definition(task_id)
    expected_fields = tuple(str(field) for field in task_spec.target_schema.fields)
    bridge_fields = tuple(str(field) for field in target_field_bridge)
    if bridge_fields != expected_fields:
        raise ValueError(
            f"{owner} target_field_bridge must preserve canonical task field order "
            f"{expected_fields!r}. Received {bridge_fields!r}."
        )

    missing_concrete_fields = [
        concrete_field
        for concrete_field in target_field_bridge.values()
        if concrete_field not in concrete_target_schema
    ]
    if missing_concrete_fields:
        missing = ", ".join(missing_concrete_fields)
        raise ValueError(
            f"{owner} target_field_bridge references undeclared dataset-local target fields: "
            f"{missing}"
        )
    return dict(target_field_bridge)


def build(request: DatasetBuildRequest) -> DatasetBuildArtifact:
    if request.dataset_root is None:
        raise ValueError("HuPR builder requires request.dataset_root to be resolved.")

    task_binding = _resolve_task_binding_contract(request.task_binding)
    binding_payload = resolve_dataset_task_binding_payload(task_binding, owner="HuPR")
    concrete_target_layout = _resolve_hupr_concrete_target_layout(
        binding_payload,
        task_binding,
    )
    task_id = str(task_binding["task_id"])
    target_field_bridge = _validate_hupr_target_field_bridge(
        _resolve_hupr_target_field_bridge(task_binding),
        task_id=task_id,
        concrete_target_schema=concrete_target_layout,
        owner=f"HuPR task binding '{task_binding['binding_id']}'",
    )
    split_scheme = _resolve_split_scheme_id(request.split_scheme)
    canonical_request = replace(
        request,
        split_scheme=split_scheme,
        task_binding=str(task_binding["binding_id"]),
    )
    build_plan = build_hupr_plan(canonical_request)
    variant = build_plan.variant
    supported_variants = task_binding.get("supported_variants", [])
    if isinstance(supported_variants, list) and supported_variants:
        if variant not in {str(value) for value in supported_variants}:
            raise ValueError(
                f"HuPR task binding '{task_binding['binding_id']}' does not support variant "
                f"{variant!r}."
            )
    return DatasetBuildArtifact(
        dataset_id=DATASET_ID,
        variant=variant,
        manifest=manifest_from_rows(
            build_plan.manifest_rows,
            dataset_id=DATASET_ID,
            variant=variant,
        ),
        dataset_root=request.dataset_root,
        materialization={
            "split_scheme": build_plan.split_scheme,
            "task_binding": str(task_binding["binding_id"]),
            "task_id": task_id,
            "target_schema": dict(concrete_target_layout),
            "target_field_bridge": target_field_bridge,
            "split_file_indexes": dict(build_plan.split_file_indexes),
        },
    )


def materialize_views_from_artifact(
    request: "DatasetLoadRequest",
    artifact: DatasetBuildArtifact,
) -> tuple[
    str,
    dict[str, int] | None,
    str,
    str,
    dict[str, object] | None,
    dict[str, "DatasetView"],
]:
    resolved_path = resolve_dataset_root(DATASET_ID, override=request.dataset_root)
    variant = str(getattr(artifact, "variant", request.variant or ""))
    materialization = _artifact_materialization(artifact)
    if not str(materialization.get("split_scheme", "")).strip():
        raise ValueError("HuPR artifact materialization payload is missing split_scheme.")
    task_binding = str(materialization.get("task_binding", "")).strip()
    if not task_binding:
        raise ValueError("HuPR artifact materialization payload is missing task_binding.")
    task_id, concrete_target_schema = resolve_materialized_task_identity(
        materialization,
        owner="HuPR artifact materialization payload",
    )
    target_field_bridge = _validate_hupr_target_field_bridge(
        _resolve_hupr_artifact_target_field_bridge(
            materialization,
            owner="HuPR artifact materialization payload",
        ),
        task_id=task_id,
        concrete_target_schema=dict(concrete_target_schema),
        owner="HuPR artifact materialization payload",
    )
    task_kind, target_kind = canonical_task_semantics(task_id)
    split_files = _resolve_split_file_indexes(materialization)
    views_by_split: dict[str, DatasetView] = {}
    for split_name, file_indexes in split_files.items():
        source_dataset = HuPRDataset(
            str(resolved_path),
            variant=variant,
            file_indexes=list(file_indexes),
            task_binding=task_binding,
            task_kind=task_kind,
            target_kind=target_kind,
            target_schema=concrete_target_schema,
        )
        metadata_rows, manifest_rows = artifact_split_payload(artifact, split=split_name)
        views_by_split[split_name] = build_manifest_backed_view(
            dataset=source_dataset,
            indices=list(range(len(source_dataset))),
            dataset_id=DATASET_ID,
            variant=variant,
            split=split_name,
            task_id=task_id,
            task_kind=task_kind,
            target_kind=target_kind,
            target_schema=dict(concrete_target_schema),
            target_field_bridge=target_field_bridge,
            metadata_rows=metadata_rows,
            manifest_rows=manifest_rows,
        )

    return (
        variant,
        None,
        task_kind,
        target_kind,
        dict(concrete_target_schema),
        views_by_split,
    )


def build_view(
    request: "DatasetLoadRequest",
    *,
    split: str | None = None,
) -> "DatasetView":
    build_request = request.to_build_request()
    if build_request.dataset_root is None:
        build_request = replace(
            build_request,
            dataset_root=resolve_dataset_root(DATASET_ID, override=request.dataset_root),
        )
    artifact = build(build_request)
    (
        variant,
        label_mapping,
        task_kind,
        target_kind,
        target_schema,
        views_by_split,
    ) = materialize_views_from_artifact(request=request, artifact=artifact)
    materialization = _artifact_materialization(artifact)
    task_id, _ = resolve_materialized_task_identity(
        materialization,
        owner="HuPR artifact materialization payload",
    )
    target_schema = dict(target_schema or {})
    target_field_bridge = _validate_hupr_target_field_bridge(
        _resolve_hupr_artifact_target_field_bridge(
            materialization,
            owner="HuPR artifact materialization payload",
        ),
        task_id=task_id,
        concrete_target_schema=target_schema,
        owner="HuPR artifact materialization payload",
    )
    return merge_materialized_split_views(
        dataset_id=DATASET_ID,
        variant=variant,
        split=split,
        task_id=task_id,
        task_kind=task_kind,
        target_kind=target_kind,
        label_mapping=label_mapping,
        target_schema=target_schema,
        target_field_bridge=target_field_bridge,
        views_by_split=views_by_split,
    )


__all__ = [
    "DATASET_ID",
    "HUPR_BANDWIDTH",
    "HUPR_CENTER_FREQ",
    "HuPRDataset",
    "build",
    "build_view",
    "materialize_views_from_artifact",
]
