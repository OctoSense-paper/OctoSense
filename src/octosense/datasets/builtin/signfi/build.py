"""Builtin owner entry for the SignFi dataset definition."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from ...base import (
    resolve_dataset_root,
)
from ...catalog import (
    get_dataset_binding_payload,
    list_dataset_binding_ids,
)
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
    SIGNFI_BANDWIDTH,
    SIGNFI_CENTER_FREQ,
    SIGNFI_NOMINAL_SAMPLE_RATE,
    SignFiDataset,
    load_signfi_dataset,
    signfi_dataset_card,
)
from .manifest import (
    _resolve_signfi_split_plan,
    build_signfi_manifest_plan,
)

DATASET_ID = "signfi"

if TYPE_CHECKING:
    from octosense.datasets.base import DatasetLoadRequest
    from octosense.datasets.views.dataset_view import DatasetView


def _resolve_task_binding_contract(binding_id: str | None) -> dict[str, object]:
    candidate = "" if binding_id in {None, ""} else str(binding_id).strip()
    available = list_dataset_binding_ids(DATASET_ID, binding_kind="task_binding")
    if not candidate:
        supported = ", ".join(available) or "<none>"
        raise ValueError(
            "SignFi requires an explicit task_binding; "
            "implicit default/singleton fallback is not supported. "
            f"Supported bindings: {supported}."
        )
    if candidate not in available:
        supported = ", ".join(available) or "<none>"
        raise ValueError(
            f"SignFi task_binding must be one of: {supported}. Received {candidate!r}."
        )
    return get_dataset_binding_payload(
        DATASET_ID,
        binding_kind="task_binding",
        binding_id=candidate,
    )


def _artifact_materialization(artifact: DatasetBuildArtifact) -> dict[str, object]:
    payload = getattr(artifact, "materialization", None)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("SignFi artifact materialization payload must be a mapping.")
    return dict(payload)


def _materialized_target_schema(
    binding_payload: dict[str, object],
    *,
    task_binding: dict[str, Any],
    owner: str,
) -> dict[str, object]:
    dataset_target_schema = binding_payload.get("dataset_target_schema")
    resolved_shapes = (
        dict(dataset_target_schema) if isinstance(dataset_target_schema, dict) else {}
    )
    target_binding = task_binding.get("target_binding")
    if not isinstance(target_binding, dict):
        raise TypeError(f"{owner} must define mapping field 'target_binding'.")
    field_mapping = target_binding.get("fields")
    if not isinstance(field_mapping, dict):
        raise TypeError(f"{owner} must define mapping field 'target_binding.fields'.")
    materialized_schema: dict[str, object] = {}
    for concrete_field_value in field_mapping.values():
        concrete_field = str(concrete_field_value or "").strip()
        if not concrete_field:
            raise ValueError(f"{owner} must not contain empty concrete target fields.")
        shape = resolved_shapes.get(concrete_field, [])
        if not isinstance(shape, list):
            raise TypeError(
                f"{owner} dataset_target_schema.{concrete_field} must resolve to a list shape."
            )
        materialized_schema[concrete_field] = list(shape)
    return materialized_schema


def _resolve_materialized_view_contract(
    materialization: dict[str, object],
) -> tuple[str, str, str, dict[str, object]]:
    task_id, target_schema = resolve_materialized_task_identity(
        materialization,
        owner="SignFi artifact materialization payload",
    )
    task_kind, target_kind = canonical_task_semantics(task_id)
    return (
        task_id,
        task_kind,
        target_kind,
        dict(target_schema),
    )


def build(request: DatasetBuildRequest) -> DatasetBuildArtifact:
    manifest_plan = build_signfi_manifest_plan(request)
    task_binding = _resolve_task_binding_contract(request.task_binding)
    supported_variants = tuple(str(item) for item in task_binding.get("supported_variants", []))
    if supported_variants and manifest_plan.variant not in supported_variants:
        raise ValueError(
            f"SignFi task binding '{task_binding['binding_id']}' does not support variant "
            f"{manifest_plan.variant!r}. Supported variants: {', '.join(supported_variants)}."
        )
    binding_payload = resolve_dataset_task_binding_payload(task_binding, owner="SignFi")
    materialized_target_schema = _materialized_target_schema(
        binding_payload,
        task_binding=task_binding,
        owner=f"SignFi task binding '{task_binding['binding_id']}'",
    )
    return DatasetBuildArtifact(
        dataset_id=DATASET_ID,
        variant=manifest_plan.variant,
        manifest=manifest_from_rows(
            manifest_plan.manifest_rows,
            dataset_id=DATASET_ID,
            variant=manifest_plan.variant,
        ),
        dataset_root=request.dataset_root,
        materialization={
            "split_scheme": manifest_plan.split_scheme,
            "task_binding": str(binding_payload["task_binding"]),
            "task_id": str(binding_payload["task_id"]),
            "target_schema": materialized_target_schema,
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
    source_dataset = load_signfi_dataset(resolved_path, variant=variant)
    materialization = _artifact_materialization(artifact)
    split_binding_id = materialization.get("split_scheme")
    task_id, task_kind, target_kind, target_schema = _resolve_materialized_view_contract(
        materialization
    )
    split_plan = _resolve_signfi_split_plan(
        source_dataset,
        split_scheme=(None if split_binding_id in {None, ""} else str(split_binding_id)),
    )
    label_mapping = source_dataset.get_label_mapping()
    split_indices = {
        "train": split_plan.train_indices,
        "val": split_plan.val_indices,
    }
    views_by_split: dict[str, DatasetView] = {}
    for split_name, indices in split_indices.items():
        metadata_rows, manifest_rows = artifact_split_payload(artifact, split=split_name)
        views_by_split[split_name] = build_manifest_backed_view(
            dataset=source_dataset,
            indices=indices,
            dataset_id=DATASET_ID,
            variant=variant,
            split=split_name,
            task_id=task_id,
            task_kind=task_kind,
            target_kind=target_kind,
            label_mapping=label_mapping,
            target_schema=dict(target_schema),
            metadata_rows=metadata_rows,
            manifest_rows=manifest_rows,
        )

    return (
        variant,
        label_mapping,
        task_kind,
        target_kind,
        dict(target_schema),
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
    task_id, _, _, _ = _resolve_materialized_view_contract(_artifact_materialization(artifact))
    return merge_materialized_split_views(
        dataset_id=DATASET_ID,
        variant=variant,
        split=split,
        task_id=task_id,
        task_kind=task_kind,
        target_kind=target_kind,
        label_mapping=label_mapping,
        target_schema=target_schema,
        views_by_split=views_by_split,
    )


__all__ = [
    "DATASET_ID",
    "SIGNFI_BANDWIDTH",
    "SIGNFI_CENTER_FREQ",
    "SIGNFI_NOMINAL_SAMPLE_RATE",
    "SignFiDataset",
    "build",
    "build_view",
    "materialize_views_from_artifact",
    "signfi_dataset_card",
]
