"""Builtin assembly entry for the Widar3 dataset definition."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from octosense.datasets.base import resolve_dataset_root
from octosense.datasets.catalog import get_dataset_binding_payload, list_dataset_binding_ids
from octosense.datasets.core.builder import stratified_train_val_indices

from ...core.builder import DatasetBuildArtifact, DatasetBuildRequest
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
from .ingest import Widar3Dataset
from .manifest import DATASET_ID, build_widar3_manifest_plan

if TYPE_CHECKING:
    from octosense.datasets.base import DatasetLoadRequest
    from octosense.datasets.views.dataset_view import DatasetView


def _resolve_widar3_binding(
    *,
    binding_kind: str,
    binding_id: str | None,
) -> dict[str, Any]:
    candidate = "" if binding_id in {None, ""} else str(binding_id).strip()
    available = list_dataset_binding_ids(DATASET_ID, binding_kind=binding_kind)
    if not candidate:
        supported = ", ".join(available) or "<none>"
        raise ValueError(
            f"Widar3 requires an explicit {binding_kind}; "
            "implicit default/singleton fallback is not supported. "
            f"Supported bindings: {supported}."
        )
    if candidate not in available:
        supported = ", ".join(available) or "<none>"
        raise ValueError(
            f"Widar3 {binding_kind} must be one of: {supported}. Received {candidate!r}."
        )
    return get_dataset_binding_payload(
        DATASET_ID,
        binding_kind=binding_kind,
        binding_id=candidate,
    )


def _resolve_config_contract(binding_id: str | None = None) -> dict[str, Any]:
    resolved = _resolve_widar3_binding(binding_kind="config", binding_id=binding_id)
    variant = resolved.get("variant")
    if variant in {None, ""}:
        raise ValueError("Widar3 config binding must declare variant.")
    if str(variant) != str(resolved["binding_id"]):
        raise ValueError(
            "Widar3 config bindings must use canonical ids. "
            f"binding_id={resolved['binding_id']!r}, variant={variant!r}"
        )
    return resolved


def _resolve_task_binding_contract(binding_id: str | None = None) -> dict[str, Any]:
    return _resolve_widar3_binding(binding_kind="task_binding", binding_id=binding_id)


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


def _coerce_int_list(payload: Any) -> list[int] | None:
    if payload is None or payload == "":
        return None
    if not isinstance(payload, list):
        raise TypeError(f"Expected integer list payload, got {type(payload)!r}")
    return [int(value) for value in payload]


def _coerce_gesture_list(payload: Any) -> list[str] | None:
    if payload is None or payload == "":
        return None
    if not isinstance(payload, list):
        raise TypeError(f"Expected gesture list payload, got {type(payload)!r}")
    return [str(value) for value in payload]


def _artifact_materialization(artifact: DatasetBuildArtifact) -> dict[str, Any]:
    payload = getattr(artifact, "materialization", None)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("Widar3 artifact materialization payload must be a mapping.")
    return dict(payload)


def _resolve_materialized_view_contract(
    materialization: dict[str, Any],
) -> tuple[str, str, str, dict[str, object]]:
    task_id, target_schema = resolve_materialized_task_identity(
        materialization,
        owner="Widar3 artifact materialization payload",
    )
    task_kind, target_kind = canonical_task_semantics(task_id)
    return (
        task_id,
        task_kind,
        target_kind,
        dict(target_schema),
    )


def _required_materialization_value(
    payload: dict[str, Any],
    *,
    key: str,
    owner: str,
) -> Any:
    value = payload.get(key)
    if value in {None, ""}:
        raise ValueError(f"{owner} must define materialization field {key!r}.")
    return value


def build(request: DatasetBuildRequest) -> DatasetBuildArtifact:
    if request.dataset_root is None:
        raise ValueError("Widar3 builder requires request.dataset_root to be resolved.")

    config = _resolve_config_contract(request.variant)
    variant = str(config["binding_id"])

    task_binding = _resolve_task_binding_contract(request.task_binding)
    binding_payload = resolve_dataset_task_binding_payload(task_binding, owner="Widar3")
    materialized_target_schema = _materialized_target_schema(
        binding_payload,
        task_binding=task_binding,
        owner=f"Widar3 task binding '{task_binding['binding_id']}'",
    )
    supported_variants = task_binding.get("supported_variants", [])
    if isinstance(supported_variants, list) and supported_variants:
        if variant not in {str(value) for value in supported_variants}:
            raise ValueError(
                f"Widar3 task binding '{task_binding['binding_id']}' does not support variant {variant!r}."
            )

    reader_payload = config.get("reader", {})
    if not isinstance(reader_payload, dict):
        raise TypeError("Widar3 config binding must define mapping field 'reader'.")
    channel = int(reader_payload.get("channel", 0))
    if channel <= 0:
        raise ValueError("Widar3 config binding must define a positive reader.channel.")

    gestures = _coerce_gesture_list(config.get("gestures"))
    manifest_plan = build_widar3_manifest_plan(split_scheme_id=request.split_scheme)
    split_variants = get_dataset_binding_payload(
        DATASET_ID,
        binding_kind="split_scheme",
        binding_id=manifest_plan.split_scheme_id,
    ).get("supported_variants", [])
    if isinstance(split_variants, list) and split_variants:
        if variant not in {str(value) for value in split_variants}:
            raise ValueError(
                f"Widar3 split scheme '{manifest_plan.split_scheme_id}' does not support variant {variant!r}."
            )

    train_filters = manifest_plan.dataset_filters("train")
    test_filters = manifest_plan.dataset_filters("test")

    train_dataset = Widar3Dataset(
        dataset_path=str(request.dataset_root),
        variant=variant,
        users=train_filters["users"],
        gestures=gestures,
        rooms=train_filters["rooms"],
        rx_ids=train_filters["rx_ids"],
        channel=channel,
    )
    test_dataset = Widar3Dataset(
        dataset_path=str(request.dataset_root),
        variant=variant,
        users=test_filters["users"],
        gestures=gestures,
        rooms=test_filters["rooms"],
        rx_ids=test_filters["rx_ids"],
        channel=channel,
    )

    train_source_rows = train_dataset.metadata_rows()
    test_source_rows = test_dataset.metadata_rows()
    rows_by_split = manifest_plan.materialize_rows(
        train_source_rows=train_source_rows,
        train_labels=train_dataset.get_labels(),
        test_source_rows=test_source_rows,
        task_binding_id=str(binding_payload["task_binding"]),
    )
    return DatasetBuildArtifact(
        dataset_id=DATASET_ID,
        variant=variant,
        manifest=manifest_from_rows(
            [
                *rows_by_split["train"],
                *rows_by_split["val"],
                *rows_by_split["test"],
            ],
            dataset_id=DATASET_ID,
            variant=variant,
        ),
        dataset_root=request.dataset_root,
        materialization={
            "channel": channel,
            "gestures": list(gestures or []),
            **manifest_plan.materialization_payload(),
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
    materialization = _artifact_materialization(artifact)
    variant = str(getattr(artifact, "variant", "")).strip()
    if not variant:
        raise ValueError("Widar3 artifact is missing a canonical variant.")
    channel = int(materialization.get("channel", 0))
    if channel <= 0:
        raise ValueError("Widar3 artifact materialization payload is missing channel.")
    gestures_payload = materialization.get("gestures")
    gestures = [str(value) for value in gestures_payload] if isinstance(gestures_payload, list) else None
    split_partitions = materialization.get("split_partitions")
    if not isinstance(split_partitions, dict):
        raise ValueError("Widar3 artifact materialization payload is missing split_partitions.")
    train_partition = split_partitions.get("train")
    test_partition = split_partitions.get("test")
    if not isinstance(train_partition, dict) or not isinstance(test_partition, dict):
        raise ValueError(
            "Widar3 artifact materialization payload must define train/test split partitions."
        )
    train_ratio = float(
        _required_materialization_value(
            materialization,
            key="train_ratio",
            owner="Widar3 artifact materialization payload",
        )
    )
    seed = int(
        _required_materialization_value(
            materialization,
            key="seed",
            owner="Widar3 artifact materialization payload",
        )
    )

    train_dataset = Widar3Dataset(
        dataset_path=str(resolved_path),
        variant=variant,
        users=_coerce_int_list(train_partition.get("users")),
        gestures=gestures,
        rooms=_coerce_int_list(train_partition.get("rooms")),
        rx_ids=_coerce_int_list(train_partition.get("rx_ids")),
        channel=channel,
        return_radiotensor=True,
    )
    test_dataset = Widar3Dataset(
        dataset_path=str(resolved_path),
        variant=variant,
        users=_coerce_int_list(test_partition.get("users")),
        gestures=gestures,
        rooms=_coerce_int_list(test_partition.get("rooms")),
        rx_ids=_coerce_int_list(test_partition.get("rx_ids")),
        channel=channel,
        return_radiotensor=True,
    )
    train_indices, val_indices = stratified_train_val_indices(
        train_dataset.get_labels(),
        train_ratio=train_ratio,
        seed=seed,
    )
    label_mapping = train_dataset.get_label_mapping()
    task_id, task_kind, target_kind, target_schema = _resolve_materialized_view_contract(
        materialization
    )
    split_datasets = {
        "train": (train_dataset, train_indices),
        "val": (train_dataset, val_indices),
        "test": (test_dataset, list(range(len(test_dataset)))),
    }
    views_by_split: dict[str, DatasetView] = {}
    for split_name, (source_dataset, indices) in split_datasets.items():
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
    "build",
    "build_view",
    "materialize_views_from_artifact",
]
