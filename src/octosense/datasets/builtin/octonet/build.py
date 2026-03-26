"""Builtin assembly entry for the OctoNet dataset definition."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from octosense.datasets.base import resolve_dataset_root
from octosense.datasets.views.dataset_view import DatasetView

from ...core.builder import DatasetBuildArtifact, DatasetBuildRequest
from ...core.materialization import (
    artifact_split_payload,
    build_manifest_backed_view,
)
from ...core.manifest import manifest_from_rows
from ...core.task_binding import canonical_task_semantics, resolve_materialized_task_identity
from .ingest import OctonetWiFiDataset, detect_octonet_wifi_node_id
from .manifest import DATASET_ID, build_octonet_manifest_plan

if TYPE_CHECKING:
    from octosense.datasets.base import DatasetLoadRequest


def _requires_label_mapping(target_kind: str) -> bool:
    return str(target_kind).strip() == "categorical_label"


def _selected_modality(request: DatasetBuildRequest) -> str | None:
    if not request.modalities:
        return None
    if len(request.modalities) != 1:
        raise NotImplementedError(
            "Current builtin dataset artifact materializes one modality per DatasetView. "
            f"Received modalities={list(request.modalities)!r}."
        )
    return str(request.modalities[0])


def _ensure_no_public_variant(request: DatasetBuildRequest) -> None:
    if request.variant in {None, ""}:
        return
    raise ValueError(
        "OctoNet does not expose a public variant selector. "
        "Select the runtime surface via modalities=['wifi'] and leave variant unset."
    )


def _build_wifi_artifact(request: DatasetBuildRequest) -> DatasetBuildArtifact:
    if request.dataset_root is None:
        raise ValueError("OctoNet wifi builder requires request.dataset_root to be resolved.")

    requested_modality = _selected_modality(request)
    if requested_modality not in {None, "wifi"}:
        raise ValueError(
            f"OctoNet wifi artifact expects modality 'wifi', got {requested_modality!r}."
        )

    selected_node_id = detect_octonet_wifi_node_id(request.dataset_root)
    dataset_instance = OctonetWiFiDataset(
        str(request.dataset_root),
        node_id=selected_node_id,
    )
    metadata_rows = dataset_instance.metadata_rows()
    manifest_plan = build_octonet_manifest_plan(
        request,
        metadata_rows=metadata_rows,
    )
    materialization = dict(manifest_plan.materialization)
    materialization["node_id"] = selected_node_id
    return DatasetBuildArtifact(
        dataset_id=DATASET_ID,
        variant=None,
        manifest=manifest_from_rows(
            manifest_plan.manifest_rows,
            dataset_id=DATASET_ID,
            variant=None,
        ),
        dataset_root=request.dataset_root,
        materialization=materialization,
    )


def _request_node_id(request: "DatasetLoadRequest") -> int | None:
    raw_value = request.options.get("node_id")
    if raw_value in {None, ""}:
        return None
    return int(raw_value)


def _resolve_artifact_task_contract(
    materialization: dict[str, object],
    *,
    owner: str,
) -> tuple[str, str, str, dict[str, object]]:
    task_id, target_schema = resolve_materialized_task_identity(
        materialization,
        owner=owner,
    )
    task_kind, target_kind = canonical_task_semantics(task_id)
    return task_id, task_kind, target_kind, target_schema


def _materialize_wifi_views_from_artifact(
    request: "DatasetLoadRequest",
    artifact: DatasetBuildArtifact,
) -> tuple[
    str | None,
    dict[str, int] | None,
    str,
    str,
    dict[str, object] | None,
    dict[str, DatasetView],
]:
    resolved_root = resolve_dataset_root(DATASET_ID, override=artifact.dataset_root or request.dataset_root)
    materialization = getattr(artifact, "materialization", None)
    if not isinstance(materialization, dict):
        raise ValueError("OctoNet wifi artifact materialization payload must be a mapping.")
    if artifact.variant not in {None, ""}:
        raise ValueError(
            "OctoNet wifi artifact materialization payload must keep public variant unset."
        )
    artifact_node_id = materialization.get("node_id")
    request_node_id = _request_node_id(request)
    if request_node_id is not None:
        resolved_node_id = request_node_id
    elif artifact_node_id not in {None, ""}:
        resolved_node_id = int(artifact_node_id)
    else:
        resolved_node_id = detect_octonet_wifi_node_id(resolved_root)
    dataset_instance = OctonetWiFiDataset(
        str(resolved_root),
        node_id=resolved_node_id,
    )
    task_id, task_kind, target_kind, target_schema = _resolve_artifact_task_contract(
        materialization,
        owner="OctoNet wifi artifact materialization payload",
    )
    label_mapping = dataset_instance.get_label_mapping() if _requires_label_mapping(target_kind) else None
    views_by_split: dict[str, DatasetView] = {}
    for split_name in ("train", "val", "test"):
        metadata_rows, manifest_rows = artifact_split_payload(artifact, split=split_name)
        if not metadata_rows:
            continue
        source_indices = [int(row["sample_index"]) for row in metadata_rows]
        users = sorted(
            {
                int(row["subject_id"])
                for row in metadata_rows
                if row.get("subject_id") not in {None, ""}
            }
        )
        views_by_split[split_name] = build_manifest_backed_view(
            dataset=dataset_instance,
            indices=source_indices,
            dataset_id=DATASET_ID,
            variant=None,
            split=split_name,
            task_id=task_id,
            label_mapping=label_mapping,
            target_schema=dict(target_schema),
            metadata_rows=metadata_rows,
            manifest_rows=manifest_rows,
            metadata_kwargs={"users": users} if users else None,
        )
    if not views_by_split:
        raise ValueError("OctoNet wifi artifact materialization produced no DatasetView splits.")
    return (
        None,
        label_mapping,
        task_kind,
        target_kind,
        dict(target_schema),
        views_by_split,
    )


def materialize_views_from_artifact(
    request: "DatasetLoadRequest",
    artifact: DatasetBuildArtifact,
) -> tuple[
    str | None,
    dict[str, int] | None,
    str,
    str,
    dict[str, object] | None,
    dict[str, DatasetView],
]:
    if artifact.variant not in {None, ""}:
        raise ValueError(f"Unsupported OctoNet artifact variant: {artifact.variant!r}")
    return _materialize_wifi_views_from_artifact(request, artifact)


def build(request: DatasetBuildRequest) -> DatasetBuildArtifact:
    _ensure_no_public_variant(request)
    return _build_wifi_artifact(request)


__all__ = [
    "DATASET_ID",
    "build",
    "materialize_views_from_artifact",
]
