"""Builtin owner entry for the XRF55 dataset definition."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...base import resolve_dataset_root
from ...catalog import get_dataset_binding_payload, list_dataset_binding_ids
from ...core.builder import DatasetBuildArtifact, DatasetBuildRequest
from ...core.materialization import (
    artifact_split_payload,
    build_manifest_backed_view,
)
from ...core.manifest import manifest_from_rows
from ...core.task_binding import (
    canonical_task_semantics,
    resolve_dataset_task_binding_payload,
    resolve_materialized_task_identity,
)
from .ingest import (
    XRF55Dataset,
    XRF55_WIFI_BANDWIDTH,
    XRF55_WIFI_CENTER_FREQ,
)
from .manifest import plan_xrf55_manifest_build

if TYPE_CHECKING:
    from octosense.datasets.base import DatasetLoadRequest
    from octosense.datasets.views.dataset_view import DatasetView

DATASET_ID = "xrf55"


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


def _resolve_variant_payload(
    variant: str | None,
    *,
    modality: str | None,
) -> tuple[str, str, dict[str, object]]:
    candidate = "" if variant in {None, ""} else str(variant).strip()
    available = list_dataset_binding_ids(DATASET_ID, binding_kind="config")
    if not candidate:
        supported = ", ".join(available) or "<none>"
        raise ValueError(
            "XRF55 requires an explicit config; "
            "implicit default/singleton fallback is not supported. "
            f"Supported bindings: {supported}."
        )
    matches: list[tuple[str, dict[str, object]]] = []
    for binding_id in available:
        payload = get_dataset_binding_payload(
            DATASET_ID,
            binding_kind="config",
            binding_id=binding_id,
        )
        payload_variant = str(payload.get("variant") or "").strip()
        payload_modality = str(payload.get("modality") or "").strip()
        if payload_variant != candidate:
            continue
        if modality is not None and payload_modality != modality:
            continue
        matches.append((binding_id, payload))
    if not matches:
        supported = ", ".join(available) or "<none>"
        if modality is None:
            raise ValueError(
                f"XRF55 variant must resolve from canonical variant '{candidate}' and an explicit modality. "
                f"Supported config bindings: {supported}."
            )
        raise ValueError(
            f"XRF55 variant '{candidate}' does not support modality {modality!r}. "
            f"Supported config bindings: {supported}."
        )
    if len(matches) > 1:
        matching_modalities = ", ".join(
            sorted({str(payload.get('modality') or '').strip() for _, payload in matches})
        )
        raise ValueError(
            f"XRF55 variant '{candidate}' is shared by multiple modalities ({matching_modalities}). "
            "Pass exactly one modality so the family can resolve the config binding deterministically."
        )
    binding_id, payload = matches[0]
    variant_key = str(payload.get("variant_key") or "").strip()
    if variant_key != binding_id:
        raise ValueError(
            f"XRF55 config binding '{binding_id}' declares mismatched variant_key {variant_key!r}."
        )
    return candidate, binding_id, payload


def _materialization_payload(artifact: DatasetBuildArtifact) -> dict[str, object]:
    payload = getattr(artifact, "materialization", None)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("XRF55 artifact materialization payload must be a mapping.")
    return payload


def _dataset_target_schema(
    binding_payload: dict[str, object],
    *,
    owner: str,
) -> dict[str, object]:
    target_schema = binding_payload.get("dataset_target_schema")
    if not isinstance(target_schema, dict):
        raise ValueError(f"{owner} is missing dataset_target_schema.")
    return dict(target_schema)


def _resolve_task_binding_contract(binding_id: str | None) -> dict[str, Any]:
    candidate = "" if binding_id in {None, ""} else str(binding_id).strip()
    available = list_dataset_binding_ids(DATASET_ID, binding_kind="task_binding")
    if not candidate:
        supported = ", ".join(available) or "<none>"
        raise ValueError(
            "XRF55 requires an explicit task_binding; "
            "implicit default/singleton fallback is not supported. "
            f"Supported bindings: {supported}."
        )
    if candidate not in available:
        supported = ", ".join(available) or "<none>"
        raise ValueError(
            f"XRF55 task_binding must be one of: {supported}. Received {candidate!r}."
        )
    return get_dataset_binding_payload(
        DATASET_ID,
        binding_kind="task_binding",
        binding_id=candidate,
    )


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


def build(request: DatasetBuildRequest) -> DatasetBuildArtifact:
    if request.dataset_root is None:
        raise ValueError("XRF55 builder requires request.dataset_root to be resolved.")

    requested_modality = _selected_modality(request)
    resolved_variant, config_binding_id, variant_payload = _resolve_variant_payload(
        request.variant,
        modality=requested_modality,
    )
    resolved_modality = str(variant_payload.get("modality", "")).strip()
    if not resolved_modality:
        raise ValueError(
            f"Dataset '{DATASET_ID}' variant '{resolved_variant}' is missing a modality declaration."
        )
    if requested_modality is not None and requested_modality != resolved_modality:
        raise ValueError(
            f"Dataset '{DATASET_ID}' runtime request variant '{resolved_variant}' resolves to modality "
            f"{resolved_modality!r}, got request modality {requested_modality!r}."
        )

    task_binding = _resolve_task_binding_contract(request.task_binding)
    binding_payload = resolve_dataset_task_binding_payload(task_binding, owner="XRF55")
    dataset_target_schema = _dataset_target_schema(
        binding_payload,
        owner="XRF55 task binding compiler payload",
    )
    manifest_plan = plan_xrf55_manifest_build(
        request,
        variant=config_binding_id,
        modality=resolved_modality,
    )
    manifest_rows = manifest_plan.build_rows(
        task_binding=str(binding_payload["task_binding"]),
    )

    return DatasetBuildArtifact(
        dataset_id=DATASET_ID,
        variant=resolved_variant,
        manifest=manifest_from_rows(
            manifest_rows,
            dataset_id=DATASET_ID,
            variant=resolved_variant,
        ),
        dataset_root=request.dataset_root,
        materialization={
            "config_binding_id": config_binding_id,
            "modality": resolved_modality,
            "task_binding": str(binding_payload["task_binding"]),
            "task_id": str(binding_payload["task_id"]),
            "target_schema": dataset_target_schema,
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
    materialization = _materialization_payload(artifact)
    resolved_variant = str(getattr(artifact, "variant", "")).strip()
    if not resolved_variant:
        raise ValueError("XRF55 artifact is missing a canonical variant.")
    config_binding_id = str(materialization.get("config_binding_id", "")).strip()
    if not config_binding_id:
        raise ValueError("XRF55 artifact materialization payload is missing config_binding_id.")
    resolved_modality = str(materialization.get("modality", "")).strip()
    if not resolved_modality:
        raise ValueError("XRF55 artifact materialization payload is missing modality.")
    task_id, task_kind, target_kind, target_schema = _resolve_artifact_task_contract(
        materialization,
        owner="XRF55 artifact materialization payload",
    )
    resolved_root = resolve_dataset_root(DATASET_ID, override=request.dataset_root)
    dataset_instance = XRF55Dataset(
        resolved_root,
        modality=resolved_modality,
        variant=config_binding_id,
    )
    label_mapping = dataset_instance.get_label_mapping() if _requires_label_mapping(target_kind) else None

    views_by_split: dict[str, DatasetView] = {}
    for split_name in ("train", "val", "test"):
        metadata_rows, manifest_rows = artifact_split_payload(artifact, split=split_name)
        if not metadata_rows:
            continue
        source_indices = [int(row["sample_index"]) for row in metadata_rows]
        views_by_split[split_name] = build_manifest_backed_view(
            dataset=dataset_instance,
            indices=source_indices,
            dataset_id=DATASET_ID,
            variant=resolved_variant,
            split=split_name,
            task_id=task_id,
            label_mapping=label_mapping,
            target_schema=dict(target_schema),
            metadata_rows=metadata_rows,
            manifest_rows=manifest_rows,
        )

    if not views_by_split:
        raise ValueError("XRF55 artifact materialization produced no DatasetView splits.")

    return (
        resolved_variant,
        label_mapping,
        task_kind,
        target_kind,
        dict(target_schema),
        views_by_split,
    )


__all__ = [
    "DATASET_ID",
    "XRF55Dataset",
    "build",
    "materialize_views_from_artifact",
]
