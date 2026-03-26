"""Builtin owner entry for the MM-Fi dataset definition."""

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
    MMFI_MMWAVE_CENTER_FREQ,
    MMFI_PROTOCOL_ACTIONS,
    MMFI_WIFI_BANDWIDTH,
    MMFI_WIFI_CENTER_FREQ,
    MMFiDataset,
)
from .manifest import build_mmfi_manifest_rows, plan_mmfi_manifest

if TYPE_CHECKING:
    from octosense.datasets.base import DatasetLoadRequest
    from octosense.datasets.views.dataset_view import DatasetView

DATASET_ID = "mmfi"


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


def _subject_codes_to_user_ids(subjects: list[str]) -> list[int]:
    user_ids: list[int] = []
    for subject in subjects:
        normalized = str(subject).strip()
        if normalized.lower().startswith("s"):
            normalized = normalized[1:]
        user_ids.append(int(normalized))
    return sorted(user_ids)


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
            "MM-Fi requires an explicit config; "
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
                f"MM-Fi variant must resolve from canonical variant '{candidate}' and an explicit modality. "
                f"Supported config bindings: {supported}."
            )
        raise ValueError(
            f"MM-Fi variant '{candidate}' does not support modality {modality!r}. "
            f"Supported config bindings: {supported}."
        )
    if len(matches) > 1:
        matching_modalities = ", ".join(
            sorted({str(payload.get('modality') or '').strip() for _, payload in matches})
        )
        raise ValueError(
            f"MM-Fi variant '{candidate}' is shared by multiple modalities ({matching_modalities}). "
            "Pass exactly one modality so the family can resolve the config binding deterministically."
        )
    binding_id, payload = matches[0]
    variant_key = str(payload.get("variant_key") or "").strip()
    if variant_key != binding_id:
        raise ValueError(
            f"MM-Fi config binding '{binding_id}' declares mismatched variant_key {variant_key!r}."
        )
    return candidate, binding_id, payload


def _materialization_payload(artifact: DatasetBuildArtifact) -> dict[str, object]:
    payload = getattr(artifact, "materialization", None)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("MM-Fi artifact materialization payload must be a mapping.")
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
            "MM-Fi requires an explicit task_binding; "
            "implicit default/singleton fallback is not supported. "
            f"Supported bindings: {supported}."
        )
    if candidate not in available:
        supported = ", ".join(available) or "<none>"
        raise ValueError(
            f"MM-Fi task_binding must be one of: {supported}. Received {candidate!r}."
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


def _resolve_request_max_points(request: "DatasetLoadRequest") -> int | None:
    payload = getattr(request, "options", None)
    if not isinstance(payload, dict):
        return None
    max_points = payload.get("max_points")
    if max_points is None:
        return None
    resolved = int(max_points)
    if resolved <= 0:
        raise ValueError("MM-Fi dataset options.max_points must be a positive integer.")
    return resolved


def build(request: DatasetBuildRequest) -> DatasetBuildArtifact:
    if request.dataset_root is None:
        raise ValueError("MM-Fi builder requires request.dataset_root to be resolved.")

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
    binding_payload = resolve_dataset_task_binding_payload(task_binding, owner="MM-Fi")
    dataset_target_schema = _dataset_target_schema(
        binding_payload,
        owner="MM-Fi task binding compiler payload",
    )
    manifest_plan = plan_mmfi_manifest(
        request,
        variant=config_binding_id,
        modality=resolved_modality,
        task_binding=str(binding_payload["task_binding"]),
    )
    manifest_rows, split_subjects = build_mmfi_manifest_rows(request, plan=manifest_plan)
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
            "split_subjects": split_subjects,
            "split_metadata_kwargs": {
                "train": {"users": _subject_codes_to_user_ids(split_subjects["train"])},
                "val": {"users": _subject_codes_to_user_ids(split_subjects["val"])},
            },
            "task_binding": str(binding_payload["task_binding"]),
            "task_id": str(binding_payload["task_id"]),
            "target_schema": dataset_target_schema,
            "split_scheme": manifest_plan.split_scheme_id,
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
    split_subjects = materialization.get("split_subjects")
    if split_subjects is None:
        split_subjects = {}
    if not isinstance(split_subjects, dict):
        raise ValueError("MM-Fi artifact materialization split_subjects must be a mapping.")
    split_metadata_kwargs = materialization.get("split_metadata_kwargs")
    if split_metadata_kwargs is None:
        split_metadata_kwargs = {}
    if not isinstance(split_metadata_kwargs, dict):
        raise ValueError("MM-Fi artifact materialization split_metadata_kwargs must be a mapping.")
    resolved_variant = str(getattr(artifact, "variant", "")).strip()
    if not resolved_variant:
        raise ValueError("MM-Fi artifact is missing a canonical variant.")
    config_binding_id = str(materialization.get("config_binding_id", "")).strip()
    if not config_binding_id:
        raise ValueError("MM-Fi artifact materialization payload is missing config_binding_id.")
    resolved_modality = str(materialization.get("modality", "")).strip()
    if not resolved_modality:
        raise ValueError("MM-Fi artifact materialization payload is missing modality.")
    task_id, task_kind, target_kind, target_schema = _resolve_artifact_task_contract(
        materialization,
        owner="MM-Fi artifact materialization payload",
    )
    resolved_root = resolve_dataset_root(DATASET_ID, override=request.dataset_root)
    max_points = _resolve_request_max_points(request)

    views_by_split: dict[str, DatasetView] = {}
    label_mapping: dict[str, int] | None = None
    for split_name in ("train", "val", "test"):
        metadata_rows, manifest_rows = artifact_split_payload(artifact, split=split_name)
        if not metadata_rows:
            continue
        subjects_payload = split_subjects.get(split_name)
        subjects = None
        if isinstance(subjects_payload, list):
            subjects = [str(subject) for subject in subjects_payload]
        source_dataset = MMFiDataset(
            resolved_root,
            modality=resolved_modality,
            variant=config_binding_id,
            subjects=subjects,
            max_points=max_points,
        )
        if label_mapping is None and _requires_label_mapping(target_kind):
            label_mapping = source_dataset.get_label_mapping()
        views_by_split[split_name] = build_manifest_backed_view(
            dataset=source_dataset,
            indices=list(range(len(source_dataset))),
            dataset_id=DATASET_ID,
            variant=resolved_variant,
            split=split_name,
            task_id=task_id,
            label_mapping=label_mapping,
            target_schema=dict(target_schema),
            metadata_rows=metadata_rows,
            manifest_rows=manifest_rows,
            metadata_kwargs=(
                dict(split_metadata_kwargs[split_name])
                if isinstance(split_metadata_kwargs.get(split_name), dict)
                else None
            ),
        )

    if not views_by_split:
        raise ValueError("MM-Fi artifact materialization produced no DatasetView splits.")
    if _requires_label_mapping(target_kind) and label_mapping is None:
        label_mapping = {}

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
    "MMFiDataset",
    "build",
    "materialize_views_from_artifact",
]
