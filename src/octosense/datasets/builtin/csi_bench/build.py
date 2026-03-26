"""Builtin assembly entry for the CSI-Bench dataset definition."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from octosense.datasets.base import resolve_dataset_root
from octosense.datasets.catalog import get_dataset_binding_payload, list_dataset_binding_ids
from octosense.datasets.core.materialization import (
    artifact_split_payload,
    build_manifest_backed_view,
)
from octosense.datasets.core.task_binding import (
    canonical_task_semantics,
    resolve_dataset_task_binding_payload,
    resolve_materialized_task_identity,
)

from ...core.builder import DatasetBuildArtifact, DatasetBuildRequest
from ...core.manifest import manifest_from_rows
from .ingest import CSIBenchDataset
from .manifest import build_csi_bench_manifest_rows, plan_csi_bench_manifest

DATASET_ID = "csi_bench"

if TYPE_CHECKING:
    from octosense.datasets.base import DatasetLoadRequest
    from octosense.datasets.views.dataset_view import DatasetView


def _requires_label_mapping(target_kind: str) -> bool:
    return str(target_kind).strip() == "categorical_label"


def _resolve_binding_contract(
    *,
    binding_kind: str,
    binding_id: str | None,
) -> dict[str, Any]:
    candidate = "" if binding_id in {None, ""} else str(binding_id).strip()
    available = list_dataset_binding_ids(DATASET_ID, binding_kind=binding_kind)
    if not candidate:
        supported = ", ".join(available) or "<none>"
        raise ValueError(
            f"CSI-Bench requires an explicit {binding_kind}; "
            "implicit default/singleton fallback is not supported. "
            f"Supported bindings: {supported}."
        )
    if candidate not in available:
        supported = ", ".join(available) or "<none>"
        raise ValueError(
            f"CSI-Bench {binding_kind} must be one of: {supported}. Received {candidate!r}."
        )
    return get_dataset_binding_payload(
        DATASET_ID,
        binding_kind=binding_kind,
        binding_id=candidate,
    )


def _resolve_config_contract(binding_id: str | None = None) -> dict[str, Any]:
    resolved = _resolve_binding_contract(
        binding_kind="config",
        binding_id=binding_id,
    )
    variant = resolved.get("variant")
    if variant in {None, ""}:
        raise ValueError("CSI-Bench config binding must declare variant.")
    if str(variant) != str(resolved["binding_id"]):
        raise ValueError(
            "CSI-Bench config bindings must use canonical ids. "
            f"binding_id={resolved['binding_id']!r}, variant={variant!r}"
        )
    return resolved


def _resolve_task_binding_contract(binding_id: str | None = None) -> dict[str, Any]:
    return _resolve_binding_contract(
        binding_kind="task_binding",
        binding_id=binding_id,
    )


def _artifact_materialization(artifact: DatasetBuildArtifact) -> dict[str, object]:
    payload = getattr(artifact, "materialization", None)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("CSI-Bench artifact materialization payload must be a mapping.")
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
        raise ValueError("CSI-Bench builder requires request.dataset_root to be resolved.")

    config = _resolve_config_contract(request.variant)
    variant = str(config["binding_id"])

    task_binding = _resolve_task_binding_contract(request.task_binding)
    binding_payload = resolve_dataset_task_binding_payload(
        task_binding,
        owner="CSI-Bench",
    )
    dataset_target_schema = _dataset_target_schema(
        binding_payload,
        owner="CSI-Bench task binding compiler payload",
    )
    task_name = task_binding.get("task_name")
    if task_name in {None, ""}:
        raise ValueError("CSI-Bench task binding must define task_name.")

    manifest_plan = plan_csi_bench_manifest(
        request,
        variant=variant,
        task_name=str(task_name),
        task_binding=str(binding_payload["task_binding"]),
    )
    manifest_rows = build_csi_bench_manifest_rows(request, plan=manifest_plan)
    return DatasetBuildArtifact(
        dataset_id=DATASET_ID,
        variant=variant,
        manifest=manifest_from_rows(
            manifest_rows,
            dataset_id=DATASET_ID,
            variant=variant,
        ),
        dataset_root=request.dataset_root,
        materialization={
            "split_scheme": manifest_plan.split_scheme_id,
            "task_binding": str(binding_payload["task_binding"]),
            "task_id": str(binding_payload["task_id"]),
            "task_name": str(task_name),
            "target_schema": dataset_target_schema,
            "resolved_split_ids": dict(manifest_plan.split_ids_by_name),
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
        raise ValueError("CSI-Bench artifact is missing a canonical variant.")
    task_name = str(materialization.get("task_name", "")).strip()
    if not task_name:
        raise ValueError("CSI-Bench artifact materialization payload is missing task_name.")
    resolved_split_ids = materialization.get("resolved_split_ids")
    if not isinstance(resolved_split_ids, dict):
        raise ValueError(
            "CSI-Bench artifact materialization payload is missing resolved_split_ids."
        )
    split_ids_by_name = {
        split_name: str(resolved_split_ids[split_name])
        for split_name in ("train", "val", "test")
    }
    split_datasets = {
        split_name: CSIBenchDataset(
            resolved_path,
            variant=variant,
            task_name=task_name,
            split_name=split_ids_by_name[split_name],
        )
        for split_name in ("train", "val", "test")
    }
    task_id, task_kind, target_kind, target_schema = _resolve_artifact_task_contract(
        materialization,
        owner="CSI-Bench artifact materialization payload",
    )
    label_mapping = (
        split_datasets["train"].get_label_mapping() if _requires_label_mapping(target_kind) else None
    )
    views_by_split: dict[str, DatasetView] = {}
    for split_name, source_dataset in split_datasets.items():
        metadata_rows, manifest_rows = artifact_split_payload(artifact, split=split_name)
        views_by_split[split_name] = build_manifest_backed_view(
            dataset=source_dataset,
            indices=list(range(len(source_dataset))),
            dataset_id=DATASET_ID,
            variant=variant,
            split=split_name,
            task_id=task_id,
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


__all__ = [
    "DATASET_ID",
    "build",
    "materialize_views_from_artifact",
]
