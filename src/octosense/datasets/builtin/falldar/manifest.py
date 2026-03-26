"""Manifest generation for the FallDar builtin dataset."""

from __future__ import annotations

from dataclasses import dataclass

from octosense.datasets.catalog import get_dataset_binding_payload, list_dataset_binding_ids
from octosense.datasets.core.builder import (
    DatasetBuildRequest,
    annotate_split_rows,
    stratified_train_val_indices,
)

from .ingest import FallDarDataset, load_falldar_dataset

DATASET_ID = "falldar"


@dataclass(frozen=True)
class FallDarSplitPlan:
    split_scheme: str
    train_indices: list[int]
    val_indices: list[int]


@dataclass(frozen=True)
class FallDarManifestPlan:
    variant: str
    split_scheme: str
    manifest_rows: list[dict[str, object]]


def _falldar_split_contract(split_scheme: str | None) -> dict[str, object]:
    candidate = "" if split_scheme in {None, ""} else str(split_scheme).strip()
    available = list_dataset_binding_ids(DATASET_ID, binding_kind="split_scheme")
    if not candidate:
        supported = ", ".join(available) or "<none>"
        raise ValueError(
            "FallDar requires an explicit split_scheme; "
            "implicit default/singleton fallback is not supported. "
            f"Supported bindings: {supported}."
        )
    if candidate not in available:
        supported = ", ".join(available) or "<none>"
        raise ValueError(
            f"FallDar split_scheme must be one of: {supported}. Received {candidate!r}."
        )
    payload = get_dataset_binding_payload(
        DATASET_ID,
        binding_kind="split_scheme",
        binding_id=candidate,
    )
    resolved_binding_id = str(payload["binding_id"])
    if str(payload.get("name") or "").strip() != resolved_binding_id:
        raise ValueError(
            f"FallDar split binding '{resolved_binding_id}' must declare name={resolved_binding_id!r}."
        )
    if str(payload.get("strategy") or "").strip() != "stratified_label":
        raise ValueError(
            f"FallDar split binding '{resolved_binding_id}' must declare strategy='stratified_label'."
        )
    resolved = dict(payload)
    resolved["train_ratio"] = float(payload["train_ratio"])
    resolved["seed"] = int(payload["seed"])
    return resolved


def _resolve_falldar_split_plan(
    dataset_instance: FallDarDataset,
    *,
    split_scheme: str | None,
) -> FallDarSplitPlan:
    split_contract = _falldar_split_contract(split_scheme)
    train_indices, val_indices = stratified_train_val_indices(
        dataset_instance.get_labels(),
        train_ratio=float(split_contract["train_ratio"]),
        seed=int(split_contract["seed"]),
    )
    return FallDarSplitPlan(
        split_scheme=str(split_contract["binding_id"]),
        train_indices=train_indices,
        val_indices=val_indices,
    )


def build_falldar_manifest_rows(
    request: DatasetBuildRequest,
) -> tuple[str, list[dict[str, object]]]:
    manifest_plan = build_falldar_manifest_plan(request)
    return manifest_plan.variant, manifest_plan.manifest_rows


def build_falldar_manifest_plan(request: DatasetBuildRequest) -> FallDarManifestPlan:
    if request.dataset_root is None:
        raise ValueError("FallDar builder requires request.dataset_root to be resolved.")

    dataset_instance = load_falldar_dataset(
        request.dataset_root,
        variant=request.variant,
    )
    split_plan = _resolve_falldar_split_plan(
        dataset_instance,
        split_scheme=request.split_scheme,
    )
    metadata_rows = []
    for index, row in enumerate(dataset_instance.metadata_rows()):
        metadata_rows.append(
            {
                "sample_id": dataset_instance.get_sample_id(index),
                **dict(row),
            }
        )
    task_binding = (
        str(request.task_binding)
        if request.task_binding not in {None, ""}
        else None
    )
    manifest_rows = (
        annotate_split_rows(
            [dict(metadata_rows[index]) for index in split_plan.train_indices],
            split="train",
            split_scheme=split_plan.split_scheme,
            task_binding=task_binding,
        )
        + annotate_split_rows(
            [dict(metadata_rows[index]) for index in split_plan.val_indices],
            split="val",
            split_scheme=split_plan.split_scheme,
            task_binding=task_binding,
        )
    )
    return FallDarManifestPlan(
        variant=dataset_instance.variant,
        split_scheme=split_plan.split_scheme,
        manifest_rows=manifest_rows,
    )


__all__ = [
    "DATASET_ID",
    "FallDarManifestPlan",
    "build_falldar_manifest_plan",
    "build_falldar_manifest_rows",
]
