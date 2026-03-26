"""Manifest generation for the SignFi builtin dataset."""

from __future__ import annotations

from dataclasses import dataclass

from octosense.datasets.catalog import get_dataset_binding_payload, list_dataset_binding_ids
from octosense.datasets.core.builder import (
    DatasetBuildRequest,
    annotate_split_rows,
    stratified_train_val_indices,
)

from .ingest import SignFiDataset, load_signfi_dataset

DATASET_ID = "signfi"


@dataclass(frozen=True)
class SignFiSplitPlan:
    split_scheme: str
    train_indices: list[int]
    val_indices: list[int]


@dataclass(frozen=True)
class SignFiManifestPlan:
    variant: str
    split_scheme: str
    manifest_rows: list[dict[str, object]]


def _resolve_signfi_split_contract(split_scheme: str | None) -> dict[str, object]:
    candidate = "" if split_scheme in {None, ""} else str(split_scheme).strip()
    available = list_dataset_binding_ids(DATASET_ID, binding_kind="split_scheme")
    if not candidate:
        supported = ", ".join(available) or "<none>"
        raise ValueError(
            "SignFi requires an explicit split_scheme; "
            "implicit default/singleton fallback is not supported. "
            f"Supported bindings: {supported}."
        )
    if candidate not in available:
        supported = ", ".join(available) or "<none>"
        raise ValueError(
            f"SignFi split_scheme must be one of: {supported}. Received {candidate!r}."
        )
    payload = get_dataset_binding_payload(
        DATASET_ID,
        binding_kind="split_scheme",
        binding_id=candidate,
    )
    if str(payload.get("name") or "").strip() != str(payload["binding_id"]):
        raise ValueError(
            "SignFi split binding "
            f"'{payload['binding_id']}' must declare name={payload['binding_id']!r}."
        )
    if str(payload.get("strategy") or "").strip() != "stratified_label":
        raise ValueError(
            "SignFi split binding "
            f"'{payload['binding_id']}' must declare strategy='stratified_label'."
        )
    payload["train_ratio"] = float(payload["train_ratio"])
    payload["seed"] = int(payload["seed"])
    return payload


def _signfi_split_indices(
    dataset_instance: SignFiDataset,
    *,
    split_scheme: str | None,
) -> tuple[str, list[int], list[int]]:
    split_plan = _resolve_signfi_split_plan(
        dataset_instance,
        split_scheme=split_scheme,
    )
    return split_plan.split_scheme, split_plan.train_indices, split_plan.val_indices


def _resolve_signfi_split_plan(
    dataset_instance: SignFiDataset,
    *,
    split_scheme: str | None,
) -> SignFiSplitPlan:
    split_contract = _resolve_signfi_split_contract(split_scheme)
    train_indices, val_indices = stratified_train_val_indices(
        dataset_instance.get_labels(),
        train_ratio=float(split_contract["train_ratio"]),
        seed=int(split_contract["seed"]),
    )
    return SignFiSplitPlan(
        split_scheme=str(split_contract["binding_id"]),
        train_indices=train_indices,
        val_indices=val_indices,
    )


def _build_signfi_manifest_rows(
    request: DatasetBuildRequest,
) -> tuple[str, list[dict[str, object]]]:
    manifest_plan = build_signfi_manifest_plan(request)
    return manifest_plan.variant, manifest_plan.manifest_rows


def build_signfi_manifest_plan(request: DatasetBuildRequest) -> SignFiManifestPlan:
    if request.dataset_root is None:
        raise ValueError("SignFi builder requires request.dataset_root to be resolved.")

    dataset_instance = load_signfi_dataset(
        request.dataset_root,
        variant=request.variant,
    )
    split_plan = _resolve_signfi_split_plan(
        dataset_instance,
        split_scheme=request.split_scheme,
    )
    metadata_rows = dataset_instance.metadata_rows()
    task_binding = None if request.task_binding in {None, ""} else str(request.task_binding)
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
    return SignFiManifestPlan(
        variant=dataset_instance.variant,
        split_scheme=split_plan.split_scheme,
        manifest_rows=manifest_rows,
    )


__all__ = [
    "DATASET_ID",
    "SignFiManifestPlan",
    "build_signfi_manifest_plan",
]
