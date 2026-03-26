"""High-level manifest planning for the MM-Fi dataset family."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from octosense.datasets.catalog import get_dataset_binding_payload, list_dataset_binding_ids
from octosense.datasets.core.builder import (
    DatasetBuildRequest,
    annotate_split_rows,
    subset_indices_by_group,
)

DATASET_ID = "mmfi"


@dataclass(frozen=True)
class MMFiManifestPlan:
    variant: str
    modality: str
    task_binding: str
    split_scheme_id: str
    group_field: str
    train_ratio: float


def _resolve_mmfi_split_contract(binding_id: str | None) -> dict[str, Any]:
    candidate = "" if binding_id in {None, ""} else str(binding_id).strip()
    available = list_dataset_binding_ids(DATASET_ID, binding_kind="split_scheme")
    if not candidate:
        supported = ", ".join(available) or "<none>"
        raise ValueError(
            "MM-Fi requires an explicit split_scheme; "
            "implicit default/singleton fallback is not supported. "
            f"Supported bindings: {supported}."
        )
    if candidate not in available:
        supported = ", ".join(available) or "<none>"
        raise ValueError(
            f"MM-Fi split_scheme must be one of: {supported}. Received {candidate!r}."
        )
    payload = get_dataset_binding_payload(
        DATASET_ID,
        binding_kind="split_scheme",
        binding_id=candidate,
    )
    binding_name = str(payload["binding_id"])
    if str(payload.get("name") or "").strip() != binding_name:
        raise ValueError(
            f"MM-Fi split binding '{binding_name}' must declare name={binding_name!r}."
        )
    strategy = str(payload.get("strategy") or "").strip()
    if strategy != "sorted_group_partition":
        raise ValueError(
            "MM-Fi split bindings must declare strategy='sorted_group_partition'. "
            f"binding_id={binding_name!r}, strategy={strategy!r}"
        )
    group_field = str(payload.get("group_field") or "").strip()
    if not group_field:
        raise ValueError(
            f"MM-Fi split binding {binding_name!r} must declare group_field."
        )
    raw_partitions = payload.get("partitions")
    if not isinstance(raw_partitions, list) or not raw_partitions:
        raise ValueError(
            f"MM-Fi split binding {binding_name!r} must declare partitions."
        )
    partitions: list[tuple[str, float]] = []
    ratio_total = 0.0
    for index, raw_partition in enumerate(raw_partitions):
        if not isinstance(raw_partition, dict):
            raise ValueError(
                "MM-Fi split binding partitions must be mappings. "
                f"binding_id={binding_name!r}, index={index}"
            )
        split_name = str(raw_partition.get("split") or "").strip()
        if not split_name:
            raise ValueError(
                "MM-Fi split binding partitions must declare split names. "
                f"binding_id={binding_name!r}, index={index}"
            )
        ratio_raw = raw_partition.get("ratio")
        if ratio_raw is None:
            raise ValueError(
                "MM-Fi split binding partitions must declare ratio. "
                f"binding_id={binding_name!r}, split={split_name!r}"
            )
        ratio = float(ratio_raw)
        if not 0.0 < ratio < 1.0:
            raise ValueError(
                "MM-Fi split binding partition ratio must be between 0 and 1. "
                f"binding_id={binding_name!r}, split={split_name!r}, ratio={ratio!r}"
            )
        partitions.append((split_name, ratio))
        ratio_total += ratio
    if len(partitions) != 2 or {name for name, _ in partitions} != {"train", "val"}:
        raise ValueError(
            "MM-Fi split binding currently supports exactly train/val partitions. "
            f"binding_id={binding_name!r}, partitions={partitions!r}"
        )
    if abs(ratio_total - 1.0) > 1e-6:
        raise ValueError(
            "MM-Fi split binding partition ratios must sum to 1.0. "
            f"binding_id={binding_name!r}, ratio_total={ratio_total!r}"
        )
    return {
        "binding_id": binding_name,
        "group_field": group_field,
        "partitions": tuple(partitions),
    }


def plan_mmfi_manifest(
    request: DatasetBuildRequest,
    *,
    variant: str,
    modality: str,
    task_binding: str,
) -> MMFiManifestPlan:
    if request.dataset_root is None:
        raise ValueError("MM-Fi builder requires request.dataset_root to be resolved.")

    split_contract = _resolve_mmfi_split_contract(request.split_scheme)
    group_field = str(split_contract["group_field"])
    if group_field != "subject":
        raise ValueError(
            "MM-Fi materialization currently requires split bindings with group_field='subject'. "
            f"binding_id={split_contract['binding_id']!r}, "
            f"group_field={group_field!r}"
        )
    return MMFiManifestPlan(
        variant=variant,
        modality=modality,
        task_binding=task_binding,
        split_scheme_id=str(split_contract["binding_id"]),
        group_field=group_field,
        train_ratio=float(dict(split_contract["partitions"])["train"]),
    )


def build_mmfi_manifest_rows(
    request: DatasetBuildRequest,
    *,
    plan: MMFiManifestPlan,
) -> tuple[list[dict[str, object]], dict[str, list[str]]]:
    if request.dataset_root is None:
        raise ValueError("MM-Fi builder requires request.dataset_root to be resolved.")

    from .ingest import MMFiDataset

    dataset_instance = MMFiDataset(
        request.dataset_root,
        modality=plan.modality,
        variant=plan.variant,
    )
    metadata_rows = dataset_instance.metadata_rows()
    missing_group_rows = [
        index for index, row in enumerate(metadata_rows) if row.get(plan.group_field) in {None, ""}
    ]
    if missing_group_rows:
        preview = ", ".join(str(index) for index in missing_group_rows[:5])
        raise ValueError(
            "MM-Fi metadata rows are missing the declared split group field. "
            f"group_field={plan.group_field!r}, rows=[{preview}]"
        )
    group_values = [row[plan.group_field] for row in metadata_rows]
    train_indices, val_indices, train_subjects, val_subjects = subset_indices_by_group(
        group_values,
        train_ratio=plan.train_ratio,
    )
    manifest_rows = (
        annotate_split_rows(
            [dict(metadata_rows[index]) for index in train_indices],
            split="train",
            split_scheme=plan.split_scheme_id,
            task_binding=plan.task_binding,
        )
        + annotate_split_rows(
            [dict(metadata_rows[index]) for index in val_indices],
            split="val",
            split_scheme=plan.split_scheme_id,
            task_binding=plan.task_binding,
        )
    )
    return manifest_rows, {
        "train": list(train_subjects),
        "val": list(val_subjects),
    }

__all__ = [
    "DATASET_ID",
    "MMFiManifestPlan",
    "build_mmfi_manifest_rows",
    "plan_mmfi_manifest",
]
