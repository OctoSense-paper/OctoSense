"""High-level manifest planning for the OctoNet builtin dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from octosense.datasets.catalog import get_dataset_binding_payload, list_dataset_binding_ids
from octosense.datasets.core.builder import (
    DatasetBuildRequest,
    annotate_split_rows,
    subset_indices_by_group,
)
from octosense.datasets.core.task_binding import resolve_dataset_task_binding_payload

DATASET_ID = "octonet"


@dataclass(frozen=True)
class OctonetManifestPlan:
    variant: str | None
    split_scheme: str
    manifest_rows: list[dict[str, object]]
    materialization: dict[str, object]


@dataclass(frozen=True)
class _OctonetSplitSpec:
    binding_id: str
    strategy: str
    group_field: str
    partitions: tuple[tuple[str, float], ...]


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
            "OctoNet requires an explicit task_binding; "
            "implicit default/singleton fallback is not supported. "
            f"Supported bindings: {supported}."
        )
    if candidate not in available:
        supported = ", ".join(available) or "<none>"
        raise ValueError(
            f"OctoNet task_binding must be one of: {supported}. Received {candidate!r}."
        )
    return get_dataset_binding_payload(
        DATASET_ID,
        binding_kind="task_binding",
        binding_id=candidate,
    )


def _resolve_octonet_split_spec(binding_id: str | None) -> _OctonetSplitSpec:
    candidate = "" if binding_id in {None, ""} else str(binding_id).strip()
    available = list_dataset_binding_ids(DATASET_ID, binding_kind="split_scheme")
    if not candidate:
        supported = ", ".join(available) or "<none>"
        raise ValueError(
            "OctoNet requires an explicit split_scheme; "
            "implicit default/singleton fallback is not supported. "
            f"Supported bindings: {supported}."
        )
    if candidate not in available:
        supported = ", ".join(available) or "<none>"
        raise ValueError(
            f"OctoNet split_scheme must be one of: {supported}. Received {candidate!r}."
        )
    payload = get_dataset_binding_payload(
        DATASET_ID,
        binding_kind="split_scheme",
        binding_id=candidate,
    )
    resolved_binding_id = str(payload["binding_id"])
    strategy = str(payload.get("strategy") or "").strip()
    if strategy != "sorted_group_partition":
        raise ValueError(
            "OctoNet split bindings must declare strategy='sorted_group_partition'. "
            f"binding_id={resolved_binding_id!r}, strategy={strategy!r}"
        )
    group_field = str(payload.get("group_field") or "").strip()
    if not group_field:
        raise ValueError(
            f"OctoNet split binding {resolved_binding_id!r} must declare group_field."
        )
    raw_partitions = payload.get("partitions")
    if not isinstance(raw_partitions, list) or not raw_partitions:
        raise ValueError(
            f"OctoNet split binding {resolved_binding_id!r} must declare partitions."
        )
    partitions: list[tuple[str, float]] = []
    ratio_total = 0.0
    for index, raw_partition in enumerate(raw_partitions):
        if not isinstance(raw_partition, dict):
            raise ValueError(
                "OctoNet split binding partitions must be mappings. "
                f"binding_id={resolved_binding_id!r}, index={index}"
            )
        split_name = str(raw_partition.get("split") or "").strip()
        if not split_name:
            raise ValueError(
                "OctoNet split binding partitions must declare split names. "
                f"binding_id={resolved_binding_id!r}, index={index}"
            )
        ratio_raw = raw_partition.get("ratio")
        if ratio_raw is None:
            raise ValueError(
                "OctoNet split binding partitions must declare ratio. "
                f"binding_id={resolved_binding_id!r}, split={split_name!r}"
            )
        ratio = float(ratio_raw)
        if not 0.0 < ratio < 1.0:
            raise ValueError(
                "OctoNet split binding partition ratio must be between 0 and 1. "
                f"binding_id={resolved_binding_id!r}, split={split_name!r}, ratio={ratio!r}"
            )
        partitions.append((split_name, ratio))
        ratio_total += ratio
    if len(partitions) != 2 or {name for name, _ in partitions} != {"train", "val"}:
        raise ValueError(
            "OctoNet split binding currently supports exactly train/val partitions. "
            f"binding_id={resolved_binding_id!r}, partitions={partitions!r}"
        )
    if abs(ratio_total - 1.0) > 1e-6:
        raise ValueError(
            "OctoNet split binding partition ratios must sum to 1.0. "
            f"binding_id={resolved_binding_id!r}, ratio_total={ratio_total!r}"
        )
    return _OctonetSplitSpec(
        binding_id=resolved_binding_id,
        strategy=strategy,
        group_field=group_field,
        partitions=tuple(partitions),
    )


def build_octonet_manifest_plan(
    request: DatasetBuildRequest,
    *,
    metadata_rows: list[dict[str, object]],
) -> OctonetManifestPlan:
    split_spec = _resolve_octonet_split_spec(request.split_scheme)
    missing_group_rows = [
        index
        for index, row in enumerate(metadata_rows)
        if row.get(split_spec.group_field) in {None, ""}
    ]
    if missing_group_rows:
        preview = ", ".join(str(index) for index in missing_group_rows[:5])
        raise ValueError(
            "OctoNet metadata rows are missing the declared split group field. "
            f"group_field={split_spec.group_field!r}, rows=[{preview}]"
        )
    train_indices, val_indices, _, _ = subset_indices_by_group(
        [row[split_spec.group_field] for row in metadata_rows],
        train_ratio=dict(split_spec.partitions)["train"],
    )
    task_binding = _resolve_task_binding_contract(request.task_binding)
    binding_payload = resolve_dataset_task_binding_payload(task_binding, owner="OctoNet")
    dataset_target_schema = _dataset_target_schema(
        binding_payload,
        owner="OctoNet task binding compiler payload",
    )
    resolved_task_binding = str(binding_payload["task_binding"])
    manifest_rows = (
        annotate_split_rows(
            [dict(metadata_rows[index]) for index in train_indices],
            split="train",
            split_scheme=split_spec.binding_id,
            task_binding=resolved_task_binding,
        )
        + annotate_split_rows(
            [dict(metadata_rows[index]) for index in val_indices],
            split="val",
            split_scheme=split_spec.binding_id,
            task_binding=resolved_task_binding,
        )
    )
    return OctonetManifestPlan(
        variant=None,
        split_scheme=split_spec.binding_id,
        manifest_rows=manifest_rows,
        materialization={
            "task_binding": resolved_task_binding,
            "task_id": str(binding_payload["task_id"]),
            "target_schema": dataset_target_schema,
        },
    )


__all__ = [
    "DATASET_ID",
    "OctonetManifestPlan",
    "build_octonet_manifest_plan",
]
