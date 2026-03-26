"""Builtin manifest row ownership for the XRFV2 dataset definition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from octosense.datasets.catalog import get_dataset_binding_payload, list_dataset_binding_ids
from octosense.datasets.core.builder import (
    DatasetBuildRequest,
    annotate_split_rows,
    stratified_train_val_indices,
)

from .ingest import XRFV2Dataset, open_xrfv2_dataset

DATASET_ID = "xrfv2"


def _row_identity_without_split(
    row: dict[str, object],
    *,
    source_split_field: str,
) -> tuple[tuple[str, object], ...]:
    excluded = {"sample_index", "sample_id", "sample_group_id", source_split_field}
    return tuple(sorted((key, row.get(key)) for key in row if key not in excluded))


def _resolve_xrfv2_split_contract(binding_id: str | None) -> dict[str, Any]:
    candidate = "" if binding_id in {None, ""} else str(binding_id).strip()
    available = list_dataset_binding_ids(DATASET_ID, binding_kind="split_scheme")
    if not candidate:
        supported = ", ".join(available) or "<none>"
        raise ValueError(
            "XRFV2 requires an explicit split_scheme; "
            "implicit default/singleton fallback is not supported. "
            f"Supported bindings: {supported}."
        )
    if candidate not in available:
        supported = ", ".join(available) or "<none>"
        raise ValueError(
            f"XRFV2 split_scheme must be one of: {supported}. Received {candidate!r}."
        )
    payload = get_dataset_binding_payload(
        DATASET_ID,
        binding_kind="split_scheme",
        binding_id=candidate,
    )
    binding_name = str(payload["binding_id"])
    if str(payload.get("name") or "").strip() != binding_name:
        raise ValueError(
            f"XRFV2 split binding '{binding_name}' must declare name={binding_name!r}."
        )
    strategy = str(payload.get("strategy") or "").strip()
    if strategy != "source_split_plus_stratified_label":
        raise ValueError(
            "XRFV2 split bindings must declare "
            "strategy='source_split_plus_stratified_label'. "
            f"binding_id={binding_name!r}, strategy={strategy!r}"
        )
    source_split_field = str(payload.get("source_split_field") or "").strip()
    if not source_split_field:
        raise ValueError(
            f"XRFV2 split binding {binding_name!r} must declare source_split_field."
        )
    source_splits = payload.get("source_splits")
    if not isinstance(source_splits, dict):
        raise ValueError(
            f"XRFV2 split binding {binding_name!r} must declare source_splits."
        )
    train_pool_split = str(source_splits.get("train_pool") or "").strip()
    heldout_split = str(source_splits.get("heldout") or "").strip()
    if not train_pool_split or not heldout_split:
        raise ValueError(
            f"XRFV2 split binding {binding_name!r} must declare train_pool and heldout source splits."
        )
    train_val = payload.get("train_val")
    if not isinstance(train_val, dict):
        raise ValueError(
            f"XRFV2 split binding {binding_name!r} must declare train_val."
        )
    train_val_strategy = str(train_val.get("strategy") or "").strip()
    if train_val_strategy != "stratified_label":
        raise ValueError(
            "XRFV2 split binding train_val must declare strategy='stratified_label'. "
            f"binding_id={binding_name!r}, strategy={train_val_strategy!r}"
        )
    seed_raw = train_val.get("seed")
    if seed_raw is None:
        raise ValueError(
            f"XRFV2 split binding {binding_name!r} train_val must declare seed."
        )
    raw_partitions = train_val.get("partitions")
    if not isinstance(raw_partitions, list) or not raw_partitions:
        raise ValueError(
            f"XRFV2 split binding {binding_name!r} train_val must declare partitions."
        )
    train_val_partitions: list[tuple[str, float]] = []
    ratio_total = 0.0
    for index, raw_partition in enumerate(raw_partitions):
        if not isinstance(raw_partition, dict):
            raise ValueError(
                "XRFV2 split binding train_val partitions must be mappings. "
                f"binding_id={binding_name!r}, index={index}"
            )
        split_name = str(raw_partition.get("split") or "").strip()
        if not split_name:
            raise ValueError(
                "XRFV2 split binding train_val partitions must declare split names. "
                f"binding_id={binding_name!r}, index={index}"
            )
        ratio_raw = raw_partition.get("ratio")
        if ratio_raw is None:
            raise ValueError(
                "XRFV2 split binding train_val partitions must declare ratio. "
                f"binding_id={binding_name!r}, split={split_name!r}"
            )
        ratio = float(ratio_raw)
        if not 0.0 < ratio < 1.0:
            raise ValueError(
                "XRFV2 split binding train_val partition ratio must be between 0 and 1. "
                f"binding_id={binding_name!r}, split={split_name!r}, ratio={ratio!r}"
            )
        train_val_partitions.append((split_name, ratio))
        ratio_total += ratio
    if len(train_val_partitions) != 2 or {name for name, _ in train_val_partitions} != {"train", "val"}:
        raise ValueError(
            "XRFV2 split binding train_val currently supports exactly train/val partitions. "
            f"binding_id={binding_name!r}, partitions={train_val_partitions!r}"
        )
    if abs(ratio_total - 1.0) > 1e-6:
        raise ValueError(
            "XRFV2 split binding train_val partition ratios must sum to 1.0. "
            f"binding_id={binding_name!r}, ratio_total={ratio_total!r}"
        )
    heldout = payload.get("heldout")
    if not isinstance(heldout, dict):
        raise ValueError(f"XRFV2 split binding {binding_name!r} must declare heldout.")
    heldout_output_split = str(heldout.get("split") or "").strip()
    if not heldout_output_split:
        raise ValueError(
            f"XRFV2 split binding {binding_name!r} heldout block must declare split."
        )
    return {
        "binding_id": binding_name,
        "source_split_field": source_split_field,
        "train_pool_split": train_pool_split,
        "heldout_source_split": heldout_split,
        "train_val_seed": int(seed_raw),
        "train_val_partitions": tuple(train_val_partitions),
        "heldout_output_split": heldout_output_split,
    }


@dataclass(frozen=True)
class XRFV2ManifestBuildPlan:
    train_dataset: XRFV2Dataset
    heldout_dataset: XRFV2Dataset
    split_contract: dict[str, Any]

    def build_rows(self, *, task_binding: str) -> list[dict[str, object]]:
        train_pool_rows = self.train_dataset.metadata_rows()
        source_split_field = str(self.split_contract["source_split_field"])
        missing_train_rows = [
            index
            for index, row in enumerate(train_pool_rows)
            if row.get(source_split_field) != self.split_contract["train_pool_split"]
        ]
        if missing_train_rows:
            preview = ", ".join(str(index) for index in missing_train_rows[:5])
            raise ValueError(
                "XRFV2 train-pool metadata rows do not match the declared source split. "
                f"source_split_field={source_split_field!r}, expected={self.split_contract['train_pool_split']!r}, rows=[{preview}]"
            )
        heldout_rows = self.heldout_dataset.metadata_rows()
        missing_heldout_rows = [
            index
            for index, row in enumerate(heldout_rows)
            if row.get(source_split_field) != self.split_contract["heldout_source_split"]
        ]
        if missing_heldout_rows:
            preview = ", ".join(str(index) for index in missing_heldout_rows[:5])
            raise ValueError(
                "XRFV2 heldout metadata rows do not match the declared source split. "
                f"source_split_field={source_split_field!r}, expected={self.split_contract['heldout_source_split']!r}, rows=[{preview}]"
            )
        train_identities = {
            _row_identity_without_split(row, source_split_field=source_split_field): index
            for index, row in enumerate(train_pool_rows)
        }
        overlapping_rows: list[tuple[int, int]] = []
        for heldout_index, row in enumerate(heldout_rows):
            train_index = train_identities.get(
                _row_identity_without_split(row, source_split_field=source_split_field)
            )
            if train_index is not None:
                overlapping_rows.append((train_index, heldout_index))
        if overlapping_rows:
            preview = ", ".join(
                f"train[{train_index}]<->heldout[{heldout_index}]"
                for train_index, heldout_index in overlapping_rows[:5]
            )
            raise ValueError(
                "XRFV2 train-pool and heldout metadata rows overlap after removing split-only "
                "fields, which would leak heldout samples into train/val. "
                f"rows=[{preview}]"
            )
        train_indices, val_indices = stratified_train_val_indices(
            self.train_dataset.get_labels(),
            train_ratio=dict(self.split_contract["train_val_partitions"])["train"],
            seed=int(self.split_contract["train_val_seed"]),
        )
        split_scheme = str(self.split_contract["binding_id"])
        return (
            annotate_split_rows(
                [dict(train_pool_rows[index]) for index in train_indices],
                split="train",
                split_scheme=split_scheme,
                task_binding=task_binding,
            )
            + annotate_split_rows(
                [dict(train_pool_rows[index]) for index in val_indices],
                split="val",
                split_scheme=split_scheme,
                task_binding=task_binding,
            )
            + annotate_split_rows(
                [dict(row) for row in heldout_rows],
                split=str(self.split_contract["heldout_output_split"]),
                split_scheme=split_scheme,
                task_binding=task_binding,
            )
        )


def plan_xrfv2_manifest_build(
    request: DatasetBuildRequest,
    *,
    variant: str,
    modality: str,
) -> XRFV2ManifestBuildPlan:
    if request.dataset_root is None:
        raise ValueError("XRFV2 builder requires request.dataset_root to be resolved.")
    split_contract = _resolve_xrfv2_split_contract(request.split_scheme)
    return XRFV2ManifestBuildPlan(
        train_dataset=open_xrfv2_dataset(
            request.dataset_root,
            modality=modality,
            variant=variant,
            split=str(split_contract["train_pool_split"]),
        ),
        heldout_dataset=open_xrfv2_dataset(
            request.dataset_root,
            modality=modality,
            variant=variant,
            split=str(split_contract["heldout_source_split"]),
        ),
        split_contract=split_contract,
    )

__all__ = [
    "DATASET_ID",
    "XRFV2ManifestBuildPlan",
    "plan_xrfv2_manifest_build",
]
