"""High-level manifest planning for the HuPR builtin dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from octosense.datasets.catalog import get_dataset_binding_payload, list_dataset_binding_ids
from octosense.datasets.core.builder import (
    DatasetBuildRequest,
    annotate_split_rows,
    partition_sorted_groups,
)

from .ingest import HuPRDataset

DATASET_ID = "hupr"


@dataclass(frozen=True)
class HuPRBuildPlan:
    variant: str
    split_scheme: str
    manifest_rows: list[dict[str, object]]
    split_file_indexes: dict[str, list[int]]


def _parse_hupr_file_index(file_path: str | Path) -> int:
    path = Path(file_path)
    try:
        return int(path.stem.split("_")[-1])
    except ValueError as exc:
        raise ValueError(f"Could not parse HuPR file index from {path}") from exc


def _hupr_split_contract(split_scheme: str | None) -> dict[str, object]:
    candidate = "" if split_scheme in {None, ""} else str(split_scheme).strip()
    available = list_dataset_binding_ids(DATASET_ID, binding_kind="split_scheme")
    if not candidate:
        supported = ", ".join(available) or "<none>"
        raise ValueError(
            "HuPR requires an explicit split_scheme; "
            "implicit default/singleton fallback is not supported. "
            f"Supported bindings: {supported}."
        )
    if candidate not in available:
        supported = ", ".join(available) or "<none>"
        raise ValueError(
            f"HuPR split_scheme must be one of: {supported}. Received {candidate!r}."
        )
    payload = get_dataset_binding_payload(
        DATASET_ID,
        binding_kind="split_scheme",
        binding_id=candidate,
    )
    resolved_binding_id = str(payload["binding_id"])
    if str(payload.get("name") or "").strip() != resolved_binding_id:
        raise ValueError(
            f"HuPR split binding '{resolved_binding_id}' must declare name={resolved_binding_id!r}."
        )
    if str(payload.get("strategy") or "").strip() != "sorted_group_partition":
        raise ValueError(
            "HuPR split binding "
            f"'{resolved_binding_id}' must declare strategy='sorted_group_partition'."
        )
    if str(payload.get("group_field") or "").strip() != "file_index":
        raise ValueError(
            f"HuPR split binding '{resolved_binding_id}' must declare group_field='file_index'."
        )
    resolved = dict(payload)
    resolved["train_ratio"] = float(payload["train_ratio"])
    return resolved


def _partition_hupr_file_indexes(
    dataset_root: Path,
    *,
    split_scheme: str | None,
) -> tuple[str, list[int], list[int]]:
    split_contract = _hupr_split_contract(split_scheme)
    radar_dir = dataset_root / "radar_maps"
    if not radar_dir.exists():
        raise FileNotFoundError(f"radar_maps not found at {radar_dir}")
    available_file_indexes = [
        _parse_hupr_file_index(path)
        for path in sorted(radar_dir.glob("*.pkl"))
    ]
    if not available_file_indexes:
        raise ValueError(f"No HuPR radar map files found under {radar_dir}")
    train_files, val_files = partition_sorted_groups(
        available_file_indexes,
        train_ratio=float(split_contract["train_ratio"]),
    )
    return str(split_contract["binding_id"]), train_files, val_files


def build_hupr_plan(
    request: DatasetBuildRequest,
) -> HuPRBuildPlan:
    if request.dataset_root is None:
        raise ValueError("HuPR builder requires request.dataset_root to be resolved.")

    split_scheme, train_files, val_files = _partition_hupr_file_indexes(
        request.dataset_root,
        split_scheme=request.split_scheme,
    )
    train_dataset = HuPRDataset(
        request.dataset_root,
        variant=request.variant,
        file_indexes=train_files,
    )
    val_dataset = HuPRDataset(
        request.dataset_root,
        variant=request.variant,
        file_indexes=val_files,
    )
    variant = train_dataset.variant
    task_binding = (
        str(request.task_binding)
        if request.task_binding not in {None, ""}
        else None
    )
    manifest_rows = (
        annotate_split_rows(
            [dict(row) for row in train_dataset.metadata_rows()],
            split="train",
            split_scheme=split_scheme,
            task_binding=task_binding,
        )
        + annotate_split_rows(
            [dict(row) for row in val_dataset.metadata_rows()],
            split="val",
            split_scheme=split_scheme,
            task_binding=task_binding,
        )
    )
    return HuPRBuildPlan(
        variant=variant,
        split_scheme=split_scheme,
        manifest_rows=manifest_rows,
        split_file_indexes={
            "train": list(train_files),
            "val": list(val_files),
        },
    )


__all__ = [
    "HuPRBuildPlan",
    "build_hupr_plan",
]
