"""High-level manifest planning for the CSI-Bench dataset family."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from octosense.datasets.catalog import get_dataset_binding_payload, list_dataset_binding_ids
from octosense.datasets.core.builder import DatasetBuildRequest

DATASET_ID = "csi_bench"


@dataclass(frozen=True)
class CSIBenchManifestPlan:
    variant: str
    task_name: str
    task_binding: str
    split_scheme_id: str
    split_ids_by_name: dict[str, str]


def _annotate_rows(rows: list[dict[str, object]], *, plan: CSIBenchManifestPlan, split: str) -> list[dict[str, object]]:
    annotated: list[dict[str, object]] = []
    for row in rows:
        annotated.append(
            {
                **dict(row),
                "partitions": {
                    "split": split,
                },
                "split": split,
                "assigned_split": split,
                "split/default": split,
                "split_scheme": plan.split_scheme_id,
                "task_binding": plan.task_binding,
            }
        )
    return annotated


def _split_scheme_contracts() -> dict[str, dict[str, Any]]:
    return {
        binding_id: get_dataset_binding_payload(
            DATASET_ID,
            binding_kind="split_scheme",
            binding_id=binding_id,
        )
        for binding_id in list_dataset_binding_ids(DATASET_ID, binding_kind="split_scheme")
    }


def _resolve_csi_bench_split_scheme_contract(binding_id: str | None = None) -> dict[str, Any]:
    candidate = "" if binding_id in {None, ""} else str(binding_id).strip()
    available = list_dataset_binding_ids(DATASET_ID, binding_kind="split_scheme")
    if not candidate:
        supported = ", ".join(available) or "<none>"
        raise ValueError(
            "CSI-Bench requires an explicit split_scheme; "
            "implicit default/singleton fallback is not supported. "
            f"Supported bindings: {supported}."
        )
    if candidate not in available:
        supported = ", ".join(available) or "<none>"
        raise ValueError(
            f"CSI-Bench split_scheme must be one of: {supported}. Received {candidate!r}."
        )
    return get_dataset_binding_payload(
        DATASET_ID,
        binding_kind="split_scheme",
        binding_id=candidate,
    )


def _split_member_contracts() -> dict[str, dict[str, str]]:
    contracts: dict[str, dict[str, str]] = {}
    for binding_id, payload in _split_scheme_contracts().items():
        split_ids = payload.get("split_ids", {})
        if not isinstance(split_ids, dict):
            raise TypeError(
                f"CSI-Bench split binding '{binding_id}' must define mapping field 'split_ids'."
            )
        contracts[binding_id] = {
            str(split_name): str(runtime_split_id)
            for split_name, runtime_split_id in split_ids.items()
            if runtime_split_id not in {None, ""}
        }
    return contracts


def _resolve_csi_bench_split_binding(
    split_name: str,
    *,
    split_scheme: str | None = None,
) -> str:
    if split_name in {None, ""}:
        raise ValueError("CSI-Bench split_name must be a non-empty string.")

    contract = _resolve_csi_bench_split_scheme_contract(split_scheme)
    split_ids = _split_member_contracts().get(str(contract["binding_id"]), {})
    if split_name not in split_ids:
        choices = ", ".join(sorted(split_ids))
        raise KeyError(
            f"CSI-Bench split scheme '{contract['binding_id']}' does not define split "
            f"member '{split_name}'. Available: {choices}"
        )
    return split_ids[split_name]


def plan_csi_bench_manifest(
    request: DatasetBuildRequest,
    *,
    variant: str,
    task_name: str,
    task_binding: str,
) -> CSIBenchManifestPlan:
    if request.dataset_root is None:
        raise ValueError("CSI-Bench builder requires request.dataset_root to be resolved.")

    split_scheme = _resolve_csi_bench_split_scheme_contract(request.split_scheme)
    resolved_split_ids = {
        split_name: _resolve_csi_bench_split_binding(
            split_name,
            split_scheme=str(split_scheme["binding_id"]),
        )
        for split_name in ("train", "val", "test")
    }
    return CSIBenchManifestPlan(
        variant=variant,
        task_name=task_name,
        task_binding=task_binding,
        split_scheme_id=str(split_scheme["binding_id"]),
        split_ids_by_name=resolved_split_ids,
    )


def build_csi_bench_manifest_rows(
    request: DatasetBuildRequest,
    *,
    plan: CSIBenchManifestPlan,
) -> list[dict[str, object]]:
    if request.dataset_root is None:
        raise ValueError("CSI-Bench builder requires request.dataset_root to be resolved.")

    from .ingest import CSIBenchDataset

    split_datasets = {
        split_name: CSIBenchDataset(
            request.dataset_root,
            variant=plan.variant,
            task_name=plan.task_name,
            split_name=plan.split_ids_by_name[split_name],
        )
        for split_name in ("train", "val", "test")
    }

    rows_by_split = {
        split_name: _annotate_rows(
            split_datasets[split_name].metadata_rows(),
            plan=plan,
            split=split_name,
        )
        for split_name in ("train", "val", "test")
    }
    return [
        *rows_by_split["train"],
        *rows_by_split["val"],
        *rows_by_split["test"],
    ]


__all__ = [
    "DATASET_ID",
    "CSIBenchManifestPlan",
    "build_csi_bench_manifest_rows",
    "plan_csi_bench_manifest",
]
