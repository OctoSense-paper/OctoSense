"""Builtin manifest row ownership for the XRF55 dataset definition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from octosense.datasets.catalog import get_dataset_binding_payload, list_dataset_binding_ids
from octosense.datasets.core.builder import (
    DatasetBuildRequest,
    annotate_split_rows,
)

from .ingest import XRF55Dataset, open_xrf55_dataset

DATASET_ID = "xrf55"


def _resolve_xrf55_split_contract(binding_id: str | None) -> dict[str, Any]:
    candidate = "" if binding_id in {None, ""} else str(binding_id).strip()
    available = list_dataset_binding_ids(DATASET_ID, binding_kind="split_scheme")
    if not candidate:
        supported = ", ".join(available) or "<none>"
        raise ValueError(
            "XRF55 requires an explicit split_scheme; "
            "implicit default/singleton fallback is not supported. "
            f"Supported bindings: {supported}."
        )
    if candidate not in available:
        supported = ", ".join(available) or "<none>"
        raise ValueError(
            f"XRF55 split_scheme must be one of: {supported}. Received {candidate!r}."
        )
    payload = get_dataset_binding_payload(
        DATASET_ID,
        binding_kind="split_scheme",
        binding_id=candidate,
    )
    resolved_binding_id = str(payload["binding_id"])
    if str(payload.get("name") or "").strip() != resolved_binding_id:
        raise ValueError(
            f"XRF55 split binding '{resolved_binding_id}' must declare name={resolved_binding_id!r}."
        )
    strategy = str(payload.get("strategy") or "").strip()
    if strategy != "metadata_declared_label":
        raise ValueError(
            "XRF55 split bindings must declare strategy='metadata_declared_label'. "
            f"binding_id={resolved_binding_id!r}, strategy={strategy!r}"
        )
    split_field = str(payload.get("split_field") or "").strip()
    if not split_field:
        raise ValueError(
            f"XRF55 split binding {resolved_binding_id!r} must declare split_field."
        )

    declared = payload.get("declared_splits")
    if not isinstance(declared, dict):
        raise ValueError(
            f"XRF55 split binding {resolved_binding_id!r} must declare declared_splits."
        )
    required_declared = tuple(str(name) for name in declared.get("required", ()))
    optional_declared = tuple(str(name) for name in declared.get("optional", ()))
    if not required_declared:
        raise ValueError(
            f"XRF55 split binding {resolved_binding_id!r} must declare required splits."
        )

    return {
        "binding_id": resolved_binding_id,
        "split_field": split_field,
        "required_declared": required_declared,
        "optional_declared": optional_declared,
    }


@dataclass(frozen=True)
class XRF55ManifestBuildPlan:
    dataset: XRF55Dataset
    split_contract: dict[str, Any]

    def build_rows(self, *, task_binding: str) -> list[dict[str, object]]:
        metadata_rows = self.dataset.metadata_rows()
        split_scheme = str(self.split_contract["binding_id"])
        split_local_indices: dict[str, list[int]] = {}
        for index, row in enumerate(metadata_rows):
            split_name = row.get(str(self.split_contract["split_field"]))
            if isinstance(split_name, str) and split_name:
                split_local_indices.setdefault(split_name, []).append(index)

        required_declared = set(self.split_contract["required_declared"])
        if not (required_declared <= set(split_local_indices)):
            missing = sorted(required_declared - set(split_local_indices))
            raise ValueError(
                "XRF55 metadata rows must declare all required splits explicitly. "
                f"Missing splits: {missing}."
            )
        supported_splits = tuple(
            split_name
            for split_name in (
                *tuple(self.split_contract["required_declared"]),
                *tuple(self.split_contract["optional_declared"]),
            )
            if split_name in split_local_indices
        )

        manifest_rows: list[dict[str, object]] = []
        for split_name in supported_splits:
            manifest_rows.extend(
                annotate_split_rows(
                    [dict(metadata_rows[index]) for index in split_local_indices[split_name]],
                    split=split_name,
                    split_scheme=split_scheme,
                    task_binding=task_binding,
                )
            )
        return manifest_rows


def plan_xrf55_manifest_build(
    request: DatasetBuildRequest,
    *,
    variant: str,
    modality: str,
) -> XRF55ManifestBuildPlan:
    if request.dataset_root is None:
        raise ValueError("XRF55 builder requires request.dataset_root to be resolved.")
    return XRF55ManifestBuildPlan(
        dataset=open_xrf55_dataset(
            request.dataset_root,
            modality=modality,
            variant=variant,
        ),
        split_contract=_resolve_xrf55_split_contract(request.split_scheme),
    )

__all__ = [
    "DATASET_ID",
    "XRF55ManifestBuildPlan",
    "plan_xrf55_manifest_build",
]
