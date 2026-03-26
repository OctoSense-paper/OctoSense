"""Dataset-owned schema objects for metadata-first manifests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Mapping

from octosense.core.describe import Describable, DescribeNode


def _sorted_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): payload[key] for key in sorted(payload)}


def _normalize_modalities(payload: Sequence[str] | None) -> tuple[str, ...]:
    if payload is None:
        return ()
    normalized: list[str] = []
    for item in payload:
        candidate = str(item).strip()
        if not candidate:
            continue
        if candidate in normalized:
            raise ValueError(
                "DatasetMetadata.modalities must not contain duplicate modality ids."
            )
        normalized.append(candidate)
    return tuple(normalized)


@dataclass(init=False)
class DatasetMetadata(Describable):
    """Dataset-level descriptive metadata attached to one materialized view."""

    name: str
    sample_count: int
    modalities: tuple[str, ...] = ()
    users: list[int] = field(default_factory=list)
    rooms: list[int] = field(default_factory=list)
    collection_dates: list[str] = field(default_factory=list)
    device_type: str = ""
    center_freq: float | None = None
    bandwidth: float | None = None
    nominal_sample_rate: float | None = None
    estimated_sample_rate: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        name: str,
        sample_count: int,
        modalities: Sequence[str] | None = None,
        users: list[int] | None = None,
        rooms: list[int] | None = None,
        collection_dates: list[str] | None = None,
        device_type: str = "",
        center_freq: float | None = None,
        bandwidth: float | None = None,
        nominal_sample_rate: float | None = None,
        estimated_sample_rate: float | None = None,
        extra: Mapping[str, Any] | None = None,
        **owner_metadata: Any,
    ) -> None:
        self.name = name
        self.sample_count = int(sample_count)
        self.modalities = _normalize_modalities(modalities)
        self.users = list(users or [])
        self.rooms = list(rooms or [])
        self.collection_dates = list(collection_dates or [])
        self.device_type = device_type
        self.center_freq = float(center_freq) if center_freq is not None else None
        self.bandwidth = float(bandwidth) if bandwidth is not None else None
        self.nominal_sample_rate = (
            float(nominal_sample_rate)
            if nominal_sample_rate is not None
            else None
        )
        self.estimated_sample_rate = (
            float(estimated_sample_rate)
            if estimated_sample_rate is not None
            else None
        )
        self.extra = dict(extra or {})
        self.extra.update(owner_metadata)

    def __str__(self) -> str:
        return (
            f"DatasetMetadata({self.name}, "
            f"samples={self.sample_count}, "
            f"users={len(self.users)}, "
            f"extra_keys={len(self.extra)})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sample_count": self.sample_count,
            "modalities": list(self.modalities),
            "users": self.users,
            "rooms": self.rooms,
            "collection_dates": self.collection_dates,
            "device_type": self.device_type,
            "center_freq": self.center_freq,
            "bandwidth": self.bandwidth,
            "nominal_sample_rate": self.nominal_sample_rate,
            "estimated_sample_rate": self.estimated_sample_rate,
            "extra": self.extra,
        }

    def describe_tree(self) -> DescribeNode:
        return DescribeNode(
            kind="dataset_metadata",
            name="dataset_metadata",
            fields=self.to_dict(),
        )


@dataclass(frozen=True)
class DatasetColumnSpec:
    """Describe one manifest field under a dataset-owned namespace."""

    name: str
    owner: str = "column"
    dtype: str | None = None
    required: bool = False
    description: str | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "owner": self.owner,
            "dtype": self.dtype,
            "required": self.required,
            "description": self.description,
            "extras": _sorted_mapping(self.extras),
        }


@dataclass(frozen=True)
class DatasetPayloadRef:
    """Reference payload materialization without owning IO decode."""

    name: str
    locator: str
    storage_kind: str = "file"
    checksum: str | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "locator": self.locator,
            "storage_kind": self.storage_kind,
            "checksum": self.checksum,
            "extras": _sorted_mapping(self.extras),
        }


@dataclass(frozen=True)
class DatasetRecord:
    """Canonical manifest record used by dataset views and fingerprints."""

    sample_id: str
    coordinates: Mapping[str, Any] = field(default_factory=dict)
    columns: Mapping[str, Any] = field(default_factory=dict)
    payload_refs: Mapping[str, str] = field(default_factory=dict)
    partitions: Mapping[str, str] = field(default_factory=dict)
    groups: Mapping[str, Any] = field(default_factory=dict)
    extras: Mapping[str, Any] = field(default_factory=dict)

    def metadata_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {"sample_id": self.sample_id}
        row.update(_sorted_mapping(self.coordinates))
        row.update(_sorted_mapping(self.columns))
        row.update(_sorted_mapping(self.groups))
        for name, value in sorted(self.partitions.items()):
            row[f"split/{name}"] = value
        return row

    def to_manifest_row(self) -> dict[str, Any]:
        row = self.metadata_row()
        row["coordinates"] = _sorted_mapping(self.coordinates)
        row["columns"] = _sorted_mapping(self.columns)
        row["payload_refs"] = _sorted_mapping(self.payload_refs)
        row["partitions"] = _sorted_mapping(self.partitions)
        row["groups"] = _sorted_mapping(self.groups)
        if self.extras:
            row["extras"] = _sorted_mapping(self.extras)
        return row


@dataclass(frozen=True)
class DatasetSplitScheme:
    """Describe one named split scheme over manifest metadata."""

    name: str
    partitions: tuple[str, ...] = ("train", "val", "test")
    group_fields: tuple[str, ...] = ()
    leakage_fields: tuple[str, ...] = ()
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "partitions": list(self.partitions),
            "group_fields": list(self.group_fields),
            "leakage_fields": list(self.leakage_fields),
            "description": self.description,
        }


@dataclass(frozen=True)
class DatasetManifestSchema:
    """Machine-readable summary of manifest field ownership."""

    dataset_id: str | None = None
    variant: str | None = None
    coordinates: tuple[DatasetColumnSpec, ...] = ()
    columns: tuple[DatasetColumnSpec, ...] = ()
    payload_refs: tuple[DatasetColumnSpec, ...] = ()
    split_schemes: tuple[DatasetSplitScheme, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "variant": self.variant,
            "coordinates": [item.to_dict() for item in self.coordinates],
            "columns": [item.to_dict() for item in self.columns],
            "payload_refs": [item.to_dict() for item in self.payload_refs],
            "split_schemes": [item.to_dict() for item in self.split_schemes],
        }


__all__ = [
    "DatasetMetadata",
    "DatasetColumnSpec",
    "DatasetManifestSchema",
    "DatasetPayloadRef",
    "DatasetRecord",
    "DatasetSplitScheme",
]
