"""Dataset-owned manifest normalization and metadata-first indexing."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Mapping

from octosense.datasets.core.schema import DatasetManifestSchema, DatasetRecord

_RESERVED_ROW_KEYS = {
    "sample_id",
    "coordinates",
    "columns",
    "payload_refs",
    "partitions",
    "groups",
    "extras",
}


def _canonical_partition_field_name(field_name: object) -> str:
    return str(field_name).strip()


def _as_mapping(row: object) -> Mapping[str, Any]:
    if isinstance(row, Mapping):
        return row
    to_dict = getattr(row, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return payload
    if is_dataclass(row):
        payload = asdict(row)
        if isinstance(payload, Mapping):
            return payload
    if hasattr(row, "__dict__"):
        payload = vars(row)
        if isinstance(payload, Mapping):
            return payload
    raise TypeError(f"Unsupported manifest row type: {type(row)!r}")


def _mapping_of_strings(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    return {str(key): value for key, value in payload.items()}


def _resolved_manifest_schema(
    schema: DatasetManifestSchema | None,
    *,
    dataset_id: str | None,
    variant: str | None,
) -> DatasetManifestSchema:
    if schema is None:
        return DatasetManifestSchema(dataset_id=dataset_id, variant=variant)
    resolved_dataset_id = schema.dataset_id if schema.dataset_id is not None else dataset_id
    resolved_variant = schema.variant if schema.variant is not None else variant
    if resolved_dataset_id == schema.dataset_id and resolved_variant == schema.variant:
        return schema
    return DatasetManifestSchema(
        dataset_id=resolved_dataset_id,
        variant=resolved_variant,
        coordinates=schema.coordinates,
        columns=schema.columns,
        payload_refs=schema.payload_refs,
        split_schemes=schema.split_schemes,
    )


def _coerce_dataset_record(row: object) -> DatasetRecord:
    if isinstance(row, DatasetRecord):
        return row

    payload = dict(_as_mapping(row))
    coordinates = _mapping_of_strings(payload.get("coordinates"))
    columns = _mapping_of_strings(payload.get("columns"))
    payload_refs = {
        str(key): str(value)
        for key, value in _mapping_of_strings(payload.get("payload_refs")).items()
        if value is not None
    }
    groups = _mapping_of_strings(payload.get("groups"))
    extras = _mapping_of_strings(payload.get("extras"))

    partitions = {
        _canonical_partition_field_name(key): str(value)
        for key, value in _mapping_of_strings(payload.get("partitions")).items()
        if value is not None and _canonical_partition_field_name(key) != ""
    }

    flat_columns = {
        str(key): value
        for key, value in payload.items()
        if key not in _RESERVED_ROW_KEYS
    }
    for key, value in flat_columns.items():
        columns.setdefault(key, value)

    sample_id = payload.get("sample_id")
    if sample_id in {None, ""}:
        raise ValueError("Manifest rows must define explicit sample_id")

    return DatasetRecord(
        sample_id=str(sample_id),
        coordinates=coordinates,
        columns=columns,
        payload_refs=payload_refs,
        partitions=partitions,
        groups=groups,
        extras=extras,
    )


@dataclass(frozen=True)
class DatasetManifest:
    """Collection-level metadata index that powers DatasetView operations."""

    records: tuple[DatasetRecord, ...]
    schema: DatasetManifestSchema
    dataset_id: str | None = None
    variant: str | None = None

    @classmethod
    def from_rows(
        cls,
        rows: list[object],
        *,
        schema: DatasetManifestSchema | None = None,
        dataset_id: str | None = None,
        variant: str | None = None,
    ) -> "DatasetManifest":
        records = tuple(_coerce_dataset_record(row) for row in rows)
        resolved_schema = _resolved_manifest_schema(
            schema,
            dataset_id=dataset_id,
            variant=variant,
        )
        return cls(
            records=records,
            schema=resolved_schema,
            dataset_id=dataset_id,
            variant=variant,
        )

    def metadata_rows(self) -> list[dict[str, Any]]:
        return [copy.deepcopy(record.metadata_row()) for record in self.records]

    def manifest_rows(self) -> list[dict[str, Any]]:
        return [copy.deepcopy(record.to_manifest_row()) for record in self.records]

    def select_positions(self, positions: list[int]) -> "DatasetManifest":
        selected_records = tuple(self.records[position] for position in positions)
        return DatasetManifest(
            records=selected_records,
            schema=_resolved_manifest_schema(
                self.schema,
                dataset_id=self.dataset_id,
                variant=self.variant,
            ),
            dataset_id=self.dataset_id,
            variant=self.variant,
        )

    def split_positions(
        self,
        split_name: str,
        *,
        split_field: str | None = None,
        candidate_fields: tuple[str, ...] = ("split",),
    ) -> list[int]:
        raw_fields = [split_field] if split_field is not None else list(candidate_fields)
        search_fields = [
            _canonical_partition_field_name(field_name)
            for field_name in raw_fields
            if field_name is not None
        ]
        positions: list[int] = []
        for index, record in enumerate(self.records):
            partition_value = None
            for field_name in search_fields:
                partition_value = record.partitions.get(field_name)
                if partition_value is not None:
                    break
            if partition_value == split_name:
                positions.append(index)
        return positions

    def overlap(
        self,
        other: "DatasetManifest",
        *,
        fields: tuple[str, ...],
    ) -> dict[str, list[object]]:
        left_rows = self.metadata_rows()
        right_rows = other.metadata_rows()
        overlaps: dict[str, list[object]] = {}
        for field in fields:
            left_values = {
                row[field]
                for row in left_rows
                if field in row and row[field] is not None
            }
            right_values = {
                row[field]
                for row in right_rows
                if field in row and row[field] is not None
            }
            overlaps[field] = sorted(left_values & right_values, key=lambda value: str(value))
        return overlaps


def manifest_from_rows(
    rows: list[object],
    *,
    schema: DatasetManifestSchema | None = None,
    dataset_id: str | None = None,
    variant: str | None = None,
) -> DatasetManifest:
    return DatasetManifest.from_rows(
        rows,
        schema=schema,
        dataset_id=dataset_id,
        variant=variant,
    )


__all__ = ["DatasetManifest", "manifest_from_rows"]
