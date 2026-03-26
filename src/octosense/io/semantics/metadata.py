"""Canonical IO-owned semantic metadata objects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from octosense.core.describe import Describable, DescribeNode

__all__ = ["CoordinateAxis", "RuntimeBindingRecord", "TransformRecord", "SignalMetadata"]


@dataclass
class CoordinateAxis(Describable):
    """Coordinate information attached to one semantic axis."""

    axis_name: str
    values: np.ndarray | None = None
    unit: str = ""

    def describe_tree(self) -> DescribeNode:
        return DescribeNode(
            kind="coord_axis",
            name=self.axis_name,
            fields={
                "available": self.values is not None,
                "length": int(len(self.values)) if self.values is not None else None,
                "unit": self.unit or None,
            },
        )


@dataclass
class TransformRecord:
    """One provenance record for a semantic transform application."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    consumed_axes: list[str] = field(default_factory=list)
    produced_axes: list[str] = field(default_factory=list)
    derivation: str = ""
    timestamp: float = 0.0


@dataclass
class RuntimeBindingRecord:
    """Runtime-facing semantic identity for one emitted artifact field."""

    exported_name: str
    semantic_id: str
    status: str | list[str] | None = None
    kind: str | None = None
    representation_id: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class SignalMetadata(Describable):
    """Semantic sample metadata owned by ``octosense.io``."""

    center_freq: float | None = None
    bandwidth: float | None = None
    sample_rate: float | None = None
    subcarrier_spacing: float | None = None
    chirp_period: float | None = None

    timestamp_start: float | None = None
    subcarrier_indices: list[int] = field(default_factory=list)
    coords: dict[str, CoordinateAxis] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    modality: str = ""
    reader_id: str = ""
    data_version: str = ""
    capture_device: str = ""
    transforms: list[TransformRecord] = field(default_factory=list)
    runtime_bindings: dict[str, RuntimeBindingRecord] = field(default_factory=dict)
    signal_runtime: RuntimeBindingRecord | None = None

    def __post_init__(self) -> None:
        for key, value in self.coords.items():
            if not isinstance(value, CoordinateAxis):
                raise TypeError(
                    f"metadata.coords['{key}'] must be CoordinateAxis, got {type(value)}. "
                    f"Use metadata.set_coord('{key}', values, unit) or move scalar values to "
                    f"metadata.extra['{key}'] instead."
                )

    def copy(self) -> "SignalMetadata":
        import copy

        return copy.deepcopy(self)

    def canonical_scalar_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        scalar_map = {
            "modality": self.modality,
            "center_freq": self.center_freq,
            "bandwidth": self.bandwidth,
            "sample_rate": self.sample_rate,
            "subcarrier_spacing": self.subcarrier_spacing,
            "timestamp": self.timestamp_start,
            "reader_id": self.reader_id,
            "capture_device": self.capture_device,
        }
        for key, value in scalar_map.items():
            if value in ("", None):
                continue
            payload[key] = value
        return payload

    def apply_canonical_scalar_payload(self, payload: Mapping[str, Any]) -> None:
        if "modality" in payload and payload["modality"] not in (None, ""):
            self.modality = str(payload["modality"])
        if "center_freq" in payload and payload["center_freq"] is not None:
            self.center_freq = float(payload["center_freq"])
        if "bandwidth" in payload and payload["bandwidth"] is not None:
            self.bandwidth = float(payload["bandwidth"])
        if "sample_rate" in payload and payload["sample_rate"] is not None:
            self.sample_rate = float(payload["sample_rate"])
        if "subcarrier_spacing" in payload and payload["subcarrier_spacing"] is not None:
            self.subcarrier_spacing = float(payload["subcarrier_spacing"])
        if "timestamp" in payload and payload["timestamp"] is not None:
            self.timestamp_start = float(payload["timestamp"])
        if "reader_id" in payload and payload["reader_id"] not in (None, ""):
            self.reader_id = str(payload["reader_id"])
        if "capture_device" in payload and payload["capture_device"] not in (None, ""):
            self.capture_device = str(payload["capture_device"])

    def add_transform(
        self,
        name: str,
        params: dict[str, Any] | None = None,
        timestamp: float | None = None,
        *,
        consumed_axes: list[str] | None = None,
        produced_axes: list[str] | None = None,
        derivation: str = "",
    ) -> None:
        import time

        self.transforms.append(
            TransformRecord(
                name=name,
                params=params or {},
                consumed_axes=consumed_axes or [],
                produced_axes=produced_axes or [],
                derivation=derivation,
                timestamp=timestamp or time.time(),
            )
        )

    def register_runtime_binding(
        self,
        exported_name: str,
        semantic_id: str,
        *,
        status: str | list[str] | None = None,
        kind: str | None = None,
        representation_id: str | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        record = self.runtime_bindings.get(exported_name)
        if record is not None and record.semantic_id != semantic_id:
            raise ValueError(
                f"runtime binding for {exported_name!r} already targets {record.semantic_id!r}, "
                f"cannot replace with {semantic_id!r}"
            )
        self.runtime_bindings[exported_name] = RuntimeBindingRecord(
            exported_name=exported_name,
            semantic_id=semantic_id,
            status=_merge_runtime_status(
                record.status if record is not None else None,
                status,
            ),
            kind=kind if kind is not None else (record.kind if record is not None else None),
            representation_id=(
                representation_id
                if representation_id is not None
                else (record.representation_id if record is not None else None)
            ),
            provenance=_merge_provenance(
                record.provenance if record is not None else {},
                provenance or {},
            ),
        )

    def set_signal_runtime_binding(
        self,
        semantic_id: str,
        *,
        exported_name: str | None = None,
        status: str | list[str] | None = None,
        representation_id: str | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        record = self.signal_runtime
        if record is not None and record.semantic_id != semantic_id:
            raise ValueError(
                f"signal runtime already targets {record.semantic_id!r}, "
                f"cannot replace with {semantic_id!r}"
            )
        self.signal_runtime = RuntimeBindingRecord(
            exported_name=(
                exported_name
                if exported_name is not None
                else (record.exported_name if record is not None else "signal")
            ),
            semantic_id=semantic_id,
            status=_merge_runtime_status(
                record.status if record is not None else None,
                status,
            ),
            kind="tensor",
            representation_id=(
                representation_id
                if representation_id is not None
                else (record.representation_id if record is not None else None)
            ),
            provenance=_merge_provenance(
                record.provenance if record is not None else {},
                provenance or {},
            ),
        )

    def apply_runtime_bridge(
        self,
        semantic_registry: Mapping[str, Any],
        *,
        binding_sources: Mapping[str, tuple[str, ...] | list[str]] | None = None,
        binding_statuses: Mapping[str, str | tuple[str, ...] | list[str]] | None = None,
        binding_provenance: Mapping[str, Mapping[str, Any]] | None = None,
        signal_semantic_id: str | None = None,
        signal_status: str | list[str] | None = None,
        signal_representation_id: str | None = None,
        signal_provenance: Mapping[str, Any] | None = None,
    ) -> None:
        signal_exported_name: str | None = None
        for entry in semantic_registry.values():
            kind = getattr(entry, "kind", None)
            if kind == "tensor":
                if signal_semantic_id and getattr(entry, "semantic_id", None) == signal_semantic_id:
                    signal_exported_name = getattr(entry, "preferred_name", None)
                continue
            preferred_name = getattr(entry, "preferred_name", None)
            semantic_id = getattr(entry, "semantic_id", None)
            if not preferred_name or not semantic_id:
                continue
            value = self._value_for_runtime_name(preferred_name)
            if not _has_runtime_value(value):
                continue
            provenance = dict(binding_provenance.get(preferred_name, {})) if binding_provenance else {}
            raw_sources = tuple(binding_sources.get(preferred_name, ())) if binding_sources else ()
            if raw_sources:
                provenance["source_fields"] = list(
                    dict.fromkeys(str(source) for source in raw_sources)
                )
            self.register_runtime_binding(
                preferred_name,
                semantic_id,
                status=_normalize_runtime_status(
                    binding_statuses.get(preferred_name) if binding_statuses else None
                ),
                kind=kind,
                representation_id=semantic_id if kind == "representation" else None,
                provenance=provenance,
            )

        if signal_semantic_id:
            self.set_signal_runtime_binding(
                signal_semantic_id,
                exported_name=signal_exported_name,
                status=signal_status,
                representation_id=signal_representation_id,
                provenance=signal_provenance,
            )

    def runtime_provenance_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.reader_id:
            payload["reader_id"] = self.reader_id
        if self.capture_device:
            payload["capture_device"] = self.capture_device
        if self.data_version:
            payload["data_version"] = self.data_version
        return payload

    def semantic_runtime_payload(self) -> dict[str, Any]:
        field_payload: dict[str, dict[str, Any]] = {}
        for exported_name, record in sorted(self.runtime_bindings.items(), key=lambda item: item[0]):
            field_payload[exported_name] = {
                "semantic_id": record.semantic_id,
                "status": record.status,
                "representation_id": record.representation_id,
                "kind": record.kind,
                "value": self._value_for_runtime_name(exported_name),
                "provenance": _merge_provenance(
                    self.runtime_provenance_payload(),
                    record.provenance,
                ),
            }

        payload = {
            "fields": field_payload,
            "provenance": self.runtime_provenance_payload(),
        }
        if self.signal_runtime is not None:
            representation_payload = {
                name: field_payload[name]
                for name, record in sorted(self.runtime_bindings.items(), key=lambda item: item[0])
                if record.kind == "representation"
            }
            payload["signal"] = {
                "exported_name": self.signal_runtime.exported_name,
                "semantic_id": self.signal_runtime.semantic_id,
                "status": self.signal_runtime.status,
                "representation_id": self.signal_runtime.representation_id,
                "provenance": _merge_provenance(
                    self.runtime_provenance_payload(),
                    self.signal_runtime.provenance,
                ),
                "representation": representation_payload,
            }
        return payload

    def get_coord(self, axis_name: str) -> CoordinateAxis | None:
        return self.coords.get(axis_name)

    def set_coord(self, axis_name: str, values: np.ndarray | None, unit: str = "") -> None:
        self.coords[axis_name] = CoordinateAxis(axis_name=axis_name, values=values, unit=unit)

    def describe_tree(self, *, include_coords: bool = True) -> DescribeNode:
        children: list[DescribeNode] = []
        if include_coords and self.coords:
            children.append(
                DescribeNode(
                    kind="coords",
                    name="coords",
                    children=tuple(
                        coord.describe_tree()
                        for _, coord in sorted(self.coords.items(), key=lambda item: item[0])
                    ),
                )
            )
        if self.transforms:
            children.append(
                DescribeNode(
                    kind="transform_provenance",
                    name="transforms",
                    fields={"count": len(self.transforms)},
                    children=tuple(
                        DescribeNode(
                            kind="transform_record",
                            name=record.name,
                            fields={
                                "params": dict(record.params),
                                "consumed_axes": list(record.consumed_axes),
                                "produced_axes": list(record.produced_axes),
                                "derivation": record.derivation or None,
                                "timestamp": float(record.timestamp),
                            },
                        )
                        for record in self.transforms
                    ),
                )
            )
        if self.runtime_bindings or self.signal_runtime is not None:
            runtime_children = []
            if self.signal_runtime is not None:
                runtime_children.append(
                    DescribeNode(
                        kind="runtime_signal_binding",
                        name="signal",
                        fields={
                            "exported_name": self.signal_runtime.exported_name,
                            "semantic_id": self.signal_runtime.semantic_id,
                            "status": self.signal_runtime.status,
                            "representation_id": self.signal_runtime.representation_id,
                            "provenance": dict(self.signal_runtime.provenance),
                        },
                    )
                )
            runtime_children.extend(
                DescribeNode(
                    kind="runtime_field_binding",
                    name=exported_name,
                    fields={
                        "semantic_id": record.semantic_id,
                        "status": record.status,
                        "kind": record.kind,
                        "representation_id": record.representation_id,
                        "provenance": dict(record.provenance),
                    },
                )
                for exported_name, record in sorted(self.runtime_bindings.items(), key=lambda item: item[0])
            )
            children.append(
                DescribeNode(
                    kind="runtime_semantics",
                    name="runtime_semantics",
                    fields={"count": len(self.runtime_bindings)},
                    children=tuple(runtime_children),
                )
            )
        fields = {
            "modality": self.modality or None,
            "reader_id": self.reader_id or None,
            "capture_device": self.capture_device or None,
            "data_version": self.data_version or None,
            "center_freq": float(self.center_freq) if self.center_freq is not None else None,
            "bandwidth": float(self.bandwidth) if self.bandwidth is not None else None,
            "sample_rate": float(self.sample_rate) if self.sample_rate is not None else None,
            "subcarrier_spacing": (
                float(self.subcarrier_spacing)
                if self.subcarrier_spacing is not None
                else None
            ),
            "chirp_period": float(self.chirp_period) if self.chirp_period is not None else None,
            "timestamp_start": (
                float(self.timestamp_start)
                if self.timestamp_start is not None
                else None
            ),
            "subcarrier_indices": list(self.subcarrier_indices) if self.subcarrier_indices else None,
            "extra": dict(self.extra) if self.extra else None,
        }
        return DescribeNode(
            kind="signal_metadata",
            name="metadata",
            fields={key: value for key, value in fields.items() if value is not None},
            children=tuple(children),
        )

    def _value_for_runtime_name(self, name: str) -> Any:
        scalar_map = {
            "modality": self.modality,
            "center_freq": self.center_freq,
            "bandwidth": self.bandwidth,
            "sample_rate": self.sample_rate,
            "subcarrier_spacing": self.subcarrier_spacing,
            "timestamp": self.timestamp_start,
            "reader_id": self.reader_id,
            "capture_device": self.capture_device,
        }
        if name in scalar_map:
            return scalar_map[name]
        return self.extra.get(name)


def _has_runtime_value(value: Any) -> bool:
    return value is not None and not (isinstance(value, str) and value == "")


def _merge_provenance(base: Mapping[str, Any], new: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in new.items():
        if key not in merged:
            merged[key] = value
            continue
        existing = merged[key]
        if existing == value:
            continue
        if key == "source_namespace":
            merged[key] = _merge_source_namespace(existing, value)
            continue
        if isinstance(existing, list):
            existing_values = list(existing)
        elif isinstance(existing, tuple):
            existing_values = list(existing)
        else:
            existing_values = [existing]
        if isinstance(value, list):
            new_values = list(value)
        elif isinstance(value, tuple):
            new_values = list(value)
        else:
            new_values = [value]
        merged[key] = list(dict.fromkeys(existing_values + new_values))
    return merged


def _normalize_runtime_status(
    value: str | tuple[str, ...] | list[str] | None,
) -> str | list[str] | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        return value
    unique_statuses = list(dict.fromkeys(str(item) for item in value if item not in (None, "")))
    if not unique_statuses:
        return None
    if len(unique_statuses) == 1:
        return unique_statuses[0]
    return unique_statuses


def _merge_runtime_status(
    base: str | list[str] | None,
    extra: str | list[str] | None,
) -> str | list[str] | None:
    return _normalize_runtime_status(
        [
            *(_status_items(base)),
            *(_status_items(extra)),
        ]
    )


def _status_items(value: str | list[str] | None) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value if item not in (None, "")]


def _merge_source_namespace(existing: Any, value: Any) -> str | list[str] | None:
    existing_values = _namespace_items(existing)
    new_values = _namespace_items(value)
    merged = list(dict.fromkeys(existing_values + new_values))
    if not merged:
        return None
    if len(merged) == 1:
        return merged[0]
    return merged


def _namespace_items(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)]
