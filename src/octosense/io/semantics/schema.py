"""Canonical IO-owned semantic axis schemas."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from octosense.core.describe import Describable, DescribeNode
from octosense.io.semantics.normalizer import (
    ResolvedBindingTarget,
    ensure_lower_snake_identifier,
    resolve_semantic_entry,
)

__all__ = ["AxisMetadata", "AxisSchema", "MetadataSchema", "build_axis_schema"]


@dataclass(frozen=True)
class AxisMetadata:
    """Metadata describing one tensor axis."""

    name: str
    unit: str | None = None
    description: str = ""
    semantic_id: str | None = None
    axis_role: str | None = None
    code: str | None = None
    kind: str | None = None
    category: str | None = None
    status: str | None = None


@dataclass(frozen=True)
class AxisSchema(Describable):
    """Ordered tensor-axis semantics owned by ``octosense.io``."""

    axes: tuple[str, ...]
    axis_metadata: dict[str, AxisMetadata] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "axes", tuple(self.axes))
        if len(self.axes) != len(set(self.axes)):
            duplicates = [axis for axis in self.axes if self.axes.count(axis) > 1]
            raise ValueError(f"Duplicate axis names: {duplicates}")
        extra_keys = set(self.axis_metadata) - set(self.axes)
        if extra_keys:
            raise ValueError(f"Metadata keys not in axes: {extra_keys}")

    def has_axis(self, name: str) -> bool:
        return self._resolve_axis_name(name) is not None

    def index(self, name: str) -> int:
        resolved_name = self._resolve_axis_name(name)
        if resolved_name is not None:
            return self.axes.index(resolved_name)
        try:
            return self.axes.index(name)
        except ValueError:
            suggestion = self.suggest_axis_name(name)
            if suggestion is not None:
                raise ValueError(
                    f"Axis '{name}' not found in schema. Use canonical axis '{suggestion}' instead. "
                    f"Available axes: {self.axes}"
                ) from None
            raise ValueError(
                f"Axis '{name}' not found in schema. Available axes: {self.axes}"
            )

    def get_metadata(self, name: str) -> AxisMetadata | None:
        resolved_name = self._resolve_axis_name(name)
        if resolved_name is None:
            return None
        return self.axis_metadata.get(resolved_name)

    def __len__(self) -> int:
        return len(self.axes)

    def to_dict(self) -> dict[str, object]:
        return {
            "axes": list(self.axes),
            "axis_metadata": {
                key: {
                    "name": value.name,
                    "unit": value.unit,
                    "description": value.description,
                    "semantic_id": value.semantic_id,
                    "axis_role": value.axis_role,
                    "code": value.code,
                    "kind": value.kind,
                    "category": value.category,
                    "status": value.status,
                }
                for key, value in self.axis_metadata.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "AxisSchema":
        axes = tuple(payload["axes"])
        axis_metadata = {
            key: AxisMetadata(
                name=value["name"],
                unit=value["unit"],
                description=value["description"],
                semantic_id=value.get("semantic_id"),
                axis_role=value.get("axis_role"),
                code=value.get("code"),
                kind=value.get("kind"),
                category=value.get("category"),
                status=value.get("status"),
            )
            for key, value in payload.get("axis_metadata", {}).items()
        }
        return cls(axes=axes, axis_metadata=axis_metadata)

    def suggest_axis_name(self, typo: str) -> str | None:
        if not self.axes:
            return None

        typo_lower = typo.lower()
        synonyms = {
            "transmit": "tx",
            "transmitter": "tx",
            "receive": "rx",
            "receiver": "rx",
            "subcarrier": "subc",
            "frequency": "freq",
            "antenna": "ant",
        }
        if typo_lower in synonyms and synonyms[typo_lower] in self.axes:
            return synonyms[typo_lower]

        def levenshtein_distance(s1: str, s2: str) -> int:
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            if len(s2) == 0:
                return len(s1)
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            return previous_row[-1]

        distances = [(axis, levenshtein_distance(typo_lower, axis.lower())) for axis in self.axes]
        distances.sort(key=lambda item: item[1])
        closest_axis, min_distance = distances[0]
        threshold = max(2, max(len(typo), len(closest_axis)) * 0.6)
        if min_distance <= threshold:
            return closest_axis
        return None

    def describe_tree(self) -> DescribeNode:
        return DescribeNode(
            kind="axis_schema",
            name="axis_schema",
            fields={"axes": list(self.axes)},
            children=tuple(
                DescribeNode(
                    kind="axis",
                    name=axis_name,
                    fields={
                        key: value
                        for key, value in {
                            "unit": metadata.unit,
                            "description": metadata.description or None,
                            "semantic_id": metadata.semantic_id,
                            "axis_role": metadata.axis_role,
                            "code": metadata.code,
                            "category": metadata.category,
                            "status": metadata.status,
                        }.items()
                        if value is not None
                    },
                )
                for axis_name in self.axes
                for metadata in [self.axis_metadata.get(axis_name)]
                if metadata is not None
            ),
        )

    def _resolve_axis_name(self, name: str) -> str | None:
        if name in self.axes:
            return name
        for axis_name in self.axes:
            metadata = self.axis_metadata.get(axis_name)
            if metadata is not None and metadata.semantic_id == name:
                return axis_name
        return None


@dataclass(frozen=True)
class MetadataSchema:
    """Required metadata-field set for one semantic signal family."""

    required_physical: list[str] = field(default_factory=list)
    optional_physical: list[str] = field(default_factory=list)
    required_coords: list[str] = field(default_factory=list)
    required_provenance: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "required_physical": list(self.required_physical),
            "optional_physical": list(self.optional_physical),
            "required_coords": list(self.required_coords),
            "required_provenance": list(self.required_provenance),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "MetadataSchema":
        return cls(
            required_physical=list(payload.get("required_physical", [])),
            optional_physical=list(payload.get("optional_physical", [])),
            required_coords=list(payload.get("required_coords", [])),
            required_provenance=list(payload.get("required_provenance", [])),
        )


def build_axis_schema(
    axes: tuple[str | ResolvedBindingTarget, ...] | list[str | ResolvedBindingTarget],
    *,
    semantic_registry: Mapping[str, Any],
    aliases: Mapping[str, str] | None = None,
) -> AxisSchema:
    del aliases
    axis_entries = {
        entry.semantic_id: entry for entry in semantic_registry.values() if getattr(entry, "kind", None) == "axis"
    }

    resolved_axes: list[tuple[str, Any]] = []
    for axis in axes:
        if isinstance(axis, ResolvedBindingTarget):
            semantic_entry = axis_entries.get(axis.semantic_id)
            if semantic_entry is None:
                resolved_axes.append((axis.exported_name, None))
                continue
            entry = semantic_entry
        else:
            raw_axis = str(axis).strip()
            semantic_entry = axis_entries.get(raw_axis)
            if semantic_entry is None:
                semantic_entry = _resolve_axis_entry(raw_axis, semantic_registry=semantic_registry)
            if semantic_entry is None:
                canonical_name = ensure_lower_snake_identifier(
                    raw_axis,
                    label="axis schema axis",
                )
                resolved_axes.append((canonical_name, None))
                continue
            entry = semantic_entry
        resolved_axes.append((entry.preferred_name, entry))

    canonical_axes = tuple(axis_name for axis_name, _ in resolved_axes)
    axis_metadata = {
        axis_name: AxisMetadata(
            name=axis_name,
            unit=entry.unit,
            description=entry.description or "",
            semantic_id=entry.semantic_id,
            axis_role=_infer_axis_role(entry),
            code=entry.code,
            kind=entry.kind,
            category=entry.category,
            status=entry.status,
        )
        for axis_name, entry in resolved_axes
        if entry is not None
    }
    return AxisSchema(
        axes=canonical_axes,
        axis_metadata=axis_metadata,
    )


def _resolve_axis_entry(
    value: str,
    *,
    semantic_registry: Mapping[str, Any],
) -> Any | None:
    return resolve_semantic_entry(
        str(value).strip(),
        semantic_registry=semantic_registry,
        kind="axis",
    )


def _infer_axis_role(entry: Any) -> str | None:
    category = getattr(entry, "category", None)
    semantic_id = getattr(entry, "semantic_id", None)
    if semantic_id == "octo.common.axis.time":
        return "temporal"
    if category in {"temporal", "time_sample", "waveform_sample"}:
        return "temporal"
    if category in {"frequency_bin", "delay_bin"}:
        return "spectral"
    if category in {"sensor_index", "source_index"}:
        return "spatial"
    if category == "frame_index":
        return "layout"
    if category == "rf_channel":
        return "channel"
    if category == "radar_pulse":
        return "pulse"
    return category
