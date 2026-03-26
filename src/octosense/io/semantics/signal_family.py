"""Canonical IO-owned signal-family schema governance objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from octosense.io.semantics.schema import AxisSchema, MetadataSchema

__all__ = ["DTypePolicy", "SignalFamilySpec", "FamilyDiff"]


@dataclass(frozen=True)
class DTypePolicy:
    preferred_dtype: str
    policy_name: str
    description: str = ""


@dataclass(frozen=True)
class SignalFamilySpec:
    """Structured semantic family descriptor rooted in ``octosense.io``."""

    name: str
    version: str
    axis_schema: AxisSchema
    metadata_schema: MetadataSchema
    dtype_policy: DTypePolicy
    canonicalization: dict[str, Any] = field(default_factory=dict)
    contract_version: str = "1.0"
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("SignalFamilySpec name cannot be empty")
        if not self.version:
            raise ValueError("SignalFamilySpec version cannot be empty")

    def get_canonical_axis_name(self, alias: str) -> str:
        for axis_name, axis_canon in self.canonicalization.items():
            if isinstance(axis_canon, dict) and "aliases" in axis_canon:
                aliases = axis_canon["aliases"]
                if isinstance(aliases, list) and alias in aliases:
                    return axis_name
        return alias


@dataclass(frozen=True)
class FamilyDiff:
    added_axes: list[str] = field(default_factory=list)
    removed_axes: list[str] = field(default_factory=list)
    reordered_axes: list[str] = field(default_factory=list)
    changed_metadata: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    version_change: tuple[str, str] | None = None
    contract_version_change: tuple[str, str] | None = None
    breaking_change: bool = False

    def is_compatible(self) -> bool:
        return not self.breaking_change and not self.removed_axes
