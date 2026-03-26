"""Declarative axis contracts shared across modules."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class MetadataRequirement:
    """Metadata field requirement with layer binding.

    Attributes:
        field_name: Metadata field name
        layer: Metadata layer ('physical', 'coords', 'provenance')
        required: Whether field is required (vs optional)
    """

    field_name: str
    layer: str  # 'physical', 'coords', 'provenance'
    required: bool = True

    def __post_init__(self) -> None:
        """Validate layer name."""
        if not self.field_name:
            raise ValueError("field_name cannot be empty")
        valid_layers = {"physical", "coords", "provenance"}
        if self.layer not in valid_layers:
            raise ValueError(f"Invalid layer '{self.layer}', must be one of {valid_layers}")


@dataclass(frozen=True)
class AxisContract:
    """Declarative axis contract for a transform boundary.

    Attributes:
        required_axes: Axes that MUST be present in input
        forbidden_axes: Axes that MUST NOT be present in input
        output_axes: Axes produced by replacing required_axes (None = no change).
            When required_axes is empty, output_axes acts as a shorthand for
            context-bound replacements and must be resolved by the caller.
        add_axes: Axes added without replacing existing ones
        remove_axes: Axes removed from input
        dtype_constraint: Required dtype ('complex', 'real', None for any)
        required_metadata: Metadata fields required (with 3-layer binding)
        required_coord_units: Expected units for named coordinate axes
        required_extra_fields: Required scalar fields inside metadata.extra

    Example:
        >>> # FFT transform contract
        >>> fft_contract = AxisContract(
        ...     required_axes=['time'],
        ...     output_axes=['freq'],  # time → freq
        ...     dtype_constraint='complex',
        ...     required_metadata=[
        ...         MetadataRequirement('sample_rate', 'physical', required=True)
        ...     ],
        ...     required_coord_units={'time': 's'}
        ... )
    """

    required_axes: list[str] = field(default_factory=list)
    forbidden_axes: list[str] = field(default_factory=list)
    output_axes: list[str] | None = None
    add_axes: list[str] = field(default_factory=list)
    remove_axes: list[str] = field(default_factory=list)
    dtype_constraint: str | None = None  # 'complex', 'real', None
    required_metadata: list[MetadataRequirement] = field(default_factory=list)
    required_coord_units: dict[str, str] = field(default_factory=dict)
    required_extra_fields: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate contract consistency."""
        duplicate_buckets = {
            "required_axes": self.required_axes,
            "forbidden_axes": self.forbidden_axes,
            "add_axes": self.add_axes,
            "remove_axes": self.remove_axes,
        }
        for label, axes in duplicate_buckets.items():
            duplicates = sorted({axis for axis in axes if axes.count(axis) > 1})
            if duplicates:
                raise ValueError(f"{label} contains duplicate axis names: {duplicates}")

        overlap = set(self.required_axes) & set(self.forbidden_axes)
        if overlap:
            raise ValueError(
                f"Inconsistent contract: required_axes and forbidden_axes overlap: {sorted(overlap)}"
            )

        # Validate output_axes/remove_axes consistency.
        if self.output_axes and self.remove_axes:
            # If output_axes is provided, ensure it doesn't conflict with remove_axes.
            overlap = set(self.output_axes) & set(self.remove_axes)
            if overlap:
                raise ValueError(
                    f"Inconsistent contract: output_axes and remove_axes overlap: {overlap}"
                )
        if (
            self.output_axes is not None
            and self.required_axes
            and len(self.output_axes) != len(self.required_axes)
        ):
            raise ValueError(
                "output_axes must align one-to-one with required_axes; "
                f"got required_axes={self.required_axes}, output_axes={self.output_axes}"
            )

        # Validate dtype_constraint
        if self.dtype_constraint and self.dtype_constraint not in ("complex", "real"):
            raise ValueError(
                f"Invalid dtype_constraint '{self.dtype_constraint}', "
                f"must be 'complex', 'real', or None"
            )

    def to_requires_dict(self) -> dict[str, object]:
        """Render this contract as explicit preconditions."""

        payload: dict[str, object] = {}
        if self.required_axes:
            payload["axes"] = list(self.required_axes)
        if self.forbidden_axes:
            payload["forbidden_axes"] = list(self.forbidden_axes)
        if self.dtype_constraint is not None:
            payload["dtype"] = self.dtype_constraint
        if self.required_metadata:
            payload["metadata"] = [
                {
                    "field": requirement.field_name,
                    "layer": requirement.layer,
                    "required": requirement.required,
                }
                for requirement in self.required_metadata
            ]
        if self.required_coord_units:
            payload["coord_units"] = dict(self.required_coord_units)
        if self.required_extra_fields:
            payload["extra_fields"] = list(self.required_extra_fields)
        return payload

    def to_updates_dict(
        self,
        *,
        source_axes: Sequence[str] | None = None,
    ) -> dict[str, object]:
        """Render this contract as explicit postconditions / updates.

        ``output_axes`` can be declared as a shorthand without ``required_axes``.
        In that case, callers should pass ``source_axes`` from the surrounding
        transform context so the canonical ``axis_replacements`` mapping can be
        materialized when available.
        """

        payload: dict[str, object] = {}
        replacement_source = list(source_axes or self.required_axes)
        if self.output_axes:
            payload["output_axes"] = list(self.output_axes)
            if len(replacement_source) == len(self.output_axes):
                payload["axis_replacements"] = {
                    src: dst
                    for src, dst in zip(replacement_source, self.output_axes, strict=True)
                }
        if self.add_axes:
            payload["add_axes"] = list(self.add_axes)
        if self.remove_axes:
            payload["remove_axes"] = list(self.remove_axes)
        return payload

    def to_dict(self) -> dict[str, object]:
        """Render the contract as a stable serialization-friendly payload."""

        return {
            "requires": self.to_requires_dict(),
            "updates": self.to_updates_dict(),
        }


__all__ = ["AxisContract", "MetadataRequirement"]
