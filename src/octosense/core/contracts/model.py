"""Shared model boundary contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class ModelInputContract:
    """Explicit assumptions about tensors consumed at model entry."""

    axes: tuple[str, ...]
    dtype_kind: str = "any"
    fixed_sizes: dict[str, int] = field(default_factory=dict)
    layout: str = ""
    channel_semantics: str = ""
    canonicalization_assumption: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "axes", tuple(self.axes))
        object.__setattr__(
            self,
            "fixed_sizes",
            {str(axis_name): int(size) for axis_name, size in self.fixed_sizes.items()},
        )
        valid_kinds = {"any", "real", "complex"}
        if self.dtype_kind not in valid_kinds:
            raise ValueError(
                f"Invalid dtype_kind '{self.dtype_kind}', expected one of {sorted(valid_kinds)}"
            )
        if not self.axes:
            raise ValueError("axes must not be empty")
        if len(self.axes) != len(set(self.axes)):
            raise ValueError(f"axes contains duplicates: {self.axes}")
        unknown_axes = set(self.fixed_sizes) - set(self.axes)
        if unknown_axes:
            raise ValueError(f"fixed_sizes references unknown axes: {sorted(unknown_axes)}")

    @property
    def batched_axes(self) -> tuple[str, ...]:
        return ("batch", *self.axes)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["axes"] = self.axes
        return payload


@dataclass(frozen=True)
class ModelOutputContract:
    """Explicit assumptions about the task-level outputs emitted by a model."""

    kind: str
    required_keys: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "required_keys", tuple(self.required_keys))
        valid_kinds = {"classification_logits", "structured_dict"}
        if self.kind not in valid_kinds:
            raise ValueError(
                f"Invalid output contract kind '{self.kind}', expected one of {sorted(valid_kinds)}"
            )
        if len(self.required_keys) != len(set(self.required_keys)):
            raise ValueError(f"required_keys contains duplicates: {self.required_keys}")
        if self.kind != "structured_dict" and self.required_keys:
            raise ValueError("required_keys is only valid for structured_dict outputs")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


__all__ = ["ModelInputContract", "ModelOutputContract"]
