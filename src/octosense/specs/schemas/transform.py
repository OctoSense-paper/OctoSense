"""Transform schemas owned by ``octosense.specs``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TransformStepSpec:
    """One explicit operator entry in a transform pipeline."""

    operator_id: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "operator_id": self.operator_id,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object] | None) -> "TransformStepSpec":
        if payload is None:
            return cls()
        params = payload.get("params", {})
        if not isinstance(params, dict):
            raise TypeError("TransformStepSpec.params must be a mapping")
        return cls(
            operator_id=str(payload.get("operator_id", "") or ""),
            params=dict(params),
        )


@dataclass(slots=True)
class TransformSpec:
    """Declarative transform selection for a benchmark run."""

    preset_id: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    steps: list[TransformStepSpec] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "preset_id": self.preset_id,
            "params": dict(self.params),
            "steps": [step.to_dict() for step in self.steps],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object] | None) -> "TransformSpec":
        if payload is None:
            return cls()
        params = payload.get("params", {})
        steps = payload.get("steps", [])
        if not isinstance(params, dict):
            raise TypeError("TransformSpec.params must be a mapping")
        if not isinstance(steps, list):
            raise TypeError("TransformSpec.steps must be a list")
        return cls(
            preset_id=_optional_str(payload.get("preset_id")),
            params=dict(params),
            steps=[TransformStepSpec.from_dict(step) for step in steps],
        )


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


__all__ = ["TransformSpec", "TransformStepSpec"]
