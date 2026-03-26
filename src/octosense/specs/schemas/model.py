"""Model schema owned by ``octosense.specs``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ModelSpec:
    """Declarative model selection for a benchmark run."""

    model_id: str = ""
    weights_id: str | None = None
    entry_overrides: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "weights_id": self.weights_id,
            "entry_overrides": dict(self.entry_overrides),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object] | None) -> "ModelSpec":
        if payload is None:
            return cls()
        entry_overrides = payload.get("entry_overrides", {})
        if not isinstance(entry_overrides, dict):
            raise TypeError("ModelSpec.entry_overrides must be a mapping")
        return cls(
            model_id=str(payload.get("model_id", "") or ""),
            weights_id=_optional_str(payload.get("weights_id")),
            entry_overrides=dict(entry_overrides),
        )


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


__all__ = ["ModelSpec"]
