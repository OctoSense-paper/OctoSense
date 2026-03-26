"""Shared transform boundary contracts."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TransformContract:
    """Declarative transform requires/updates contract."""

    requires: dict[str, object] = field(default_factory=dict)
    updates: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "requires", {str(key): value for key, value in self.requires.items()})
        object.__setattr__(self, "updates", {str(key): value for key, value in self.updates.items()})

    def to_dict(self) -> dict[str, dict[str, object]]:
        return {
            "requires": dict(self.requires),
            "updates": dict(self.updates),
        }


__all__ = ["TransformContract"]
