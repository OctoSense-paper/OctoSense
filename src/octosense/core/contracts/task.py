"""Shared task/output contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TaskOutputSpec:
    """Stable task-level output semantics shared across owners."""

    task_type: str
    output_kind: str
    primary_metric: str

    def __post_init__(self) -> None:
        if not self.task_type:
            raise ValueError("task_type cannot be empty")
        if not self.output_kind:
            raise ValueError("output_kind cannot be empty")
        if not self.primary_metric:
            raise ValueError("primary_metric cannot be empty")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


__all__ = ["TaskOutputSpec"]
