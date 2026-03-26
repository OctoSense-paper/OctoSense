"""Task schema owned by ``octosense.specs``."""

from dataclasses import dataclass


@dataclass(slots=True)
class TaskSpec:
    """Declarative task selection for a benchmark run."""

    task_id: str = ""
    task_binding: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "task_binding": self.task_binding,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object] | None) -> "TaskSpec":
        if payload is None:
            return cls()
        return cls(
            task_id=str(payload.get("task_id", "") or ""),
            task_binding=str(payload.get("task_binding", "") or ""),
        )


__all__ = ["TaskSpec"]
