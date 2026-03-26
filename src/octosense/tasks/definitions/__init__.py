"""Canonical task definitions and task-owned semantic schema types."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from octosense.core import TaskOutputSpec

TaskKind = Literal["classification", "pose", "localization", "respiration", "gait"]
LabelSchemaKind = Literal["categorical", "structured", "continuous"]


def _freeze_strings(values: Sequence[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    return tuple(str(value) for value in values)


def _require_declared_task_id(task_id: str, *, kind: TaskKind | None = None) -> str:
    if not isinstance(task_id, str):
        raise TypeError("Task identity must be provided as a canonical task_id string.")
    if not task_id:
        raise ValueError("Task identity must be a non-empty canonical task id.")
    if kind is not None and not task_id.startswith(f"{kind}/"):
        raise ValueError(
            f"Task '{task_id}' does not match declared kind '{kind}'."
        )
    return task_id


@dataclass(frozen=True)
class LabelSpaceSpec:
    """Task-owned semantic label namespace.

    Concrete dataset-local label ids, names, vocabularies, and payload fields
    are owned by dataset task bindings, not by ``octosense.tasks``.
    """

    namespace: str
    schema_kind: LabelSchemaKind
    description: str = ""


@dataclass(frozen=True)
class TargetSchemaSpec:
    """Task-owned semantic target contract.

    ``fields`` and ``shape`` describe canonical semantic slots only. Dataset
    task bindings own concrete metadata columns, payload fields, and tensor
    shapes used to instantiate these semantics.
    """

    target_kind: str
    fields: tuple[str, ...] = ()
    shape: tuple[str | int, ...] = ()
    description: str = ""


@dataclass(frozen=True, init=False)
class TaskSpec:
    """Canonical task schema owned by ``octosense.tasks``."""

    task_id: str
    kind: TaskKind
    label_space: LabelSpaceSpec
    target_schema: TargetSchemaSpec
    output_schema: TaskOutputSpec
    default_metrics: tuple[str, ...]
    display_name: str

    def __init__(
        self,
        *,
        task_id: str,
        kind: TaskKind,
        label_space: LabelSpaceSpec | None = None,
        target_schema: TargetSchemaSpec | None = None,
        output_schema: TaskOutputSpec | None = None,
        default_metrics: Sequence[str] | None = None,
        display_name: str = "",
    ) -> None:
        from octosense.tasks.label_spaces import get_label_space
        from octosense.tasks.outputs import get_target_schema, get_task_output_spec

        resolved_task_id = _require_declared_task_id(task_id, kind=kind)
        resolved_label_space = label_space or get_label_space(resolved_task_id)
        resolved_target_schema = target_schema or get_target_schema(resolved_task_id)
        resolved_output_schema = output_schema or get_task_output_spec(resolved_task_id)
        resolved_metrics = default_metrics or (resolved_output_schema.primary_metric,)

        object.__setattr__(self, "task_id", resolved_task_id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "label_space", resolved_label_space)
        object.__setattr__(self, "target_schema", resolved_target_schema)
        object.__setattr__(self, "output_schema", resolved_output_schema)
        object.__setattr__(self, "default_metrics", _freeze_strings(resolved_metrics))
        object.__setattr__(self, "display_name", display_name or resolved_task_id)

    @property
    def owner_scope(self) -> tuple[str, ...]:
        return (
            "task_id",
            "label_space",
            "target_schema",
            "output_schema",
            "default_metrics",
        )


from octosense.tasks.definitions.classification import CLASSIFICATION_TASKS
from octosense.tasks.definitions.gait import GAIT_TASK_DEFINITIONS
from octosense.tasks.definitions.localization import LOCALIZATION_TASK_DEFINITIONS
from octosense.tasks.definitions.pose import POSE_TASK_DEFINITIONS
from octosense.tasks.definitions.respiration import RESPIRATION_TASK_DEFINITIONS

TASK_DEFINITIONS: dict[str, TaskSpec] = {
    **CLASSIFICATION_TASKS,
    **POSE_TASK_DEFINITIONS,
    **LOCALIZATION_TASK_DEFINITIONS,
    **RESPIRATION_TASK_DEFINITIONS,
    **GAIT_TASK_DEFINITIONS,
}

TASK_IDS: tuple[str, ...] = tuple(sorted(TASK_DEFINITIONS))


def load(task_id: str) -> TaskSpec:
    """Load a canonical task schema by canonical semantic task id."""

    resolved = _require_declared_task_id(task_id)
    if resolved not in TASK_DEFINITIONS:
        supported = ", ".join(sorted(TASK_DEFINITIONS))
        raise ValueError(
            f"Unsupported canonical task identity '{task_id}'. "
            f"Supported canonical ids: {supported}"
        )
    return TASK_DEFINITIONS[resolved]


__all__ = ["TaskSpec", "load"]
