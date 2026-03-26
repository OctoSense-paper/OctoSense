"""Canonical task ids and task-owned semantic label-space contracts."""

from __future__ import annotations

from octosense.tasks.definitions import LabelSpaceSpec, TaskKind

TASK_LABEL_SPACES: dict[str, LabelSpaceSpec] = {
    "classification/gesture": LabelSpaceSpec(
        namespace="gesture",
        schema_kind="categorical",
        description=(
            "Gesture label semantics only. Dataset task bindings own concrete "
            "gesture vocabularies and metadata/payload fields."
        ),
    ),
    "classification/activity": LabelSpaceSpec(
        namespace="activity",
        schema_kind="categorical",
        description=(
            "Activity label semantics only. Dataset task bindings own concrete "
            "activity/action vocabularies and field names."
        ),
    ),
    "pose/human_pose": LabelSpaceSpec(
        namespace="human_pose",
        schema_kind="structured",
        description=(
            "Human-pose semantic namespace only. Dataset task bindings own "
            "joint vocabularies, ids, and payload field mappings."
        ),
    ),
    "localization/position": LabelSpaceSpec(
        namespace="position",
        schema_kind="continuous",
        description=(
            "Continuous position semantics only. Dataset task bindings own the "
            "concrete coordinate fields and identifiers."
        ),
    ),
    "respiration/rate": LabelSpaceSpec(
        namespace="respiration",
        schema_kind="continuous",
        description=(
            "Respiration semantics only. Dataset task bindings own concrete "
            "rate/waveform field names and payload layout."
        ),
    ),
    "gait/identification": LabelSpaceSpec(
        namespace="identity",
        schema_kind="categorical",
        description=(
            "Identity-label semantics only. Dataset task bindings own subject "
            "or track vocabularies and metadata field mappings."
        ),
    ),
}


def supported_task_ids() -> tuple[str, ...]:
    return tuple(sorted(TASK_LABEL_SPACES))


def _require_canonical_task_id(task_id: str, *, kind: TaskKind | None = None) -> str:
    if not isinstance(task_id, str):
        raise TypeError("Task identity must be provided as a canonical task_id string.")
    if not task_id:
        raise ValueError("Task identity must be a non-empty canonical task id.")
    if task_id not in TASK_LABEL_SPACES:
        supported = ", ".join(supported_task_ids())
        raise ValueError(
            f"Unsupported task identity '{task_id}'. "
            f"Expected a canonical task id. Supported canonical ids: {supported}"
        )
    if kind is not None and not task_id.startswith(f"{kind}/"):
        raise ValueError(
            f"Task '{task_id}' does not match declared kind '{kind}'."
        )
    return task_id


def get_label_space(task_id: str) -> LabelSpaceSpec:
    return TASK_LABEL_SPACES[_require_canonical_task_id(task_id)]


__all__ = [
    "TASK_LABEL_SPACES",
    "get_label_space",
    "supported_task_ids",
]
