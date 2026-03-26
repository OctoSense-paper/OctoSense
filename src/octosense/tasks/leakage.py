"""Task-level leakage ownership markers and dataset-binding gates.

Concrete leakage grouping keys are dataset-binding-owned. The task layer owns
the canonical rule that these keys must remain dataset-local declarations
rather than task-schema fields.
"""

from __future__ import annotations

from octosense.tasks.label_spaces import _require_canonical_task_id

TASK_LEAKAGE_OWNERS: dict[str, str] = {
    "classification/gesture": "dataset_binding",
    "classification/activity": "dataset_binding",
    "pose/human_pose": "dataset_binding",
    "localization/position": "dataset_binding",
    "respiration/rate": "dataset_binding",
    "gait/identification": "dataset_binding",
}


def get_leakage_owner(task_id: str) -> str:
    """Return the owner scope for leakage grouping declarations."""

    return TASK_LEAKAGE_OWNERS[_require_canonical_task_id(task_id)]


def resolve_dataset_leakage_keys(
    task_id: str,
    leakage_keys: list[str],
    *,
    owner: str,
) -> list[str]:
    """Normalize dataset-binding leakage keys for one canonical task."""

    leakage_owner = get_leakage_owner(task_id)
    if leakage_owner != "dataset_binding":
        raise ValueError(
            f"{owner} cannot declare leakage keys for canonical task_id {task_id!r}; "
            f"owner scope is {leakage_owner!r}."
        )

    normalized: list[str] = []
    seen: set[str] = set()
    for raw_key in leakage_keys:
        key = str(raw_key).strip()
        if not key:
            raise ValueError(f"{owner} leakage_keys entries must be non-empty strings.")
        if key in seen:
            continue
        seen.add(key)
        normalized.append(key)
    return normalized


__all__ = ["TASK_LEAKAGE_OWNERS", "get_leakage_owner", "resolve_dataset_leakage_keys"]
