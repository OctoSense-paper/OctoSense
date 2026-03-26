"""Canonical gait task definitions."""

from __future__ import annotations

from octosense.tasks.definitions import TaskSpec

GAIT_TASK_DEFINITIONS: dict[str, TaskSpec] = {
    "gait/identification": TaskSpec(
        task_id="gait/identification",
        kind="gait",
        display_name="Gait Identification",
    ),
}

__all__ = ["GAIT_TASK_DEFINITIONS"]
