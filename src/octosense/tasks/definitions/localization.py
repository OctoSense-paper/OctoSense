"""Canonical localization task definitions."""

from __future__ import annotations

from octosense.tasks.definitions import TaskSpec

LOCALIZATION_TASK_DEFINITIONS: dict[str, TaskSpec] = {
    "localization/position": TaskSpec(
        task_id="localization/position",
        kind="localization",
        display_name="Position Localization",
    ),
}

__all__ = ["LOCALIZATION_TASK_DEFINITIONS"]
