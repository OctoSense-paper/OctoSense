"""Canonical respiration task definitions."""

from __future__ import annotations

from octosense.tasks.definitions import TaskSpec

RESPIRATION_TASK_DEFINITIONS: dict[str, TaskSpec] = {
    "respiration/rate": TaskSpec(
        task_id="respiration/rate",
        kind="respiration",
        display_name="Respiration Estimation",
    ),
}

__all__ = ["RESPIRATION_TASK_DEFINITIONS"]
