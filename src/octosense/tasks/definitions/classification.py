"""Canonical classification task definitions."""

from __future__ import annotations

from octosense.tasks.definitions import TaskSpec

CLASSIFICATION_TASKS: dict[str, TaskSpec] = {
    "classification/gesture": TaskSpec(
        task_id="classification/gesture",
        kind="classification",
        display_name="Gesture Classification",
    ),
    "classification/activity": TaskSpec(
        task_id="classification/activity",
        kind="classification",
        display_name="Activity Classification",
    ),
}

__all__ = ["CLASSIFICATION_TASKS"]
