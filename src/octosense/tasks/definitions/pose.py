"""Canonical pose task definitions."""

from __future__ import annotations

from octosense.tasks.definitions import TaskSpec

POSE_TASK_DEFINITIONS: dict[str, TaskSpec] = {
    "pose/human_pose": TaskSpec(
        task_id="pose/human_pose",
        kind="pose",
        display_name="Human Pose Estimation",
    ),
}

__all__ = ["POSE_TASK_DEFINITIONS"]
