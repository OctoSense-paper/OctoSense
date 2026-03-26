"""Canonical task semantic target/output contracts."""

from __future__ import annotations

from octosense.core import TaskOutputSpec
from octosense.tasks.definitions import TargetSchemaSpec
from octosense.tasks.label_spaces import _require_canonical_task_id


TASK_TARGET_SCHEMAS: dict[str, TargetSchemaSpec] = {
    "classification/gesture": TargetSchemaSpec(
        target_kind="categorical_label",
        fields=("class_label",),
        shape=(),
        description=(
            "Single gesture-class semantic label. Dataset task bindings own the "
            "concrete id/name/value fields that materialize this slot."
        ),
    ),
    "classification/activity": TargetSchemaSpec(
        target_kind="categorical_label",
        fields=("class_label",),
        shape=(),
        description=(
            "Single activity-class semantic label. Dataset task bindings own the "
            "concrete activity/action field names."
        ),
    ),
    "pose/human_pose": TargetSchemaSpec(
        target_kind="structured_pose",
        fields=("pose_keypoints", "pose_extent"),
        shape=("joint_axis", "coordinate_axis"),
        description=(
            "Human-pose semantic target slots. Dataset task bindings own the "
            "concrete joint payload fields and exact tensor layout."
        ),
    ),
    "localization/position": TargetSchemaSpec(
        target_kind="coordinates",
        fields=("position",),
        shape=("coordinate_axis",),
        description=(
            "Continuous position semantic target. Dataset task bindings own the "
            "concrete coordinate field names and dimensionality source."
        ),
    ),
    "respiration/rate": TargetSchemaSpec(
        target_kind="respiration_signal",
        fields=("respiration_signal",),
        shape=("time_axis",),
        description=(
            "Respiration semantic target. Dataset task bindings own whether the "
            "instantiation exposes rate, waveform, or both."
        ),
    ),
    "gait/identification": TargetSchemaSpec(
        target_kind="categorical_label",
        fields=("identity_label",),
        shape=(),
        description=(
            "Identity-class semantic label. Dataset task bindings own the "
            "concrete subject/track metadata fields."
        ),
    ),
}

TASK_OUTPUT_SPECS: dict[str, TaskOutputSpec] = {
    "classification/gesture": TaskOutputSpec(
        task_type="classification",
        output_kind="logits",
        primary_metric="accuracy",
    ),
    "classification/activity": TaskOutputSpec(
        task_type="classification",
        output_kind="logits",
        primary_metric="accuracy",
    ),
    "pose/human_pose": TaskOutputSpec(
        task_type="pose",
        output_kind="joint_coordinates",
        primary_metric="mpjpe",
    ),
    "localization/position": TaskOutputSpec(
        task_type="localization",
        output_kind="coordinates",
        primary_metric="mae",
    ),
    "respiration/rate": TaskOutputSpec(
        task_type="respiration",
        output_kind="waveform",
        primary_metric="rmse",
    ),
    "gait/identification": TaskOutputSpec(
        task_type="gait",
        output_kind="identity_logits",
        primary_metric="accuracy",
    ),
}

TASK_OUTPUT_TASK_IDS: tuple[str, ...] = tuple(sorted(TASK_OUTPUT_SPECS))
TASK_TARGET_SCHEMA_IDS: tuple[str, ...] = tuple(sorted(TASK_TARGET_SCHEMAS))


def get_target_schema(task_id: str) -> TargetSchemaSpec:
    return TASK_TARGET_SCHEMAS[_require_canonical_task_id(task_id)]


def get_task_output_spec(task_id: str) -> TaskOutputSpec:
    return TASK_OUTPUT_SPECS[_require_canonical_task_id(task_id)]


__all__ = [
    "TASK_OUTPUT_SPECS",
    "TASK_OUTPUT_TASK_IDS",
    "TASK_TARGET_SCHEMAS",
    "TASK_TARGET_SCHEMA_IDS",
    "get_target_schema",
    "get_task_output_spec",
]
