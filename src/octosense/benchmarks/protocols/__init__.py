"""Internal protocol package.

The only canonical public benchmark surface is ``octosense.benchmarks`` via
``evaluate(run)``. The canonical protocol-family catalog still lives here so
other owners can consume benchmark contracts without redefining them.
"""

from __future__ import annotations

from octosense.benchmarks.protocols._catalog import (
    canonical_execution_adapter_for_task_kind,
    canonical_protocol_for_protocol_id,
    canonical_protocol_for_task_kind,
    canonical_protocol_id_for_primary_metric,
    canonical_protocol_id_for_task_kind,
    canonical_target_key_for_task_kind,
    canonical_task_kinds_for_protocol_id,
    resolve_runtime_protocol_payload,
)

CLASSIFICATION_PROTOCOL = canonical_protocol_for_task_kind("classification")
LOCALIZATION_PROTOCOL = canonical_protocol_for_task_kind("localization")
POSE_PROTOCOL = canonical_protocol_for_task_kind("pose")
RESPIRATION_PROTOCOL = canonical_protocol_for_task_kind("respiration")

__all__ = [
    "CLASSIFICATION_PROTOCOL",
    "LOCALIZATION_PROTOCOL",
    "POSE_PROTOCOL",
    "RESPIRATION_PROTOCOL",
    "canonical_execution_adapter_for_task_kind",
    "canonical_protocol_for_protocol_id",
    "canonical_protocol_for_task_kind",
    "canonical_protocol_id_for_primary_metric",
    "canonical_protocol_id_for_task_kind",
    "canonical_target_key_for_task_kind",
    "canonical_task_kinds_for_protocol_id",
    "resolve_runtime_protocol_payload",
]
