"""Pipeline/kernel description helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from octosense.core import DescribeNode, ensure_describe_node


def describe_pipeline(pipeline: object) -> dict[str, object]:
    provider = getattr(pipeline, "describe_tree", None)
    if callable(provider):
        return ensure_describe_node(provider()).to_dict()
    return {"kind": type(pipeline).__name__}


def describe_tree(pipeline: object) -> DescribeNode:
    provider = getattr(pipeline, "describe_tree", None)
    if callable(provider):
        return ensure_describe_node(provider())
    return DescribeNode(kind="pipeline", name=type(pipeline).__name__)


def describe_execution_dataloaders(dataloaders: object) -> dict[str, object]:
    provider = getattr(dataloaders, "describe_tree", None)
    if callable(provider):
        return ensure_describe_node(provider()).to_dict()
    return {"kind": type(dataloaders).__name__}


def _summary_metrics_payload(result: Mapping[str, Any]) -> Mapping[str, Any] | None:
    metrics_payload = result.get("metrics")
    if not isinstance(metrics_payload, Mapping):
        return None
    nested_metrics = metrics_payload.get("metrics")
    if not isinstance(nested_metrics, Mapping):
        return None
    return nested_metrics


def summarize_execution_result(result: Mapping[str, Any]) -> dict[str, object]:
    summary: dict[str, object] = {}
    metrics_payload = _summary_metrics_payload(result)
    for key in (
        "device",
        "train_loss",
        "val_accuracy",
        "test_accuracy",
        "num_classes",
        "run_id",
        "run_name",
    ):
        if key in result:
            summary[key] = result[key]
        if metrics_payload is not None and key in metrics_payload:
            summary[key] = metrics_payload[key]
    if isinstance(result.get("metrics"), Mapping):
        metrics_envelope = result["metrics"]
        for key in ("primary_metric", "primary_metric_value"):
            if key in metrics_envelope:
                summary[key] = metrics_envelope[key]
    if "timing" in result and isinstance(result["timing"], Mapping):
        timing_payload = result["timing"]
        timing_summary: dict[str, object] = {
            key: timing_payload[key]
            for key in ("duration_sec", "peak_memory_mb", "epochs_completed")
            if key in timing_payload
        }
        if timing_summary:
            summary["timing"] = timing_summary
    if "history" in result and isinstance(result["history"], list):
        summary["history_length"] = len(result["history"])
    if "split_manifest" in result:
        summary["split_manifest"] = result["split_manifest"]
    return summary


__all__ = [
    "describe_execution_dataloaders",
    "describe_pipeline",
    "describe_tree",
    "summarize_execution_result",
]
