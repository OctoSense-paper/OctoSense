"""Metrics artifact helpers for benchmark results."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def build_metrics_artifact(
    metrics: Mapping[str, Any],
    *,
    primary_metric: str | None = None,
    task_kind: str | None = None,
    protocol_id: str | None = None,
) -> dict[str, Any]:
    """Normalize metric values into a stable artifact payload."""
    payload: dict[str, Any] = {"metrics": {str(key): float(value) for key, value in metrics.items()}}
    if primary_metric is not None:
        payload["primary_metric"] = str(primary_metric)
        if primary_metric in payload["metrics"]:
            payload["primary_metric_value"] = float(payload["metrics"][primary_metric])
    if task_kind is not None:
        payload["task_kind"] = str(task_kind)
    if protocol_id is not None:
        payload["protocol_id"] = str(protocol_id)
    return payload


def write_metrics_artifact(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Persist a metrics artifact to JSON."""

    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact_path


def read_metrics_artifact(path: str | Path) -> dict[str, Any]:
    """Load a metrics artifact from JSON."""

    return dict(json.loads(Path(path).read_text(encoding="utf-8")))


__all__ = [
    "build_metrics_artifact",
    "read_metrics_artifact",
    "write_metrics_artifact",
]
