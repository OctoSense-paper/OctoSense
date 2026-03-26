"""Canonical benchmark API centered on completed-run evaluation.

``octosense.benchmarks`` evaluates completed benchmark runs only. Execution,
spec loading, and pipeline orchestration remain owned by ``octosense.cli.run``
and ``octosense.pipelines``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from octosense.benchmarks.artifacts.materialization import (
    BenchmarkArtifactMaterializationOptions,
    finalize_completed_benchmark_run,
    load_benchmark_protocol,
)
from octosense.benchmarks.artifacts.run_manifest import run_manifest_from_dict
from octosense.benchmarks.artifacts.metrics import read_metrics_artifact
from octosense.specs.schemas.run_manifest import RUN_MANIFEST_KIND


def _read_json_file(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _build_evaluation_result(
    *,
    protocol: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    resolved_protocol = dict(protocol)
    normalized_metrics = {str(key): float(value) for key, value in metrics.items()}
    primary_metric = str(resolved_protocol.get("primary_metric", ""))
    result: dict[str, Any] = {
        "protocol": resolved_protocol,
        "metrics": normalized_metrics,
        "task_kind": resolved_protocol.get("task_kind"),
        "primary_metric": primary_metric,
    }
    if primary_metric in normalized_metrics:
        result["primary_metric_value"] = float(normalized_metrics[primary_metric])
    return result


def _attach_run_context(
    result: Mapping[str, Any],
    *,
    run: Mapping[str, Any],
    source: str,
) -> dict[str, Any]:
    attached = dict(result)
    attached["source"] = source
    for key in ("run_id", "status", "artifact_root"):
        value = run.get(key)
        if value is not None:
            attached[key] = value
    artifacts = run.get("artifacts")
    if isinstance(artifacts, Mapping) and artifacts:
        attached["artifacts"] = dict(artifacts)
    run_manifest = run.get("run_manifest")
    if isinstance(run_manifest, Mapping) and run_manifest:
        attached["run_manifest"] = dict(run_manifest)
    return attached


def _normalize_metrics_payload(run: Mapping[str, Any]) -> dict[str, Any] | None:
    metrics = run.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    nested_metrics = metrics.get("metrics")
    if not isinstance(nested_metrics, Mapping):
        return None
    payload: dict[str, Any] = {"metrics": dict(nested_metrics)}
    for key in ("primary_metric", "primary_metric_value", "protocol_id", "task_kind"):
        value = metrics.get(key)
        if value is not None:
            payload[key] = value
    return payload


def _resolve_artifact_path(
    reference: object,
    *,
    artifact_root: Path | None,
    base_dir: Path | None,
) -> Path | None:
    if reference is None:
        return None
    candidate = Path(str(reference)).expanduser()
    if candidate.is_absolute():
        return candidate if candidate.exists() else None
    for anchor in (artifact_root, base_dir):
        if anchor is None:
            continue
        resolved = (anchor / candidate).resolve()
        if resolved.exists():
            return resolved
    return None


def _mapping_from_run_manifest_artifact(path: Path, payload: Mapping[str, Any]) -> Mapping[str, Any]:
    run_manifest = run_manifest_from_dict(payload)
    run_manifest_payload = run_manifest.to_dict()
    base_dir = path.parent
    artifact_root = _resolve_artifact_path(run_manifest.artifact_root, artifact_root=None, base_dir=base_dir)
    metrics_path = _resolve_artifact_path(
        run_manifest.metrics_path,
        artifact_root=artifact_root,
        base_dir=base_dir,
    )
    environment_path = _resolve_artifact_path(
        run_manifest.environment_path,
        artifact_root=artifact_root,
        base_dir=base_dir,
    )
    timing_path = _resolve_artifact_path(
        run_manifest.timing_path,
        artifact_root=artifact_root,
        base_dir=base_dir,
    )
    protocol_path = _resolve_artifact_path(
        run_manifest.protocol_path,
        artifact_root=artifact_root,
        base_dir=base_dir,
    )

    artifacts: dict[str, str] = {}
    if metrics_path is not None:
        artifacts["metrics"] = str(metrics_path)
    if environment_path is not None:
        artifacts["environment"] = str(environment_path)
    if timing_path is not None:
        artifacts["timing"] = str(timing_path)
    if protocol_path is not None:
        artifacts["protocol"] = str(protocol_path)

    resolved: dict[str, Any] = {
        "run_id": run_manifest.run_id,
        "status": run_manifest.status,
        "artifact_root": str(artifact_root) if artifact_root is not None else run_manifest.artifact_root,
        "run_manifest": run_manifest_payload,
        "artifacts": artifacts,
    }
    if metrics_path is not None:
        resolved["metrics"] = read_metrics_artifact(metrics_path)
    return resolved


def _load_run_path(path: Path) -> Mapping[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.name != "run_manifest.json":
        raise TypeError(
            "octosense.benchmarks.evaluate(...) accepts only completed run payloads or "
            "a canonical run_manifest.json artifact path"
        )
    payload = _read_json_file(resolved)
    if str(payload.get("kind", "")).strip() != RUN_MANIFEST_KIND:
        raise ValueError(f"{resolved} is not a valid RunManifest artifact")
    run = _mapping_from_run_manifest_artifact(resolved, payload)
    _validate_run_status(run)
    return run


def _validate_run_status(run: Mapping[str, Any]) -> None:
    status = str(run.get("status", "") or "").strip()
    if status and status != "completed":
        raise ValueError(
            "Canonical benchmark run payloads must use status='completed'; "
            f"got status={status!r}."
        )


def _as_run_mapping(run: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    if isinstance(run, Mapping):
        if str(run.get("kind", "")).strip() == RUN_MANIFEST_KIND:
            raise TypeError(
                "Pass a canonical run_manifest.json path instead of a detached RunManifest mapping"
            )
        _validate_run_status(run)
        return run
    if isinstance(run, (str, Path)):
        return _load_run_path(Path(run))
    raise TypeError(
        "octosense.benchmarks.evaluate(...) accepts only completed run payloads or "
        "a canonical run_manifest.json artifact path"
    )


def _protocol_from_artifact_reference(run: Mapping[str, Any]) -> Mapping[str, Any] | None:
    artifacts = run.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return None
    protocol_reference = artifacts.get("protocol")
    if protocol_reference is None:
        return None
    artifact_root_value = run.get("artifact_root")
    artifact_root = None if artifact_root_value is None else Path(str(artifact_root_value)).expanduser()
    protocol_path = _resolve_artifact_path(
        protocol_reference,
        artifact_root=artifact_root,
        base_dir=artifact_root,
    )
    if protocol_path is None:
        raise ValueError(
            "Run payload declares artifacts.protocol, but the referenced protocol artifact does not exist"
        )
    return _read_json_file(protocol_path)


def _resolve_protocol_payload(run: Mapping[str, Any]) -> Mapping[str, Any]:
    protocol_payload = run.get("protocol")
    if isinstance(protocol_payload, Mapping):
        return protocol_payload
    artifact_protocol = _protocol_from_artifact_reference(run)
    if artifact_protocol is not None:
        return artifact_protocol
    raise ValueError(
        "Run payload must provide an explicit benchmark protocol contract via run.protocol, "
        "or artifacts.protocol"
    )


def _evaluate_from_metrics(run: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    metric_payload = _normalize_metrics_payload(run)
    if metric_payload is None:
        raise ValueError("Metrics-backed benchmark runs must include metrics.metrics")
    resolved_protocol = dict(protocol)
    declared_primary_metric = metric_payload.get("primary_metric")
    protocol_primary_metric = str(resolved_protocol.get("primary_metric", "")).strip()
    if declared_primary_metric is not None and str(declared_primary_metric).strip() != protocol_primary_metric:
        raise ValueError(
            "Metrics artifact primary_metric does not match the canonical benchmark protocol: "
            f"{declared_primary_metric!r} != {protocol_primary_metric!r}"
        )
    metric_values = dict(metric_payload["metrics"])
    if protocol_primary_metric and protocol_primary_metric not in metric_values:
        raise ValueError(
            "Metrics artifact does not contain the canonical primary metric "
            f"{protocol_primary_metric!r}"
        )
    return _attach_run_context(
        _build_evaluation_result(
            protocol=resolved_protocol,
            metrics=metric_values,
        ),
        run=run,
        source="artifact_metrics",
    )


def evaluate(run: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    """Evaluate one canonical completed run payload."""

    run_mapping = _as_run_mapping(run)
    protocol = load_benchmark_protocol(_resolve_protocol_payload(run_mapping))
    if isinstance(run_mapping.get("metrics"), Mapping):
        return _evaluate_from_metrics(run_mapping, protocol)
    raise ValueError(
        "octosense.benchmarks.evaluate(...) accepts completed runs with persisted metrics only; "
        "execution-phase {protocol, outputs} payloads must stay on the internal runner path"
    )


def materialize(
    run: Mapping[str, Any],
    *,
    options: BenchmarkArtifactMaterializationOptions | None = None,
) -> dict[str, Any]:
    """Finalize one completed benchmark run and optionally persist artifacts."""

    return finalize_completed_benchmark_run(run, options=options)


__all__ = [
    "BenchmarkArtifactMaterializationOptions",
    "evaluate",
    "materialize",
]
