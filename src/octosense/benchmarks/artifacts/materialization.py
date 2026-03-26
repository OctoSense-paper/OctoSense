"""Benchmark-owned completed-run finalization and optional artifact materialization."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from octosense.benchmarks.artifacts.environment import collect_environment, write_environment_artifact
from octosense.benchmarks.artifacts.metrics import build_metrics_artifact, write_metrics_artifact
from octosense.benchmarks.artifacts.provenance import (
    build_provenance_artifact,
    resolve_git_sha,
    write_provenance_artifact,
)
from octosense.benchmarks.artifacts.reports import render_markdown_report, write_markdown_report
from octosense.benchmarks.artifacts.run_manifest import build_run_manifest, write_run_manifest
from octosense.benchmarks.evaluators.classification import evaluate_classification
from octosense.benchmarks.evaluators.localization import evaluate_localization
from octosense.benchmarks.evaluators.pose import evaluate_pose
from octosense.benchmarks.evaluators.respiration import evaluate_respiration
from octosense.benchmarks.protocols import (
    canonical_protocol_for_protocol_id,
    canonical_protocol_for_task_kind,
    canonical_protocol_id_for_task_kind,
    canonical_task_kinds_for_protocol_id,
)
from octosense.specs.schemas.run_manifest import RunManifest


@dataclass(frozen=True)
class BenchmarkArtifactMaterializationOptions:
    """Explicit benchmark-owned options required to persist a completed run."""

    output_root: Path
    run_name: str
    task_id: str
    spec_digest: str
    dataset_digest: str
    requested_workers: int
    effective_workers: int
    provenance_root: Path | None = None
    worker_note: str | None = None
    worker_sharing_strategy: str | None = None
    worker_shm_manager: str | None = None

_EVALUATORS: dict[str, Callable[[Mapping[str, Any], Mapping[str, Any] | None], dict[str, Any]]] = {
    "classification": evaluate_classification,
    "pose": evaluate_pose,
    "localization": evaluate_localization,
    "respiration": evaluate_respiration,
}
_FORBIDDEN_PREMATERIALIZED_FIELDS = (
    "artifact_root",
    "artifacts",
    "commit",
    "environment",
    "provenance",
    "run_manifest",
)
_REQUIRED_TIMING_FIELDS = (
    "first_batch_sec",
    "duration_sec",
    "mean_epoch_sec",
    "epochs_completed",
    "train_samples_processed",
)
_ALLOWED_TRACE_EVENTS = frozenset({"epoch_end", "eval", "first_batch_profile"})


def _write_json(path: str | Path, payload: Any) -> Path:
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact_path


def _write_jsonl(path: str | Path, events: list[dict[str, object]]) -> Path:
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("w", encoding="utf-8") as handle:
        for payload in events:
            handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")
    return artifact_path


def write_json_artifact(path: str | Path, payload: Any) -> Path:
    """Write a JSON artifact using the canonical benchmark helper."""

    return _write_json(path, payload)


def write_jsonl_artifact(path: str | Path, events: list[dict[str, object]]) -> Path:
    """Write a JSONL artifact using the canonical benchmark helper."""

    return _write_jsonl(path, events)


def _write_optional_json_artifact(
    artifacts: dict[str, str],
    *,
    output_dir: Path,
    key: str,
    filename: str,
    payload: Mapping[str, Any] | None,
) -> None:
    if payload is None:
        return
    path = _write_json(output_dir / filename, dict(payload))
    artifacts[key] = str(path)


def write_optional_json_artifact(
    artifacts: dict[str, str],
    *,
    output_dir: Path,
    key: str,
    filename: str,
    payload: Mapping[str, Any] | None,
) -> None:
    """Write one optional JSON artifact into an outward artifact map."""

    _write_optional_json_artifact(
        artifacts,
        output_dir=output_dir,
        key=key,
        filename=filename,
        payload=payload,
    )


def _run_dir(output_root: Path, task_id: str) -> tuple[str, Path]:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"{timestamp}_{task_id}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


def _completed_run_mapping(run: Mapping[str, Any]) -> dict[str, Any]:
    return dict(run)


def _canonical_protocol_id(value: object) -> str:
    protocol_id = str(value).strip()
    return str(canonical_protocol_for_protocol_id(protocol_id)["protocol_id"])


def _protocol_id_from_task_kind(value: object) -> str:
    task_kind = str(value).strip()
    return canonical_protocol_id_for_task_kind(task_kind)


def _resolve_protocol_identity(
    *,
    protocol_id: object | None = None,
    task_kind: object | None = None,
    source: str,
) -> str:
    candidates: set[str] = set()
    if protocol_id is not None and str(protocol_id).strip():
        candidates.add(_canonical_protocol_id(protocol_id))
    if task_kind is not None and str(task_kind).strip():
        candidates.add(_protocol_id_from_task_kind(task_kind))
    if not candidates:
        raise ValueError(
            f"{source} must provide at least one canonical benchmark identity field via "
            "protocol_id or task_kind"
        )
    if len(candidates) != 1:
        raise ValueError(
            f"{source} declares conflicting canonical benchmark identity fields: {sorted(candidates)!r}"
        )
    return next(iter(candidates))


def _resolve_protocol_id(protocol: Mapping[str, Any]) -> str:
    return _resolve_protocol_identity(
        protocol_id=protocol.get("protocol_id"),
        task_kind=protocol.get("task_kind"),
        source="Benchmark protocol contract",
    )


def load_benchmark_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    protocol_id = _resolve_protocol_id(protocol)
    resolved = dict(protocol)
    task_kind = resolved.get("task_kind")
    if task_kind is not None:
        normalized_task_kind = str(task_kind).strip()
        expected_protocol_id = _protocol_id_from_task_kind(normalized_task_kind)
        if expected_protocol_id != protocol_id:
            raise ValueError(
                f"Benchmark protocol '{protocol_id}' declares task_kind={task_kind!r}, "
                f"but task kind {normalized_task_kind!r} is owned by protocol "
                f"{expected_protocol_id!r}."
            )
        canonical = canonical_protocol_for_task_kind(normalized_task_kind)
        resolved["task_kind"] = normalized_task_kind
    else:
        owned_task_kinds = canonical_task_kinds_for_protocol_id(protocol_id)
        if len(owned_task_kinds) != 1:
            raise ValueError(
                f"Benchmark protocol '{protocol_id}' must declare task_kind because protocol family "
                f"ownership is shared by task kinds {list(owned_task_kinds)!r}."
            )
        resolved["task_kind"] = owned_task_kinds[0]
        canonical = canonical_protocol_for_task_kind(owned_task_kinds[0])
    for key in ("name", "protocol_id", "primary_metric", "prediction_key", "target_key"):
        value = resolved.get(key)
        if value is not None:
            normalized_value = str(value).strip()
            if normalized_value and normalized_value != str(canonical[key]):
                raise ValueError(
                    f"Benchmark protocol '{protocol_id}' declares {key}={value!r}, "
                    f"expected canonical value {canonical[key]!r}"
                )
    metric_keys = resolved.get("metric_keys")
    if metric_keys is not None and list(metric_keys) != list(canonical["metric_keys"]):
        raise ValueError(
            f"Benchmark protocol '{protocol_id}' declares metric_keys={list(metric_keys)!r}, "
            f"expected canonical metric_keys={list(canonical['metric_keys'])!r}"
        )
    resolved["name"] = canonical["name"]
    resolved["protocol_id"] = canonical["protocol_id"]
    for key, value in canonical.items():
        resolved.setdefault(key, value)
    return resolved


def load_benchmark_evaluator(protocol_id: str) -> Any:
    evaluator = _EVALUATORS.get(protocol_id)
    if evaluator is None:
        supported = ", ".join(sorted(_EVALUATORS))
        raise ValueError(
            f"Benchmark evaluator '{protocol_id}' is not implemented. "
            f"Supported evaluator families: {supported}"
        )
    return evaluator


def _evaluate_outputs_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Compute evaluation metrics from one execution-phase ``{protocol, outputs}`` payload."""

    protocol_payload = payload.get("protocol")
    if not isinstance(protocol_payload, Mapping):
        raise ValueError("Execution-phase benchmark payload must include a protocol mapping")
    outputs = payload.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("Execution-phase benchmark payload must include an outputs mapping")
    protocol = load_benchmark_protocol(protocol_payload)
    evaluator = load_benchmark_evaluator(str(protocol["protocol_id"]))
    return evaluator(outputs, protocol=protocol)


def _metrics_artifact_from_completed_run(
    completed_run: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    metrics_payload = completed_run.get("metrics")
    if isinstance(metrics_payload, Mapping):
        nested_metrics = metrics_payload.get("metrics")
        numeric_metrics = nested_metrics if isinstance(nested_metrics, Mapping) else metrics_payload
        return build_metrics_artifact(
            numeric_metrics,
            primary_metric=str(protocol.get("primary_metric") or "") or None,
            task_kind=str(protocol.get("task_kind") or "") or None,
            protocol_id=str(protocol.get("protocol_id") or "") or None,
        )

    outputs = completed_run.get("outputs")
    if isinstance(outputs, Mapping):
        evaluation = load_benchmark_evaluator(str(protocol["protocol_id"]))(outputs, protocol=protocol)
        evaluated_metrics = evaluation.get("metrics")
        if not isinstance(evaluated_metrics, Mapping):
            raise ValueError(
                "Benchmark evaluator did not return a canonical metrics mapping for the completed run."
            )
        return build_metrics_artifact(
            evaluated_metrics,
            primary_metric=str(protocol.get("primary_metric") or "") or None,
            task_kind=str(protocol.get("task_kind") or "") or None,
            protocol_id=str(protocol.get("protocol_id") or "") or None,
        )

    raise ValueError(
        "Completed benchmark run payload must provide either a metrics mapping or "
        "benchmark outputs for benchmark-owned finalization."
    )


def _finalized_completed_run_mapping(
    completed_run: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    finalized = dict(completed_run)
    finalized["status"] = "completed"
    finalized["protocol"] = dict(protocol)
    finalized["metrics"] = dict(metrics)
    return finalized


def _public_materialized_run_mapping(
    completed_run: Mapping[str, Any],
    *,
    run_id: str,
    run_name: str,
    task_id: str,
    artifact_root: str,
    provenance: Mapping[str, Any],
    artifacts: Mapping[str, str],
    run_manifest: RunManifest,
    environment: Mapping[str, Any],
    commit: str,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    public_run = dict(completed_run)
    public_run.update(
        {
            "run_id": run_id,
            "run_name": run_name,
            "task_id": task_id,
            "status": "completed",
            "artifact_root": artifact_root,
            "provenance": dict(provenance),
            "artifacts": dict(artifacts),
            "run_manifest": run_manifest.to_dict(),
            "environment": dict(environment),
            "commit": commit,
            "metrics": dict(metrics),
        }
    )
    return public_run


def _required_mapping(run: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = run.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(
            f"Completed benchmark run payload must provide a mapping at run[{key!r}] "
            "before artifact materialization."
        )
    return dict(value)


def _required_history_rows(run: Mapping[str, Any]) -> list[dict[str, object]]:
    value = run.get("history")
    if not isinstance(value, list):
        raise ValueError(
            "Completed benchmark run payload must provide a list field 'history' "
            "before artifact materialization."
        )
    normalized: list[dict[str, object]] = []
    for index, row in enumerate(value):
        if not isinstance(row, Mapping):
            raise ValueError(
                "Completed benchmark run payload field 'history' must contain only mappings; "
                f"item {index} is {type(row).__name__}."
            )
        normalized.append(dict(row))
    return normalized


def _required_int(run: Mapping[str, Any], key: str) -> int:
    value = run.get(key)
    if value is not None and value != "":
        return int(value)
    raise ValueError(
        f"Completed benchmark run payload must provide integer field {key!r}."
    )


def _required_str(run: Mapping[str, Any], key: str) -> str:
    value = run.get(key)
    if value is not None:
        text = str(value).strip()
        if text:
            return text
    raise ValueError(
        f"Completed benchmark run payload must provide non-empty string field {key!r}."
    )


def _train_events(run: Mapping[str, Any]) -> list[dict[str, object]]:
    value = run.get("train_events")
    if value is None and run.get("train_log") is not None:
        raise ValueError(
            "Completed benchmark run payload must provide 'train_events'; "
            "'train_log' is an artifact output, not an accepted input alias."
        )
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(
            "Completed benchmark run payload field 'train_events' must be a list of mapping events."
        )
    normalized: list[dict[str, object]] = []
    for index, event in enumerate(value):
        if not isinstance(event, Mapping):
            raise ValueError(
                "Completed benchmark run payload field 'train_events' must contain only mappings; "
                f"item {index} is {type(event).__name__}."
            )
        normalized.append(dict(event))
    return normalized


def _optional_mapping(run: Mapping[str, Any], key: str) -> dict[str, Any] | None:
    value = run.get(key)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(
            f"Completed benchmark run payload field {key!r} must be a mapping when provided."
        )
    return dict(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _extract_task_identity(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        for key in ("task_id", "id", "task"):
            nested = _optional_str(value.get(key))
            if nested is not None:
                return nested
        return None
    return _optional_str(value)


def _validate_optional_match(*, field: str, actual: object, expected: object, source: str) -> None:
    if actual is None:
        return
    if actual != expected:
        raise ValueError(
            f"Completed benchmark run payload field {field!r} from {source} must match "
            f"materialization options: expected {expected!r}, got {actual!r}."
        )


def _validate_completed_run_contract(
    completed_run: Mapping[str, Any],
    *,
    options: BenchmarkArtifactMaterializationOptions | None = None,
) -> None:
    status = _optional_str(completed_run.get("status"))
    if status is not None and status != "completed":
        raise ValueError(
            "octosense.benchmarks.materialize(...) accepts only completed run payloads; "
            f"got status={status!r}."
        )
    if options is None:
        return

    _validate_optional_match(
        field="task_id",
        actual=_optional_str(completed_run.get("task_id")),
        expected=options.task_id,
        source="run payload",
    )
    _validate_optional_match(
        field="run_name",
        actual=_optional_str(completed_run.get("run_name")),
        expected=options.run_name,
        source="run payload",
    )
    _validate_optional_match(
        field="spec_digest",
        actual=_optional_str(completed_run.get("spec_digest")),
        expected=options.spec_digest,
        source="run payload",
    )
    _validate_optional_match(
        field="dataset_digest",
        actual=_optional_str(completed_run.get("dataset_digest")),
        expected=options.dataset_digest,
        source="run payload",
    )
    _validate_optional_match(
        field="requested_workers",
        actual=_optional_int(completed_run.get("requested_workers")),
        expected=options.requested_workers,
        source="run payload",
    )
    _validate_optional_match(
        field="effective_workers",
        actual=_optional_int(completed_run.get("effective_workers")),
        expected=options.effective_workers,
        source="run payload",
    )
    _validate_optional_match(
        field="worker_note",
        actual=_optional_str(completed_run.get("worker_note")),
        expected=options.worker_note,
        source="run payload",
    )
    _validate_optional_match(
        field="worker_sharing_strategy",
        actual=_optional_str(completed_run.get("worker_sharing_strategy")),
        expected=options.worker_sharing_strategy,
        source="run payload",
    )
    _validate_optional_match(
        field="worker_shm_manager",
        actual=_optional_str(completed_run.get("worker_shm_manager")),
        expected=options.worker_shm_manager,
        source="run payload",
    )

    provenance = _optional_mapping(completed_run, "provenance")
    if provenance is not None:
        _validate_optional_match(
            field="provenance.spec_digest",
            actual=_optional_str(provenance.get("spec_digest")),
            expected=options.spec_digest,
            source="run provenance",
        )
        _validate_optional_match(
            field="provenance.dataset_digest",
            actual=_optional_str(provenance.get("dataset_digest")),
            expected=options.dataset_digest,
            source="run provenance",
        )

    environment = _optional_mapping(completed_run, "environment")
    if environment is not None:
        _validate_optional_match(
            field="environment.dataloader_workers_requested",
            actual=_optional_int(environment.get("dataloader_workers_requested")),
            expected=options.requested_workers,
            source="run environment",
        )
        _validate_optional_match(
            field="environment.dataloader_workers_effective",
            actual=_optional_int(environment.get("dataloader_workers_effective")),
            expected=options.effective_workers,
            source="run environment",
        )
        _validate_optional_match(
            field="environment.torch_sharing_strategy",
            actual=_optional_str(environment.get("torch_sharing_strategy")),
            expected=options.worker_sharing_strategy,
            source="run environment",
        )
        _validate_optional_match(
            field="environment.torch_shm_manager",
            actual=_optional_str(environment.get("torch_shm_manager")),
            expected=options.worker_shm_manager,
            source="run environment",
        )
        _validate_optional_match(
            field="environment.worker_backend_note",
            actual=_optional_str(environment.get("worker_backend_note")),
            expected=options.worker_note,
            source="run environment",
        )

    run_manifest = _optional_mapping(completed_run, "run_manifest")
    if run_manifest is not None:
        _validate_optional_match(
            field="run_manifest.spec_digest",
            actual=_optional_str(run_manifest.get("spec_digest")),
            expected=options.spec_digest,
            source="run manifest",
        )
        _validate_optional_match(
            field="run_manifest.dataset_digest",
            actual=_optional_str(run_manifest.get("dataset_digest")),
            expected=options.dataset_digest,
            source="run manifest",
        )


def _validate_no_prematerialized_fields(completed_run: Mapping[str, Any]) -> None:
    blocked = [
        field
        for field in _FORBIDDEN_PREMATERIALIZED_FIELDS
        if completed_run.get(field) not in (None, {}, [], "")
    ]
    if blocked:
        blocked_fields = ", ".join(sorted(blocked))
        raise ValueError(
            "octosense.benchmarks.materialize(...) only accepts raw completed-run payloads "
            "from pipelines.execution and refuses pre-materialized artifact fields: "
            f"{blocked_fields}."
        )


def _validate_timing_payload(timing: Mapping[str, Any]) -> None:
    missing = [field for field in _REQUIRED_TIMING_FIELDS if field not in timing]
    if missing:
        raise ValueError(
            "Completed benchmark run payload timing is missing required execution-owned fields: "
            f"{', '.join(missing)}."
        )


def _validate_trace_events(
    train_events: list[dict[str, object]],
    *,
    mode: str,
) -> None:
    event_names: list[str] = []
    for index, event in enumerate(train_events):
        event_name = _optional_str(event.get("event"))
        if event_name is None:
            raise ValueError(
                "Completed benchmark run payload field 'train_events' must contain canonical "
                f"execution trace mappings with an 'event' key; item {index} is missing one."
            )
        if event_name not in _ALLOWED_TRACE_EVENTS:
            raise ValueError(
                "Completed benchmark run payload field 'train_events' contains unsupported "
                f"event {event_name!r}; benchmark artifact materialization only accepts "
                "canonical pipelines.execution trace events."
            )
        event_names.append(event_name)
        if event_name == "epoch_end":
            for field in ("epoch", "train_loss", "epoch_duration_sec"):
                if field not in event:
                    raise ValueError(
                        "Completed benchmark run payload field 'train_events' contains an "
                        f"'epoch_end' event missing required field {field!r}."
                    )
        elif "split" not in event:
            raise ValueError(
                "Completed benchmark run payload field 'train_events' contains a trace event "
                f"{event_name!r} without required field 'split'."
            )
    if mode == "train" and "epoch_end" not in event_names:
        raise ValueError(
            "Training completed-run payloads must include at least one canonical 'epoch_end' "
            "trace event before artifact materialization."
        )


def _validate_history_against_trace(
    history: list[dict[str, object]],
    train_events: list[dict[str, object]],
    *,
    mode: str,
) -> None:
    epoch_rows = [event for event in train_events if event.get("event") == "epoch_end"]
    if mode == "train":
        if not history:
            raise ValueError(
                "Training completed-run payloads must include non-empty epoch history before "
                "artifact materialization."
            )
        if len(history) != len(epoch_rows):
            raise ValueError(
                "Completed benchmark run payload history length does not match the number of "
                "canonical 'epoch_end' trace events."
            )
    elif history and len(history) != len(epoch_rows):
        raise ValueError(
            "Completed benchmark run payload history must stay aligned with the canonical "
            "'epoch_end' trace events."
        )


def _validate_execution_owned_materialization_input(
    completed_run: Mapping[str, Any],
) -> None:
    if completed_run.get("outputs") is not None:
        raise ValueError(
            "octosense.benchmarks.materialize(...) no longer accepts execution-phase "
            "{protocol, outputs} payloads for artifact materialization. Materialize only "
            "raw completed runs emitted by pipelines.execution."
        )
    _validate_no_prematerialized_fields(completed_run)
    mode = _required_str(completed_run, "mode")
    if mode not in {"train", "evaluate"}:
        raise ValueError(
            "Completed benchmark run payload must provide canonical execution mode "
            f"'train' or 'evaluate' before artifact materialization; got {mode!r}."
        )
    for key in ("primary_metric", "protocol_id", "task_kind"):
        _required_str(completed_run, key)
    history = _required_history_rows(completed_run)
    timing = _required_mapping(completed_run, "timing")
    train_events = _train_events(completed_run)
    _validate_timing_payload(timing)
    if not train_events:
        raise ValueError(
            "Completed benchmark run payload must include canonical execution trace events in "
            "'train_events' before artifact materialization."
        )
    _validate_trace_events(train_events, mode=mode)
    _validate_history_against_trace(history, train_events, mode=mode)


def finalize_completed_benchmark_run(
    run: Mapping[str, Any],
    *,
    options: BenchmarkArtifactMaterializationOptions | None = None,
) -> dict[str, Any]:
    """Finalize one completed benchmark run and optionally materialize artifacts."""

    completed_run = _completed_run_mapping(run)
    _validate_completed_run_contract(completed_run, options=options)
    protocol = load_benchmark_protocol(_required_mapping(completed_run, "protocol"))
    metrics_artifact = _metrics_artifact_from_completed_run(completed_run, protocol=protocol)
    finalized_run = _finalized_completed_run_mapping(
        completed_run,
        protocol=protocol,
        metrics=metrics_artifact,
    )
    if options is None:
        return finalized_run

    _validate_execution_owned_materialization_input(completed_run)
    timing = _required_mapping(completed_run, "timing")
    train_events = _train_events(completed_run)
    runtime_device = _required_str(completed_run, "device")
    execution_seed = _required_int(completed_run, "seed")
    split_manifest = _optional_mapping(completed_run, "split_manifest")
    sample_describe_tree = _optional_mapping(completed_run, "sample_describe_tree")

    run_id, run_dir = _run_dir(options.output_root, options.task_id)
    commit = resolve_git_sha(options.provenance_root or options.output_root)

    metrics_payload = dict(metrics_artifact["metrics"])
    metrics_payload["duration_sec"] = float(timing["duration_sec"])

    environment_extra: dict[str, object] = {
        "dataloader_workers_requested": options.requested_workers,
        "dataloader_workers_effective": options.effective_workers,
    }
    if options.worker_sharing_strategy is not None:
        environment_extra["torch_sharing_strategy"] = options.worker_sharing_strategy
    if options.worker_shm_manager is not None:
        environment_extra["torch_shm_manager"] = options.worker_shm_manager
    if options.worker_note is not None:
        environment_extra["worker_backend_note"] = options.worker_note
    environment = collect_environment(extra=environment_extra)

    artifacts: dict[str, str] = {}
    timing_path = _write_json(run_dir / "timing.json", dict(timing))
    artifacts["timing"] = str(timing_path)
    metrics_artifact = build_metrics_artifact(
        metrics_payload,
        primary_metric=str(protocol.get("primary_metric") or "") or None,
        task_kind=str(protocol.get("task_kind") or "") or None,
        protocol_id=str(protocol.get("protocol_id") or "") or None,
    )
    metrics_path = write_metrics_artifact(run_dir / "metrics.json", metrics_artifact)
    artifacts["metrics"] = str(metrics_path)
    environment_path = write_environment_artifact(run_dir / "environment.json", environment)
    artifacts["environment"] = str(environment_path)
    protocol_path = _write_json(run_dir / "protocol.json", dict(protocol))
    artifacts["protocol"] = str(protocol_path)
    _write_optional_json_artifact(
        artifacts,
        output_dir=run_dir,
        key="split_manifest",
        filename="split_manifest.json",
        payload=split_manifest,
    )
    _write_optional_json_artifact(
        artifacts,
        output_dir=run_dir,
        key="sample_describe_tree",
        filename="sample_describe_tree.json",
        payload=sample_describe_tree,
    )

    train_log_path = _write_jsonl(run_dir / "train_log.jsonl", train_events)
    artifacts["train_log"] = str(train_log_path)

    provenance = build_provenance_artifact(
        spec_digest=options.spec_digest,
        dataset_digest=options.dataset_digest,
        seed=execution_seed,
        git_sha=commit,
    )
    provenance_path = write_provenance_artifact(run_dir / "provenance.json", provenance)
    artifacts["provenance"] = str(provenance_path)
    run_manifest: RunManifest = build_run_manifest(
        run_id=run_id,
        spec_digest=options.spec_digest,
        dataset_digest=options.dataset_digest,
        git_sha=commit,
        seed=execution_seed,
        device=runtime_device,
        status="completed",
        finished_at=datetime.now(UTC).isoformat(),
        artifact_root=str(run_dir),
        metrics_path=str(metrics_path),
        environment_path=str(environment_path),
        timing_path=str(timing_path),
        protocol_path=str(protocol_path),
    )
    run_manifest_path = write_run_manifest(run_dir / "run_manifest.json", run_manifest)
    artifacts["run_manifest"] = str(run_manifest_path)
    report_path = write_markdown_report(
        run_dir / "report.md",
        render_markdown_report(
            run_manifest,
            metrics_artifact["metrics"],
            protocol=protocol,
            provenance=provenance,
            environment=environment,
        ),
    )
    artifacts["report"] = str(report_path)

    return _public_materialized_run_mapping(
        finalized_run,
        run_id=run_id,
        run_name=options.run_name,
        task_id=options.task_id,
        artifact_root=str(run_dir),
        provenance=provenance,
        artifacts=artifacts,
        run_manifest=run_manifest,
        environment=environment,
        commit=commit,
        metrics=metrics_artifact,
    )


__all__ = [
    "BenchmarkArtifactMaterializationOptions",
    "finalize_completed_benchmark_run",
    "load_benchmark_evaluator",
    "load_benchmark_protocol",
    "write_json_artifact",
    "write_jsonl_artifact",
    "write_optional_json_artifact",
]
