"""Canonical execution runtime for spec-native pipeline targets."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any, cast

import numpy as np
import octosense.benchmarks as benchmark_api
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from octosense.benchmarks import BenchmarkArtifactMaterializationOptions
from octosense.benchmarks.artifacts.materialization import _evaluate_outputs_payload
from octosense.benchmarks.artifacts.metrics import build_metrics_artifact
from octosense.datasets.fingerprint import dataset_digest as compute_dataset_digest
from octosense.datasets import DatasetView
from octosense.pipelines._runtime_spec import (
    PipelineExecutionHandle as _PipelineExecutionHandle,
    RuntimePipelineSpec as _RuntimePipelineSpec,
)
from octosense.pipelines.execution.evaluate import evaluate_loader
from octosense.pipelines.execution.infer import infer_batches, infer_pipeline
from octosense.pipelines.execution.train import (
    MetricTrace,
    peak_memory_mb,
    synchronize_device,
)
from octosense.pipelines.dataloading.datamodule import build_sample_ids
from octosense.specs.compiler.freezer import freeze_spec
from octosense.specs.serde.canonical import spec_digest as compute_spec_digest
from octosense.specs.schemas.benchmark import BenchmarkSpec
from octosense.specs.schemas.runtime import RuntimeSpec
from octosense.tasks import load as load_task
from octosense.tasks.definitions import TaskSpec
from octosense._internal.worker_backend import resolve_num_workers


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(user_device: str | None) -> torch.device:
    return resolve_device(user_device)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible execution runs."""
    import random

    random.seed(seed)
    _set_seed(seed)


def resolve_device(preferred: str | None, *, require_cuda: bool = False) -> torch.device:
    """Resolve a runtime device while enforcing remote CUDA requirements when requested."""
    if preferred:
        device = torch.device(preferred)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "Requested CUDA device for remote experiment, but torch.cuda.is_available() is false"
            )
        if require_cuda and device.type != "cuda":
            raise RuntimeError(
                f"CUDA is required for remote experiment execution, but resolved device is {device}"
            )
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if require_cuda:
        raise RuntimeError(
            "CUDA is required for remote experiment execution, but no CUDA device is available"
        )
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def reset_peak_memory_stats(device: torch.device) -> None:
    """Reset peak memory accounting when the backend supports it."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _runtime_payload_from_mapping(
    payload: Mapping[str, Any] | None,
    *,
    field_names: set[str] | None = None,
) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        return {}
    runtime_keys = {"device", "batch_size", "epochs", "seed", "num_workers"}
    if field_names is not None:
        runtime_keys &= field_names
    return {
        key: payload[key]
        for key in runtime_keys
        if key in payload and payload[key] is not None
    }


def _require_non_empty_runtime_field(runtime_payload: Mapping[str, object], field_name: str) -> object:
    value = runtime_payload.get(field_name)
    if value is None:
        raise ValueError(
            "pipelines.execution requires explicit runtime fields. "
            f"Missing spec.runtime['{field_name}'] and no runtime override was supplied."
        )
    if isinstance(value, str) and not value.strip():
        raise ValueError(
            "pipelines.execution requires explicit runtime fields. "
            f"spec.runtime['{field_name}'] must be a non-empty string."
        )
    return value


def _runtime_spec_for_execution(
    spec: _RuntimePipelineSpec,
    *,
    mode: str,
) -> RuntimeSpec:
    required_fields = {"device", "batch_size", "seed", "num_workers"}
    if mode == "train":
        required_fields.add("epochs")
    runtime_payload = _runtime_payload_from_mapping(spec.runtime, field_names=required_fields)
    for field_name in required_fields:
        _require_non_empty_runtime_field(runtime_payload, field_name)
    return RuntimeSpec.from_dict(dict(runtime_payload))


def _dataset_rows_for_digest(dataset_source: Any | None) -> list[dict[str, Any]]:
    if dataset_source is None:
        return []
    if isinstance(dataset_source, Mapping):
        rows: list[dict[str, Any]] = []
        for split_name in ("train", "val", "test"):
            split_dataset = dataset_source.get(split_name)
            metadata_rows = getattr(split_dataset, "metadata_rows", None)
            if callable(metadata_rows):
                rows.extend(
                    {
                        **dict(row),
                        "_split": str(split_name),
                    }
                    for row in metadata_rows()
                )
        return rows
    metadata_rows = getattr(dataset_source, "metadata_rows", None)
    if callable(metadata_rows):
        return [dict(row) for row in metadata_rows()]
    return []


def _dataset_digest(dataset_source: Any | None, spec: _RuntimePipelineSpec) -> str:
    rows = _dataset_rows_for_digest(dataset_source)
    if rows:
        return compute_dataset_digest(rows)
    return compute_dataset_digest([dict(spec.dataset)])


def _normalized_output_root(output_root: str | Path | None) -> str | None:
    if output_root is None:
        return None
    normalized = str(output_root).strip()
    return normalized or None


def _resolved_output_root(output_root: str | Path | None) -> Path | None:
    normalized = _normalized_output_root(output_root)
    if normalized is None:
        return None
    candidate = Path(normalized).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.resolve()


def _materialize_completed_run(
    *,
    completed_run: Mapping[str, Any],
    options: BenchmarkArtifactMaterializationOptions | None = None,
) -> Mapping[str, Any]:
    finalized_run = benchmark_api.materialize(completed_run, options=options)
    if not isinstance(finalized_run, Mapping):
        raise TypeError("octosense.benchmarks.materialize(...) must return a mapping payload.")
    return finalized_run

def _resolve_canonical_benchmark_spec(
    handle: _PipelineExecutionHandle,
    benchmark_spec: BenchmarkSpec | None,
) -> BenchmarkSpec | None:
    handle_spec = handle.get_benchmark_spec()
    if handle_spec is None:
        return None if benchmark_spec is None else freeze_spec(benchmark_spec).spec
    if benchmark_spec is not None:
        handle_digest = compute_spec_digest(handle_spec)
        argument_digest = compute_spec_digest(benchmark_spec)
        if handle_digest != argument_digest:
            raise ValueError(
                "execute_pipeline_handle(...) received a benchmark_spec that does not match "
                "the canonical BenchmarkSpec bound to the pipeline handle. "
                f"handle={handle_digest} argument={argument_digest}"
            )
    return handle_spec


def _resolved_run_name(
    spec: _RuntimePipelineSpec,
    *,
    run_name: str | None,
) -> str:
    if run_name is not None and str(run_name).strip():
        return str(run_name).strip()
    runtime_name = str(spec.runtime.get("run_name", "")).strip()
    if runtime_name:
        return runtime_name
    return str(spec.task.get("task_id", "pipeline"))


def _build_completed_run_payload(
    result: Mapping[str, Any],
    *,
    benchmark_spec: BenchmarkSpec,
    execution_spec: _RuntimePipelineSpec,
    execution_runtime: RuntimeSpec,
    dataset_source: Any,
    resolved_mode: str,
    run_name: str,
    requested_workers: int,
    effective_workers: int,
    worker_status: Any | None,
) -> dict[str, Any]:
    task_spec = _resolve_task_spec(execution_spec)
    trainable_outputs = _task_outputs_from_trainable_result(result)
    protocol = _resolved_protocol(execution_spec, task_spec)
    completed_run = dict(result)
    completed_run.update(
        {
            "mode": resolved_mode,
            "status": "completed",
            "run_name": run_name,
            "task_id": str(benchmark_spec.task.task_id),
            "device": str(execution_runtime.device),
            "seed": int(execution_runtime.seed),
            "protocol": protocol,
            "primary_metric": str(protocol["primary_metric"]),
            "task_kind": str(protocol["task_kind"]),
            "protocol_id": str(protocol["protocol_id"]),
            "spec_digest": compute_spec_digest(benchmark_spec),
            "dataset_digest": _dataset_digest(dataset_source, execution_spec),
            "requested_workers": requested_workers,
            "effective_workers": effective_workers,
        }
    )
    worker_note = worker_status.reason if worker_status is not None else None
    if worker_note is not None:
        completed_run["worker_note"] = worker_note
    worker_sharing_strategy = (
        worker_status.sharing_strategy if worker_status is not None else None
    )
    if worker_sharing_strategy is not None:
        completed_run["worker_sharing_strategy"] = worker_sharing_strategy
    worker_shm_manager = (
        worker_status.torch_shm_manager if worker_status is not None else None
    )
    if worker_shm_manager is not None:
        completed_run["worker_shm_manager"] = worker_shm_manager
    if trainable_outputs.split_manifest is not None:
        completed_run["split_manifest"] = cast(
            dict[str, object], trainable_outputs.split_manifest
        )
    if trainable_outputs.sample_describe_tree is not None:
        completed_run["sample_describe_tree"] = trainable_outputs.sample_describe_tree
    return completed_run


def _resolve_execution_workers(requested_workers: int) -> tuple[int, Any]:
    effective_workers, worker_status = resolve_num_workers(requested_workers, strict=True)
    return int(effective_workers), worker_status


def _canonical_completed_run_result(
    completed_run: Mapping[str, Any],
) -> dict[str, Any]:
    canonical_result = dict(completed_run)
    canonical_result["metrics"] = _canonical_metrics_artifact_payload(
        canonical_result.get("metrics"),
        protocol=canonical_result.get("protocol"),
        completed_run=canonical_result,
    )
    artifacts = canonical_result.get("artifacts")
    if artifacts is None:
        canonical_result["artifacts"] = {}
    elif not isinstance(artifacts, Mapping):
        raise ValueError(
            "pipelines.execution canonical result payload requires a mapping at result['artifacts']."
        )
    else:
        canonical_result["artifacts"] = {str(key): str(value) for key, value in artifacts.items()}
    for key in ("environment", "provenance"):
        payload = canonical_result.get(key)
        if payload is None:
            continue
        if not isinstance(payload, Mapping):
            raise ValueError(
                "pipelines.execution canonical result payload requires mappings at "
                f"result['{key}'] when the field is present."
            )
        canonical_result[key] = {str(field): value for field, value in payload.items()}
    canonical_result.setdefault("status", "completed")
    canonical_result.setdefault("artifact_root", None)
    canonical_result.setdefault("run_manifest", None)
    canonical_result.setdefault("environment", None)
    canonical_result.setdefault("provenance", None)
    canonical_result.setdefault("commit", None)
    return canonical_result


def _canonical_metrics_artifact_payload(
    metrics: object,
    *,
    protocol: object,
    completed_run: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(metrics, Mapping):
        raise ValueError(
            "pipelines.execution canonical result payload requires a mapping at result['metrics']."
        )
    nested_metrics = metrics.get("metrics")
    if isinstance(nested_metrics, Mapping):
        metrics_source = metrics
        numeric_metrics = _merged_numeric_metrics_payload(metrics)
        missing_shape = False
    else:
        metrics_source = None
        numeric_metrics = _merged_numeric_metrics_payload(metrics)
        missing_shape = True
    primary_metric = str(
        (metrics_source or {}).get("primary_metric")
        or (protocol if isinstance(protocol, Mapping) else {}).get("primary_metric")
        or completed_run.get("primary_metric")
        or ""
    ).strip() or None
    task_kind = str(
        (metrics_source or {}).get("task_kind")
        or (protocol if isinstance(protocol, Mapping) else {}).get("task_kind")
        or completed_run.get("task_kind")
        or ""
    ).strip() or None
    protocol_id = str(
        (metrics_source or {}).get("protocol_id")
        or (protocol if isinstance(protocol, Mapping) else {}).get("protocol_id")
        or completed_run.get("protocol_id")
        or ""
    ).strip() or None
    missing_fields = [
        field_name
        for field_name, value in (
            ("primary_metric", primary_metric),
            ("task_kind", task_kind),
            ("protocol_id", protocol_id),
        )
        if value is None
    ]
    if missing_fields:
        missing = ", ".join(f"metrics['{field_name}']" for field_name in missing_fields)
        raise ValueError(
            "pipelines.execution canonical result payload requires a fully materialized metrics "
            f"envelope. Missing {missing}."
        )
    if missing_shape and not numeric_metrics:
        raise ValueError(
            "pipelines.execution canonical result payload requires metrics.metrics in the "
            "materialized metrics envelope."
        )
    return build_metrics_artifact(
        numeric_metrics,
        primary_metric=primary_metric,
        task_kind=task_kind,
        protocol_id=protocol_id,
    )


_METRICS_ARTIFACT_METADATA_KEYS = frozenset(
    {
        "metrics",
        "primary_metric",
        "primary_metric_value",
        "protocol_id",
        "task_kind",
    }
)

_CANONICAL_RUNTIME_PROTOCOL_FIELDS = frozenset(
    {
        "task_kind",
        "protocol_id",
        "primary_metric",
        "prediction_key",
        "target_key",
    }
)


def _merged_numeric_metrics_payload(metrics: Mapping[str, Any]) -> dict[str, float]:
    merged_metrics = {
        str(key): float(value)
        for key, value in metrics.items()
        if str(key) not in _METRICS_ARTIFACT_METADATA_KEYS and isinstance(value, int | float)
    }
    nested_metrics = metrics.get("metrics")
    if not isinstance(nested_metrics, Mapping):
        return merged_metrics
    merged_metrics.update(
        {
            str(key): float(value)
            for key, value in nested_metrics.items()
            if isinstance(value, int | float)
        }
    )
    return merged_metrics


def _has_canonical_runtime_protocol(protocol: object) -> bool:
    if not isinstance(protocol, Mapping):
        return False
    for field in _CANONICAL_RUNTIME_PROTOCOL_FIELDS:
        value = protocol.get(field)
        if not isinstance(value, str) or not value.strip():
            return False
    return True


def _apply_execution_runtime_overlay(
    spec: _RuntimePipelineSpec,
    runtime: RuntimeSpec | Mapping[str, Any] | None,
) -> _RuntimePipelineSpec:
    overlay_source = runtime.to_dict() if isinstance(runtime, RuntimeSpec) else runtime
    runtime_overlay = _runtime_payload_from_mapping(overlay_source)
    if not runtime_overlay:
        return spec
    updated = _RuntimePipelineSpec.from_dict(spec.to_dict())
    updated.runtime.update(runtime_overlay)
    return updated


def _apply_execution_mode_overlay(
    spec: _RuntimePipelineSpec,
    mode: str | None,
) -> _RuntimePipelineSpec:
    if mode not in {"train", "evaluate"}:
        return spec
    updated = _RuntimePipelineSpec.from_dict(spec.to_dict())
    updated.protocol["mode"] = mode
    return updated


def _resolved_execution_spec(
    execution_payload: Mapping[str, Any],
    *,
    mode: str | None,
    runtime: RuntimeSpec | Mapping[str, Any] | None,
) -> _RuntimePipelineSpec:
    execution_spec = _RuntimePipelineSpec.from_dict(dict(execution_payload))
    execution_spec = _apply_execution_mode_overlay(execution_spec, mode)
    execution_spec = _apply_execution_runtime_overlay(execution_spec, runtime)
    execution_spec.validate()
    return execution_spec


def _runtime_request_overlay(
    *,
    runtime: RuntimeSpec | Mapping[str, Any] | None,
    device: str | torch.device | None,
    seed: int | None,
) -> RuntimeSpec | Mapping[str, Any] | None:
    overlay = _runtime_payload_from_mapping(
        runtime.to_dict() if isinstance(runtime, RuntimeSpec) else runtime
    )
    if device is not None:
        overlay["device"] = str(device)
    if seed is not None:
        overlay["seed"] = int(seed)
    if not overlay:
        return None
    return overlay


def _describe_tree_payload(value: object) -> dict[str, object] | None:
    if isinstance(value, Mapping):
        return {str(key): payload for key, payload in value.items()}
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return {str(key): item for key, item in payload.items()}
    return None


def _sample_describe_tree_payload(dataset_source: Any | None) -> dict[str, object] | None:
    if dataset_source is None:
        return None
    candidates: list[Any] = []
    if isinstance(dataset_source, Mapping):
        train_dataset = dataset_source.get("train")
        if train_dataset is not None:
            candidates.append(train_dataset)
        candidates.extend(
            dataset
            for split_name, dataset in dataset_source.items()
            if split_name != "train"
        )
    else:
        candidates.append(dataset_source)

    for candidate in candidates:
        describe_provider = getattr(candidate, "sample_describe_tree", None)
        if callable(describe_provider):
            describe_payload = _describe_tree_payload(describe_provider())
            if describe_payload is not None:
                return describe_payload
        if not isinstance(candidate, Dataset) or len(candidate) == 0:
            continue
        sample, _ = candidate[0]
        sample_describe = getattr(sample, "describe_tree", None)
        if callable(sample_describe):
            describe_payload = _describe_tree_payload(sample_describe())
            if describe_payload is not None:
                return describe_payload
    return None


@dataclass(frozen=True)
class _ExecutionAdapter:
    adapter_id: str
    target_mode: str


_EXECUTION_ADAPTERS: tuple[_ExecutionAdapter, ...] = (
    _ExecutionAdapter(adapter_id="scalar_index_supervised", target_mode="tensor"),
    _ExecutionAdapter(adapter_id="structured_tensor_mapping", target_mode="mapping"),
)

def _resolved_protocol(spec: _RuntimePipelineSpec, task_spec: TaskSpec) -> dict[str, Any]:
    protocol = dict(spec.protocol)
    required_fields = ("task_kind", "protocol_id", "primary_metric")
    missing_fields = [
        field for field in required_fields if not str(protocol.get(field, "")).strip()
    ]
    if missing_fields:
        missing = ", ".join(f"spec.protocol['{field}']" for field in missing_fields)
        raise ValueError(
            "pipelines.execution requires explicit protocol contract fields. "
            f"Missing {missing}; runner no longer fills task_kind/protocol_id/primary_metric defaults."
        )
    if str(protocol["task_kind"]).strip() != str(task_spec.kind).strip():
        raise ValueError(
            "spec.protocol['task_kind'] must match the canonical task contract. "
            f"Got {protocol['task_kind']!r} for task kind {task_spec.kind!r}."
        )
    if str(protocol["primary_metric"]).strip() != str(task_spec.output_schema.primary_metric).strip():
        raise ValueError(
            "spec.protocol['primary_metric'] must match TaskSpec.output_schema.primary_metric. "
            f"Got {protocol['primary_metric']!r} for canonical primary metric "
            f"{task_spec.output_schema.primary_metric!r}."
        )
    target_key = str(protocol.get("target_key", "")).strip()
    if target_key:
        declared_fields = {str(field) for field in task_spec.target_schema.fields}
        if target_key not in declared_fields:
            raise ValueError(
                "spec.protocol['target_key'] must match TaskSpec.target_schema.fields. "
                f"Got {target_key!r}; declared fields: {sorted(declared_fields)!r}."
            )
    return protocol


def _resolve_execution_adapter(spec: _RuntimePipelineSpec) -> _ExecutionAdapter:
    explicit_adapter = str(spec.protocol.get("execution_adapter", "")).strip()
    if not explicit_adapter:
        raise ValueError(
            "pipelines.execution requires explicit spec.protocol['execution_adapter']; "
            "runner no longer infers adapters from task taxonomy or spec.task fallbacks."
        )
    for adapter in _EXECUTION_ADAPTERS:
        if adapter.adapter_id == explicit_adapter:
            return adapter
    supported = ", ".join(adapter.adapter_id for adapter in _EXECUTION_ADAPTERS)
    raise ValueError(
        f"Unsupported execution adapter '{explicit_adapter}'. Supported adapters: {supported}"
    )


def _resolve_task_spec(spec: _RuntimePipelineSpec) -> TaskSpec:
    task_id = str(spec.task.get("task_id", "")).strip()
    if not task_id:
        raise ValueError(
            "pipelines.execution requires spec.task['task_id'] to load the canonical TaskSpec."
        )
    return load_task(task_id)


def _require_train_or_evaluate_mode(mode: str) -> str:
    if mode not in {"train", "evaluate"}:
        raise ValueError(
            "pipelines.execution only supports explicit mode='train' or mode='evaluate' for "
            "task execution."
        )
    return mode


def _resolve_execution_mode(spec: _RuntimePipelineSpec, mode: str | None) -> str:
    if mode is not None:
        if mode == "infer":
            return mode
        return _require_train_or_evaluate_mode(mode)

    protocol_mode = str(spec.protocol.get("mode", "")).strip()
    if protocol_mode:
        return _require_train_or_evaluate_mode(protocol_mode)

    raise ValueError(
        "pipelines.execution requires an explicit execution mode. "
        "Pass mode='train'/'evaluate' or set spec.protocol['mode'] accordingly."
    )


def load_execution_target(target: BenchmarkSpec) -> BenchmarkSpec:
    if not isinstance(target, BenchmarkSpec):
        raise TypeError(
            "pipelines.execution only accepts canonical BenchmarkSpec objects. "
            "Spec serde/loading belongs to octosense.specs."
        )
    return target


def _epoch_history(
    events: list[dict[str, object]],
    *,
    metric_field: str | None = None,
) -> list[dict[str, float]]:
    history: list[dict[str, float]] = []
    for event in events:
        if event.get("event") != "epoch_end":
            continue
        payload: dict[str, float] = {
            "epoch": float(event["epoch"]),
            "train_loss": float(event["train_loss"]),
        }
        if metric_field is not None and metric_field in event:
            payload[metric_field] = float(event[metric_field])
        if "val_loss" in event:
            payload["val_loss"] = float(event["val_loss"])
        if "epoch_duration_sec" in event:
            payload["epoch_duration_sec"] = float(event["epoch_duration_sec"])
        history.append(payload)
    return history


def _require_training_field(spec: _RuntimePipelineSpec, field_name: str) -> object:
    value = spec.training.get(field_name)
    if value is None:
        raise ValueError(
            "pipelines.execution requires explicit training config fields. "
            f"Missing spec.training['{field_name}']."
        )
    if isinstance(value, str) and not value.strip():
        raise ValueError(
            f"spec.training['{field_name}'] must be a non-empty string when provided."
        )
    return value


def _mapping_protocol_keys(task_spec: TaskSpec, spec: _RuntimePipelineSpec) -> tuple[str, str]:
    protocol = dict(spec.protocol)
    protocol_prediction_key = str(protocol.get("prediction_key", "")).strip()
    target_key = str(protocol.get("target_key", "")).strip()
    if not protocol_prediction_key or not target_key:
        raise ValueError(
            "structured tensor execution requires explicit spec.protocol['prediction_key'] "
            "and spec.protocol['target_key']."
        )
    declared_fields = {str(field) for field in task_spec.target_schema.fields}
    if target_key not in declared_fields:
        raise ValueError(
            "spec.protocol['target_key'] must match TaskSpec.target_schema.fields. "
            f"Got {target_key!r}; declared fields: {sorted(declared_fields)!r}."
        )
    # Structured mapping models emit canonical task fields (for example
    # "pose_keypoints"), not the evaluator envelope key ("predictions").
    # Keep the outward protocol canonical and resolve the model-side prediction
    # tensor from the canonical structured target field instead.
    return target_key, target_key


def _reference_dataset_view(dataset_source: object) -> DatasetView | None:
    if isinstance(dataset_source, DatasetView):
        return dataset_source
    if not isinstance(dataset_source, Mapping):
        return None
    for split_name in ("train", "val", "test"):
        split_view = dataset_source.get(split_name)
        if isinstance(split_view, DatasetView):
            return split_view
    return None


def _resolve_structured_target_bridge(
    dataset_source: object | None,
    *,
    task_spec: TaskSpec,
) -> dict[str, str] | None:
    split_view = _reference_dataset_view(dataset_source)
    if split_view is None:
        return None
    try:
        bridge = split_view.get_execution_target_bridge()
    except AttributeError:
        return None
    if not isinstance(bridge, Mapping):
        return None
    declared_fields = {str(field).strip() for field in task_spec.target_schema.fields if str(field).strip()}
    if not declared_fields:
        return None
    canonical_bridge = {
        canonical_field: str(concrete_field).strip()
        for canonical_field, concrete_field in bridge.items()
        if canonical_field in declared_fields
        and str(concrete_field).strip()
    }
    if not canonical_bridge:
        return None
    if all(canonical_field == concrete_field for canonical_field, concrete_field in canonical_bridge.items()):
        return None
    return canonical_bridge


def _canonicalize_structured_mapping(
    payload: Mapping[str, object],
    *,
    target_field_bridge: Mapping[str, str] | None,
) -> dict[str, object]:
    normalized = {str(key): value for key, value in payload.items()}
    if not isinstance(target_field_bridge, Mapping):
        return normalized
    canonicalized = dict(normalized)
    for canonical_field, concrete_field in target_field_bridge.items():
        if canonical_field in canonicalized:
            continue
        concrete_name = str(concrete_field).strip()
        if concrete_name and concrete_name in normalized:
            canonicalized[str(canonical_field)] = normalized[concrete_name]
    return canonicalized


def _resolve_loss_function(
    spec: _RuntimePipelineSpec,
) -> tuple[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
    objective = str(_require_training_field(spec, "loss")).strip()
    if not objective:
        raise ValueError("spec.training['loss'] must be a non-empty string.")
    loss_params = spec.training.get("loss_params", {})
    if loss_params is None:
        loss_params = {}
    if not isinstance(loss_params, Mapping):
        raise TypeError("spec.training['loss_params'] must be a mapping when provided.")

    functional_candidates = [objective]
    if not objective.endswith("_loss"):
        functional_candidates.append(f"{objective}_loss")
    for function_name in functional_candidates:
        function = getattr(torch.nn.functional, function_name, None)
        if callable(function):
            resolved_name = function_name
            resolved_params = dict(loss_params)
            return (
                resolved_name,
                lambda prediction, target, fn=function, kwargs=resolved_params: cast(
                    torch.Tensor,
                    fn(prediction, target, **kwargs),
                ),
            )

    loss_class = getattr(nn, objective, None)
    if isinstance(loss_class, type) and issubclass(loss_class, nn.Module):
        loss_module = cast(nn.Module, loss_class(**dict(loss_params)))
        return (
            objective,
            lambda prediction, target, module=loss_module: cast(
                torch.Tensor,
                module(prediction, target),
            ),
        )

    supported = sorted(
        name
        for name in dir(torch.nn.functional)
        if name.endswith("_loss") and callable(getattr(torch.nn.functional, name, None))
    )
    raise ValueError(
        "Unsupported spec.training['loss'] for execution. "
        f"Got {objective!r}; provide a torch.nn.functional '*_loss' name or a torch.nn loss "
        f"class name. Available functional losses include: {', '.join(supported)}."
    )


def _resolve_optimizer(spec: _RuntimePipelineSpec, model: nn.Module) -> torch.optim.Optimizer:
    optimizer_name = str(_require_training_field(spec, "optimizer")).strip()
    optimizer_class = getattr(torch.optim, optimizer_name, None)
    if not isinstance(optimizer_class, type) or not issubclass(
        optimizer_class,
        torch.optim.Optimizer,
    ):
        supported = sorted(
            name
            for name in dir(torch.optim)
            if isinstance(getattr(torch.optim, name, None), type)
            and issubclass(getattr(torch.optim, name), torch.optim.Optimizer)
        )
        raise ValueError(
            "Unsupported spec.training['optimizer'] for execution. "
            f"Got {optimizer_name!r}. Available optimizers: {', '.join(supported)}."
        )
    optimizer_params = spec.training.get("optimizer_params", {})
    if optimizer_params is None:
        optimizer_params = {}
    if not isinstance(optimizer_params, Mapping):
        raise TypeError("spec.training['optimizer_params'] must be a mapping when provided.")
    resolved_params = dict(optimizer_params)
    learning_rate = spec.training.get("learning_rate")
    if learning_rate is not None and "lr" not in resolved_params:
        resolved_params["lr"] = float(learning_rate)
    return cast(torch.optim.Optimizer, optimizer_class(model.parameters(), **resolved_params))


def _resolve_batch_contract(
    adapter: _ExecutionAdapter,
    *,
    task_spec: TaskSpec,
    spec: _RuntimePipelineSpec,
) -> tuple[str | None, str | None]:
    if adapter.target_mode != "mapping":
        return (None, None)
    return _mapping_protocol_keys(task_spec, spec)


def _resolve_prediction_target_pair(
    *,
    adapter: _ExecutionAdapter,
    outputs: Any,
    targets: Any,
    prediction_key: str | None,
    target_key: str | None,
    target_field_bridge: Mapping[str, str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if adapter.target_mode == "tensor":
        if not isinstance(outputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
            raise TypeError(
                "Tensor execution adapters require both model outputs and batch targets to be "
                f"torch.Tensor, got outputs={type(outputs)!r}, targets={type(targets)!r}."
            )
        return outputs, targets
    if not isinstance(outputs, Mapping) or not isinstance(targets, Mapping):
        raise TypeError(
            "Mapping execution adapters require model outputs and targets to be mappings, "
            f"got outputs={type(outputs)!r}, targets={type(targets)!r}."
        )
    if prediction_key is None or target_key is None:
        raise ValueError("Mapping execution adapters require explicit prediction/target keys.")
    canonical_outputs = _canonicalize_structured_mapping(
        outputs,
        target_field_bridge=target_field_bridge,
    )
    canonical_targets = _canonicalize_structured_mapping(
        targets,
        target_field_bridge=target_field_bridge,
    )
    prediction = canonical_outputs.get(prediction_key)
    if prediction is None:
        raise ValueError(
            "Model outputs are missing the structured execution field required by the protocol. "
            f"Expected prediction_key={prediction_key!r}; available output fields: "
            f"{sorted(canonical_outputs)!r}."
        )
    if target_key not in canonical_targets:
        raise ValueError(
            "Batch targets are missing the canonical structured target field required by the protocol. "
            f"Expected {target_key!r}; available target fields: {sorted(canonical_targets)!r}."
        )
    target = canonical_targets[target_key]
    if not isinstance(prediction, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError(
            "Selected prediction/target fields must resolve to torch.Tensor values, got "
            f"prediction={type(prediction)!r}, target={type(target)!r}."
        )
    return prediction, target


def _resolve_progress_callback(
    progress: bool | Callable[[str], None],
) -> Callable[[str], None] | None:
    if callable(progress):
        return progress
    if progress:
        return lambda message: print(message, flush=True)
    return None


@dataclass
class _TaskRunOutputs:
    metrics: dict[str, float]
    timing: dict[str, object]
    train_events: list[dict[str, object]]
    history: list[dict[str, float]]
    split_manifest: dict[str, object] | None
    sample_describe_tree: dict[str, object] | None = None


@dataclass
class _ExecutedTrainableTask:
    payload: dict[str, Any]
    dataset_source: Any | None


@dataclass(frozen=True)
class _ExecutionResult:
    trace: MetricTrace
    train_loss: float
    val: "_TaskEvaluation"
    test: "_TaskEvaluation"
    duration_sec: float


@dataclass(frozen=True)
class _TaskEvaluation:
    loss: float | None
    batches: int
    samples: int
    metrics: dict[str, float]
    benchmark: dict[str, Any]


def _protocol_output_keys(protocol: Mapping[str, Any]) -> tuple[str, str]:
    prediction_key = str(protocol.get("prediction_key", "")).strip()
    target_key = str(protocol.get("target_key", "")).strip()
    if not prediction_key or not target_key:
        raise ValueError(
            "pipelines.execution requires canonical benchmark protocol keys "
            "'prediction_key' and 'target_key' for evaluator-backed metrics."
        )
    return prediction_key, target_key


def _empty_split_summary(*, split_name: str, reason: str) -> dict[str, Any]:
    return {
        "split": split_name,
        "status": "skipped",
        "reason": reason,
        "metrics": {},
    }


def _evaluate_benchmark_metrics(
    *,
    protocol: Mapping[str, Any],
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, Any]:
    prediction_key, target_key = _protocol_output_keys(protocol)
    return _evaluate_outputs_payload(
        {
            "protocol": dict(protocol),
            "outputs": {
                prediction_key: predictions,
                target_key: targets,
            },
        }
    )


def _prefixed_metrics(split_name: str, metrics: Mapping[str, float]) -> dict[str, float]:
    return {f"{split_name}_{key}": float(value) for key, value in metrics.items()}


def _merge_task_metrics(
    *,
    train_loss: float | None,
    train_samples: int | None,
    val: _TaskEvaluation,
    test: _TaskEvaluation,
    primary_metric: str,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if train_loss is not None:
        metrics["train_loss"] = float(train_loss)
    if train_samples is not None:
        metrics["train_samples"] = float(train_samples)
    if val.loss is not None:
        metrics["val_loss"] = float(val.loss)
    metrics["val_batches"] = float(val.batches)
    metrics["val_samples"] = float(val.samples)
    metrics.update(_prefixed_metrics("val", val.metrics))
    if primary_metric in val.metrics:
        primary_metric_value = float(val.metrics[primary_metric])
        metrics[primary_metric] = primary_metric_value
        metrics["primary_metric_value"] = primary_metric_value

    if test.loss is not None:
        metrics["test_loss"] = float(test.loss)
    metrics["test_batches"] = float(test.batches)
    metrics["test_samples"] = float(test.samples)
    metrics.update(_prefixed_metrics("test", test.metrics))
    if primary_metric in test.metrics:
        metrics["test_metric_value"] = float(test.metrics[primary_metric])
    return metrics


def _full_label_histogram(labels: list[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        key = str(int(label))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def _dataset_split_manifest(
    dataset_source: Any | None,
    *,
    seed: int,
) -> dict[str, object] | None:
    if dataset_source is None:
        return None

    def _split_mapping(source: Any) -> dict[str, Dataset] | None:
        if isinstance(source, dict):
            return source if {"train", "val"}.issubset(set(source)) else None
        getter = getattr(source, "get_split", None)
        if not callable(getter):
            return None
        split_mapping: dict[str, Dataset] = {}
        for split_name in ("train", "val", "test"):
            try:
                split_dataset = getter(split_name)
            except Exception:
                continue
            if isinstance(split_dataset, Dataset):
                split_mapping[split_name] = cast(Dataset, split_dataset)
        return split_mapping if {"train", "val"}.issubset(set(split_mapping)) else None

    def _labels_for(dataset: Dataset) -> list[int] | None:
        if hasattr(dataset, "get_labels"):
            try:
                labels = dataset.get_labels()  # type: ignore[attr-defined]
            except (AttributeError, TypeError, ValueError):
                return None
            return [int(label) for label in labels]
        return None

    split_mapping = _split_mapping(dataset_source)
    if split_mapping is None:
        return None

    source_sample_count: int | None = None
    getter = getattr(dataset_source, "get_split", None)
    if callable(getter):
        try:
            source_sample_count = len(getter("all"))
        except Exception:
            source_sample_count = None

    splits_payload: dict[str, dict[str, object]] = {}
    sample_count = 0
    for split_name in ("train", "val", "test"):
        split_dataset = split_mapping.get(split_name)
        if split_dataset is None:
            continue
        split_indices = list(range(len(split_dataset)))
        labels = _labels_for(split_dataset)
        split_payload: dict[str, object] = {
            "sample_count": len(split_indices),
            "sample_ids": build_sample_ids(split_dataset, split_indices),
        }
        if labels is not None:
            split_payload["label_histogram"] = _full_label_histogram(labels)
        splits_payload[split_name] = split_payload
        sample_count += len(split_indices)

    source_metadata = dataset_source
    if isinstance(dataset_source, dict):
        source_metadata = split_mapping.get("train")

    return {
        "seed": int(seed),
        "dataset_id": str(getattr(source_metadata, "dataset_id", "")),
        "variant": str(getattr(source_metadata, "variant", "")),
        "profile_id": str(getattr(source_metadata, "profile_id", "")),
        "split_strategy": "predefined_splits",
        "protocol": "dataset_defined",
        "sample_unit": "dataset_defined",
        "dataset_path": (
            str(source_metadata.dataset_path)
            if getattr(source_metadata, "dataset_path", None) is not None
            else None
        ),
        "sample_count": sample_count,
        "source_sample_count": source_sample_count if source_sample_count is not None else sample_count,
        "splits": splits_payload,
    }

def _task_outputs_from_trainable_result(result: dict[str, Any]) -> _TaskRunOutputs:
    metrics_payload = result.get("metrics")
    if not isinstance(metrics_payload, Mapping):
        raise ValueError(
            "pipelines.execution result payload is missing a canonical 'metrics' mapping."
        )
    nested_metrics = metrics_payload.get("metrics")
    numeric_source = nested_metrics if isinstance(nested_metrics, Mapping) else metrics_payload
    metrics = {
        str(key): float(value)
        for key, value in numeric_source.items()
        if isinstance(value, int | float)
    }
    return _TaskRunOutputs(
        metrics=metrics,
        timing=cast(dict[str, object], result["timing"]),
        train_events=cast(list[dict[str, object]], result["train_events"]),
        history=cast(list[dict[str, float]], result["history"]),
        split_manifest=cast(dict[str, object] | None, result.get("split_manifest")),
        sample_describe_tree=cast(dict[str, object] | None, result.get("sample_describe_tree")),
    )


def _load_trainable_pipeline_handle(
    spec: BenchmarkSpec,
    *,
    dataset: Any | None,
) -> _PipelineExecutionHandle:
    from octosense.pipelines.api import load as load_pipeline

    handle = load_pipeline(spec=spec, dataset=dataset)
    if not isinstance(handle, _PipelineExecutionHandle):
        raise TypeError(
            "Canonical trainable pipeline build did not return an executable pipeline handle, "
            f"got {type(handle)!r}."
        )
    return handle


def _pipeline_dataset_source(handle: _PipelineExecutionHandle) -> Any | None:
    dataset_source = handle.get_execution_dataset_source()
    if dataset_source is None:
        return None
    if isinstance(dataset_source, DatasetView):
        return dataset_source
    if isinstance(dataset_source, Mapping):
        split_mapping: dict[str, DatasetView] = {}
        for split_name, split_dataset in dataset_source.items():
            if not isinstance(split_dataset, DatasetView):
                raise TypeError(
                    "Trainable pipeline handles must expose dataset_source as a DatasetView "
                    "or a split mapping of DatasetView objects."
                )
            split_mapping[str(split_name)] = split_dataset
        return split_mapping
    raise TypeError(
        "Trainable pipeline handles must expose dataset_source as a DatasetView or a "
        f"split mapping of DatasetView objects, got {type(dataset_source)!r}."
    )


def _run_execution(
    *,
    adapter: _ExecutionAdapter,
    protocol: Mapping[str, Any],
    primary_metric: str,
    model: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    train_loader: Any,
    val_loader: Any,
    test_loader: Any,
    epochs: int,
    log_every_steps: int,
    prefetched_train_batch: tuple[torch.Tensor, Any] | None,
    remaining_train_batches: Any,
    progress: Callable[[str], None] | None,
    prediction_key: str | None,
    target_key: str | None,
    target_field_bridge: Mapping[str, str] | None,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> _ExecutionResult:
    trace = MetricTrace()
    start = time.perf_counter()
    train_loss = 0.0

    def _progress(message: str) -> None:
        if progress is not None:
            progress(message)

    def _compute_loss(outputs: Any, targets: Any) -> torch.Tensor:
        prediction, target = _resolve_prediction_target_pair(
            adapter=adapter,
            outputs=outputs,
            targets=targets,
            prediction_key=prediction_key,
            target_key=target_key,
            target_field_bridge=target_field_bridge,
        )
        return loss_fn(prediction, target)

    def _evaluate(loader: Any, split_name: str) -> _TaskEvaluation:
        if loader is None:
            benchmark = _empty_split_summary(
                split_name=split_name,
                reason="loader_unavailable",
            )
            evaluation = _TaskEvaluation(
                loss=0.0,
                batches=0,
                samples=0,
                metrics={},
                benchmark=benchmark,
            )
        else:
            forward_evaluation = evaluate_loader(
                model=model,
                device=device,
                loader=loader,
                target_mode=adapter.target_mode,
                resolve_prediction_target=lambda outputs, targets: _resolve_prediction_target_pair(
                    adapter=adapter,
                    outputs=outputs,
                    targets=targets,
                    prediction_key=prediction_key,
                    target_key=target_key,
                    target_field_bridge=target_field_bridge,
                ),
                compute_loss=lambda _outputs, _targets, prediction, target: loss_fn(
                    prediction,
                    target,
                ),
            )
            prediction_tensor = forward_evaluation.predictions
            target_tensor = forward_evaluation.targets
            if prediction_tensor is None or target_tensor is None:
                benchmark = _empty_split_summary(
                    split_name=split_name,
                    reason="empty_loader",
                )
                metrics: dict[str, float] = {}
            else:
                benchmark = _evaluate_benchmark_metrics(
                    protocol=protocol,
                    predictions=prediction_tensor,
                    targets=target_tensor,
                )
                metrics = cast(dict[str, float], benchmark["metrics"])
            evaluation = _TaskEvaluation(
                loss=forward_evaluation.loss,
                batches=forward_evaluation.batches,
                samples=forward_evaluation.samples,
                metrics=metrics,
                benchmark=benchmark,
            )
        trace.events.append(
            {
                "event": "eval",
                "split": split_name,
                "loss": float(evaluation.loss or 0.0),
                "batches": int(evaluation.batches),
                "samples": int(evaluation.samples),
                **{str(key): float(value) for key, value in evaluation.metrics.items()},
            }
        )
        return evaluation

    val_evaluation = _TaskEvaluation(
        loss=0.0,
        batches=0,
        samples=0,
        metrics={},
        benchmark=_empty_split_summary(split_name="val", reason="loader_unavailable"),
    )
    for epoch_idx in range(epochs):
        model.train()
        epoch_start = time.perf_counter()
        epoch_loss = 0.0
        if epoch_idx == 0 and prefetched_train_batch is not None and remaining_train_batches is None:
            raise ValueError(
                "remaining_train_batches is required when prefetched_train_batch is provided"
            )
        epoch_batches = (
            chain((prefetched_train_batch,), remaining_train_batches)
            if epoch_idx == 0 and prefetched_train_batch is not None
            else train_loader
        )
        for step_idx, (batch_x, batch_targets) in enumerate(epoch_batches, start=1):
            batch_start = time.perf_counter()
            batch_x = batch_x.to(device)
            if adapter.target_mode == "mapping":
                targets = {key: value.to(device) for key, value in batch_targets.items()}
            else:
                targets = batch_targets.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = _compute_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            step_loss = float(loss.item())
            epoch_loss += step_loss
            trace.maybe_record_first_batch(float(time.perf_counter() - batch_start))
            if log_every_steps and (
                step_idx == 1
                or step_idx == len(train_loader)
                or (step_idx % log_every_steps) == 0
            ):
                _progress(
                    f"epoch {epoch_idx + 1}/{epochs} "
                    f"step {step_idx}/{len(train_loader)} loss={step_loss:.4f}"
                )
        train_loss = epoch_loss / max(1, len(train_loader))
        val_evaluation = _evaluate(val_loader, "val")
        trace.record_epoch_end(
            epoch=epoch_idx + 1,
            train_loss=train_loss,
            epoch_duration_sec=float(time.perf_counter() - epoch_start),
            val_loss=val_evaluation.loss,
            metric_name=f"val_{primary_metric}",
            metric_value=val_evaluation.metrics.get(primary_metric),
        )
        _progress(
            f"epoch {epoch_idx + 1}/{epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_evaluation.loss:.4f} "
            f"val_{primary_metric}={val_evaluation.metrics.get(primary_metric, 0.0):.4f}"
        )

    test_evaluation = _evaluate(test_loader, "test")
    return _ExecutionResult(
        trace=trace,
        train_loss=train_loss,
        val=val_evaluation,
        test=test_evaluation,
        duration_sec=float(time.perf_counter() - start),
    )


def _evaluate_loader_without_training_fields(
    *,
    adapter: _ExecutionAdapter,
    protocol: Mapping[str, Any],
    model: nn.Module,
    device: torch.device,
    loader: Any,
    split_name: str,
    trace: MetricTrace,
    prediction_key: str | None,
    target_key: str | None,
    target_field_bridge: Mapping[str, str] | None,
) -> dict[str, Any]:
    if loader is None:
        benchmark = _empty_split_summary(
            split_name=split_name,
            reason="loader_unavailable",
        )
        metrics: dict[str, float] = {}
        trace.events.append(
            {
                "event": "eval",
                "split": split_name,
                "batches": 0,
                "samples": 0,
                **metrics,
            }
        )
        return {
            "batches": 0,
            "samples": 0,
            "benchmark": benchmark,
        }

    forward_evaluation = evaluate_loader(
        model=model,
        device=device,
        loader=loader,
        target_mode=adapter.target_mode,
        resolve_prediction_target=lambda outputs, targets: _resolve_prediction_target_pair(
            adapter=adapter,
            outputs=outputs,
            targets=targets,
            prediction_key=prediction_key,
            target_key=target_key,
            target_field_bridge=target_field_bridge,
        ),
    )

    if forward_evaluation.predictions is None or forward_evaluation.targets is None:
        benchmark = _empty_split_summary(
            split_name=split_name,
            reason="empty_loader",
        )
        metrics: dict[str, float] = {}
    else:
        benchmark = _evaluate_benchmark_metrics(
            protocol=protocol,
            predictions=forward_evaluation.predictions,
            targets=forward_evaluation.targets,
        )
        metrics = cast(dict[str, float], benchmark["metrics"])

    trace.events.append(
        {
            "event": "eval",
            "split": split_name,
            "batches": forward_evaluation.batches,
            "samples": forward_evaluation.samples,
            **metrics,
        }
    )
    return {
        "batches": forward_evaluation.batches,
        "samples": forward_evaluation.samples,
        "benchmark": benchmark,
    }


def _execute_trainable_task_spec(
    spec: _RuntimePipelineSpec,
    *,
    handle: Any,
    runtime: RuntimeSpec,
    device: torch.device,
    mode: str,
    progress: Callable[[str], None] | None,
) -> _ExecutedTrainableTask:
    task_spec = _resolve_task_spec(spec)
    adapter = _resolve_execution_adapter(spec)
    protocol = _resolved_protocol(spec, task_spec)
    model, dataloaders = handle._build_execution_components(
        seed=runtime.seed,
        device=str(device),
        batch_size=runtime.batch_size,
        num_workers=runtime.num_workers,
    )
    prediction_key, target_key = _resolve_batch_contract(
        adapter,
        task_spec=task_spec,
        spec=spec,
    )
    dataset_source = _pipeline_dataset_source(handle)
    target_field_bridge = _resolve_structured_target_bridge(
        dataset_source,
        task_spec=task_spec,
    )
    train_iter: Any | None = None
    prefetched_train_batch: tuple[torch.Tensor, Any] | None = None
    if mode == "train":
        train_iter = iter(dataloaders.train_loader)
        prefetched_train_batch = next(train_iter)
        first_train_x, first_train_targets = prefetched_train_batch
    else:
        first_train_x, first_train_targets = next(iter(dataloaders.train_loader))
    val_x: torch.Tensor | None = None
    if dataloaders.val_loader is not None:
        val_x, _ = next(iter(dataloaders.val_loader))
    split_manifest = _dataset_split_manifest(
        dataset_source,
        seed=int(runtime.seed),
    )
    sample_describe_tree = _sample_describe_tree_payload(dataset_source)
    common_payload = {
        "device": str(device),
        "train_batch_shape": tuple(first_train_x.shape),
        "val_batch_shape": tuple(val_x.shape) if val_x is not None else None,
        "split_manifest": split_manifest,
        "sample_describe_tree": sample_describe_tree,
        "mode": mode,
    }
    if mode == "train":
        log_every_steps = int(_require_training_field(spec, "log_every_steps"))
        optimizer = _resolve_optimizer(spec, model)
        loss_name, loss_fn = _resolve_loss_function(spec)
        epochs = int(runtime.epochs)
        execution = _run_execution(
            adapter=adapter,
            protocol=protocol,
            primary_metric=str(protocol["primary_metric"]),
            model=model,
            device=device,
            optimizer=optimizer,
            train_loader=dataloaders.train_loader,
            val_loader=dataloaders.val_loader,
            test_loader=dataloaders.test_loader,
            epochs=epochs,
            log_every_steps=log_every_steps,
            progress=progress,
            prefetched_train_batch=prefetched_train_batch,
            remaining_train_batches=train_iter,
            prediction_key=prediction_key,
            target_key=target_key,
            target_field_bridge=target_field_bridge,
            loss_fn=loss_fn,
        )
        payload = {
            **common_payload,
            "metrics": _merge_task_metrics(
                train_loss=execution.train_loss,
                train_samples=len(getattr(dataloaders.train_loader, "dataset", [])),
                val=execution.val,
                test=execution.test,
                primary_metric=str(protocol["primary_metric"]),
            ),
            "history": _epoch_history(
                execution.trace.events,
                metric_field=f"val_{str(protocol['primary_metric'])}",
            ),
            "timing": execution.trace.build_timing_payload(
                duration_sec=execution.duration_sec,
                peak_memory_mb_value=peak_memory_mb(device),
                train_samples=len(getattr(dataloaders.train_loader, "dataset", [])),
                epochs=epochs,
            ),
            "train_events": list(execution.trace.events),
            "loss_name": loss_name,
        }
        return _ExecutedTrainableTask(
            payload=payload,
            dataset_source=dataset_source,
        )

    trace = MetricTrace()
    start = time.perf_counter()
    val_evaluation = _evaluate_loader_without_training_fields(
        adapter=adapter,
        protocol=protocol,
        model=model,
        device=device,
        loader=dataloaders.val_loader,
        split_name="val",
        trace=trace,
        prediction_key=prediction_key,
        target_key=target_key,
        target_field_bridge=target_field_bridge,
    )
    test_evaluation = _evaluate_loader_without_training_fields(
        adapter=adapter,
        protocol=protocol,
        model=model,
        device=device,
        loader=dataloaders.test_loader,
        split_name="test",
        trace=trace,
        prediction_key=prediction_key,
        target_key=target_key,
        target_field_bridge=target_field_bridge,
    )
    payload = {
        **common_payload,
        "metrics": _merge_task_metrics(
            train_loss=None,
            train_samples=None,
            val=_TaskEvaluation(
                loss=None,
                batches=int(val_evaluation["batches"]),
                samples=int(val_evaluation["samples"]),
                metrics=cast(dict[str, float], val_evaluation["benchmark"]["metrics"]),
                benchmark=cast(dict[str, Any], val_evaluation["benchmark"]),
            ),
            test=_TaskEvaluation(
                loss=None,
                batches=int(test_evaluation["batches"]),
                samples=int(test_evaluation["samples"]),
                metrics=cast(dict[str, float], test_evaluation["benchmark"]["metrics"]),
                benchmark=cast(dict[str, Any], test_evaluation["benchmark"]),
            ),
            primary_metric=str(protocol["primary_metric"]),
        ),
        "history": [],
        "timing": trace.build_timing_payload(
            duration_sec=float(time.perf_counter() - start),
            peak_memory_mb_value=peak_memory_mb(device),
            train_samples=0,
            epochs=0,
        ),
        "train_events": list(trace.events),
    }
    return _ExecutedTrainableTask(
        payload=payload,
        dataset_source=dataset_source,
    )


def _execute_trainable_inference_spec(
    spec: _RuntimePipelineSpec,
    *,
    handle: Any,
    runtime: RuntimeSpec,
    device: torch.device,
    inputs: Any,
) -> torch.Tensor | list[torch.Tensor]:
    model, _ = handle._build_execution_components(
        seed=runtime.seed,
        device=str(device),
        batch_size=runtime.batch_size,
        num_workers=runtime.num_workers,
    )
    prepared = handle._prepare_inference_inputs(inputs)
    if isinstance(prepared, list):
        return infer_batches(model, prepared, device=device)
    return infer_pipeline(model, prepared, device=device)


def _reject_stale_execution_payload_keys(execution_payload: Mapping[str, Any]) -> None:
    if "artifacts" in execution_payload:
        raise ValueError(
            "RuntimePipelineSpec.artifacts is no longer accepted at the builder/runner boundary. "
            "Pass output_root=... to handle.train()/evaluate() or execute_pipeline_handle(...)."
        )


def execute_pipeline_handle(
    handle: _PipelineExecutionHandle,
    *,
    benchmark_spec: BenchmarkSpec | None = None,
    output_root: str | Path | None = None,
    run_name: str | None = None,
    device: str | torch.device | None = None,
    seed: int | None = None,
    runtime: RuntimeSpec | Mapping[str, Any] | None = None,
    inputs: Any | None = None,
    progress: bool | Callable[[str], None] | None = None,
    mode: str | None = None,
) -> dict[str, Any] | torch.Tensor | list[torch.Tensor]:
    """Execute a previously materialized canonical pipeline handle."""

    if not isinstance(handle, _PipelineExecutionHandle):
        raise TypeError(
            "execute_pipeline_handle(...) requires an executable pipeline handle."
        )
    runtime_overlay = _runtime_request_overlay(runtime=runtime, device=device, seed=seed)
    execution_payload = handle._resolve_execution_payload(mode=mode, runtime=None)
    if not isinstance(execution_payload, Mapping):
        raise TypeError(
            "Pipeline handles must resolve a mapping payload before execution."
        )
    _reject_stale_execution_payload_keys(execution_payload)
    canonical_benchmark_spec = _resolve_canonical_benchmark_spec(handle, benchmark_spec)
    execution_spec = _resolved_execution_spec(
        execution_payload,
        mode=mode,
        runtime=runtime_overlay,
    )
    materialization_output_root = _resolved_output_root(output_root)
    resolved_mode = _resolve_execution_mode(execution_spec, mode)

    runtime_spec = _runtime_spec_for_execution(
        execution_spec,
        mode=resolved_mode,
    )

    requested_workers = int(runtime_spec.num_workers)
    effective_workers, worker_status = _resolve_execution_workers(requested_workers)

    execution_runtime = RuntimeSpec.from_dict(runtime_spec.to_dict())
    execution_runtime.num_workers = effective_workers

    resolved_device = _resolve_device(execution_runtime.device)
    reset_peak_memory_stats(resolved_device)
    _set_seed(int(execution_runtime.seed))
    progress_callback = _resolve_progress_callback(
        execution_spec.runtime.get("progress", False) if progress is None else progress
    )
    if resolved_mode == "infer":
        if inputs is None:
            raise ValueError("execute_pipeline_handle(..., mode='infer') requires explicit inputs")
        return _execute_trainable_inference_spec(
            execution_spec,
            handle=handle,
            runtime=execution_runtime,
            device=resolved_device,
            inputs=inputs,
        )

    executed = _execute_trainable_task_spec(
        spec=execution_spec,
        handle=handle,
        runtime=execution_runtime,
        device=resolved_device,
        mode=resolved_mode,
        progress=progress_callback,
    )
    completed_run = dict(executed.payload)
    resolved_run_name = _resolved_run_name(execution_spec, run_name=run_name)
    if canonical_benchmark_spec is not None:
        completed_run = _build_completed_run_payload(
            completed_run,
            benchmark_spec=canonical_benchmark_spec,
            execution_spec=execution_spec,
            execution_runtime=execution_runtime,
            dataset_source=executed.dataset_source,
            resolved_mode=resolved_mode,
            run_name=resolved_run_name,
            requested_workers=requested_workers,
            effective_workers=effective_workers,
            worker_status=worker_status,
        )
        materialization_options: BenchmarkArtifactMaterializationOptions | None = None
        if materialization_output_root is not None:
            materialization_options = BenchmarkArtifactMaterializationOptions(
                output_root=materialization_output_root,
                run_name=resolved_run_name,
                task_id=str(canonical_benchmark_spec.task.task_id),
                spec_digest=str(completed_run["spec_digest"]),
                dataset_digest=str(completed_run["dataset_digest"]),
                requested_workers=requested_workers,
                effective_workers=effective_workers,
                provenance_root=Path.cwd(),
                worker_note=cast(str | None, completed_run.get("worker_note")),
                worker_sharing_strategy=cast(
                    str | None, completed_run.get("worker_sharing_strategy")
                ),
                worker_shm_manager=cast(str | None, completed_run.get("worker_shm_manager")),
            )
        finalized_run = _materialize_completed_run(
            completed_run=completed_run,
            options=materialization_options,
        )
        return _canonical_completed_run_result(dict(finalized_run))
    if _has_canonical_runtime_protocol(execution_spec.protocol):
        task_spec = _resolve_task_spec(execution_spec)
        protocol = _resolved_protocol(execution_spec, task_spec)
        completed_run["protocol"] = protocol
    return _canonical_completed_run_result(completed_run)


def execute_target(
    spec: BenchmarkSpec,
    *,
    output_root: str | Path | None = None,
    run_name: str | None = None,
    device: str | torch.device | None = None,
    seed: int | None = None,
    runtime: RuntimeSpec | Mapping[str, Any] | None = None,
    dataset: Any | None = None,
    inputs: Any | None = None,
    progress: bool | Callable[[str], None] | None = None,
    mode: str | None = None,
) -> dict[str, Any] | torch.Tensor | list[torch.Tensor]:
    """Execute a canonical pipeline target from a BenchmarkSpec object."""

    benchmark_spec = load_execution_target(spec)
    handle = _load_trainable_pipeline_handle(benchmark_spec, dataset=dataset)
    canonical_benchmark_spec = handle.get_benchmark_spec()
    return execute_pipeline_handle(
        handle,
        benchmark_spec=canonical_benchmark_spec if canonical_benchmark_spec is not None else benchmark_spec,
        output_root=output_root,
        run_name=run_name,
        device=device,
        seed=seed,
        runtime=runtime,
        inputs=inputs,
        progress=progress,
        mode=mode,
    )


__all__ = [
    "load_execution_target",
    "reset_peak_memory_stats",
    "resolve_device",
    "set_seed",
    "synchronize_device",
    "write_json",
]
