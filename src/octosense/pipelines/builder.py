"""Canonical pipeline handle assembly behind ``octosense.pipelines.load(...)``."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import octosense.datasets as dataset_api
from octosense.core import Describable, DescribeNode
from octosense.datasets.catalog import (
    get_dataset_binding_payload,
    get_dataset_card,
    list_dataset_binding_ids,
)
from octosense.core.contracts.model import ModelInputContract
from octosense.datasets import DatasetView
from octosense.io.tensor import RadioTensor
from octosense.models.handles import ModelHandle
from octosense.models.boundary import get_model_input_contract, validate_model_entry
from octosense.models.handles import instantiate_model_handle
from octosense.pipelines.dataloading.datamodule import (
    ExecutionDataLoaders,
    build_execution_dataloaders,
)
from octosense.pipelines._runtime_spec import (
    PipelineExecutionHandle,
    RuntimePipelineSpec as _RuntimePipelineSpec,
    RuntimeTransformStep,
    _build_execution_spec,
)
from octosense.pipelines.execution.runner import execute_pipeline_handle
from octosense.pipelines.recipes import _resolve_pipeline_recipe
from octosense.specs.compiler.freezer import freeze_spec
from octosense.specs.compiler.resolver import (
    ReferenceResolverContext,
    load_transform_preset_payload,
)
from octosense.specs.schemas.benchmark import BenchmarkSpec
from octosense.benchmarks.protocols import (
    canonical_execution_adapter_for_task_kind,
    resolve_runtime_protocol_payload,
)
from octosense.specs.schemas.dataset import DatasetSpec
from octosense.specs.schemas.model import ModelSpec
from octosense.specs.schemas.runtime import RuntimeSpec
from octosense.specs.schemas.task import TaskSpec as BenchmarkTaskSpec
from octosense.specs.schemas.transform import TransformSpec, TransformStepSpec
from octosense.tasks import load as load_task
from octosense.tasks.definitions import TaskSpec
from octosense.transforms.api import Sequential
from octosense.transforms.adapters.auto_align import AutoAlign  # noqa: F401

@dataclass(frozen=True)
class _PipelineBuildRequest:
    recipe_id: str | None = None
    task_id: str | None = None
    task_binding: str | None = None
    model_id: str | None = None
    weights_id: str | None = None
    entry_overrides: dict[str, Any] | None = None
    transform_steps: tuple[RuntimeTransformStep, ...] | None = None
    dataset_id: str | None = None
    modalities: tuple[str, ...] | None = None
    dataset: Any | None = None
    variant: str | None = None
    path: str | Path | None = None
    input_selection: dict[str, object] | None = None


@dataclass(frozen=True)
class _TrainablePipelineBlueprint:
    pipeline_id: str
    spec: _RuntimePipelineSpec
    description: str = ""


def _normalize_optional_string_selector(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"pipelines.load(...) requires non-empty {name} selectors")
    return normalized


def _load_task_spec(task_id: str) -> TaskSpec:
    return load_task(_normalize_optional_string_selector("task_id", task_id))


def _resolve_requested_dataset(
    *,
    dataset: Any | None,
    dataset_id: str | None,
    modalities: Sequence[str] | str | None,
    variant: str | None,
    split_scheme: str | None = None,
    task_binding: str | None = None,
    path: str | Path | None,
) -> Any | None:
    if dataset is not None and dataset_id is not None:
        raise ValueError(
            "pipelines.load(...) accepts either dataset=... or dataset_id=..., not both"
        )
    if dataset is not None or dataset_id is None:
        return dataset

    resolved_modalities = _normalize_modalities_selector(modalities)
    return dataset_api.load(
        dataset_id,
        modalities=resolved_modalities,
        variant=variant,
        split_scheme=_normalize_optional_string_selector("split_scheme", split_scheme),
        task_binding=_normalize_optional_string_selector("task_binding", task_binding),
        path=str(path) if isinstance(path, Path) else path,
    )


def materialize_dataset_selector(
    *,
    dataset: Any | None = None,
    dataset_id: str | None = None,
    modalities: Sequence[str] | str | None = None,
    variant: str | None = None,
    split_scheme: str | None = None,
    task_binding: str | None = None,
    path: str | Path | None = None,
    input_selection: Mapping[str, object] | None = None,
) -> Any | None:
    dataset_selector = _resolve_requested_dataset(
        dataset=dataset,
        dataset_id=_normalize_optional_string_selector("dataset_id", dataset_id),
        modalities=modalities,
        variant=_normalize_optional_string_selector("variant", variant),
        split_scheme=split_scheme,
        task_binding=task_binding,
        path=path,
    )
    if dataset_selector is None:
        return None
    if not isinstance(input_selection, Mapping):
        return dataset_selector

    modality = input_selection.get("modality")
    if not isinstance(modality, str) or not modality.strip():
        return dataset_selector
    node_id = input_selection.get("node_id")
    return _project_dataset_input(
        _resolve_dataset_source(dataset_selector),
        modality=modality.strip(),
        node_id=None if node_id in {None, ""} else int(node_id),
    )


def _mapping_view(value: object) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
    return {}


def _resolve_canonical_task_binding(
    *,
    dataset_id: str | None,
    task_id: str | None,
) -> str | None:
    if dataset_id is None or task_id is None:
        return None
    supported_bindings = list_dataset_binding_ids(dataset_id, binding_kind="task_binding")
    matching_bindings = [
        binding_id
        for binding_id in supported_bindings
        if str(
            get_dataset_binding_payload(
                dataset_id,
                binding_kind="task_binding",
                binding_id=binding_id,
            ).get("task_id", "")
            or ""
        ).strip()
        == task_id
    ]
    if not matching_bindings:
        supported = ", ".join(sorted(supported_bindings)) or "<none>"
        raise ValueError(
            f"Dataset '{dataset_id}' does not declare a task_binding for task_id={task_id!r}. "
            f"Supported task_bindings: {supported}"
        )
    if len(matching_bindings) > 1:
        bindings = ", ".join(sorted(matching_bindings))
        raise ValueError(
            f"Dataset '{dataset_id}' maps task_id={task_id!r} to multiple task_bindings: "
            f"{bindings}. Pass task_binding explicitly."
        )
    return matching_bindings[0]


def _resolve_canonical_modalities(
    *,
    dataset_id: str | None,
    task_id: str | None,
    modalities: Sequence[str] | None,
    variant: str | None,
    path: str | Path | None,
) -> tuple[str, ...] | None:
    if modalities is not None:
        return tuple(modalities)
    if dataset_id is None:
        return None
    del task_id, variant, path
    supported_modalities = tuple(get_dataset_card(dataset_id).modalities)
    if len(supported_modalities) == 1:
        return supported_modalities
    if not supported_modalities:
        return None
    available = ", ".join(supported_modalities)
    raise ValueError(
        f"Dataset '{dataset_id}' requires explicit modalities because it exposes multiple "
        f"modalities. Supported modalities: {available}"
    )


def _normalize_modalities_selector(
    value: Sequence[str] | str | None,
) -> tuple[str, ...] | None:
    if value is None:
        return None
    raw_items: Sequence[str] | tuple[str, ...]
    if isinstance(value, str):
        raw_items = (value,)
    elif isinstance(value, Sequence):
        raw_items = value
    else:
        raise TypeError("dataset modalities selector must be a string or sequence of strings")

    normalized = tuple(str(item).strip() for item in raw_items if str(item).strip())
    return normalized or None


def _normalize_transform_selector(
    transform: object | None,
) -> tuple[RuntimeTransformStep, ...] | None:
    if transform is None:
        return None

    if isinstance(transform, Sequential):
        transforms = list(transform.transforms)
    elif isinstance(transform, (list, tuple)):
        transforms = list(transform)
    else:
        transforms = [transform]

    return tuple(RuntimeTransformStep.from_transform(item) for item in transforms)


def _update_present_fields(
    target: dict[str, Any],
    payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return target
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        target[str(key)] = copy.deepcopy(value)
    return target


def _task_payload_for_lane(spec: Any) -> dict[str, Any]:
    return _mapping_view(getattr(spec, "task", None))


def _model_payload_for_lane(spec: Any) -> dict[str, Any]:
    return _mapping_view(getattr(spec, "model", None))


def _spec_task_id_for_lane(spec: Any) -> str:
    task = _task_payload_for_lane(spec)
    value = task.get("task_id")
    return value.strip() if isinstance(value, str) and value.strip() else ""


def _normalize_trainable_model(model: str) -> str:
    normalized = model.strip()
    if not normalized:
        raise ValueError("Trainable pipeline model ids must be non-empty strings")
    return normalized.lower()


def _normalize_trainable_model_ref(model: str | ModelHandle | nn.Module) -> str:
    if isinstance(model, ModelHandle):
        return _normalize_trainable_model(model.model_id)
    if isinstance(model, nn.Module):
        raise TypeError("_normalize_trainable_model_ref expects a registry id or ModelHandle")
    return _normalize_trainable_model(model)


def _render_model_selector(model_ref: str | ModelHandle | nn.Module) -> str:
    if isinstance(model_ref, nn.Module):
        return model_ref.__class__.__name__
    if isinstance(model_ref, ModelHandle):
        if model_ref.weights_id is not None:
            return f"{model_ref.model_id}:{model_ref.weights_id}"
        return model_ref.model_id
    return _normalize_trainable_model(model_ref)


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _coerce_dataset_leaf(value: object | None) -> DatasetView | None:
    if isinstance(value, DatasetView):
        return value
    return None


def _dataset_split_mapping(dataset_source: object | None) -> dict[str, DatasetView] | None:
    if dataset_source is None:
        return None
    if isinstance(dataset_source, Mapping):
        mapping: dict[str, DatasetView] = {}
        for split_name, split_value in dataset_source.items():
            dataset = _coerce_dataset_leaf(split_value)
            if dataset is not None:
                mapping[str(split_name)] = dataset
        return mapping if {"train", "val"}.issubset(set(mapping)) else None
    if not isinstance(dataset_source, DatasetView):
        return None

    split_mapping: dict[str, DatasetView] = {}
    for split_name in ("train", "val", "test"):
        try:
            split_view = dataset_source.get_split(split_name)
        except Exception:
            continue
        if isinstance(split_view, DatasetView):
            split_mapping[split_name] = split_view
    return split_mapping if {"train", "val"}.issubset(set(split_mapping)) else None


def _reference_dataset_source(dataset_source: object | None) -> DatasetView | None:
    if dataset_source is None:
        return None
    dataset_leaf = _coerce_dataset_leaf(dataset_source)
    if dataset_leaf is not None:
        return dataset_leaf
    split_mapping = _dataset_split_mapping(dataset_source)
    if split_mapping is not None:
        return split_mapping.get("train") or next(iter(split_mapping.values()), None)
    return None


def _source_dataset_id(dataset_source: object | None) -> str | None:
    source = _reference_dataset_source(dataset_source)
    candidate = getattr(source, "dataset_id", None) if source is not None else None
    if isinstance(candidate, str) and candidate.strip():
        return candidate
    return None


def _normalize_input_selection_payload(value: object | None) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    modality = value.get("modality")
    if not isinstance(modality, str) or not modality.strip():
        return None
    selection: dict[str, object] = {"modality": modality.strip()}
    node_id = value.get("node_id")
    if node_id is not None:
        selection["node_id"] = int(node_id)
    return selection


def _source_input_selection(dataset_source: object | None) -> dict[str, object] | None:
    source = _reference_dataset_source(dataset_source)
    if source is None:
        return None

    selection = _normalize_input_selection_payload(getattr(source, "input_selection", None))
    if selection is not None:
        return selection

    get_input_selection = getattr(source, "get_input_selection", None)
    if callable(get_input_selection):
        selection = _normalize_input_selection_payload(get_input_selection())
        if selection is not None:
            return selection

    dataset_metadata = getattr(source, "dataset_metadata", None)
    return _normalize_input_selection_payload(
        getattr(dataset_metadata, "input_selection", None)
    )


def _source_modalities(dataset_source: object | None) -> tuple[str, ...] | None:
    source = _reference_dataset_source(dataset_source)
    if source is None:
        return None

    for candidate in (
        getattr(source, "modalities", None),
        getattr(getattr(source, "dataset_metadata", None), "modalities", None),
    ):
        normalized = _normalize_modalities_selector(cast(Sequence[str] | str | None, candidate))
        if normalized is not None:
            return normalized
    input_selection = _source_input_selection(source)
    if input_selection is None:
        return None
    modality = input_selection.get("modality")
    if not isinstance(modality, str) or not modality.strip():
        return None
    return (modality.strip(),)


def _explicit_dataset_selector_payload(
    *,
    dataset_id: str | None,
    dataset_source: object | None,
    variant: str | None,
    path: str | Path | None,
) -> dict[str, object]:
    payload: dict[str, object] = {}
    resolved_dataset_id = dataset_id or _source_dataset_id(dataset_source)
    if resolved_dataset_id is not None:
        payload["dataset_id"] = resolved_dataset_id

    resolved_variant = variant
    if resolved_variant is None:
        source = _reference_dataset_source(dataset_source)
        candidate = getattr(source, "variant", None) if source is not None else None
        if isinstance(candidate, str) and candidate.strip():
            resolved_variant = candidate
    if resolved_variant is not None:
        payload["variant"] = str(resolved_variant)

    resolved_path = path
    if resolved_path is None:
        source = _reference_dataset_source(dataset_source)
        candidate = getattr(source, "dataset_path", None) if source is not None else None
        if isinstance(candidate, (str, Path)):
            resolved_path = candidate
    if isinstance(resolved_path, (str, Path)):
        payload["path"] = str(resolved_path)

    resolved_modalities = _source_modalities(dataset_source)
    if resolved_modalities is not None:
        payload["modalities"] = list(resolved_modalities)

    input_selection = _source_input_selection(dataset_source)
    if input_selection is not None:
        payload["input_selection"] = input_selection
    return payload


def _canonical_model_selector_payload(
    model_ref: str | ModelHandle | nn.Module | None,
    model_id: str | None = None,
    weights_id: str | None = None,
) -> dict[str, object]:
    if isinstance(model_ref, nn.Module):
        raise TypeError(
            "Canonical pipeline specs only persist explicit model selectors. "
            "Pass model_id=... instead of an instantiated nn.Module."
        )
    if isinstance(model_ref, ModelHandle):
        payload: dict[str, object] = {
            "model_id": model_ref.model_id,
        }
        if model_ref.weights_id is not None:
            payload["weights_id"] = model_ref.weights_id
        if model_ref.entry_overrides:
            payload["entry_overrides"] = copy.deepcopy(model_ref.entry_overrides)
        return payload

    resolved_model_id = model_id
    if resolved_model_id is None and isinstance(model_ref, str):
        resolved_model_id = model_ref
    if resolved_model_id is None:
        raise ValueError("Canonical pipeline specs require a model_id selector")
    normalized_model_id = _normalize_trainable_model(resolved_model_id)
    payload = {
        "model_id": normalized_model_id,
    }
    normalized_weights_id = _normalize_optional_string_selector("weights_id", weights_id)
    if normalized_weights_id is not None:
        payload["weights_id"] = normalized_weights_id
    return payload


def _freeze_benchmark_spec(spec: BenchmarkSpec) -> BenchmarkSpec:
    return BenchmarkSpec.from_dict(copy.deepcopy(spec.to_dict()))


def _normalize_benchmark_task_binding(spec: BenchmarkSpec) -> BenchmarkSpec:
    normalized = _freeze_benchmark_spec(spec)
    task_binding = str(normalized.task.task_binding or "").strip() or None
    if task_binding is None:
        raise ValueError(
            "pipelines.load(spec=...) requires spec.task.task_binding to be set explicitly; "
            "canonical task bindings are not inferred from task_id or dataset selectors."
        )
    normalized.task.task_binding = task_binding
    return normalized


def _resolve_and_freeze_benchmark_spec(
    spec: BenchmarkSpec,
    *,
    resolver_context: ReferenceResolverContext | None = None,
) -> BenchmarkSpec:
    return freeze_spec(
        _normalize_benchmark_task_binding(spec),
        resolver_context=resolver_context,
    ).spec


def _default_resolver_context_for_model_ref(
    model_ref: str | ModelHandle | nn.Module,
) -> ReferenceResolverContext:
    if isinstance(model_ref, nn.Module):
        input_contract = get_model_input_contract(model_ref)
    elif isinstance(model_ref, ModelHandle):
        input_contract = model_ref.get_input_contract()
    else:
        input_contract = ModelHandle(model_id=_normalize_trainable_model_ref(model_ref)).get_input_contract()
    return ReferenceResolverContext(
        load_transform_preset=load_transform_preset_payload,
        resolve_model_input_contract=lambda _spec: input_contract.to_dict(),
    )


def _benchmark_spec_with_transform_steps(
    *,
    base_spec: BenchmarkSpec,
    transform_steps: Sequence[RuntimeTransformStep],
) -> BenchmarkSpec:
    updated = _freeze_benchmark_spec(base_spec)
    updated.transform = TransformSpec(
        steps=[
            TransformStepSpec(
                operator_id=step.operator_id,
                params=copy.deepcopy(step.params),
            )
            for step in transform_steps
        ]
    )
    return updated


def _benchmark_spec_from_execution_payload(
    *,
    base_spec: BenchmarkSpec,
    execution_spec: _RuntimePipelineSpec,
) -> BenchmarkSpec:
    updated = _benchmark_spec_with_transform_steps(
        base_spec=base_spec,
        transform_steps=execution_spec.transforms,
    )

    task_id = str(
        execution_spec.task.get("task_id")
        or updated.task.task_id
        or ""
    ).strip()
    if task_id:
        updated.task.task_id = task_id
    task_binding = str(execution_spec.task.get("task_binding") or "").strip()
    if task_binding:
        updated.task.task_binding = task_binding

    dataset_payload = dict(execution_spec.dataset)
    dataset_id = str(dataset_payload.get("dataset_id", updated.dataset.dataset_id) or "").strip()
    if dataset_id:
        updated.dataset.dataset_id = dataset_id
    updated.dataset.variant = cast(str | None, dataset_payload.get("variant"))
    updated.dataset.split_scheme = cast(str | None, dataset_payload.get("split_scheme"))
    updated.dataset.dataset_root = cast(str | None, dataset_payload.get("path"))
    updated.dataset.modalities = [
        str(item).strip()
        for item in cast(Sequence[object], dataset_payload.get("modalities", []) or [])
        if str(item).strip()
    ]
    updated.dataset.input_selection = copy.deepcopy(dataset_payload.get("input_selection"))

    model_payload = dict(execution_spec.model)
    model_id = str(
        model_payload.get("model_id")
        or updated.model.model_id
        or ""
    ).strip()
    if model_id:
        updated.model.model_id = model_id
    updated.model.weights_id = cast(str | None, model_payload.get("weights_id"))
    updated.model.entry_overrides = copy.deepcopy(
        cast(Mapping[str, Any], model_payload.get("entry_overrides", {}))
        if isinstance(model_payload.get("entry_overrides"), Mapping)
        else {}
    )

    updated.runtime = RuntimeSpec.from_dict(copy.deepcopy(execution_spec.runtime))
    updated.protocol = copy.deepcopy(execution_spec.protocol)
    return updated


def _resolve_dataset_source(dataset: Any | None) -> DatasetView | dict[str, DatasetView]:
    if dataset is None:
        raise ValueError(
            "pipelines.load(...) requires an explicit dataset handle or dataset_id; "
            "the canonical pipelines owner does not provide a default dataset fallback."
        )
    split_mapping = _dataset_split_mapping(dataset)
    if split_mapping is not None and isinstance(dataset, Mapping):
        return split_mapping
    if isinstance(dataset, DatasetView):
        return dataset
    raise TypeError(
        "pipelines.load(...) dataset sources must be a DatasetView or an explicit "
        "split mapping of DatasetView objects."
    )


def _project_dataset_input(
    dataset_source: DatasetView | dict[str, DatasetView],
    *,
    modality: str,
    node_id: int | None = None,
) -> DatasetView | dict[str, DatasetView]:
    if isinstance(dataset_source, Mapping):
        projected: dict[str, DatasetView] = {}
        for split_name, split_view in dataset_source.items():
            projected_view = split_view.select_input(modality, node_id=node_id)
            if not isinstance(projected_view, DatasetView):
                raise TypeError(
                    "DatasetView.select_input(...) must return a DatasetView, "
                    f"got {type(projected_view)!r}"
                )
            projected[str(split_name)] = projected_view
        return projected

    projected = dataset_source.select_input(modality, node_id=node_id)
    if not isinstance(projected, DatasetView):
        raise TypeError(
            "DatasetView.select_input(...) must return a DatasetView, "
            f"got {type(projected)!r}"
        )
    return projected


def _as_plain_tensor(value: RadioTensor | torch.Tensor) -> torch.Tensor:
    if isinstance(value, RadioTensor):
        return value.to_tensor(contiguous=True)
    if torch.is_tensor(value):
        return value.contiguous()
    raise TypeError(f"Expected RadioTensor or torch.Tensor, got {type(value)!r}")


def _apply_inference_transform(
    sample: RadioTensor | torch.Tensor,
    transform: object | None,
) -> object:
    if transform is None:
        return sample
    if not callable(transform):
        raise TypeError(
            f"inference transform must be callable and return RadioTensor or torch.Tensor, got {type(transform)!r}"
        )
    return transform(sample)


def _materialize_inference_tensor(value: object) -> torch.Tensor:
    try:
        return _as_plain_tensor(cast(RadioTensor | torch.Tensor, value))
    except TypeError as exc:
        raise TypeError(
            "Inference transforms must return RadioTensor or torch.Tensor values; "
            f"got {type(value)!r}"
        ) from exc


def _train_dataset_leaf(dataset_source: object) -> Dataset[Any]:
    split_mapping = _dataset_split_mapping(dataset_source)
    if split_mapping is not None:
        return split_mapping["train"]
    dataset = _coerce_dataset_leaf(dataset_source)
    if dataset is not None:
        return dataset
    raise TypeError(
        "Pipeline dataset source must expose a DatasetView directly or through train/val splits."
    )


def _dataset_describe_node(dataset_source: object) -> DescribeNode | None:
    describe_tree = getattr(dataset_source, "describe_tree", None)
    if callable(describe_tree):
        return describe_tree().with_name("dataset")
    split_mapping = _dataset_split_mapping(dataset_source)
    if split_mapping is not None:
        train_describe_tree = getattr(split_mapping["train"], "describe_tree", None)
        if callable(train_describe_tree):
            return train_describe_tree().with_name("dataset")
    return None


def _sample_radio_tensor(dataset_source: object) -> RadioTensor:
    sample, _ = _train_dataset_leaf(dataset_source)[0]
    if not isinstance(sample, RadioTensor):
        raise TypeError(
            "Generic pipeline assembly requires DatasetView samples to be RadioTensor instances. "
            f"Got {type(sample)!r}."
        )
    return sample


def _sample_input_axes(dataset_source: object) -> tuple[str, ...]:
    return tuple(str(axis) for axis in _sample_radio_tensor(dataset_source).axis_schema.axes)


def _parse_alignment_axes(expr: Any) -> tuple[str, ...]:
    if isinstance(expr, str):
        axes = tuple(part.strip() for part in expr.split("*") if part.strip())
        if axes:
            return axes
    if isinstance(expr, (list, tuple)):
        axes = tuple(str(item).strip() for item in expr if str(item).strip())
        if axes:
            return axes
    raise ValueError(
        "alignment expressions must be a non-empty axis string like "
        "'subc*tx*rx' or a sequence of axis names"
    )


def _replace_last_transform(
    blueprint: _TrainablePipelineBlueprint,
    transform_spec: RuntimeTransformStep,
    *,
    description: str,
    metadata: dict[str, Any],
) -> _TrainablePipelineBlueprint:
    if not blueprint.spec.transforms:
        raise ValueError(f"Blueprint {blueprint.pipeline_id} has no transforms to align")
    spec = copy.deepcopy(blueprint.spec)
    spec.transforms = [*spec.transforms[:-1], transform_spec]
    spec.metadata = metadata
    return _TrainablePipelineBlueprint(
        pipeline_id=f"{blueprint.pipeline_id}_aligned",
        spec=spec,
        description=description,
    )


def _auto_align_runtime_step(
    contract: ModelInputContract,
    **params: Any,
) -> RuntimeTransformStep:
    return RuntimeTransformStep(
        operator_id="AutoAlign",
        params={
            "model_or_contract": contract.to_dict(),
            **params,
        },
    )


def _sequence_alignment_blueprint(
    blueprint: _TrainablePipelineBlueprint,
    contract: ModelInputContract,
    hint: dict[str, Any],
) -> _TrainablePipelineBlueprint:
    model_time_axis, model_feature_axis = contract.axes[:2]
    if model_time_axis not in hint:
        raise ValueError(
            f"Sequence auto_align requires a non-empty binding for '{model_time_axis}'"
        )
    semantic_time_axes = _parse_alignment_axes(hint.get(model_time_axis))
    if len(semantic_time_axes) != 1:
        raise ValueError(
            "Sequence auto_align requires the first model axis to bind to exactly one "
            f"semantic axis, got {semantic_time_axes}"
        )
    if model_feature_axis not in hint:
        raise ValueError(
            f"Sequence auto_align requires a non-empty binding for '{model_feature_axis}'"
        )
    flatten_axes = _parse_alignment_axes(hint.get(model_feature_axis))
    return _replace_last_transform(
        blueprint,
        _auto_align_runtime_step(
            contract,
            axis_map={
                model_time_axis: semantic_time_axes[0],
                model_feature_axis: flatten_axes,
            },
        ),
        description=(
            f"{blueprint.description} with explicit sequence auto_align "
            f"{model_time_axis}<-{semantic_time_axes[0]}, "
            f"{model_feature_axis}<-{'*'.join(flatten_axes)}"
        ),
        metadata={
            "input_axes": list(blueprint.spec.metadata.get("input_axes", [])),
            "output_axes": list(contract.axes),
            "semantic_binding": {
                model_time_axis: semantic_time_axes[0],
                model_feature_axis: list(flatten_axes),
            },
        },
    )


def _image_alignment_blueprint(
    blueprint: _TrainablePipelineBlueprint,
    contract: ModelInputContract,
    hint: dict[str, Any],
) -> _TrainablePipelineBlueprint:
    model_channel_axis, model_height_axis, model_width_axis = contract.axes
    channel_axes = _parse_alignment_axes(hint.get(model_channel_axis, ()))
    height_axes = _parse_alignment_axes(hint.get(model_height_axis, ()))
    width_axes = _parse_alignment_axes(hint.get(model_width_axis, ()))
    if len(height_axes) != 1 or len(width_axes) != 1:
        raise ValueError(
            "Image auto_align requires single-axis bindings for height/width, "
            f"got height={height_axes}, width={width_axes}"
        )
    return _replace_last_transform(
        blueprint,
        _auto_align_runtime_step(
            contract,
            axis_map={
                model_channel_axis: channel_axes,
                model_height_axis: height_axes[0],
                model_width_axis: width_axes[0],
            },
        ),
        description=(
            f"{blueprint.description} with explicit image auto_align "
            f"{model_channel_axis}<-{'*'.join(channel_axes)}, "
            f"{model_height_axis}<-{height_axes[0]}, "
            f"{model_width_axis}<-{width_axes[0]}"
        ),
        metadata={
            "input_axes": list(blueprint.spec.metadata.get("input_axes", [])),
            "output_axes": list(contract.axes),
            "semantic_binding": {
                model_channel_axis: list(channel_axes),
                model_height_axis: height_axes[0],
                model_width_axis: width_axes[0],
            },
        },
    )


def _build_generic_trainable_spec(
    *,
    task_spec: TaskSpec,
    dataset_source: object,
    model_ref: str | ModelHandle | nn.Module,
) -> _RuntimePipelineSpec:
    if isinstance(model_ref, nn.Module):
        contract = get_model_input_contract(model_ref)
    elif isinstance(model_ref, ModelHandle):
        contract = model_ref.get_input_contract()
    else:
        contract = ModelHandle(
            model_id=_normalize_trainable_model_ref(model_ref)
        ).get_input_contract()
    input_axes = _sample_input_axes(dataset_source)
    dataset_payload = _explicit_dataset_selector_payload(
        dataset_id=None,
        dataset_source=dataset_source,
        variant=None,
        path=None,
    )
    model_payload = _canonical_model_selector_payload(model_ref)
    benchmark_spec = BenchmarkSpec(
        dataset=DatasetSpec.from_dict(dataset_payload),
        task=BenchmarkTaskSpec(
            task_id=task_spec.task_id,
            task_binding="",
        ),
        transform=TransformSpec(
            steps=[
                TransformStepSpec(
                    operator_id="AutoAlign",
                    params={"model_or_contract": contract.to_dict(), "value": "auto"},
                )
            ]
        ),
        model=ModelSpec.from_dict(model_payload),
    )
    spec = _build_execution_spec(benchmark_spec)
    spec.metadata = {
        "input_axes": list(input_axes),
        "output_axes": list(contract.axes),
    }
    return spec


def _build_auto_aligned_spec_from_existing_payload(
    *,
    execution_payload: _RuntimePipelineSpec,
    dataset_source: object,
    contract: ModelInputContract,
    custom_input_shape: Any | None,
) -> _RuntimePipelineSpec:
    spec = copy.deepcopy(execution_payload)
    spec.metadata = {
        "input_axes": list(_sample_input_axes(dataset_source)),
        "output_axes": list(contract.axes),
    }
    base_blueprint = _TrainablePipelineBlueprint(
        pipeline_id="spec_lane",
        spec=spec,
        description="Auto-align derived from existing canonical BenchmarkSpec.",
    )
    if custom_input_shape is None:
        spec.transforms = [
            _auto_align_runtime_step(contract, value="auto"),
        ]
        return spec
    return _blueprint_for_alignment_hint(
        custom_input_shape,
        contract=contract,
        base=base_blueprint,
    ).spec


def _resolve_trainable_blueprint(
    *,
    task_spec: TaskSpec,
    dataset_source: object | None,
    model_ref: str | ModelHandle | nn.Module,
) -> _TrainablePipelineBlueprint:
    if dataset_source is None:
        raise ValueError(
            "Generic pipeline assembly requires an explicit dataset-bound source. "
            "Canonical pipelines require an explicit dataset-bound source."
        )
    spec = _build_generic_trainable_spec(
        task_spec=task_spec,
        dataset_source=dataset_source,
        model_ref=model_ref,
    )
    dataset_id = _source_dataset_id(dataset_source) or "dataset"
    model_name = _render_model_selector(model_ref)
    pipeline_id = f"{task_spec.task_id}::{dataset_id}::{model_name}"
    description = (
        "Generic pipeline assembly from dataset semantics and model contract. "
        "Implicit axis guessing is disabled; mismatched axes require an explicit auto_align mapping."
    )
    return _TrainablePipelineBlueprint(
        pipeline_id=pipeline_id,
        spec=spec,
        description=description,
    )


def _blueprint_for_alignment_hint(
    hint: Any,
    *,
    contract: ModelInputContract,
    base: _TrainablePipelineBlueprint,
) -> _TrainablePipelineBlueprint:
    if isinstance(hint, dict):
        if len(contract.axes) == 2:
            return _sequence_alignment_blueprint(base, contract, hint)
        if len(contract.axes) == 3:
            return _image_alignment_blueprint(base, contract, hint)
        raise ValueError(
            "pipeline.auto_align(dict) supports only 2D/3D model-entry contracts, "
            f"got {contract.axes}"
        )
    raise ValueError(f"Unsupported custom_input_shape hint: {hint!r}")


def _instantiate_trainable_model(
    model_ref: str | ModelHandle | nn.Module,
    sample_tensor: torch.Tensor,
) -> nn.Module:
    if isinstance(model_ref, nn.Module):
        return copy.deepcopy(model_ref)
    handle = (
        model_ref
        if isinstance(model_ref, ModelHandle)
        else ModelHandle(model_id=_normalize_trainable_model_ref(model_ref))
    )
    if "num_classes" in handle.unresolved_entry_overrides:
        raise ValueError(
            f"Pipeline materialization for model '{handle.model_id}' requires an explicit "
            "'num_classes' entry override from the task/model contract; pipelines no longer "
            "infer classification cardinality from datasets."
    )
    return instantiate_model_handle(handle, sample_tensor)


def _infer_execution_adapter_id(task_spec: TaskSpec) -> str:
    return canonical_execution_adapter_for_task_kind(task_spec.kind)


def _dataset_label_mapping_for_prediction(dataset_source: object) -> dict[int, str]:
    dataset = dataset_source.get("train", dataset_source) if isinstance(dataset_source, Mapping) else dataset_source
    mapping_getter = getattr(dataset, "get_label_mapping", None)
    if not callable(mapping_getter):
        return {}
    payload = mapping_getter()
    if not isinstance(payload, Mapping):
        return {}
    resolved: dict[int, str] = {}
    for label_name, class_index in payload.items():
        try:
            resolved[int(class_index)] = str(label_name)
        except (TypeError, ValueError):
            continue
    return resolved


def _decode_classification_predictions(
    logits: torch.Tensor,
    *,
    top_k: int,
    index_to_label: Mapping[int, str] | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    if logits.ndim == 1:
        batch_logits = logits.unsqueeze(0)
        single_prediction = True
    elif logits.ndim == 2:
        batch_logits = logits
        single_prediction = logits.shape[0] == 1
    else:
        raise ValueError(
            "predict(...) currently expects classification logits with shape [classes] or [batch, classes], "
            f"got {tuple(logits.shape)}."
        )

    probabilities = torch.softmax(batch_logits, dim=-1)
    top_count = max(1, min(int(top_k), int(probabilities.shape[-1])))
    scores, indices = torch.topk(probabilities, k=top_count, dim=-1)

    decoded_rows: list[dict[str, Any]] = []
    for row_scores, row_indices in zip(scores, indices, strict=True):
        top_predictions: list[dict[str, Any]] = []
        for score, index in zip(
            row_scores.detach().cpu().tolist(),
            row_indices.detach().cpu().tolist(),
            strict=True,
        ):
            prediction = {
                "index": int(index),
                "score": float(score),
            }
            if index_to_label is not None and int(index) in index_to_label:
                prediction["label"] = index_to_label[int(index)]
            top_predictions.append(prediction)

        predicted = dict(top_predictions[0])
        decoded_rows.append(
            {
                "predicted_index": int(predicted["index"]),
                "predicted_score": float(predicted["score"]),
                "predicted_label": predicted.get("label"),
                "top_predictions": top_predictions,
            }
        )

    return decoded_rows[0] if single_prediction else decoded_rows


def _default_training_payload(task_spec: TaskSpec, dataset_source: object) -> dict[str, Any]:
    del dataset_source
    adapter_id = _infer_execution_adapter_id(task_spec)
    default_loss = "cross_entropy" if adapter_id == "scalar_index_supervised" else "mse_loss"
    return {
        "loss": default_loss,
        "optimizer": "Adam",
        "learning_rate": 1e-3,
        "log_every_steps": 10,
    }


def _explicit_runtime_overlay(
    runtime: RuntimeSpec | Mapping[str, Any] | None,
) -> dict[str, Any]:
    if isinstance(runtime, RuntimeSpec):
        payload: object = runtime.to_dict()
    else:
        payload = runtime
    if not isinstance(payload, Mapping):
        return {}
    return _update_present_fields({}, payload)


def _filter_runtime_overlay_for_mode(
    runtime_overlay: Mapping[str, Any] | None,
    *,
    mode: str | None,
) -> dict[str, Any]:
    if not isinstance(runtime_overlay, Mapping):
        return {}
    filtered = dict(runtime_overlay)
    if mode != "train":
        filtered.pop("epochs", None)
    return filtered


def _runtime_payload_for_mode(
    runtime_payload: RuntimeSpec | Mapping[str, Any] | None,
    *,
    mode: str | None,
) -> dict[str, Any]:
    return _filter_runtime_overlay_for_mode(
        _explicit_runtime_overlay(runtime_payload),
        mode=mode,
    )


def _canonical_protocol_payload(
    *,
    task_spec: TaskSpec,
    protocol_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    execution_adapter = canonical_execution_adapter_for_task_kind(task_spec.kind)
    resolved = resolve_runtime_protocol_payload(
        protocol_payload,
        task_kind=task_spec.kind,
        primary_metric=task_spec.output_schema.primary_metric,
        target_fields=task_spec.target_schema.fields,
        execution_adapter=execution_adapter,
    )
    mode = str(dict(protocol_payload or {}).get("mode", "") or "").strip()
    if mode:
        resolved["mode"] = mode
    return resolved


def _canonicalize_pipeline_spec_for_execution(
    spec: _RuntimePipelineSpec | Mapping[str, Any],
    *,
    task_spec: TaskSpec,
    dataset_source: object,
) -> _RuntimePipelineSpec:
    canonical = (
        _RuntimePipelineSpec.from_dict(dict(spec))
        if isinstance(spec, Mapping)
        else copy.deepcopy(spec)
    )

    canonical.training = _update_present_fields(
        _default_training_payload(task_spec, dataset_source),
        canonical.training,
    )
    canonical.runtime = _explicit_runtime_overlay(canonical.runtime)
    canonical.protocol = _canonical_protocol_payload(
        task_spec=task_spec,
        protocol_payload=canonical.protocol,
    )
    return canonical




@dataclass
class TrainableTaskPipeline(PipelineExecutionHandle, Describable):
    """Trainable pipeline handle for canonical dataset/task/model assembly."""

    task: str
    dataset_source: DatasetView | dict[str, DatasetView]
    model_ref: str | ModelHandle | nn.Module
    spec: BenchmarkSpec | None = None
    task_spec: TaskSpec = field(init=False)
    pipeline_id: str = field(init=False)
    transform: Sequential = field(init=False)
    _execution_payload: _RuntimePipelineSpec = field(init=False, repr=False)
    benchmark_spec: BenchmarkSpec | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.task_spec = _load_task_spec(self.task)
        public_spec = (
            _resolve_and_freeze_benchmark_spec(
                self.spec,
                resolver_context=_default_resolver_context_for_model_ref(self.model_ref),
            )
            if self.spec is not None
            else None
        )
        if self.spec is None:
            blueprint = _resolve_trainable_blueprint(
                task_spec=self.task_spec,
                dataset_source=self.dataset_source,
                model_ref=self.model_ref,
            )
        else:
            blueprint = _TrainablePipelineBlueprint(
                pipeline_id=_pipeline_id_from_benchmark(public_spec),
                spec=_build_execution_spec(public_spec),
                description="Execution payload lowered from canonical BenchmarkSpec.",
            )
        self._set_pipeline_spec(
            blueprint.spec,
            pipeline_id=blueprint.pipeline_id,
        )
        public_spec = (
            _benchmark_spec_from_execution_payload(
                base_spec=public_spec,
                execution_spec=self._execution_payload,
            )
            if public_spec is not None
            else None
        )
        self._set_public_benchmark_spec(public_spec)

    def _set_pipeline_spec(
        self,
        spec: _RuntimePipelineSpec | Mapping[str, Any],
        *,
        pipeline_id: str,
    ) -> None:
        self._execution_payload = _canonicalize_pipeline_spec_for_execution(
            spec,
            task_spec=self.task_spec,
            dataset_source=self.dataset_source,
        )
        self.pipeline_id = pipeline_id
        self.transform = self._execution_payload.build_transform()

    def _set_public_benchmark_spec(self, spec: BenchmarkSpec | None) -> None:
        frozen_spec = None if spec is None else _freeze_benchmark_spec(spec)
        self.spec = frozen_spec
        self.benchmark_spec = frozen_spec

    def get_benchmark_spec(self) -> BenchmarkSpec | None:
        return None if self.benchmark_spec is None else _freeze_benchmark_spec(self.benchmark_spec)

    def auto_align(self, custom_input_shape: Any | None = None) -> "TrainableTaskPipeline":
        if isinstance(self.model_ref, nn.Module):
            contract = get_model_input_contract(self.model_ref)
        elif isinstance(self.model_ref, ModelHandle):
            contract = self.model_ref.get_input_contract()
        else:
            contract = ModelHandle(
                model_id=_normalize_trainable_model_ref(self.model_ref)
            ).get_input_contract()
        if self.spec is not None:
            aligned_spec = _build_auto_aligned_spec_from_existing_payload(
                execution_payload=self._execution_payload,
                dataset_source=self.dataset_source,
                contract=contract,
                custom_input_shape=custom_input_shape,
            )
            pipeline_id = (
                self.pipeline_id if custom_input_shape is None else f"{self.pipeline_id}_aligned"
            )
            blueprint = _TrainablePipelineBlueprint(
                pipeline_id=pipeline_id,
                spec=aligned_spec,
                description="Auto-align derived from existing canonical BenchmarkSpec.",
            )
        elif custom_input_shape is None:
            blueprint = _resolve_trainable_blueprint(
                task_spec=self.task_spec,
                dataset_source=self.dataset_source,
                model_ref=self.model_ref,
            )
        else:
            base_blueprint = _resolve_trainable_blueprint(
                task_spec=self.task_spec,
                dataset_source=self.dataset_source,
                model_ref=self.model_ref,
            )
            blueprint = _blueprint_for_alignment_hint(
                custom_input_shape,
                contract=contract,
                base=base_blueprint,
            )
        self._set_pipeline_spec(
            blueprint.spec,
            pipeline_id=blueprint.pipeline_id,
        )
        public_spec = (
            _benchmark_spec_from_execution_payload(
                base_spec=self.benchmark_spec if self.benchmark_spec is not None else self.spec,
                execution_spec=self._execution_payload,
            )
            if (self.benchmark_spec is not None or self.spec is not None)
            else None
        )
        self._set_public_benchmark_spec(public_spec)
        return self

    def _sample_tensor(self) -> torch.Tensor:
        sample, _ = _train_dataset_leaf(self.dataset_source)[0]
        transformed = self.transform(sample)
        return _as_plain_tensor(transformed)

    def describe_tree(self) -> DescribeNode:
        sample_tensor = self._sample_tensor()
        if isinstance(self.model_ref, nn.Module):
            contract = get_model_input_contract(self.model_ref)
            model_name = _render_model_selector(self.model_ref)
        elif isinstance(self.model_ref, ModelHandle):
            model_name = _render_model_selector(self.model_ref)
            contract = self.model_ref.get_input_contract()
        else:
            model_name = _render_model_selector(self.model_ref)
            contract = ModelHandle(
                model_id=_normalize_trainable_model_ref(self.model_ref)
            ).get_input_contract()
        children: list[DescribeNode] = []
        dataset_node = _dataset_describe_node(self.dataset_source)
        if dataset_node is not None:
            children.append(dataset_node)
        children.extend(
            [
                DescribeNode(
                    kind="transform_chain",
                    name="transform_chain",
                    fields={"steps": [transform.operator_id for transform in self._execution_payload.transforms]},
                ),
                DescribeNode(
                    kind="model_contract",
                    name="model_contract",
                    fields={
                        "layout": contract.layout,
                        "axes": list(contract.axes),
                        "fixed_sizes": dict(contract.fixed_sizes),
                        "sample_tensor_shape": [int(dim) for dim in sample_tensor.shape],
                    },
                ),
                DescribeNode(
                    kind="output_contract",
                    name="output_contract",
                    fields={"semantic_binding": self._execution_payload.metadata.get("semantic_binding", {})},
                ),
            ]
        )
        return DescribeNode(
            kind="pipeline",
            name=str(model_name),
            fields={
                "task": self.task_spec.task_id,
                "task_kind": self.task_spec.kind,
                "model": str(model_name),
                "runtime_note": "trainable task pipeline",
            },
            children=tuple(children),
        )

    def _build_execution_components(
        self,
        *,
        seed: int | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        num_workers: int | None = None,
    ) -> tuple[nn.Module, ExecutionDataLoaders]:
        if seed is None:
            raise ValueError("_build_execution_components(...) requires an explicit runtime seed")
        if batch_size is None:
            raise ValueError(
                "_build_execution_components(...) requires an explicit runtime batch_size"
            )
        if num_workers is None:
            raise ValueError(
                "_build_execution_components(...) requires an explicit runtime num_workers"
            )
        dataloaders = build_execution_dataloaders(
            self.dataset_source,
            task_spec=self.task_spec,
            transform=self.transform,
            seed=int(seed),
            batch_size=int(batch_size),
            num_workers=int(num_workers),
            device=str(_resolve_device(device)),
        )
        model = _instantiate_trainable_model(
            self.model_ref,
            dataloaders.sample_tensor,
        ).to(_resolve_device(device))
        validate_model_entry(model, dataloaders.sample_tensor)
        return model, dataloaders

    def _prepare_inference_inputs(
        self,
        inputs: RadioTensor | torch.Tensor | list[RadioTensor | torch.Tensor],
    ) -> torch.Tensor | list[torch.Tensor]:
        def _prepare_one(value: RadioTensor | torch.Tensor) -> torch.Tensor:
            tensor = _materialize_inference_tensor(
                _apply_inference_transform(value, self.transform)
            )
            return tensor.unsqueeze(0) if tensor.ndim == 2 or tensor.ndim == 3 else tensor

        if isinstance(inputs, list):
            return [_prepare_one(item) for item in inputs]
        return _prepare_one(inputs)

    def get_execution_dataset_source(self) -> DatasetView | dict[str, DatasetView]:
        return self.dataset_source

    def _runtime_overlay(
        self,
        *,
        mode: str | None,
        device: str | None = None,
        seed: int | None = None,
        batch_size: int | None = None,
        num_workers: int | None = None,
        epochs: int | None = None,
    ) -> dict[str, Any] | None:
        runtime_overlay = _filter_runtime_overlay_for_mode(
            {
                key: value
                for key, value in {
                    "device": str(device) if device is not None else None,
                    "seed": int(seed) if seed is not None else None,
                    "batch_size": int(batch_size) if batch_size is not None else None,
                    "num_workers": int(num_workers) if num_workers is not None else None,
                    "epochs": int(epochs) if epochs is not None else None,
                }.items()
                if value is not None
            },
            mode=mode,
        )
        return runtime_overlay or None

    def _resolve_execution_payload(
        self,
        *,
        mode: str | None,
        runtime: RuntimeSpec | Mapping[str, Any] | None = None,
    ) -> dict[str, object]:
        execution_spec = copy.deepcopy(self._execution_payload)
        if mode in {"train", "evaluate"}:
            execution_spec.protocol = _update_present_fields(
                dict(execution_spec.protocol),
                {"mode": mode},
            )
        execution_spec.runtime = _runtime_payload_for_mode(
            execution_spec.runtime,
            mode=mode,
        )
        runtime_overlay = _runtime_payload_for_mode(runtime, mode=mode)
        if runtime_overlay:
            execution_spec.runtime = _update_present_fields(
                dict(execution_spec.runtime),
                runtime_overlay,
            )
        return execution_spec.to_dict()

    def evaluate(
        self,
        *,
        output_root: str | Path | None = None,
        run_name: str | None = None,
        device: str | None = None,
        seed: int | None = None,
        batch_size: int | None = None,
        num_workers: int | None = None,
        progress: bool | None = None,
    ) -> dict[str, Any]:
        return execute_pipeline_handle(
            self,
            benchmark_spec=self.get_benchmark_spec(),
            output_root=output_root,
            run_name=run_name,
            runtime=self._runtime_overlay(
                mode="evaluate",
                device=device,
                seed=seed,
                batch_size=batch_size,
                num_workers=num_workers,
            ),
            progress=progress,
            mode="evaluate",
        )

    def train(
        self,
        *,
        output_root: str | Path | None = None,
        run_name: str | None = None,
        device: str | None = None,
        seed: int | None = None,
        batch_size: int | None = None,
        num_workers: int | None = None,
        epochs: int | None = None,
        progress: bool | None = None,
    ) -> dict[str, Any]:
        return execute_pipeline_handle(
            self,
            benchmark_spec=self.get_benchmark_spec(),
            output_root=output_root,
            run_name=run_name,
            runtime=self._runtime_overlay(
                mode="train",
                device=device,
                seed=seed,
                batch_size=batch_size,
                num_workers=num_workers,
                epochs=epochs,
            ),
            progress=progress,
            mode="train",
        )

    def infer(
        self,
        inputs: RadioTensor | torch.Tensor | list[RadioTensor | torch.Tensor],
        *,
        device: str | None = None,
        seed: int | None = None,
        batch_size: int | None = None,
        num_workers: int | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        return cast(
            torch.Tensor | list[torch.Tensor],
            execute_pipeline_handle(
                self,
                benchmark_spec=self.get_benchmark_spec(),
                runtime=self._runtime_overlay(
                    mode="infer",
                    device=device,
                    seed=seed,
                    batch_size=batch_size,
                    num_workers=num_workers,
                ),
                mode="infer",
                inputs=inputs,
            ),
        )

    def predict(
        self,
        inputs: RadioTensor | torch.Tensor | list[RadioTensor | torch.Tensor],
        *,
        top_k: int = 3,
        device: str | None = None,
        seed: int | None = None,
        batch_size: int | None = None,
        num_workers: int | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        if self.task_spec.kind != "classification":
            raise NotImplementedError(
                "predict(...) currently implements task-aware decoding only for classification pipelines."
            )

        outputs = self.infer(
            inputs,
            device=device,
            seed=seed,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        index_to_label = _dataset_label_mapping_for_prediction(self.dataset_source)

        if isinstance(outputs, list):
            return [
                cast(
                    dict[str, Any],
                    _decode_classification_predictions(
                        cast(torch.Tensor, output),
                        top_k=top_k,
                        index_to_label=index_to_label,
                    ),
                )
                for output in outputs
            ]
        return cast(
            dict[str, Any] | list[dict[str, Any]],
            _decode_classification_predictions(
                cast(torch.Tensor, outputs),
                top_k=top_k,
                index_to_label=index_to_label,
            ),
        )


def _pipeline_id_from_benchmark(spec: BenchmarkSpec) -> str:
    task_id = spec.task.task_id.strip() or "task"
    dataset_id = spec.dataset.dataset_id.strip() or "dataset"
    model_id = spec.model.model_id.strip() or "model"
    weights_id = (spec.model.weights_id or "").strip()
    model_name = f"{model_id}:{weights_id}" if weights_id else model_id
    return f"{task_id}::{dataset_id}::{model_name}"


def _build_request(
    *,
    recipe_id: str | None = None,
    task_id: str | None = None,
    task_binding: str | None = None,
    model_id: str | None = None,
    weights_id: str | None = None,
    entry_overrides: Mapping[str, Any] | None = None,
    transform: object | None = None,
    dataset_id: str | None = None,
    modalities: Sequence[str] | str | None = None,
    dataset: Any | None = None,
    variant: str | None = None,
    path: str | Path | None = None,
) -> _PipelineBuildRequest:
    resolved_recipe_id = _normalize_optional_string_selector("recipe_id", recipe_id)
    resolved_task_id = (
        _load_task_spec(task_id).task_id
        if task_id is not None
        else None
    )
    resolved_task_binding = _normalize_optional_string_selector("task_binding", task_binding)
    if resolved_task_id is not None and resolved_task_binding is None:
        raise ValueError(
            "pipelines.load(...) now requires an explicit task_binding selector when "
            "task_id=... is provided."
        )
    if resolved_task_id is None and resolved_task_binding is not None:
        raise ValueError(
            "pipelines.load(...) does not accept task_binding without task_id."
        )
    resolved_model_id = _normalize_optional_string_selector("model_id", model_id)
    resolved_weights_id = _normalize_optional_string_selector("weights_id", weights_id)
    resolved_dataset_id = _normalize_optional_string_selector("dataset_id", dataset_id)
    requested_modalities = _normalize_modalities_selector(modalities)
    resolved_variant = _normalize_optional_string_selector("variant", variant)
    return _PipelineBuildRequest(
        recipe_id=resolved_recipe_id,
        model_id=resolved_model_id,
        weights_id=resolved_weights_id,
        entry_overrides=copy.deepcopy(dict(entry_overrides)) if isinstance(entry_overrides, Mapping) else None,
        transform_steps=_normalize_transform_selector(transform),
        task_id=resolved_task_id,
        task_binding=resolved_task_binding,
        dataset_id=resolved_dataset_id,
        modalities=requested_modalities,
        dataset=dataset,
        variant=resolved_variant,
        path=path,
    )


def _resolve_recipe_request(recipe_id: str, **overrides: Any) -> _PipelineBuildRequest:
    recipe = _resolve_pipeline_recipe(recipe_id, **overrides)
    resolved_modalities = _resolve_canonical_modalities(
        dataset_id=recipe.dataset_id,
        task_id=recipe.task_id,
        modalities=recipe.modalities,
        variant=recipe.variant,
        path=recipe.path,
    )
    resolved_task_binding = _resolve_canonical_task_binding(
        dataset_id=recipe.dataset_id,
        task_id=recipe.task_id,
    )
    resolved_dataset = materialize_dataset_selector(
        dataset=overrides.get("dataset"),
        dataset_id=recipe.dataset_id,
        task_binding=resolved_task_binding,
        modalities=recipe.modalities,
        variant=recipe.variant,
        path=recipe.path,
        input_selection=recipe.input_selection,
    )
    return _PipelineBuildRequest(
        recipe_id=recipe.recipe_id,
        task_id=recipe.task_id,
        task_binding=resolved_task_binding,
        model_id=recipe.model_id,
        weights_id=recipe.weights_id,
        entry_overrides=(
            copy.deepcopy(dict(cast(Mapping[str, Any], overrides["entry_overrides"])))
            if isinstance(overrides.get("entry_overrides"), Mapping)
            else None
        ),
        transform_steps=_normalize_transform_selector(overrides.get("transform")),
        dataset_id=recipe.dataset_id,
        modalities=resolved_modalities,
        dataset=resolved_dataset,
        variant=recipe.variant,
        path=recipe.path,
        input_selection=(
            copy.deepcopy(dict(recipe.input_selection))
            if isinstance(recipe.input_selection, Mapping)
            else None
        ),
    )


def _reject_recipe_identity_overrides(
    *,
    dataset_id: str | None,
    task_id: str | None,
    task_binding: str | None,
    model_id: str | None,
    weights_id: str | None,
) -> None:
    forbidden = tuple(
        name
        for name, value in (
            ("dataset_id", dataset_id),
            ("model_id", model_id),
            ("task_id", task_id),
            ("task_binding", task_binding),
            ("weights_id", weights_id),
        )
        if value is not None
    )
    if not forbidden:
        return
    names = ", ".join(forbidden)
    raise ValueError(
        "pipelines.load(recipe_id=...) treats recipe_id as immutable identity "
        f"'<dataset_id>/<model_id>@<task_id>' and does not accept same-level "
        f"selector overrides: {names}."
    )


def _selector_benchmark_spec(
    request: _PipelineBuildRequest,
    *,
    dataset_source: object,
    model_handle: ModelHandle,
) -> BenchmarkSpec:
    resolved_task_binding = request.task_binding
    source_payload = _explicit_dataset_selector_payload(
        dataset_id=None,
        dataset_source=dataset_source,
        variant=None,
        path=None,
    )
    dataset_id = request.dataset_id or str(source_payload.get("dataset_id", "") or "").strip()
    variant = request.variant or _normalize_optional_string_selector(
        "variant",
        cast(str | None, source_payload.get("variant")),
    )
    canonical_modalities = _resolve_canonical_modalities(
        dataset_id=request.dataset_id,
        task_id=request.task_id,
        modalities=request.modalities,
        variant=request.variant,
        path=request.path,
    )
    dataset_root_value = request.path if request.path is not None else source_payload.get("path")
    if request.modalities is not None:
        modalities = list(request.modalities)
    elif canonical_modalities is not None:
        modalities = list(canonical_modalities)
    else:
        modalities = list(source_payload.get("modalities", []) or [])
    input_selection = request.input_selection
    if input_selection is None and isinstance(source_payload.get("input_selection"), Mapping):
        input_selection = copy.deepcopy(
            dict(cast(Mapping[str, object], source_payload["input_selection"]))
        )
    transform_steps = (
        request.transform_steps
        if request.transform_steps is not None
        else (
            RuntimeTransformStep(
                operator_id="AutoAlign",
                params={
                    "model_or_contract": model_handle.get_input_contract().to_dict(),
                    "value": "auto",
                },
            ),
        )
    )
    return BenchmarkSpec(
        dataset=DatasetSpec(
            dataset_id=dataset_id,
            modalities=modalities,
            variant=variant,
            dataset_root=(
                str(dataset_root_value)
                if isinstance(dataset_root_value, (str, Path)) and str(dataset_root_value).strip()
                else None
            ),
            input_selection=copy.deepcopy(input_selection) if input_selection is not None else None,
        ),
        task=BenchmarkTaskSpec(
            task_id=request.task_id or "",
            task_binding=resolved_task_binding or "",
        ),
        transform=TransformSpec(
            steps=[
                TransformStepSpec(
                    operator_id=step.operator_id,
                    params=copy.deepcopy(step.params),
                )
                for step in transform_steps
            ]
        ),
        model=ModelSpec(
            model_id=model_handle.model_id,
            weights_id=model_handle.weights_id,
            entry_overrides=copy.deepcopy(model_handle.entry_overrides),
        ),
    )


def _materialize_build_request(request: _PipelineBuildRequest):
    if request.task_id is None:
        raise ValueError("pipelines.load(...) requires a canonical task_id after request resolution")
    if request.model_id is None:
        raise ValueError(
            "pipelines.load(...) requires task_id/model_id after request resolution"
        )
    dataset_selector = (
        request.dataset
        if request.dataset is not None
        else materialize_dataset_selector(
            dataset_id=request.dataset_id,
            task_binding=request.task_binding,
            modalities=request.modalities,
            variant=request.variant,
            path=request.path,
        )
    )
    model_ref = ModelHandle(
        model_id=request.model_id,
        weights_id=request.weights_id,
        entry_overrides=copy.deepcopy(request.entry_overrides or {}),
    )
    resolved_dataset_source = _resolve_dataset_source(dataset_selector)
    selector_spec = _selector_benchmark_spec(
        request,
        dataset_source=resolved_dataset_source,
        model_handle=model_ref,
    )
    return TrainableTaskPipeline(
        task=request.task_id,
        dataset_source=resolved_dataset_source,
        model_ref=model_ref,
        spec=selector_spec,
    )


def _load_from_spec(
    spec: BenchmarkSpec,
    *,
    dataset: Any | None = None,
):
    if not isinstance(spec, BenchmarkSpec):
        raise TypeError(
            "pipelines.builder only accepts canonical BenchmarkSpec objects"
        )
    model_payload = {
        "model_id": spec.model.model_id,
        "weights_id": spec.model.weights_id,
        "entry_overrides": dict(spec.model.entry_overrides),
    }
    model_id = _normalize_optional_string_selector(
        "model_id",
        str(model_payload.get("model_id") or ""),
    )
    if model_id is None:
        raise ValueError("BenchmarkSpec model selector is missing model.model_id")
    model_handle = ModelHandle(
        model_id=model_id,
        weights_id=cast(str | None, model_payload.get("weights_id")),
        entry_overrides=(
            dict(cast(Mapping[str, Any], model_payload.get("entry_overrides")))
            if isinstance(model_payload.get("entry_overrides"), Mapping)
            else {}
        ),
    )
    frozen_spec = _resolve_and_freeze_benchmark_spec(
        spec,
        resolver_context=_default_resolver_context_for_model_ref(model_handle),
    )
    task_spec = _load_task_spec(frozen_spec.task.task_id)

    dataset_payload = frozen_spec.dataset.to_dict()
    dataset_selector = (
        materialize_dataset_selector(
            dataset=dataset,
            input_selection=cast(Mapping[str, object] | None, dataset_payload.get("input_selection")),
        )
        if dataset is not None
        else materialize_dataset_selector(
            dataset_id=cast(str | None, dataset_payload.get("dataset_id")),
            modalities=cast(Sequence[str] | str | None, dataset_payload.get("modalities")),
            variant=cast(str | None, dataset_payload.get("variant")),
            split_scheme=cast(str | None, dataset_payload.get("split_scheme")),
            task_binding=cast(str | None, frozen_spec.task.task_binding),
            path=cast(str | Path | None, dataset_payload.get("dataset_root")),
            input_selection=cast(Mapping[str, object] | None, dataset_payload.get("input_selection")),
        )
    )
    if dataset_selector is None:
        raise ValueError("BenchmarkSpec dataset selector is missing dataset_id")

    resolved_dataset_source = _resolve_dataset_source(dataset_selector)

    return TrainableTaskPipeline(
        task=task_spec.task_id,
        dataset_source=resolved_dataset_source,
        model_ref=model_handle,
        spec=frozen_spec,
    )


def load(
    spec: BenchmarkSpec | None = None,
    *,
    recipe_id: str | None = None,
    dataset_id: str | None = None,
    modalities: Sequence[str] | str | None = None,
    task_id: str | None = None,
    task_binding: str | None = None,
    model_id: str | None = None,
    weights_id: str | None = None,
    entry_overrides: Mapping[str, Any] | None = None,
    transform: object | None = None,
    variant: str | None = None,
    path: str | Path | None = None,
    dataset: Any | None = None,
):
    if spec is not None:
        if any(
            value is not None
            for value in (
                recipe_id,
                dataset_id,
                modalities,
                task_id,
                task_binding,
                model_id,
                weights_id,
                variant,
                path,
            )
        ):
            raise ValueError(
                "pipelines.load(spec=...) cannot be combined with recipe_id/dataset_id/modalities/"
                "task_id/task_binding/model_id/weights_id/variant/path selectors"
            )
        if entry_overrides is not None:
            raise ValueError(
                "pipelines.load(spec=...) cannot be combined with entry_overrides; "
                "persist model overrides inside spec.model.entry_overrides"
            )
        if transform is not None:
            raise ValueError(
                "pipelines.load(spec=...) cannot be combined with transform=...; "
                "persist transform steps inside spec.transform"
            )
        return _load_from_spec(spec, dataset=dataset)
    if recipe_id is not None and transform is not None:
        raise ValueError(
            "pipelines.load(recipe_id=...) cannot be combined with transform=...; "
            "use task_id/model_id selectors or persist transforms inside BenchmarkSpec"
        )
    request = (
        (
            _reject_recipe_identity_overrides(
                dataset_id=dataset_id,
                task_id=task_id,
                task_binding=task_binding,
                model_id=model_id,
                weights_id=weights_id,
            )
            or _resolve_recipe_request(
                recipe_id,
                weights_id=weights_id,
                entry_overrides=entry_overrides,
                transform=transform,
                modalities=modalities,
                dataset=dataset,
                variant=variant,
                path=path,
            )
        )
        if recipe_id is not None
        else _build_request(
            task_id=task_id,
            task_binding=task_binding,
            model_id=model_id,
            weights_id=weights_id,
            entry_overrides=entry_overrides,
            transform=transform,
            dataset_id=dataset_id,
            modalities=modalities,
            dataset=dataset,
            variant=variant,
            path=path,
        )
    )
    return _materialize_build_request(request)


__all__: list[str] = []
