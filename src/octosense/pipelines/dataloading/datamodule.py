"""Execution-owned dataloader assembly for pipeline runtimes.

This module only assembles runtime loaders from explicit dataset-owned splits
and target contracts. Task- or dataset-specific split policy stays outside the
pipeline dataloading layer.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from octosense._internal.worker_backend import resolve_num_workers
from octosense.core import Describable, DescribeNode
from octosense.datasets import DatasetView
from octosense.io.tensor import RadioTensor
from octosense.pipelines.dataloading.collate import (
    collate_scalar_index_batch,
    collate_structured_target_batch,
)
from octosense.pipelines.dataloading.samplers import build_sampler
from octosense.tasks.definitions import TaskSpec

CollateFn = Callable[[list[Any]], Any]
DatasetSplitMapping = Mapping[str, DatasetView]
DatasetSource = DatasetView | DatasetSplitMapping
_SCALAR_TARGET_KINDS = frozenset({"categorical_label"})
_STRUCTURED_TARGET_KINDS = frozenset({"structured_pose", "coordinates", "respiration_signal"})


@dataclass(frozen=True)
class DataLoaderConfig:
    batch_size: int = 32
    num_workers: int = 0
    shuffle: bool = False
    device: str | torch.device | None = None
    collate_fn: CollateFn | None = None
    sampler_weights: Sequence[float] | torch.Tensor | None = None
    drop_last: bool = False


@dataclass(frozen=True)
class ExecutionRuntimeConfig:
    seed: int
    batch_size: int
    num_workers: int


@dataclass(frozen=True)
class _ResolvedTargetAdapter:
    adapter_id: Literal["scalar_index", "structured_tensor_mapping"]


def _adapter_from_target_kind(target_kind: object) -> Literal["scalar_index", "structured_tensor_mapping"] | None:
    normalized = str(target_kind or "").strip()
    if normalized in _SCALAR_TARGET_KINDS:
        return "scalar_index"
    if normalized in _STRUCTURED_TARGET_KINDS:
        return "structured_tensor_mapping"
    return None


def _declared_target_schema(dataset_source: object) -> dict[str, object] | None:
    candidates: list[object] = [dataset_source]
    reference_dataset = _reference_dataset(dataset_source)
    if reference_dataset is not None:
        candidates.append(reference_dataset)
    for candidate in candidates:
        provider = getattr(candidate, "get_target_schema", None)
        if not callable(provider):
            continue
        try:
            payload = provider()
        except AttributeError:
            continue
        if isinstance(payload, Mapping):
            return {
                str(key): value
                for key, value in payload.items()
                if str(key) not in {"task_id", "task_kind", "target_kind"}
            }
    return None


def _declared_target_kind(dataset_source: object) -> str | None:
    candidates: list[object] = [dataset_source]
    reference_dataset = _reference_dataset(dataset_source)
    if reference_dataset is not None:
        candidates.append(reference_dataset)
    for candidate in candidates:
        target_kind = getattr(candidate, "target_kind", None)
        normalized = str(target_kind or "").strip()
        if normalized:
            return normalized
        provider = getattr(candidate, "get_target_schema", None)
        if not callable(provider):
            continue
        try:
            payload = provider()
        except AttributeError:
            continue
        if isinstance(payload, Mapping):
            normalized = str(payload.get("target_kind", "") or "").strip()
            if normalized:
                return normalized
    return None


def _resolve_target_adapter(
    task_spec: TaskSpec | None,
    dataset_source: DatasetSource,
) -> _ResolvedTargetAdapter:
    declared_target_kind = (
        task_spec.target_schema.target_kind if task_spec is not None else _declared_target_kind(dataset_source)
    )
    declared_adapter = _adapter_from_target_kind(declared_target_kind)

    if declared_adapter == "scalar_index":
        return _ResolvedTargetAdapter(adapter_id="scalar_index")

    if declared_adapter != "structured_tensor_mapping":
        raise ValueError(
            "build_execution_dataloaders(...) requires a canonical task_spec or dataset-owned "
            "target metadata with a supported target kind. Sample-target inference is not supported, "
            f"got {declared_target_kind!r}."
        )
    declared_schema = _declared_target_schema(dataset_source)
    if declared_schema:
        return _ResolvedTargetAdapter(
            adapter_id="structured_tensor_mapping",
        )
    target_owner = (
        f"structured canonical task '{task_spec.task_id}'"
        if task_spec is not None
        else "structured dataset targets"
    )
    raise ValueError(
        "build_execution_dataloaders(...) requires dataset-local concrete target layout for "
        f"{target_owner}. Datasets must expose "
        "dataset.get_target_schema() as concrete layout metadata without carrying canonical semantics."
    )


def _resolve_collate_fn(
    adapter_id: Literal["scalar_index", "structured_tensor_mapping"],
) -> CollateFn:
    if adapter_id == "scalar_index":
        return collate_scalar_index_batch
    if adapter_id == "structured_tensor_mapping":
        return collate_structured_target_batch
    raise NotImplementedError(f"Unsupported target adapter: {adapter_id!r}")


def _resolve_mapped_execution_datasets(
    split_mapping: Mapping[str, DatasetView],
    *,
    adapter_id: Literal["scalar_index", "structured_tensor_mapping"],
    transform: nn.Module | None,
) -> tuple[
    Dataset[tuple[torch.Tensor, object]],
    Dataset[tuple[torch.Tensor, object]],
    Dataset[tuple[torch.Tensor, object]] | None,
    CollateFn,
]:
    dataset_cls: type[Dataset[tuple[torch.Tensor, object]]]
    if adapter_id == "scalar_index":
        dataset_cls = cast(type[Dataset[tuple[torch.Tensor, object]]], _MappedScalarTargetDataset)
    elif adapter_id == "structured_tensor_mapping":
        dataset_cls = cast(
            type[Dataset[tuple[torch.Tensor, object]]],
            _MappedStructuredTargetDataset,
        )
    else:
        raise NotImplementedError(
            "build_execution_dataloaders(...) could not resolve a supported target "
            "adapter from the dataset target contract."
        )

    train_dataset = split_mapping["train"]
    val_dataset = split_mapping["val"]
    test_dataset = split_mapping.get("test")
    return (
        dataset_cls(train_dataset, transform),
        dataset_cls(val_dataset, transform),
        dataset_cls(test_dataset, transform) if test_dataset is not None else None,
        _resolve_collate_fn(adapter_id),
    )


def build_sample_ids(dataset: Dataset, indices: list[int]) -> list[str]:
    if isinstance(dataset, DatasetView):
        rows = dataset.metadata_rows()
        if rows:
            sample_ids = [
                row.get("sample_id", row.get("sample", row.get("record_id")))
                for row in rows
            ]
            if all(value is not None for value in sample_ids):
                return [str(sample_ids[idx]) for idx in indices]
    if hasattr(dataset, "get_sample_id"):
        return [str(dataset.get_sample_id(idx)) for idx in indices]  # type: ignore[attr-defined]
    return [f"sample-{idx:06d}" for idx in indices]


@dataclass
class ExecutionDataLoaders(Describable):
    """Canonical runtime dataloaders for pipeline-owned execution flows."""

    train_loader: DataLoader[Any]
    val_loader: DataLoader[Any]
    test_loader: DataLoader[Any] | None
    sample_tensor: torch.Tensor
    runtime_config: ExecutionRuntimeConfig

    def describe_tree(self) -> DescribeNode:
        split_children: list[DescribeNode] = []
        for split_name, loader in (
            ("train", self.train_loader),
            ("val", self.val_loader),
            ("test", self.test_loader),
        ):
            if loader is None:
                continue
            dataset = getattr(loader, "dataset", None)
            sample_count = len(dataset) if dataset is not None else 0
            split_children.append(
                DescribeNode(
                    kind="loader_split",
                    name=split_name,
                    fields={
                        "samples": int(sample_count),
                        "batches": int(len(loader)),
                        "batch_size": int(loader.batch_size or 0),
                        "num_workers": int(loader.num_workers),
                    },
                )
            )
        return DescribeNode(
            kind="execution_dataloaders",
            name="dataloaders",
            fields={
                "sample_tensor_shape": [int(dim) for dim in self.sample_tensor.shape],
                "sample_tensor_dtype": str(self.sample_tensor.dtype),
            },
            children=tuple(split_children),
        )


class _MappedScalarTargetDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, dataset: Dataset[tuple[Any, int]], transform: nn.Module | None) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample, label = self.dataset[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        return _execution_sample_tensor(sample), int(label)


class _MappedStructuredTargetDataset(Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    def __init__(
        self,
        dataset: Dataset[tuple[Any, Mapping[str, object]]],
        transform: nn.Module | None,
    ) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        sample, target = self.dataset[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        if not isinstance(target, Mapping):
            raise TypeError(
                "Structured-target execution expects mapping targets with tensor-valued fields."
            )
        structured_target: dict[str, torch.Tensor] = {}
        for key, value in target.items():
            if not torch.is_tensor(value):
                raise TypeError(
                    "Structured-target execution expects every target field to be a tensor, "
                    f"got {type(value)!r} for field {key!r}"
                )
            structured_target[str(key)] = value
        return _execution_sample_tensor(sample), structured_target


class _MappedSampleTensorDataset(Dataset[torch.Tensor]):
    def __init__(self, dataset: Dataset[Any], transform: nn.Module | None) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        sample: Any = self.dataset[idx]
        if isinstance(sample, tuple):
            sample = sample[0]
        if self.transform is not None:
            sample = self.transform(sample)
        if isinstance(sample, RadioTensor):
            return sample.as_tensor()
        if torch.is_tensor(sample):
            return sample
        raise TypeError(f"Expected RadioTensor or torch.Tensor, got {type(sample)!r}")


def _execution_sample_tensor(sample: object) -> torch.Tensor:
    if not isinstance(sample, RadioTensor):
        raise TypeError(
            "Execution dataloaders expect DatasetView samples to remain RadioTensor instances "
            f"after optional transforms, got {type(sample)!r}."
        )
    tensor = sample.to_tensor(contiguous=True)
    return tensor if tensor.is_complex() else tensor.float()


def _is_dataset_mapping(value: object) -> bool:
    return isinstance(value, Mapping)


def _coerce_dataset_leaf(value: object) -> DatasetView | None:
    if isinstance(value, DatasetView):
        return value
    return None


def _reference_dataset(dataset_source: object) -> DatasetView | None:
    if _is_dataset_mapping(dataset_source):
        mapping = dict(cast(Mapping[str, object], dataset_source))
        for split_name in ("train", "val", "test"):
            dataset = _coerce_dataset_leaf(mapping.get(split_name))
            if dataset is not None:
                return dataset
        return None

    dataset = _coerce_dataset_leaf(dataset_source)
    if dataset is not None:
        return dataset
    return None

def _split_dataset_mapping(
    dataset_source: DatasetSource,
) -> dict[str, DatasetView] | None:
    if _is_dataset_mapping(dataset_source):
        mapping: dict[str, DatasetView] = {}
        for split_name, split_value in cast(Mapping[str, object], dataset_source).items():
            dataset = _coerce_dataset_leaf(split_value)
            if dataset is not None:
                mapping[str(split_name)] = dataset
        return mapping if {"train", "val"}.issubset(set(mapping)) else None

    if not isinstance(dataset_source, DatasetView):
        return None

    split_mapping: dict[str, DatasetView] = {}
    for split_name in ("train", "val", "test"):
        try:
            split_value = dataset_source.get_split(split_name)
        except Exception:
            continue
        dataset = _coerce_dataset_leaf(split_value)
        if dataset is not None:
            split_mapping[split_name] = dataset
    return split_mapping if {"train", "val"}.issubset(set(split_mapping)) else None

def resolve_loader_runtime_kwargs(
    requested_num_workers: int,
    *,
    device: str | torch.device | None = None,
) -> tuple[int, dict[str, Any]]:
    effective_num_workers, _ = resolve_num_workers(requested_num_workers, strict=True)
    runtime_kwargs: dict[str, Any] = {}
    if device is not None:
        resolved_device = torch.device(device)
    elif torch.cuda.is_available():
        resolved_device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        resolved_device = torch.device("mps")
    else:
        resolved_device = torch.device("cpu")
    enable_pin_memory = os.environ.get("OCTOSENSE_ENABLE_PIN_MEMORY", "").strip().lower()
    if resolved_device.type == "cuda" and enable_pin_memory in {"1", "true", "yes", "on"}:
        runtime_kwargs["pin_memory"] = True
    if effective_num_workers > 0:
        runtime_kwargs["persistent_workers"] = True
        runtime_kwargs["prefetch_factor"] = 2
    return effective_num_workers, runtime_kwargs


def build_data_loader(
    dataset: object,
    *,
    config: DataLoaderConfig,
) -> DataLoader:
    effective_num_workers, runtime_kwargs = resolve_loader_runtime_kwargs(
        config.num_workers,
        device=config.device,
    )
    sampler = build_sampler(
        dataset,
        config.shuffle,
        weights=config.sampler_weights,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=effective_num_workers,
        collate_fn=config.collate_fn,
        drop_last=config.drop_last,
        **runtime_kwargs,
    )


def build_execution_dataloaders(
    dataset: DatasetSource,
    *,
    task_spec: TaskSpec | None = None,
    transform: nn.Module | None = None,
    batch_size: int,
    num_workers: int,
    seed: int,
    shuffle: bool = True,
    device: str | torch.device | None = None,
) -> ExecutionDataLoaders:
    """Assemble canonical dataloaders from explicit runtime values and dataset contracts."""

    if batch_size <= 0:
        raise ValueError("build_execution_dataloaders(...) requires batch_size > 0")
    if num_workers < 0:
        raise ValueError("build_execution_dataloaders(...) requires num_workers >= 0")
    if seed < 0:
        raise ValueError("build_execution_dataloaders(...) requires seed >= 0")

    split_mapping = _split_dataset_mapping(dataset)
    if split_mapping is None:
        raise ValueError(
            "build_execution_dataloaders(...) requires explicit dataset-owned 'train'/'val' "
            "splits; pipeline dataloading no longer derives split policy from targets."
        )
    resolved_adapter = _resolve_target_adapter(task_spec, dataset)
    mapped_train, mapped_val, mapped_test, collate_fn = _resolve_mapped_execution_datasets(
        split_mapping,
        adapter_id=resolved_adapter.adapter_id,
        transform=transform,
    )

    sample_tensor, _ = mapped_train[0]
    return ExecutionDataLoaders(
        train_loader=build_data_loader(
            mapped_train,
            config=DataLoaderConfig(
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle,
                device=device,
                collate_fn=collate_fn,
            ),
        ),
        val_loader=build_data_loader(
            mapped_val,
            config=DataLoaderConfig(
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
                device=device,
                collate_fn=collate_fn,
            ),
        ),
        test_loader=(
            build_data_loader(
                mapped_test,
                config=DataLoaderConfig(
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=False,
                    device=device,
                    collate_fn=collate_fn,
                ),
            )
            if mapped_test is not None
            else None
        ),
        sample_tensor=sample_tensor,
        runtime_config=ExecutionRuntimeConfig(
            seed=seed,
            batch_size=batch_size,
            num_workers=num_workers,
        ),
    )


def load_tensor_batch(
    dataset: Dataset[Any],
    *,
    transform: nn.Module | None,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = False,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    mapped = _MappedSampleTensorDataset(dataset, transform)
    loader = build_data_loader(
        mapped,
        config=DataLoaderConfig(
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            device=device,
        ),
    )
    batch = next(iter(loader))
    if not torch.is_tensor(batch):
        raise TypeError(f"Expected torch.Tensor batch, got {type(batch)!r}")
    return batch


__all__ = [
    "DataLoaderConfig",
    "DatasetSource",
    "DatasetSplitMapping",
    "build_data_loader",
    "build_execution_dataloaders",
    "load_tensor_batch",
    "resolve_loader_runtime_kwargs",
]
