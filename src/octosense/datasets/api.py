"""Internal loader implementation for ``octosense.datasets``.

The package root owns the canonical public surface. This module exists only to
host the root-backed loader implementation consumed by
``octosense.datasets.__init__``.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from collections.abc import Mapping, Sequence
from inspect import isroutine
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import Dataset

from octosense.core import DescribeNode, ensure_describe_node
from octosense.datasets.base import (
    BaseDatasetAdapter,
    DatasetLoadRequest,
    freeze_public_dataset_options_for_cache,
    normalize_public_dataset_options,
)
from octosense.datasets.core.schema import DatasetMetadata
from octosense.datasets.views.dataset_view import DatasetView, copy_dataset_metadata
from octosense.datasets.views.in_memory import (
    InMemoryClassificationDataset,
    radiotensor_samples_from_tensor_input,
)
from octosense.io.semantics.metadata import SignalMetadata
from octosense.io.tensor import RadioTensor

_DEFAULT_FROM_TENSOR_SEED = 41
_DEFAULT_FROM_TENSOR_TRAIN_RATIO = 0.8


def _annotate_split_rows(
    rows: Sequence[dict[str, object]],
    *,
    split: str,
) -> list[dict[str, object]]:
    annotated_rows: list[dict[str, object]] = []
    for row in rows:
        partitions = dict(row.get("partitions", {})) if isinstance(row.get("partitions"), Mapping) else {}
        partitions["split"] = split
        annotated_rows.append(
            {
                **dict(row),
                "split": split,
                "assigned_split": split,
                "partitions": partitions,
            }
        )
    return annotated_rows

def _resolve_from_tensor_split_controls(
    payload: object | None,
    *,
    seed: int | None,
) -> tuple[int, float]:
    if payload is None:
        mapping: Mapping[str, object] = {}
    elif isinstance(payload, Mapping):
        mapping = payload
    else:
        raise TypeError("datasets.from_tensor(..., defaults=...) expects a mapping when provided.")

    unsupported_keys = {
        str(key)
        for key in mapping
        if str(key) not in {"seed", "train_ratio"}
    }
    if unsupported_keys:
        unsupported = ", ".join(sorted(unsupported_keys))
        raise ValueError(
            "datasets.from_tensor(..., defaults=...) only accepts dataset split controls "
            f"('seed', 'train_ratio'); got unsupported keys: {unsupported}"
        )

    resolved_seed = int(mapping.get("seed", _DEFAULT_FROM_TENSOR_SEED) if seed is None else seed)
    train_ratio = float(mapping.get("train_ratio", _DEFAULT_FROM_TENSOR_TRAIN_RATIO))
    return resolved_seed, train_ratio


def _normalize_from_tensor_modalities(
    modalities: Sequence[str] | None,
) -> tuple[str, ...]:
    if modalities is None:
        return ()
    if isinstance(modalities, str):
        raise TypeError(
            "datasets.from_tensor(..., modalities=...) expects a sequence of modality names."
        )
    normalized: list[str] = []
    for item in modalities:
        candidate = str(item).strip()
        if not candidate:
            continue
        if candidate in normalized:
            raise ValueError(
                "datasets.from_tensor(..., modalities=...) received duplicate modality entries."
            )
        normalized.append(candidate)
    return tuple(normalized)


def _sorted_unique(values: Sequence[Any]) -> list[Any]:
    unique_values = set(values)
    if not unique_values:
        return []
    try:
        return sorted(unique_values)
    except TypeError:
        return sorted(unique_values, key=lambda value: (type(value).__name__, str(value)))


def _partition_sorted_groups(
    groups: Sequence[Any],
    *,
    train_ratio: float,
) -> tuple[list[Any], list[Any]]:
    unique_groups = _sorted_unique(groups)
    if not unique_groups:
        raise ValueError("Cannot split an empty group list")
    if len(unique_groups) == 1:
        return unique_groups, []

    train_count = int(round(len(unique_groups) * train_ratio))
    train_count = max(1, min(len(unique_groups) - 1, train_count))
    train_groups = list(unique_groups[:train_count])
    val_groups = list(unique_groups[train_count:])
    return train_groups, val_groups


def _subset_indices_by_group(
    groups: Sequence[Any],
    *,
    train_ratio: float,
) -> tuple[list[int], list[int], list[Any], list[Any]]:
    train_groups, val_groups = _partition_sorted_groups(groups, train_ratio=train_ratio)
    train_group_set = set(train_groups)
    train_indices = [idx for idx, group in enumerate(groups) if group in train_group_set]
    val_indices = [idx for idx, group in enumerate(groups) if group not in train_group_set]
    return train_indices, val_indices, train_groups, val_groups


def _schema_declared_field_names(schema: object) -> list[str]:
    declared: list[str] = []
    for collection_name in ("coordinates", "columns"):
        collection = getattr(schema, collection_name, None)
        if not isinstance(collection, Sequence):
            continue
        for item in collection:
            field_name = getattr(item, "name", None)
            if isinstance(field_name, str) and field_name:
                declared.append(field_name)
    return list(dict.fromkeys(declared))


def _declared_metadata_field_names(dataset: Dataset[Any]) -> list[str]:
    declared: list[str] = []

    manifest = getattr(dataset, "manifest", None)
    schema = getattr(manifest, "schema", None)
    if schema is not None:
        declared.extend(_schema_declared_field_names(schema))

    for provider_name in ("manifest_schema", "get_manifest_schema"):
        provider = getattr(dataset, provider_name, None)
        payload = provider() if callable(provider) else provider
        if payload is not None:
            declared.extend(_schema_declared_field_names(payload))

    metadata = getattr(dataset, "dataset_metadata", None)
    metadata_extra = getattr(metadata, "extra", None)
    if isinstance(metadata_extra, Mapping):
        metadata_fields = metadata_extra.get("metadata_fields")
        if isinstance(metadata_fields, Sequence) and not isinstance(metadata_fields, str):
            declared.extend(str(field_name) for field_name in metadata_fields)

    return list(dict.fromkeys(field_name for field_name in declared if field_name))


def _label_metadata_rows(dataset: Dataset[Any]) -> list[dict[str, object]]:
    labels = _dataset_labels(dataset)
    declared_fields = _declared_metadata_field_names(dataset)
    provider = getattr(dataset, "metadata_rows", None)
    if callable(provider):
        rows = provider()
        if isinstance(rows, list) and len(rows) == len(labels):
            normalized_rows: list[dict[str, object]] = []
            for index, (row, label) in enumerate(zip(rows, labels, strict=True)):
                normalized = {
                    str(key): _normalize_metadata_value(value)
                    for key, value in dict(row).items()
                }
                normalized.setdefault("sample_index", int(index))
                normalized["label"] = int(label)
                normalized_rows.append(normalized)
            return normalized_rows

    rows: list[dict[str, object]] = []
    get_sample_id = getattr(dataset, "get_sample_id", None)
    get_group_id = getattr(dataset, "get_group_id", None)
    samples = getattr(dataset, "samples", None)
    for index, label in enumerate(labels):
        row = {
            "sample_index": int(index),
            "label": int(label),
        }
        if callable(get_sample_id):
            row["sample_id"] = str(get_sample_id(index))
        if callable(get_group_id):
            row["sample_group_id"] = str(get_group_id(index))
        if samples is not None and index < len(samples):
            row.update(
                _sample_record_metadata(
                    samples[index],
                    declared_fields=declared_fields or None,
                )
            )
        rows.append(row)
    return rows


def _normalize_metadata_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _is_metadata_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, (str, int, float, bool, Path, np.generic)):
        return True
    return False


def _mapping_metadata_row(mapping: Mapping[object, object]) -> dict[str, object]:
    return {
        str(key): _normalize_metadata_value(value)
        for key, value in mapping.items()
        if value is not None
    }


def _public_sample_attribute_names(sample: object) -> list[str]:
    try:
        attribute_names = vars(sample).keys()
    except TypeError:
        return []
    return [
        str(name)
        for name in sorted(attribute_names)
        if isinstance(name, str) and name and not name.startswith("_")
    ]


def _sample_record_metadata(
    sample: object,
    *,
    declared_fields: Sequence[str] | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {}

    for provider_name in ("metadata_row", "to_metadata_row"):
        provider = getattr(sample, provider_name, None)
        if callable(provider):
            payload = provider()
            if isinstance(payload, Mapping):
                row.update(_mapping_metadata_row(payload))

    record = getattr(sample, "record", None)
    if record is not None:
        for provider_name in ("metadata_row", "to_metadata_row"):
            provider = getattr(record, provider_name, None)
            if callable(provider):
                payload = provider()
                if isinstance(payload, Mapping):
                    row.update(_mapping_metadata_row(payload))

    for method_name, field_name in (
        ("sample_id", "sample_id"),
        ("group_id", "sample_group_id"),
    ):
        method = getattr(sample, method_name, None)
        if callable(method):
            row[field_name] = str(method())

    field_names = (
        list(dict.fromkeys(str(field_name) for field_name in declared_fields))
        if declared_fields is not None
        else _public_sample_attribute_names(sample)
    )
    for field_name in field_names:
        if hasattr(sample, field_name):
            value = getattr(sample, field_name)
            if isroutine(value):
                continue
            if _is_metadata_value(value):
                row[field_name] = _normalize_metadata_value(value)
    return row


def _dataset_labels(dataset: Dataset[Any]) -> list[int]:
    if hasattr(dataset, "get_labels"):
        labels = dataset.get_labels()  # type: ignore[attr-defined]
        return [int(label) for label in labels]
    return [int(dataset[idx][1]) for idx in range(len(dataset))]


def _infer_view_sample_describe_tree(
    dataset: Dataset[Any],
) -> DescribeNode:
    for provider_name in ("sample_describe_tree", "get_sample_describe_tree"):
        provider = getattr(dataset, provider_name, None)
        if callable(provider):
            payload = provider()
            if isinstance(payload, (dict, DescribeNode)):
                return ensure_describe_node(payload).with_name("sample")

    sample, _ = dataset[0]
    if not isinstance(sample, RadioTensor):
        raise TypeError(
            "Expected dataset sample to be RadioTensor when inferring view describe tree, "
            f"got {type(sample)!r}"
        )
    return sample.describe_tree().with_name("sample")


def _stratified_train_val_indices(
    dataset: Dataset[Any],
    *,
    train_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    labels = _dataset_labels(dataset)
    by_label: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_label[int(label)].append(idx)

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for _, indices in sorted(by_label.items()):
        shuffled = list(indices)
        rng.shuffle(shuffled)
        if len(shuffled) == 1:
            train_indices.extend(shuffled)
            continue
        train_count = int(round(len(shuffled) * train_ratio))
        train_count = max(1, min(len(shuffled) - 1, train_count))
        train_indices.extend(shuffled[:train_count])
        val_indices.extend(shuffled[train_count:])

    if not val_indices and len(train_indices) > 1:
        val_indices.append(train_indices.pop())

    train_indices.sort()
    val_indices.sort()
    return train_indices, val_indices


def _infer_label_mapping_and_targets(
    targets: Sequence[int | str] | torch.Tensor,
) -> tuple[list[int], dict[str, int]]:
    if torch.is_tensor(targets):
        target_values = targets.detach().cpu().tolist()
    else:
        target_values = list(targets)
    if not target_values:
        raise ValueError("targets must not be empty")

    if all(isinstance(value, str) for value in target_values):
        label_mapping: dict[str, int] = {}
        encoded: list[int] = []
        for value in target_values:
            key = str(value)
            if key not in label_mapping:
                label_mapping[key] = len(label_mapping)
            encoded.append(label_mapping[key])
        return encoded, label_mapping

    encoded = [int(value) for value in target_values]
    unique_labels = sorted(set(encoded))
    return encoded, {str(label): int(label) for label in unique_labels}


def _stratified_trainval_test_indices(
    targets: Sequence[int],
    *,
    test_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if not 0.0 <= float(test_ratio) < 1.0:
        raise ValueError(f"test_ratio must be in [0, 1), got {test_ratio}")

    by_label: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(targets):
        by_label[int(label)].append(idx)

    rng = np.random.default_rng(seed)
    trainval_indices: list[int] = []
    test_indices: list[int] = []
    largest_label: int | None = None
    largest_size = -1

    for label, indices in sorted(by_label.items()):
        shuffled = list(indices)
        rng.shuffle(shuffled)
        if len(shuffled) > largest_size:
            largest_label = int(label)
            largest_size = len(shuffled)

        if len(shuffled) >= 3 and test_ratio > 0.0:
            proposed = int(round(len(shuffled) * test_ratio))
            test_count = max(1, min(len(shuffled) - 2, proposed))
        else:
            test_count = 0

        if test_count > 0:
            test_indices.extend(shuffled[:test_count])
            trainval_indices.extend(shuffled[test_count:])
        else:
            trainval_indices.extend(shuffled)

    if not test_indices and test_ratio > 0.0 and len(targets) >= 3 and largest_label is not None:
        source = by_label[largest_label]
        moved_index = int(source[0])
        if moved_index in trainval_indices:
            trainval_indices.remove(moved_index)
        test_indices.append(moved_index)

    trainval_indices.sort()
    test_indices.sort()
    return trainval_indices, test_indices


def _build_from_tensor_view(
    tensor: RadioTensor | torch.Tensor | Sequence[RadioTensor],
    targets: Sequence[int | str] | torch.Tensor,
    *,
    axis_names: Sequence[str] | None = None,
    metadata: SignalMetadata | None = None,
    modalities: Sequence[str] | None = None,
    dataset_id: str,
    variant: str,
    path: str | Path | None = None,
    defaults: Mapping[str, object] | None = None,
    seed: int | None = None,
    test_ratio: float | None = 0.1,
) -> DatasetView:
    resolved_modalities = _normalize_from_tensor_modalities(modalities)
    samples = radiotensor_samples_from_tensor_input(
        tensor,
        axis_names=axis_names,
        metadata=metadata,
    )
    encoded_targets, label_mapping = _infer_label_mapping_and_targets(targets)
    if len(samples) != len(encoded_targets):
        raise ValueError(
            f"dataset.from_tensor length mismatch: {len(samples)} samples vs {len(encoded_targets)} targets"
        )
    dataset = InMemoryClassificationDataset(
        samples,
        encoded_targets,
        label_mapping=label_mapping,
        dataset_metadata=DatasetMetadata(
            name=dataset_id,
            sample_count=len(samples),
            modalities=resolved_modalities,
            extra={
                "variant": variant,
                "source": "dataset.from_tensor",
            },
        ),
    )
    resolved_path = Path(path).expanduser().resolve() if path is not None else None
    resolved_seed, train_ratio = _resolve_from_tensor_split_controls(
        defaults,
        seed=seed,
    )
    metadata_rows = dataset.metadata_rows()

    trainval_indices = list(range(len(dataset)))
    test_indices: list[int] = []
    if test_ratio is not None and float(test_ratio) > 0.0 and len(dataset) >= 3:
        trainval_indices, test_indices = _stratified_trainval_test_indices(
            encoded_targets,
            test_ratio=float(test_ratio),
            seed=resolved_seed,
        )
    trainval_labels = [encoded_targets[index] for index in trainval_indices]
    split_train_indices, split_val_indices = _stratified_train_val_indices(
        InMemoryClassificationDataset(
            [samples[index] for index in trainval_indices],
            trainval_labels,
            label_mapping=label_mapping,
            dataset_metadata=copy.deepcopy(dataset.dataset_metadata),
        ),
        train_ratio=train_ratio,
        seed=resolved_seed,
    )
    train_indices = [trainval_indices[index] for index in split_train_indices]
    val_indices = [trainval_indices[index] for index in split_val_indices]

    ordered_splits: list[tuple[str, list[int]]] = [
        ("train", train_indices),
        ("val", val_indices),
    ]
    if test_indices:
        ordered_splits.append(("test", test_indices))

    merged_indices: list[int] = []
    merged_rows: list[dict[str, object]] = []
    for split_name, indices in ordered_splits:
        merged_indices.extend(indices)
        merged_rows.extend(
            _annotate_split_rows(
                [metadata_rows[index] for index in indices],
                split=split_name,
            )
        )

    root_split = "all" if test_indices else "train+val"
    root_view = DatasetView(
        dataset,
        merged_indices,
        dataset_id=dataset_id,
        variant=variant,
        split=root_split,
        label_mapping=label_mapping,
        target_schema=dataset.get_target_schema(),
        dataset_metadata=copy_dataset_metadata(dataset, sample_count=len(merged_indices)),
        metadata_rows=merged_rows or None,
        sample_describe_tree=dataset.sample_describe_tree(),
    )
    return root_view


def _build_load_request(
    dataset_id: str,
    *,
    modalities: Sequence[str] | None,
    variant: str | None,
    split_scheme: str | None,
    task_binding: str | None,
    path: str | Path | None,
    options: Mapping[str, object] | None,
) -> DatasetLoadRequest:
    return DatasetLoadRequest.from_public_load(
        dataset_id,
        modalities=modalities,
        variant=variant,
        split_scheme=split_scheme,
        task_binding=task_binding,
        path=path,
        options=options,
    )


def _resolve_dataset_adapter(request: DatasetLoadRequest) -> BaseDatasetAdapter:
    from octosense.datasets.registry import resolve_dataset_adapter

    return resolve_dataset_adapter(request)


def _request_cache_key(request: DatasetLoadRequest) -> tuple[object, ...]:
    normalized_root = (
        str(request.dataset_root.expanduser().resolve())
        if request.dataset_root is not None
        else None
    )
    return (
        request.dataset_id,
        request.modalities,
        request.variant,
        request.split_scheme,
        request.task_binding,
        normalized_root,
        freeze_public_dataset_options_for_cache(request.options),
    )


def _view_cache_key(
    request: DatasetLoadRequest,
) -> tuple[object, ...]:
    return _request_cache_key(request)


class _DatasetNamespace:
    def __init__(self) -> None:
        self._view_cache: dict[tuple[object, ...], DatasetView] = {}

    def clear_cache(self) -> None:
        self._view_cache.clear()

    def _resolve_request_and_adapter(
        self,
        dataset_id: str,
        *,
        modalities: Sequence[str] | None,
        variant: str | None = None,
        split_scheme: str | None = None,
        task_binding: str | None = None,
        path: str | Path | None = None,
        options: Mapping[str, object] | None = None,
    ) -> tuple[DatasetLoadRequest, BaseDatasetAdapter]:
        request = _build_load_request(
            dataset_id,
            modalities=modalities,
            variant=variant,
            split_scheme=split_scheme,
            task_binding=task_binding,
            path=path,
            options=options,
        )
        adapter = _resolve_dataset_adapter(request)
        return request, adapter

    def load(
        self,
        dataset_id: str,
        *,
        modalities: Sequence[str] | None = None,
        variant: str | None = None,
        split_scheme: str | None = None,
        task_binding: str | None = None,
        path: str | Path | None = None,
        options: Mapping[str, object] | None = None,
    ) -> DatasetView:
        request, adapter = self._resolve_request_and_adapter(
            dataset_id,
            modalities=modalities,
            variant=variant,
            split_scheme=split_scheme,
            task_binding=task_binding,
            path=path,
            options=options,
        )
        cache_key = _view_cache_key(request)
        cached = self._view_cache.get(cache_key)
        if cached is not None:
            return cached
        view = adapter.build_view(request)
        self._view_cache[cache_key] = view
        return view

    def from_tensor(
        self,
        tensor: RadioTensor | torch.Tensor | Sequence[RadioTensor],
        targets: Sequence[int | str] | torch.Tensor,
        *,
        axis_names: Sequence[str] | None = None,
        metadata: SignalMetadata | None = None,
        modalities: Sequence[str] | None = None,
        dataset_id: str = "custom",
        variant: str = "from_tensor",
        path: str | Path | None = None,
        defaults: Mapping[str, object] | None = None,
        seed: int | None = None,
        test_ratio: float | None = 0.1,
    ) -> DatasetView:
        return _build_from_tensor_view(
            tensor,
            targets,
            axis_names=axis_names,
            metadata=metadata,
            modalities=modalities,
            dataset_id=dataset_id,
            variant=variant,
            path=path,
            defaults=defaults,
            seed=seed,
            test_ratio=test_ratio,
        )


dataset = _DatasetNamespace()


def load(
    dataset_id: str,
    *,
    modalities: Sequence[str] | None = None,
    variant: str | None = None,
    split_scheme: str | None = None,
    task_binding: str | None = None,
    path: str | Path | None = None,
    options: Mapping[str, object] | None = None,
) -> DatasetView:
    """Load a dataset and return ``DatasetView`` as the canonical public object.

    ``modalities``, ``variant``, ``split_scheme``, and ``task_binding`` are the
    dataset-side selectors. Concrete split views are accessed via
    ``DatasetView.get_split(...)`` on the returned root view.
    """

    return dataset.load(
        dataset_id,
        modalities=modalities,
        variant=variant,
        split_scheme=split_scheme,
        task_binding=task_binding,
        path=path,
        options=normalize_public_dataset_options(options),
    )


def from_tensor(
    tensor: RadioTensor | torch.Tensor | Sequence[RadioTensor],
    targets: Sequence[int | str] | torch.Tensor,
    *,
    axis_names: Sequence[str] | None = None,
    metadata: SignalMetadata | None = None,
    modalities: Sequence[str] | None = None,
    dataset_id: str = "custom",
    variant: str = "from_tensor",
    path: str | Path | None = None,
    defaults: Mapping[str, object] | None = None,
    seed: int | None = None,
    test_ratio: float | None = 0.1,
) -> DatasetView:
    """Build an in-memory ``DatasetView`` from tensor-backed samples."""

    return dataset.from_tensor(
        tensor,
        targets,
        axis_names=axis_names,
        metadata=metadata,
        modalities=modalities,
        dataset_id=dataset_id,
        variant=variant,
        path=path,
        defaults=defaults,
        seed=seed,
        test_ratio=test_ratio,
    )
