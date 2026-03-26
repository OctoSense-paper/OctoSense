"""In-memory view owner for ``datasets.from_tensor(...)`` inputs."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.utils.data import Dataset

from octosense.core import DescribeNode
from octosense.datasets.core.schema import DatasetMetadata
from octosense.io.semantics.metadata import SignalMetadata
from octosense.io.semantics.schema import AxisSchema
from octosense.io.tensor import RadioTensor


class InMemoryClassificationDataset(Dataset[tuple[RadioTensor, int]]):
    """In-memory sample source used by ``datasets.from_tensor(...)``.

    This dataset intentionally stays task-agnostic: it owns concrete label
    storage and encoding details for scalar targets, but it does not claim a
    canonical task identity or task-owned semantic target kind.
    """

    def __init__(
        self,
        samples: Sequence[RadioTensor],
        labels: Sequence[int],
        *,
        label_mapping: dict[str, int],
        sample_ids: Sequence[str] | None = None,
        dataset_metadata: DatasetMetadata | None = None,
    ) -> None:
        if len(samples) != len(labels):
            raise ValueError(f"samples/labels length mismatch: {len(samples)} != {len(labels)}")
        self.samples = list(samples)
        self.label_mapping = dict(label_mapping)
        self.sample_ids = (
            list(sample_ids)
            if sample_ids is not None
            else [f"sample-{index}" for index in range(len(self.samples))]
        )
        self._dataset_metadata = dataset_metadata
        self._target_schema = _in_memory_label_schema(self.label_mapping)
        self._metadata_rows = [
            {
                "sample_index": int(index),
                "sample_id": str(self.sample_ids[index]),
                "label": int(label),
            }
            for index, label in enumerate(labels)
        ]
        self._sample_describe_tree = (
            self.samples[0].describe_tree().with_name("sample") if self.samples else None
        )
        if self._dataset_metadata is not None:
            self._dataset_metadata.extra.setdefault(
                "target_coordinates",
                {
                    "attached_axis": "sample",
                    "fields": {
                        "label": {
                            "encoding": "index",
                            "label_mapping": dict(self.label_mapping),
                        }
                    },
                },
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[RadioTensor, int]:
        return self.samples[idx], int(self._metadata_rows[idx]["label"])

    def get_labels(self) -> list[int]:
        return [int(row["label"]) for row in self._metadata_rows]

    def get_label_mapping(self) -> dict[str, int]:
        return dict(self.label_mapping)

    def get_target_schema(self) -> dict[str, object]:
        return dict(self._target_schema)

    def get_sample_id(self, idx: int) -> str:
        return self.sample_ids[idx]

    def metadata_rows(self) -> list[dict[str, object]]:
        return [dict(row) for row in self._metadata_rows]

    def sample_describe_tree(self) -> DescribeNode:
        if self._sample_describe_tree is None:
            raise ValueError("InMemoryClassificationDataset has no samples")
        return self._sample_describe_tree

    @property
    def dataset_metadata(self) -> DatasetMetadata | None:
        return self._dataset_metadata


def _in_memory_label_schema(label_mapping: dict[str, int]) -> dict[str, object]:
    ordered_labels = [
        label
        for label, _ in sorted(
            label_mapping.items(),
            key=lambda item: int(item[1]),
        )
    ]
    return {
        "source": "metadata_rows",
        "label_field": "label",
        "encoding": "index",
        "labels": ordered_labels,
        "label_mapping": dict(label_mapping),
    }


def radiotensor_samples_from_tensor_input(
    tensor: RadioTensor | torch.Tensor | Sequence[RadioTensor],
    *,
    axis_names: Sequence[str] | None = None,
    metadata: SignalMetadata | None = None,
) -> list[RadioTensor]:
    """Convert ``datasets.from_tensor(...)`` input into sample-level ``RadioTensor`` objects."""

    def _slice_batched_radiotensor(rt: RadioTensor, sample_idx: int) -> RadioTensor:
        if len(rt.shape) < 2:
            raise ValueError("Batched RadioTensor must include a leading sample axis")
        sample_axis = rt.axis_schema.axes[0]
        if sample_axis not in {"sample", "item", "batch"}:
            raise ValueError(
                "dataset.from_tensor expected the leading RadioTensor axis to be one of "
                "('sample', 'item', 'batch'), "
                f"got {sample_axis!r}"
            )
        sample_tensor = rt.to_tensor(contiguous=True)[sample_idx]
        sample_axes = tuple(rt.axis_schema.axes[1:])
        axis_metadata = {
            name: meta
            for name, meta in rt.axis_schema.axis_metadata.items()
            if name in sample_axes
        }
        copied_metadata = rt.metadata.copy()
        copied_metadata.coords.pop(sample_axis, None)
        return RadioTensor(
            sample_tensor,
            AxisSchema(sample_axes, axis_metadata=axis_metadata),
            copied_metadata,
        )

    if isinstance(tensor, Sequence) and not torch.is_tensor(tensor):
        samples = list(tensor)
        if not samples:
            raise ValueError("tensor sequence must not be empty")
        if not all(isinstance(sample, RadioTensor) for sample in samples):
            raise TypeError("tensor sequence must contain only RadioTensor samples")
        return samples

    if isinstance(tensor, RadioTensor):
        leading_axis = tensor.axis_schema.axes[0] if tensor.axis_schema.axes else None
        if leading_axis in {"sample", "item", "batch"}:
            return [_slice_batched_radiotensor(tensor, idx) for idx in range(int(tensor.shape[0]))]
        return [tensor]

    if not torch.is_tensor(tensor):
        raise TypeError(
            "dataset.from_tensor expects RadioTensor, torch.Tensor, or a sequence of RadioTensor samples, "
            f"got {type(tensor)!r}"
        )
    if tensor.ndim < 2:
        raise ValueError("Plain torch.Tensor input must have a leading sample axis")
    if axis_names is None:
        raise ValueError("axis_names are required when dataset.from_tensor receives a plain torch.Tensor")

    axis_tuple = tuple(str(axis_name) for axis_name in axis_names)
    if len(axis_tuple) == tensor.ndim:
        if axis_tuple[0] not in {"sample", "item", "batch"}:
            raise ValueError(
                "When axis_names includes the leading sample axis, it must be named "
                "'sample', 'item', or 'batch'"
            )
        sample_axes = axis_tuple[1:]
    elif len(axis_tuple) == tensor.ndim - 1:
        sample_axes = axis_tuple
    else:
        raise ValueError(
            f"axis_names length must be {tensor.ndim - 1} or {tensor.ndim}, got {len(axis_tuple)}"
        )

    base_metadata = metadata.copy() if metadata is not None else SignalMetadata()
    schema = AxisSchema(sample_axes)
    return [
        RadioTensor(
            tensor[index].contiguous(),
            schema,
            base_metadata.copy(),
        )
        for index in range(int(tensor.shape[0]))
    ]


__all__ = [
    "InMemoryClassificationDataset",
    "radiotensor_samples_from_tensor_input",
]
