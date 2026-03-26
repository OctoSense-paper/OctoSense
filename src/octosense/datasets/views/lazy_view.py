"""Lazy dataset-view owner for deferred sample access."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from torch.utils.data import Dataset

from octosense.core import DescribeNode, Describable, ensure_describe_node
from octosense.io.tensor import RadioTensor


class LazyDatasetView(Dataset[tuple[RadioTensor, Any]], Describable):
    """Resolve an expensive dataset or view only when accessed."""

    def __init__(
        self,
        factory: Callable[[], Dataset[tuple[RadioTensor, Any]]],
        *,
        describe_tree_payload: DescribeNode | dict[str, object] | None = None,
    ) -> None:
        self._factory = factory
        self._dataset: Dataset[tuple[RadioTensor, Any]] | None = None
        self._describe_tree_payload = (
            ensure_describe_node(describe_tree_payload)
            if describe_tree_payload is not None
            else None
        )

    def _resolve(self) -> Dataset[tuple[RadioTensor, Any]]:
        if self._dataset is None:
            self._dataset = self._factory()
        return self._dataset

    def __len__(self) -> int:
        return len(self._resolve())

    def __getitem__(self, idx: int) -> tuple[RadioTensor, Any]:
        return self._resolve()[idx]

    def get_labels(self) -> list[int]:
        dataset = self._resolve()
        if not hasattr(dataset, "get_labels"):
            raise AttributeError("Underlying dataset does not expose get_labels()")
        labels = dataset.get_labels()  # type: ignore[attr-defined]
        return [int(label) for label in labels]

    def get_label_mapping(self) -> dict[str, int]:
        dataset = self._resolve()
        if not hasattr(dataset, "get_label_mapping"):
            raise AttributeError("Underlying dataset does not expose get_label_mapping()")
        return dict(dataset.get_label_mapping())  # type: ignore[attr-defined]

    def get_target_schema(self) -> dict[str, object]:
        dataset = self._resolve()
        if not hasattr(dataset, "get_target_schema"):
            raise AttributeError("Underlying dataset does not expose get_target_schema()")
        target_schema = dataset.get_target_schema()  # type: ignore[attr-defined]
        if not isinstance(target_schema, Mapping):
            raise TypeError(
                "Underlying dataset get_target_schema() must return dataset-local target layout."
            )
        return {
            str(key): value
            for key, value in target_schema.items()
            if str(key) not in {"task_id", "task_kind", "target_kind"}
        }

    @property
    def task_id(self) -> str | None:
        return getattr(self._resolve(), "task_id", None)

    @property
    def task_kind(self) -> str | None:
        return getattr(self._resolve(), "task_kind", None)

    @property
    def target_kind(self) -> str | None:
        return getattr(self._resolve(), "target_kind", None)

    @property
    def dataset_metadata(self) -> Any:
        return getattr(self._resolve(), "dataset_metadata", None)

    def describe_tree(self) -> DescribeNode:
        if self._describe_tree_payload is not None:
            return self._describe_tree_payload
        dataset = self._resolve()
        describe_tree = getattr(dataset, "describe_tree", None)
        if callable(describe_tree):
            return ensure_describe_node(describe_tree())
        raise AttributeError("Resolved dataset does not expose describe_tree()")


__all__ = ["LazyDatasetView"]
