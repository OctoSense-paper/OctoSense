"""Metadata partition helpers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from octosense.datasets.views.filter import MetadataFilter, compile_metadata_filter


@dataclass(frozen=True)
class PartitionPlan:
    """Two-way partition over metadata rows."""

    equals: Mapping[str, object] = field(default_factory=dict)
    predicate: Callable[[dict[str, object]], bool] | None = None
    matched_split_name: str | None = None
    remainder_split_name: str | None = None

    def selector(self) -> MetadataFilter:
        return MetadataFilter(equals=dict(self.equals), predicate=self.predicate)


def partition_positions(
    rows: list[dict[str, Any]],
    *,
    predicate: Callable[[dict[str, object]], bool] | None = None,
    **equals: object,
) -> tuple[list[int], list[int]]:
    selected = compile_metadata_filter(predicate=predicate, **equals).select_positions(rows)
    selected_set = set(selected)
    remainder = [index for index in range(len(rows)) if index not in selected_set]
    return selected, remainder


def partition_rows(rows: list[dict[str, Any]], key: str) -> dict[Any, list[int]]:
    partitions: dict[Any, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        partitions[row.get(key)].append(index)
    return dict(partitions)


__all__ = ["PartitionPlan", "partition_positions", "partition_rows"]
