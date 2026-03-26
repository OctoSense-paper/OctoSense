"""Metadata filter helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MetadataFilter:
    """Declarative metadata filter used by DatasetView."""

    equals: Mapping[str, object] = field(default_factory=dict)
    predicate: Callable[[dict[str, object]], bool] | None = None

    def matches(self, row: Mapping[str, Any]) -> bool:
        materialized = {str(key): value for key, value in row.items()}
        if self.predicate is not None and not bool(self.predicate(dict(materialized))):
            return False
        return all(materialized.get(key) == value for key, value in self.equals.items())

    def select_positions(self, rows: list[dict[str, Any]]) -> list[int]:
        return [
            index
            for index, row in enumerate(rows)
            if self.matches(row)
        ]


def compile_metadata_filter(
    *,
    predicate: Callable[[dict[str, object]], bool] | None = None,
    **equals: object,
) -> MetadataFilter:
    return MetadataFilter(equals=dict(equals), predicate=predicate)


def filter_rows(rows: list[dict[str, Any]], **equals: Any) -> list[int]:
    return compile_metadata_filter(**equals).select_positions(rows)


__all__ = ["MetadataFilter", "compile_metadata_filter", "filter_rows"]
