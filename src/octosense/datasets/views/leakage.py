"""Metadata overlap helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LeakageReport:
    fields: tuple[str, ...]
    overlapping_keys: tuple[tuple[object, ...], ...]

    @property
    def has_leakage(self) -> bool:
        return bool(self.overlapping_keys)

    def to_dict(self) -> dict[str, object]:
        return {
            "fields": list(self.fields),
            "has_leakage": self.has_leakage,
            "overlapping_keys": [list(item) for item in self.overlapping_keys],
        }


def normalize_leakage_fields(
    fields: Sequence[str],
    *,
    owner: str,
) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for field in fields:
        candidate = str(field).strip()
        if not candidate:
            raise ValueError(f"{owner} leakage fields must be non-empty strings.")
        if candidate in seen:
            raise ValueError(
                f"{owner} leakage fields must be unique after normalization; got duplicate {candidate!r}."
            )
        seen.add(candidate)
        normalized.append(candidate)
    return tuple(normalized)


def _row_overlap_key(
    row: dict[str, Any],
    *,
    fields: tuple[str, ...],
) -> tuple[object, ...]:
    return tuple(row.get(field) for field in fields)


def _available_metadata_fields(rows: list[dict[str, Any]]) -> set[str]:
    return {
        str(field_name)
        for row in rows
        for field_name in row
        if str(field_name).strip()
    }


def _normalize_declared_metadata_fields(
    fields: Sequence[str] | None,
) -> tuple[str, ...]:
    if fields is None:
        return ()
    normalized: list[str] = []
    for field in fields:
        candidate = str(field).strip()
        if candidate and candidate not in normalized:
            normalized.append(candidate)
    return tuple(normalized)


def _validate_overlap_fields_exist(
    rows: list[dict[str, Any]],
    *,
    fields: tuple[str, ...],
    side: str,
    declared_fields: Sequence[str] | None = None,
) -> None:
    available_fields = set(_normalize_declared_metadata_fields(declared_fields))
    if not available_fields:
        available_fields = _available_metadata_fields(rows)
    if not available_fields:
        return
    missing_fields = tuple(field for field in fields if field not in available_fields)
    if not missing_fields:
        return
    missing_text = ", ".join(repr(field) for field in missing_fields)
    available_text = ", ".join(sorted(available_fields))
    raise ValueError(
        f"metadata overlap fields must exist in {side} metadata rows; missing {missing_text}. "
        f"Available fields: {available_text}"
    )


def summarize_metadata_overlap(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    fields: Sequence[str],
    left_declared_fields: Sequence[str] | None = None,
    right_declared_fields: Sequence[str] | None = None,
) -> dict[str, list[object]]:
    report = build_leakage_report(
        left,
        right,
        fields=fields,
        left_declared_fields=left_declared_fields,
        right_declared_fields=right_declared_fields,
    )
    overlaps: dict[str, list[object]] = {}
    for index, field in enumerate(report.fields):
        values = {
            key[index]
            for key in report.overlapping_keys
            if key[index] is not None
        }
        overlaps[field] = sorted(values, key=lambda value: str(value))
    return overlaps


def build_leakage_report(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    fields: Sequence[str],
    left_declared_fields: Sequence[str] | None = None,
    right_declared_fields: Sequence[str] | None = None,
) -> LeakageReport:
    normalized_fields = normalize_leakage_fields(
        fields,
        owner="metadata overlap",
    )
    _validate_overlap_fields_exist(
        left,
        fields=normalized_fields,
        side="left",
        declared_fields=left_declared_fields,
    )
    _validate_overlap_fields_exist(
        right,
        fields=normalized_fields,
        side="right",
        declared_fields=right_declared_fields,
    )
    right_keys = {
        _row_overlap_key(row, fields=normalized_fields)
        for row in right
    }
    overlapping_keys = tuple(
        sorted(
            {
                _row_overlap_key(row, fields=normalized_fields)
                for row in left
                if _row_overlap_key(row, fields=normalized_fields) in right_keys
            },
            key=lambda item: tuple(str(value) for value in item),
        )
    )
    return LeakageReport(fields=normalized_fields, overlapping_keys=overlapping_keys)


def metadata_overlap(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    fields: Sequence[str],
) -> list[dict[str, Any]]:
    report = build_leakage_report(left, right, fields=fields)
    right_keys = set(report.overlapping_keys)
    overlaps: list[dict[str, Any]] = []
    for row in left:
        key = _row_overlap_key(row, fields=report.fields)
        if key in right_keys:
            overlaps.append(dict(row))
    return overlaps


__all__ = [
    "LeakageReport",
    "build_leakage_report",
    "metadata_overlap",
    "normalize_leakage_fields",
    "summarize_metadata_overlap",
]
