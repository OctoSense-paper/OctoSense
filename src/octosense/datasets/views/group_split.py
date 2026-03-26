"""Group split helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _stable_sort_key(value: object) -> tuple[str, str]:
    return type(value).__name__, str(value)


def _normalize_group_value(value: object, width: int) -> tuple[object, ...]:
    if width == 1:
        return (value,)
    if isinstance(value, tuple) and len(value) == width:
        return value
    raise ValueError(
        f"Expected group assignment with width={width}, got {value!r}"
    )


@dataclass(frozen=True)
class GroupSplitPlan:
    fields: tuple[str, ...]
    ratio: float = 0.8
    left_values: tuple[tuple[object, ...], ...] | None = None
    right_values: tuple[tuple[object, ...], ...] | None = None


@dataclass(frozen=True)
class GroupSplitResult:
    left_positions: tuple[int, ...]
    right_positions: tuple[int, ...]
    left_groups: tuple[tuple[object, ...], ...]
    right_groups: tuple[tuple[object, ...], ...]


def resolve_group_split(
    rows: list[dict[str, Any]],
    *,
    plan: GroupSplitPlan,
) -> GroupSplitResult:
    if not rows:
        return GroupSplitResult((), (), (), ())
    if not plan.fields:
        raise ValueError("GroupSplitPlan.fields must not be empty")

    grouped_positions: dict[tuple[object, ...], list[int]] = {}
    for index, row in enumerate(rows):
        key = tuple(row.get(field) for field in plan.fields)
        if any(value is None for value in key):
            raise ValueError(
                f"group_split requires every row to define {plan.fields}, got missing values in row {index}"
            )
        grouped_positions.setdefault(key, []).append(index)

    ordered_groups = sorted(grouped_positions, key=lambda item: tuple(_stable_sort_key(value) for value in item))
    if len(ordered_groups) == 1:
        raise ValueError("group_split requires at least two distinct metadata groups")

    width = len(plan.fields)
    if plan.left_values is None and plan.right_values is None:
        if not 0.0 < float(plan.ratio) < 1.0:
            raise ValueError(f"ratio must be between 0 and 1, got {plan.ratio}")
        cutoff = int(round(len(ordered_groups) * float(plan.ratio)))
        cutoff = max(1, min(len(ordered_groups) - 1, cutoff))
        left_groups = tuple(ordered_groups[:cutoff])
        right_groups = tuple(ordered_groups[cutoff:])
    else:
        left_groups = (
            tuple(_normalize_group_value(value, width) for value in plan.left_values)
            if plan.left_values is not None
            else None
        )
        right_groups = (
            tuple(_normalize_group_value(value, width) for value in plan.right_values)
            if plan.right_values is not None
            else None
        )
        if left_groups is None:
            right_group_set = set(right_groups or ())
            left_groups = tuple(group for group in ordered_groups if group not in right_group_set)
        elif right_groups is None:
            left_group_set = set(left_groups)
            right_groups = tuple(group for group in ordered_groups if group not in left_group_set)
        else:
            overlap = set(left_groups) & set(right_groups)
            if overlap:
                overlap_list = sorted(overlap, key=lambda item: tuple(str(value) for value in item))
                raise ValueError(
                    f"group_split received overlapping group assignments: {overlap_list}"
                )
            assigned = set(left_groups) | set(right_groups)
            missing = [group for group in ordered_groups if group not in assigned]
            if missing:
                raise ValueError(
                    f"group_split requires explicit coverage of every group, missing: {missing}"
                )
        assert right_groups is not None

    left_group_set = set(left_groups)
    left_positions = tuple(
        index
        for group, positions in grouped_positions.items()
        if group in left_group_set
        for index in positions
    )
    right_positions = tuple(
        index
        for group, positions in grouped_positions.items()
        if group not in left_group_set
        for index in positions
    )
    if not left_positions or not right_positions:
        raise ValueError("group_split requires both partitions to contain at least one group")
    return GroupSplitResult(
        left_positions=left_positions,
        right_positions=right_positions,
        left_groups=left_groups,
        right_groups=tuple(group for group in ordered_groups if group not in left_group_set),
    )


def group_split(rows: list[dict[str, Any]], by: tuple[str, ...], ratio: float) -> tuple[list[int], list[int]]:
    result = resolve_group_split(rows, plan=GroupSplitPlan(fields=by, ratio=ratio))
    return list(result.left_positions), list(result.right_positions)


__all__ = ["GroupSplitPlan", "GroupSplitResult", "group_split", "resolve_group_split"]
