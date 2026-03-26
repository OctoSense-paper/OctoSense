"""Tree-structured describe primitives for OctoSense introspection."""

from __future__ import annotations

import json
import operator
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _normalize_describe_value(value: Any) -> Any:
    """Normalize values so describe trees stay JSON serializable and stable."""
    if isinstance(value, DescribeNode):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _normalize_describe_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize_describe_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Size):
        return [int(item) for item in value]
    if isinstance(value, (torch.dtype, torch.device, Path)):
        return str(value)
    return value


def _truncate_describe_value(
    value: Any,
    *,
    max_items: int = 12,
    max_string: int = 120,
) -> Any:
    if isinstance(value, str):
        if len(value) <= max_string:
            return value
        return f"{value[: max_string - 3]}..."
    if isinstance(value, list):
        if len(value) <= max_items:
            return [_truncate_describe_value(item, max_items=max_items, max_string=max_string) for item in value]
        head = max_items // 2
        tail = max_items - head - 1
        return [
            *[_truncate_describe_value(item, max_items=max_items, max_string=max_string) for item in value[:head]],
            f"... ({len(value) - head - tail} more)",
            *[_truncate_describe_value(item, max_items=max_items, max_string=max_string) for item in value[-tail:]],
        ]
    if isinstance(value, dict):
        items = list(value.items())
        if len(items) <= max_items:
            return {
                key: _truncate_describe_value(item, max_items=max_items, max_string=max_string)
                for key, item in items
            }
        head = max_items // 2
        tail = max_items - head - 1
        truncated: dict[str, Any] = {
            str(key): _truncate_describe_value(item, max_items=max_items, max_string=max_string)
            for key, item in items[:head]
        }
        truncated["..."] = f"{len(items) - head - tail} more"
        for key, item in items[-tail:]:
            truncated[str(key)] = _truncate_describe_value(item, max_items=max_items, max_string=max_string)
        return truncated
    return value


def _format_describe_value(value: Any, *, verbose: bool = False) -> str:
    normalized = _normalize_describe_value(value)
    if not verbose:
        normalized = _truncate_describe_value(normalized)
    if isinstance(normalized, str):
        return normalized
    return json.dumps(normalized, ensure_ascii=True)


@dataclass(frozen=True)
class DescribeNode:
    """Structured describe node with stable text and JSON rendering."""

    kind: str
    name: str
    fields: dict[str, Any] = field(default_factory=dict)
    children: tuple["DescribeNode", ...] = ()

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("DescribeNode.kind cannot be empty")
        if not self.name:
            raise ValueError("DescribeNode.name cannot be empty")
        object.__setattr__(self, "fields", {str(key): value for key, value in self.fields.items()})
        normalized_children = tuple(
            child if isinstance(child, DescribeNode) else DescribeNode.from_dict(child)
            for child in self.children
        )
        object.__setattr__(self, "children", normalized_children)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "fields": _normalize_describe_value(self.fields),
            "children": [child.to_dict() for child in self.children],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DescribeNode":
        return cls(
            kind=str(payload["kind"]),
            name=str(payload["name"]),
            fields=dict(payload.get("fields", {})),
            children=tuple(
                cls.from_dict(child)
                for child in payload.get("children", [])
            ),
        )

    def with_name(self, name: str) -> "DescribeNode":
        return DescribeNode(kind=self.kind, name=name, fields=dict(self.fields), children=self.children)

    def with_kind(self, kind: str) -> "DescribeNode":
        return DescribeNode(kind=kind, name=self.name, fields=dict(self.fields), children=self.children)

    def append_child(self, child: "DescribeNode") -> "DescribeNode":
        return DescribeNode(
            kind=self.kind,
            name=self.name,
            fields=dict(self.fields),
            children=(*self.children, child),
        )

    def child(self, name: str) -> "DescribeNode":
        for child in self.children:
            if child.name == name:
                return child
        available = ", ".join(node.name for node in self.children)
        raise KeyError(f"DescribeNode child '{name}' not found. Available children: {available}")

    def render(self, *, depth: int | None = None, verbose: bool = False) -> str:
        """Render the node as a stable ASCII tree."""
        if depth is not None:
            try:
                depth = operator.index(depth)
            except TypeError as exc:
                raise TypeError("DescribeNode.render(depth=...) expects an int >= 0") from exc
            if depth < 0:
                raise ValueError("DescribeNode.render(depth=...) expects an int >= 0")

        lines = [f"{self.name} [{self.kind}]"]

        def _render_entry(
            prefix: str,
            is_last: bool,
            entry: tuple[str, Any, Any],
            *,
            current_depth: int,
        ) -> list[str]:
            branch = "`- " if is_last else "|- "
            next_prefix = prefix + ("   " if is_last else "|  ")
            entry_type, key, payload = entry
            if entry_type == "field":
                return [f"{prefix}{branch}{key}: {_format_describe_value(payload, verbose=verbose)}"]
            child = payload
            child_lines = [f"{prefix}{branch}{child.name} [{child.kind}]"]
            if depth is not None and current_depth + 1 >= depth:
                return child_lines
            child_entries = _node_entries(child)
            for index, child_entry in enumerate(child_entries):
                child_lines.extend(
                    _render_entry(
                        next_prefix,
                        index == len(child_entries) - 1,
                        child_entry,
                        current_depth=current_depth + 1,
                    )
                )
            return child_lines

        def _node_entries(node: "DescribeNode") -> list[tuple[str, Any, Any]]:
            entries: list[tuple[str, Any, Any]] = [
                ("field", key, value) for key, value in node.fields.items()
            ]
            entries.extend(("child", child.name, child) for child in node.children)
            return entries

        if depth is None or depth > 0:
            entries = _node_entries(self)
            for index, entry in enumerate(entries):
                lines.extend(
                    _render_entry(
                        "",
                        index == len(entries) - 1,
                        entry,
                        current_depth=0,
                    )
                )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()


def ensure_describe_node(value: DescribeNode | Mapping[str, Any]) -> DescribeNode:
    if isinstance(value, DescribeNode):
        return value
    return DescribeNode.from_dict(value)


class Describable(ABC):
    """Mixin for OctoSense objects that expose a tree-structured description."""

    @abstractmethod
    def describe_tree(self) -> DescribeNode:
        """Return the structured describe tree for this object."""

    def describe(self, *, depth: int | None = None, verbose: bool = False) -> str:
        return self.describe_tree().render(depth=depth, verbose=verbose)


__all__ = ["Describable", "DescribeNode", "ensure_describe_node"]
