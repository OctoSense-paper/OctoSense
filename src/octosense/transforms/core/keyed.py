"""Canonical dictionary-based transform entry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch.nn as nn

from octosense.io.tensor import RadioTensor

if TYPE_CHECKING:
    from octosense.transforms.core.base import BaseTransform


class KeyedTransform(nn.Module):
    """Apply one ``BaseTransform`` to designated keys of a dictionary payload."""

    def __init__(
        self,
        transform: BaseTransform,
        *,
        keys: Sequence[str],
        output_keys: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        if not keys:
            raise ValueError("KeyedTransform requires at least one input key")
        if output_keys is not None and len(output_keys) != len(keys):
            raise ValueError("output_keys must match keys length when provided")
        self.transform = transform
        self.keys = tuple(str(key) for key in keys)
        self.output_keys = (
            tuple(str(key) for key in output_keys)
            if output_keys is not None
            else self.keys
        )

    @property
    def requires(self) -> dict[str, object]:
        return {
            "payload_keys": list(self.keys),
            "per_key": {key: self.transform.requires for key in self.keys},
        }

    @property
    def updates(self) -> dict[str, object]:
        update_payload: dict[str, object] = {
            "payload_keys": list(self.output_keys),
            "per_key": {
                output_key: self.transform.updates
                for output_key in self.output_keys
            },
        }
        if self.keys != self.output_keys:
            update_payload["key_mapping"] = {
                src: dst
                for src, dst in zip(self.keys, self.output_keys, strict=True)
            }
        return update_payload

    def semantic_contract(self) -> dict[str, object]:
        return {
            "requires": self.requires,
            "updates": self.updates,
        }

    def forward(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        output = dict(payload)
        for input_key, output_key in zip(self.keys, self.output_keys, strict=True):
            if input_key not in payload:
                raise KeyError(
                    f"Dictionary payload is missing required key '{input_key}'. "
                    f"Available keys: {sorted(payload.keys())}"
                )
            value = payload[input_key]
            if not isinstance(value, RadioTensor):
                raise TypeError(
                    f"KeyedTransform expects payload['{input_key}'] to be a RadioTensor, "
                    f"got {type(value)!r}"
                )
            output[output_key] = self.transform(value)
        return output


class DictSequential(nn.Module):
    """Compose multiple dictionary-based transforms."""

    def __init__(self, transforms: Sequence[nn.Module]) -> None:
        super().__init__()
        self.transforms = nn.ModuleList(list(transforms))

    def forward(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        current = dict(payload)
        for transform in self.transforms:
            current = dict(transform(current))
        return current

    @property
    def requires(self) -> dict[str, object]:
        if not self.transforms:
            return {}
        first = self.transforms[0]
        requires = getattr(first, "requires", None)
        return dict(requires) if isinstance(requires, dict) else {}

    @property
    def updates(self) -> dict[str, object]:
        if not self.transforms:
            return {}
        last = self.transforms[-1]
        updates = getattr(last, "updates", None)
        return dict(updates) if isinstance(updates, dict) else {}


__all__ = ["DictSequential", "KeyedTransform"]
