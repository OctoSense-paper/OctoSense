"""Canonical sequential composition kernel."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import torch.nn as nn

from octosense.core.describe import Describable, DescribeNode
from octosense.io.tensor import RadioTensor
from octosense.transforms.core.base import BaseTransform


class Sequential(nn.Module, Describable):
    """Sequential composition of transforms."""

    def __init__(self, transforms: list[nn.Module]):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(self, x: RadioTensor | Mapping[str, Any]) -> Any:
        payload_mode = isinstance(x, Mapping)
        current: Any = dict(x) if payload_mode else cast(RadioTensor, x)

        for transform in self.transforms:
            if isinstance(current, dict):
                if isinstance(transform, BaseTransform):
                    signal = current.get("signal")
                    if not isinstance(signal, RadioTensor):
                        raise TypeError(
                            "Sequential expected payload['signal'] to be a RadioTensor "
                            f"before applying {transform.__class__.__name__}, got {type(signal)!r}"
                        )
                    next_payload = dict(current)
                    next_payload["signal"] = cast(BaseTransform, transform)(signal)
                    current = next_payload
                    continue

                produced = transform(current)
                if isinstance(produced, Mapping):
                    current = dict(produced)
                    continue

                if "signal" not in current:
                    raise TypeError(
                        f"Dictionary-mode Sequential expects {transform.__class__.__name__} "
                        "to return a mapping when payloads do not expose a 'signal' slot."
                    )
                next_payload = dict(current)
                next_payload["signal"] = produced
                current = next_payload
                continue

            if isinstance(transform, BaseTransform):
                current = cast(BaseTransform, transform)(current)
                continue

            produced = transform(cast(RadioTensor, current))
            if isinstance(produced, Mapping):
                if "signal" not in produced:
                    raise TypeError(
                        "Sequential expected dictionary-mode transforms to preserve a "
                        "'signal' entry when invoked with a RadioTensor input."
                    )
                current = produced["signal"]
                continue
            current = produced

        if payload_mode:
            return cast(dict[str, Any], current)

        if isinstance(current, dict):
            if "signal" not in current:
                raise TypeError(
                    "Sequential expected dictionary-mode transforms to preserve a "
                    "'signal' entry when invoked with a RadioTensor input."
                )
            return current["signal"]
        return current

    def __repr__(self) -> str:
        transform_names = [t.__class__.__name__ for t in self.transforms]
        return f"Sequential({' -> '.join(transform_names)})"

    def describe_tree(self) -> DescribeNode:
        children: list[DescribeNode] = []
        for index, transform in enumerate(self.transforms):
            fields: dict[str, Any] = {
                "index": int(index),
                "class_name": transform.__class__.__name__,
            }
            to_dict = getattr(transform, "to_dict", None)
            if callable(to_dict):
                payload = to_dict()
                if isinstance(payload, Mapping):
                    transform_name = payload.get("operator_id")
                    if transform_name is not None:
                        fields["transform_id"] = str(transform_name)
                    params = payload.get("params")
                    if isinstance(params, Mapping) and params:
                        fields["params"] = {
                            str(key): value
                            for key, value in params.items()
                        }
            children.append(
                DescribeNode(
                    kind="transform_step",
                    name=f"{index}:{transform.__class__.__name__}",
                    fields=fields,
                )
            )
        return DescribeNode(
            kind="transform_pipeline",
            name="transform",
            fields={"step_count": len(self.transforms)},
            children=tuple(children),
        )
__all__ = ["Sequential"]
