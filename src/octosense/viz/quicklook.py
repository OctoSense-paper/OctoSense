"""Lightweight tensor inspection helpers."""

from __future__ import annotations

from typing import TypedDict

from octosense.io.tensor import RadioTensor
from octosense.viz.axes import get_axis_info


class AxisQuicklook(TypedDict):
    label: str
    size: int


class QuicklookPayload(TypedDict):
    axes: tuple[str, ...]
    shape: tuple[int, ...]
    axes_info: dict[str, AxisQuicklook]


def quicklook(tensor: RadioTensor) -> QuicklookPayload:
    if not isinstance(tensor, RadioTensor):
        raise TypeError("quicklook expects a RadioTensor")

    axes = tensor.axis_schema.axes
    payload: QuicklookPayload = {
        "axes": tuple(axes),
        "shape": tuple(int(dim) for dim in tensor.shape),
        "axes_info": {},
    }
    for axis_name in axes:
        values, label = get_axis_info(tensor, axis_name)
        payload["axes_info"][axis_name] = {"label": label, "size": int(len(values))}
    return payload
